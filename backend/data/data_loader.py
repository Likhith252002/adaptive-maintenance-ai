"""
data_loader.py
Loads and preprocesses the NASA CMAPSS turbofan degradation dataset
(or any similarly structured CSV) into sliding-window sequences for LSTM training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# NASA CMAPSS operational settings + 21 sensor columns
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS  = [f"s{i}" for i in range(1, 22)]
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]   # near-zero variance
FEATURE_COLS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]


class DataLoader:
    """
    Loads CMAPSS FD001–FD004 train/test text files and produces
    (X, y) numpy arrays suitable for LSTMModel.fit().

    Parameters
    ----------
    data_dir : path containing train_FD00*.txt and test_FD00*.txt
    seq_len  : sliding window length (timesteps)
    max_rul  : clip RUL target at this value (piecewise-linear target)
    """

    def __init__(
        self,
        data_dir: str  = "data/raw",
        seq_len:  int  = 30,
        max_rul:  int  = 125,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len  = seq_len
        self.max_rul  = max_rul
        self._scaler  = MinMaxScaler()

    # ── Public API ──────────────────────────────────────────────────────────

    def load(self, subset: str = "FD001") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns X_train, y_train, X_test, y_test as numpy arrays.
        X shape: (N, seq_len, n_features)
        y shape: (N,)
        """
        train_df = self._read_file(f"train_{subset}.txt")
        test_df  = self._read_file(f"test_{subset}.txt")
        rul_df   = self._read_rul(f"RUL_{subset}.txt")

        train_df = self._add_rul(train_df)
        test_df  = self._add_rul_test(test_df, rul_df)

        train_df[FEATURE_COLS] = self._scaler.fit_transform(train_df[FEATURE_COLS])
        test_df[FEATURE_COLS]  = self._scaler.transform(test_df[FEATURE_COLS])

        X_train, y_train = self._make_sequences(train_df)
        X_test,  y_test  = self._make_sequences(test_df, last_only=True)

        logger.info("Loaded %s — train: %s, test: %s", subset, X_train.shape, X_test.shape)
        return X_train, y_train, X_test, y_test

    def get_feature_names(self) -> List[str]:
        return FEATURE_COLS

    # ── Private helpers ─────────────────────────────────────────────────────

    COLUMNS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS

    def _read_file(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            logger.warning("DataLoader: %s not found — generating synthetic data", path)
            return self._synthetic_df()
        df = pd.read_csv(path, sep=r"\s+", header=None, names=self.COLUMNS)
        return df.drop(columns=SETTING_COLS + DROP_SENSORS)

    def _read_rul(self, filename: str) -> pd.Series:
        path = self.data_dir / filename
        if not path.exists():
            return pd.Series(dtype=float)
        return pd.read_csv(path, header=None, squeeze=True)

    def _add_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycle, on="unit")
        df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=self.max_rul)
        return df.drop(columns=["max_cycle"])

    def _add_rul_test(self, df: pd.DataFrame, rul_series: pd.Series) -> pd.DataFrame:
        max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycle, on="unit")
        if len(rul_series):
            unit_rul = {u+1: r for u, r in enumerate(rul_series)}
            df["RUL"] = df.apply(
                lambda r: min(unit_rul.get(r["unit"], 0) + (r["max_cycle"] - r["cycle"]),
                              self.max_rul), axis=1
            )
        else:
            df["RUL"] = 0
        return df.drop(columns=["max_cycle"])

    def _make_sequences(
        self, df: pd.DataFrame, last_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for _, grp in df.groupby("unit"):
            feats = grp[FEATURE_COLS].values
            ruls  = grp["RUL"].values
            if last_only:
                if len(feats) >= self.seq_len:
                    xs.append(feats[-self.seq_len:])
                    ys.append(ruls[-1])
            else:
                for i in range(self.seq_len, len(feats) + 1):
                    xs.append(feats[i-self.seq_len:i])
                    ys.append(ruls[i-1])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def _synthetic_df(self, n_units: int = 5, max_cycles: int = 200) -> pd.DataFrame:
        """Fallback synthetic data when real dataset is absent."""
        rows = []
        for u in range(1, n_units + 1):
            for c in range(1, max_cycles + 1):
                row = {"unit": u, "cycle": c}
                for s in SENSOR_COLS:
                    row[s] = np.random.randn()
                rows.append(row)
        return pd.DataFrame(rows).drop(columns=SETTING_COLS + DROP_SENSORS, errors="ignore")
