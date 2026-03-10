"""
data_loader.py
Complete data pipeline for the NASA CMAPSS turbofan engine dataset (FD001 subset).
Downloads automatically if not present, parses train/test files, computes RUL,
normalises sensor readings, and returns clean DataFrames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger(__name__)

# NASA CMAPSS column names
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS  = [f"s{i}" for i in range(1, 22)]
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]   # near-zero variance
FEATURE_COLS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]

ALL_COLS = ["engine_id", "cycle"] + SETTING_COLS + SENSOR_COLS

_BASE_URL = (
    "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"
)
_DATASET_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]


class DataLoader:
    """
    Loads and preprocesses the NASA CMAPSS FD001 turbofan dataset.

    Parameters
    ----------
    data_dir : directory to store raw + processed files
    seq_len  : sliding window length (for LSTM sequences)
    max_rul  : clip RUL at this ceiling (piecewise-linear target)
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        seq_len:  int = 30,
        max_rul:  int = 125,
    ):
        self.data_dir  = Path(data_dir)
        self.seq_len   = seq_len
        self.max_rul   = max_rul
        self._scaler   = MinMaxScaler()
        self._scaler_path = self.data_dir / "scaler.pkl"

    # ── Public API ──────────────────────────────────────────────────────────

    def load(self, subset: str = "FD001") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download (if needed), parse, compute RUL, and normalise.

        Returns
        -------
        train_df, test_df  — clean DataFrames with columns:
            engine_id, cycle, s2, s3, s4, s7, s8, s9, s11, s12, s13, s14,
            s15, s17, s20, s21, RUL
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_downloaded(subset)

        train_df = self._parse_file(f"train_{subset}.txt")
        test_df  = self._parse_file(f"test_{subset}.txt")

        train_df = self._compute_rul(train_df)
        test_df  = self._compute_rul_test(test_df, subset)

        # Fit scaler on train, transform both
        train_df[FEATURE_COLS] = self._scaler.fit_transform(train_df[FEATURE_COLS])
        test_df[FEATURE_COLS]  = self._scaler.transform(test_df[FEATURE_COLS])

        joblib.dump(self._scaler, self._scaler_path)
        logger.info("Scaler saved to %s", self._scaler_path)
        logger.info(
            "Loaded %s — train: %d rows, %d engines | test: %d rows, %d engines",
            subset,
            len(train_df), train_df["engine_id"].nunique(),
            len(test_df),  test_df["engine_id"].nunique(),
        )
        return train_df, test_df

    def load_scaler(self) -> MinMaxScaler:
        """Load scaler from disk (after load() has been called at least once)."""
        if self._scaler_path.exists():
            self._scaler = joblib.load(self._scaler_path)
        return self._scaler

    def get_feature_names(self) -> List[str]:
        return FEATURE_COLS

    # ── LSTM sequence helpers (used by LSTMModel) ───────────────────────────

    def make_sequences(
        self, df: pd.DataFrame, last_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sliding-window sequences from a DataFrame.

        Returns X (N, seq_len, n_features) and y (N,).
        """
        xs, ys = [], []
        for _, grp in df.groupby("engine_id"):
            feats = grp[FEATURE_COLS].values
            ruls  = grp["RUL"].values
            if last_only:
                if len(feats) >= self.seq_len:
                    xs.append(feats[-self.seq_len:])
                    ys.append(ruls[-1])
            else:
                for i in range(self.seq_len, len(feats) + 1):
                    xs.append(feats[i - self.seq_len : i])
                    ys.append(ruls[i - 1])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    # ── Download ─────────────────────────────────────────────────────────────

    def _ensure_downloaded(self, subset: str) -> None:
        train_path = self.data_dir / f"train_{subset}.txt"
        if train_path.exists():
            return

        logger.info("Downloading NASA CMAPSS %s …", subset)
        for filename in _DATASET_FILES:
            dest = self.data_dir / filename
            if dest.exists():
                continue
            url = f"{_BASE_URL}/{filename}"
            logger.info("  GET %s", url)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info("  Saved → %s", dest)

        logger.info("Download complete.")

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _parse_file(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=ALL_COLS,
            engine="python",
        )
        # Drop near-constant operational settings and low-variance sensors
        df = df.drop(columns=SETTING_COLS + DROP_SENSORS)
        return df

    # ── RUL computation ──────────────────────────────────────────────────────

    def _compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Training RUL: max_cycle − current_cycle, clipped at max_rul."""
        max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycle, on="engine_id")
        df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=self.max_rul)
        return df.drop(columns=["max_cycle"])

    def _compute_rul_test(self, df: pd.DataFrame, subset: str) -> pd.DataFrame:
        """
        Test RUL: uses the RUL_{subset}.txt ground-truth file when present,
        otherwise falls back to 0 for all engines.
        """
        rul_path = self.data_dir / f"RUL_{subset}.txt"
        max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycle, on="engine_id")

        if rul_path.exists():
            rul_series = pd.read_csv(rul_path, header=None).squeeze()
            unit_rul   = {u + 1: int(r) for u, r in enumerate(rul_series)}
            df["RUL"] = df.apply(
                lambda r: min(
                    unit_rul.get(int(r["engine_id"]), 0) + (r["max_cycle"] - r["cycle"]),
                    self.max_rul,
                ),
                axis=1,
            )
        else:
            df["RUL"] = 0

        return df.drop(columns=["max_cycle"])
