"""
anomaly_detector.py
Isolation Forest-based anomaly detector with online partial-fit capability.
Returns a normalised anomaly score in [0, 1].
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Wraps sklearn IsolationForest with:
    - fit / partial_update (re-fit on new data window)
    - score  → float in [0, 1]  (higher = more anomalous)
    - threshold tuning via contamination parameter
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators:  int   = 100,
        random_state:  int   = 42,
    ):
        self.contamination = contamination
        self._model  = IsolationForest(
            contamination = contamination,
            n_estimators  = n_estimators,
            random_state  = random_state,
            n_jobs        = -1,
        )
        self._scaler  = StandardScaler()
        self._fitted  = False
        self._buffer  = []
        self._buf_max = 5000

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Initial fit on baseline (normal) data."""
        X_s = self._scaler.fit_transform(X)
        self._model.fit(X_s)
        self._fitted = True
        logger.info("AnomalyDetector fitted on %d samples", len(X))
        return self

    def partial_update(self, X_new: np.ndarray) -> "AnomalyDetector":
        """Re-fit on a rolling buffer including new observations."""
        self._buffer.extend(X_new.tolist())
        if len(self._buffer) > self._buf_max:
            self._buffer = self._buffer[-self._buf_max:]
        X_all = np.array(self._buffer)
        return self.fit(X_all)

    # ── Inference ──────────────────────────────────────────────────────────

    def score(self, x: np.ndarray) -> float:
        """
        Return anomaly score in [0, 1].
        IsolationForest decision_function returns negative anomaly scores;
        we invert and normalise.
        """
        if not self._fitted:
            logger.warning("AnomalyDetector not fitted — returning 0.0")
            return 0.0

        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_s   = self._scaler.transform(x)
        raw   = self._model.decision_function(x_s)[0]  # more negative = more anomalous
        # Map to [0, 1]: raw in roughly [-0.5, 0.5]
        score = float(np.clip(0.5 - raw, 0, 1))
        return round(score, 4)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 = normal, -1 = anomaly (sklearn convention)."""
        if not self._fitted:
            return np.ones(len(X), dtype=int)
        X_s = self._scaler.transform(X)
        return self._model.predict(X_s)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import pickle
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler,
                         "fitted": self._fitted}, f)
        logger.info("AnomalyDetector saved → %s", path)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model   = state["model"]
        self._scaler  = state["scaler"]
        self._fitted  = state["fitted"]
        logger.info("AnomalyDetector loaded ← %s", path)
