"""
drift_detector.py
Statistical data-drift detection using Evidently (batch) and
a custom Page-Hinkley test for online streaming detection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Page-Hinkley online drift detector ────────────────────────────────────

class PageHinkley:
    """
    Page-Hinkley change-detection test for univariate streams.
    Detects an upward shift in the mean.

    Parameters
    ----------
    delta    : minimal detectable change magnitude
    lambda_  : detection threshold
    alpha    : forgetting factor (for running mean)
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 0.9999):
        self.delta   = delta
        self.lambda_ = lambda_
        self.alpha   = alpha
        self.reset()

    def reset(self) -> None:
        self._n   = 0
        self._sum = 0.0
        self._ph  = 0.0
        self._min_ph = 0.0
        self._mean = 0.0

    def update(self, value: float) -> bool:
        """Update with new observation. Returns True if drift detected."""
        self._n   += 1
        self._mean = self.alpha * self._mean + (1 - self.alpha) * value
        self._sum += value - self._mean - self.delta
        self._ph   = self._sum - min(self._sum, self._ph + self._sum)

        if self._ph > self.lambda_:
            logger.info("PageHinkley: drift detected at sample %d (PH=%.2f)", self._n, self._ph)
            self.reset()
            return True
        return False


# ── Main DriftDetector ────────────────────────────────────────────────────

class DriftDetector:
    """
    Multivariate drift detection combining:
    1. Page-Hinkley test on each feature (online)
    2. Evidently DataDriftPreset on reference vs current windows (batch)

    Parameters
    ----------
    feature_names     : list of feature column names
    reference_window  : number of samples in the reference distribution
    test_window       : number of samples in the current distribution
    drift_threshold   : fraction of drifted features to declare global drift
    """

    def __init__(
        self,
        feature_names:    List[str],
        reference_window: int   = 500,
        test_window:      int   = 100,
        drift_threshold:  float = 0.3,
    ):
        self.feature_names   = feature_names
        self.reference_window = reference_window
        self.test_window      = test_window
        self.drift_threshold  = drift_threshold

        self._ph_detectors: Dict[str, PageHinkley] = {
            f: PageHinkley() for f in feature_names
        }
        self._reference_data: Optional[np.ndarray] = None
        self._drift_count = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def set_reference(self, X: np.ndarray) -> None:
        """Set baseline distribution from initial healthy data."""
        self._reference_data = X[-self.reference_window:]
        logger.info("DriftDetector: reference set (%d samples)", len(self._reference_data))

    def detect(self, window: np.ndarray) -> bool:
        """
        Run both PH and Evidently checks on the latest window.
        Returns True if significant drift is detected.
        """
        ph_drift   = self._check_page_hinkley(window[-1])
        ev_drift   = self._check_evidently(window)
        is_drift   = ph_drift or ev_drift

        if is_drift:
            self._drift_count += 1
            logger.warning("DriftDetector: drift #%d detected (PH=%s, Evidently=%s)",
                           self._drift_count, ph_drift, ev_drift)
        return is_drift

    def get_drift_report(self, current: np.ndarray) -> Dict:
        """Return a detailed drift report dict (uses Evidently if available)."""
        if self._reference_data is None:
            return {"error": "no_reference_set"}
        try:
            return self._evidently_report(current)
        except Exception as exc:
            logger.warning("DriftDetector: Evidently report failed — %s", exc)
            return {"error": str(exc), "drift_count": self._drift_count}

    # ── Private helpers ────────────────────────────────────────────────────

    def _check_page_hinkley(self, latest_row: np.ndarray) -> bool:
        drifted = 0
        for i, feat in enumerate(self.feature_names):
            val = float(latest_row[i]) if i < len(latest_row) else 0.0
            if self._ph_detectors[feat].update(val):
                drifted += 1
        return (drifted / len(self.feature_names)) >= self.drift_threshold

    def _check_evidently(self, window: np.ndarray) -> bool:
        if self._reference_data is None or len(window) < self.test_window:
            return False
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            ref_df = pd.DataFrame(self._reference_data, columns=self.feature_names)
            cur_df = pd.DataFrame(window[-self.test_window:], columns=self.feature_names)
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)
            result = report.as_dict()
            share  = result["metrics"][0]["result"]["share_of_drifted_columns"]
            return share >= self.drift_threshold
        except ImportError:
            return False

    def _evidently_report(self, current: np.ndarray) -> Dict:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        ref_df = pd.DataFrame(self._reference_data, columns=self.feature_names)
        cur_df = pd.DataFrame(current[-self.test_window:], columns=self.feature_names)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        return report.as_dict()
