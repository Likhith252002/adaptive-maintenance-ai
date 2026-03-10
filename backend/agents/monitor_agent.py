"""
monitor_agent.py
Continuously monitors incoming sensor streams, computes health metrics,
and triggers downstream agents when anomalies or drift are detected.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SensorReading(BaseModel):
    timestamp: datetime
    sensor_id: str
    values: Dict[str, float]
    source: str = "stream"


class HealthMetrics(BaseModel):
    sensor_id: str
    timestamp: datetime
    rul_estimate: float          # Remaining Useful Life (cycles)
    health_score: float          # 0.0 (failed) → 1.0 (healthy)
    anomaly_score: float         # 0.0 (normal) → 1.0 (severe anomaly)
    drift_detected: bool
    confidence: float


class MonitorAgent:
    """
    LangGraph-compatible agent node that ingests sensor readings,
    runs the LSTM + anomaly detector, and emits HealthMetrics.

    State keys consumed : "latest_reading"
    State keys produced : "health_metrics", "alert_required", "retrain_required"
    """

    ALERT_THRESHOLD   = 0.75   # anomaly_score above this triggers alert
    RETRAIN_THRESHOLD = 0.60   # health_score below this triggers retraining
    LOW_RUL_THRESHOLD = 50.0   # cycles — low RUL triggers alert regardless

    def __init__(self, lstm_model=None, anomaly_detector=None, drift_detector=None):
        self.lstm_model       = lstm_model
        self.anomaly_detector = anomaly_detector
        self.drift_detector   = drift_detector
        self._window: List[np.ndarray] = []
        self._window_size = 30

    # ── Public API ──────────────────────────────────────────────────────────

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node entry point."""
        reading: SensorReading = state.get("latest_reading")
        if reading is None:
            logger.warning("MonitorAgent: no reading in state")
            return state

        metrics = await self._process_reading(reading)

        return {
            **state,
            "health_metrics":    metrics,
            "alert_required":    self._should_alert(metrics),
            "retrain_required":  self._should_retrain(metrics),
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _process_reading(self, reading: SensorReading) -> HealthMetrics:
        feature_vec = np.array(list(reading.values.values()), dtype=np.float32)
        self._window.append(feature_vec)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        rul          = self._estimate_rul(feature_vec)
        anomaly_score = self._compute_anomaly_score(feature_vec)
        health_score  = max(0.0, 1.0 - anomaly_score * 0.8)
        drift_flag    = self._check_drift(feature_vec)

        return HealthMetrics(
            sensor_id     = reading.sensor_id,
            timestamp     = reading.timestamp,
            rul_estimate  = rul,
            health_score  = round(health_score, 4),
            anomaly_score = round(anomaly_score, 4),
            drift_detected= drift_flag,
            confidence    = 0.90,
        )

    def _estimate_rul(self, vec: np.ndarray) -> float:
        if self.lstm_model is not None:
            return float(self.lstm_model.predict(vec))
        # Placeholder heuristic
        degradation = float(np.clip(np.mean(np.abs(vec)), 0, 1))
        return round(max(0.0, 300.0 * (1.0 - degradation)), 2)

    def _compute_anomaly_score(self, vec: np.ndarray) -> float:
        if self.anomaly_detector is not None:
            return float(self.anomaly_detector.score(vec))
        return float(np.clip(np.random.normal(0.1, 0.05), 0, 1))

    def _check_drift(self, vec: np.ndarray) -> bool:
        if self.drift_detector is not None and len(self._window) == self._window_size:
            return self.drift_detector.detect(np.array(self._window))
        return False

    def _should_alert(self, m: HealthMetrics) -> bool:
        return m.anomaly_score > self.ALERT_THRESHOLD or m.rul_estimate < self.LOW_RUL_THRESHOLD

    def _should_retrain(self, m: HealthMetrics) -> bool:
        return m.health_score < self.RETRAIN_THRESHOLD or m.drift_detected
