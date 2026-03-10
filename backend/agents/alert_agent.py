"""
alert_agent.py
Generates human-readable alerts using an LLM (Claude via langchain-anthropic)
and broadcasts them over WebSocket to the frontend dashboard.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Alert(BaseModel):
    id: str
    sensor_id: str
    severity: str           # "low" | "medium" | "high" | "critical"
    title: str
    message: str
    recommended_action: str
    timestamp: datetime
    rul_estimate: Optional[float] = None
    anomaly_score: Optional[float] = None


SEVERITY_MAP = {
    (0.0, 0.4):  "low",
    (0.4, 0.65): "medium",
    (0.65, 0.85):"high",
    (0.85, 1.0): "critical",
}


class AlertAgent:
    """
    LangGraph-compatible agent node that generates and dispatches alerts.

    State keys consumed : "alert_required", "health_metrics"
    State keys produced : "active_alerts"
    """

    def __init__(self, websocket_manager=None, llm=None):
        self.ws_manager  = websocket_manager
        self.llm         = llm
        self._alert_log: List[Alert] = []

    # ── Public API ──────────────────────────────────────────────────────────

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph node entry point."""
        if not state.get("alert_required", False):
            return {**state, "active_alerts": self._alert_log[-10:]}

        metrics = state.get("health_metrics")
        if metrics is None:
            return state

        alert = await self._generate_alert(metrics)
        self._alert_log.append(alert)

        if self.ws_manager:
            await self.ws_manager.broadcast({"type": "alert", "data": alert.dict()})

        logger.warning("AlertAgent [%s] %s — %s", alert.severity.upper(),
                       alert.sensor_id, alert.title)

        return {**state, "active_alerts": self._alert_log[-10:]}

    def get_recent_alerts(self, n: int = 20) -> List[Alert]:
        return self._alert_log[-n:]

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _generate_alert(self, metrics) -> Alert:
        severity = self._classify_severity(metrics.anomaly_score)
        title, message, action = await self._compose_message(metrics, severity)

        return Alert(
            id=f"alert-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            sensor_id=metrics.sensor_id,
            severity=severity,
            title=title,
            message=message,
            recommended_action=action,
            timestamp=metrics.timestamp,
            rul_estimate=metrics.rul_estimate,
            anomaly_score=metrics.anomaly_score,
        )

    def _classify_severity(self, score: float) -> str:
        for (lo, hi), label in SEVERITY_MAP.items():
            if lo <= score < hi:
                return label
        return "critical"

    async def _compose_message(self, metrics, severity: str):
        """Use LLM if available, otherwise use rule-based templates."""
        if self.llm:
            prompt = (
                f"Sensor {metrics.sensor_id} has anomaly_score={metrics.anomaly_score:.2f}, "
                f"RUL={metrics.rul_estimate:.0f} cycles, health={metrics.health_score:.2f}. "
                f"Generate a concise maintenance alert title, message, and recommended action."
            )
            try:
                response = await self.llm.ainvoke(prompt)
                lines = response.content.strip().split("\n")
                return (
                    lines[0] if len(lines) > 0 else "Anomaly Detected",
                    lines[1] if len(lines) > 1 else prompt,
                    lines[2] if len(lines) > 2 else "Inspect sensor immediately.",
                )
            except Exception as exc:
                logger.warning("AlertAgent LLM call failed: %s", exc)

        # Rule-based fallback
        templates = {
            "low":      ("Minor Deviation Detected",
                         f"Sensor {metrics.sensor_id} shows slight irregularity (score={metrics.anomaly_score:.2f}).",
                         "Monitor over next 24 hours."),
            "medium":   ("Elevated Anomaly — Attention Required",
                         f"Sensor {metrics.sensor_id} anomaly_score={metrics.anomaly_score:.2f}, RUL={metrics.rul_estimate:.0f}.",
                         "Schedule inspection within 72 hours."),
            "high":     ("High Anomaly — Maintenance Recommended",
                         f"Sensor {metrics.sensor_id} degrading rapidly. RUL={metrics.rul_estimate:.0f} cycles.",
                         "Schedule maintenance within 24 hours."),
            "critical": ("CRITICAL FAILURE RISK",
                         f"Sensor {metrics.sensor_id} at critical state. Immediate intervention required.",
                         "Take equipment offline and inspect NOW."),
        }
        return templates[severity]
