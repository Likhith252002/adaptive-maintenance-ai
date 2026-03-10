"""
routes.py
REST API endpoints + WebSocket endpoint.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Health ─────────────────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok", "service": "adaptive-maintenance-ai"}


# ── Metrics ────────────────────────────────────────────────────────────────

@router.get("/api/v1/metrics/latest", tags=["metrics"])
async def get_latest_metrics(request: Request):
    """Return the most recent health metrics for all sensors."""
    # In production, fetch from a short-lived cache or time-series DB
    return {"message": "Connect via WebSocket at /ws for real-time metrics."}


@router.get("/api/v1/sensors", tags=["sensors"])
async def list_sensors():
    """List all active sensor IDs."""
    return {"sensors": [f"sensor_{i:02d}" for i in range(1, 4)]}


# ── Alerts ─────────────────────────────────────────────────────────────────

@router.get("/api/v1/alerts", tags=["alerts"])
async def get_alerts(request: Request, limit: int = 20):
    """Return recent alerts from AlertAgent."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(503, "Orchestrator not initialised")
    alerts = orchestrator.alerter.get_recent_alerts(n=limit)
    return {"alerts": [a.dict() for a in alerts]}


# ── Drift ──────────────────────────────────────────────────────────────────

@router.get("/api/v1/drift/status", tags=["drift"])
async def get_drift_status():
    """Return current drift detection status."""
    return {
        "drift_detected": False,
        "last_check":     None,
        "detail":         "Connect via WebSocket for live drift events.",
    }


# ── Retraining ─────────────────────────────────────────────────────────────

class RetrainRequest(BaseModel):
    force: bool = False
    reason: str = "manual"


@router.post("/api/v1/retrain", tags=["model"])
async def trigger_retrain(request: Request, body: RetrainRequest):
    """Manually trigger a model retraining cycle."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(503, "Orchestrator not initialised")
    summary = await orchestrator.retrainer._retrain()
    return {"status": "triggered", "summary": summary}


# ── WebSocket ──────────────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, request: Request):
    """
    Persistent WebSocket connection.
    Clients receive real-time JSON messages:
      { "type": "metrics" | "alert" | "drift", "data": {...} }
    """
    ws_manager = getattr(request.app.state, "ws_manager", None)
    if ws_manager is None:
        await websocket.close(code=1011)
        return

    await ws_manager.connect(websocket)
    logger.info("WebSocket client connected. Total: %d", ws_manager.count())
    try:
        while True:
            # Keep-alive: accept ping messages from client
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected. Total: %d", ws_manager.count())
