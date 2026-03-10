"""
main.py
FastAPI application entry point.
Initialises all agents, wires the Orchestrator, and starts the sensor stream.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .websocket_manager import WebSocketManager
from ..agents import AlertAgent, MonitorAgent, Orchestrator, RetrainingAgent
from ..data.stream_simulator import StreamSimulator
from ..models import AnomalyDetector, LSTMModel
from ..drift.drift_detector import DriftDetector
from ..data.data_loader import FEATURE_COLS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Globals (populated during lifespan) ───────────────────────────────────
ws_manager:   WebSocketManager | None  = None
orchestrator: Orchestrator      | None = None
_stream_task: asyncio.Task      | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise models + agents. Shutdown: cancel stream task."""
    global ws_manager, orchestrator, _stream_task

    logger.info("Starting Adaptive Maintenance AI backend …")

    # 1. Shared infrastructure
    ws_manager = WebSocketManager()

    # 2. ML models
    lstm    = LSTMModel(input_size=len(FEATURE_COLS), seq_len=30)
    anomaly = AnomalyDetector(contamination=0.05)
    drift   = DriftDetector(feature_names=FEATURE_COLS)

    # 3. LLM (optional — requires ANTHROPIC_API_KEY)
    llm = None
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3)
        logger.info("LLM loaded: Claude claude-sonnet-4-6")

    # 4. Agents
    monitor   = MonitorAgent(lstm_model=lstm, anomaly_detector=anomaly, drift_detector=drift)
    alerter   = AlertAgent(websocket_manager=ws_manager, llm=llm)
    retrainer = RetrainingAgent(lstm_model=lstm)

    # 5. Orchestrator
    orchestrator = Orchestrator(monitor, alerter, retrainer)

    # 6. Background stream task
    simulator = StreamSimulator(interval_sec=1.0, n_sensors=3, inject_faults=True)
    _stream_task = asyncio.create_task(_run_stream(simulator))
    logger.info("Sensor stream task started.")

    # Inject dependencies into app state
    app.state.ws_manager   = ws_manager
    app.state.orchestrator = orchestrator

    yield

    # Shutdown
    if _stream_task:
        _stream_task.cancel()
        try:
            await _stream_task
        except asyncio.CancelledError:
            pass
    logger.info("Backend shutdown complete.")


async def _run_stream(simulator: StreamSimulator) -> None:
    """Background task: feed each sensor tick through the Orchestrator."""
    from ..agents.monitor_agent import SensorReading
    from datetime import datetime

    async for raw in simulator.stream():
        try:
            reading = SensorReading(
                timestamp = datetime.fromisoformat(raw["timestamp"]),
                sensor_id = raw["sensor_id"],
                values    = raw["values"],
                source    = raw["source"],
            )
            state = {"latest_reading": reading}
            result = await orchestrator.run(state)

            # Broadcast health metrics to all connected clients
            if ws_manager and result.get("health_metrics"):
                metrics = result["health_metrics"]
                await ws_manager.broadcast({
                    "type": "metrics",
                    "data": metrics.dict(),
                })
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.exception("Stream loop error: %s", exc)


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Adaptive Maintenance AI",
    description = "Real-time predictive maintenance with multi-agent AI and drift detection.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router)
