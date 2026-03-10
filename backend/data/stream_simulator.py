"""
stream_simulator.py
Simulates a real-time sensor data stream by replaying historical records
(or generating synthetic degradation signals) at configurable intervals.
Feeds the Orchestrator via an async generator.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np

from .data_loader import FEATURE_COLS, DataLoader

logger = logging.getLogger(__name__)


class StreamSimulator:
    """
    Async generator that yields SensorReading-like dicts.

    Parameters
    ----------
    interval_sec   : seconds between ticks (float for sub-second)
    n_sensors      : number of parallel sensor units to simulate
    inject_faults  : randomly inject fault patterns for testing
    replay_data    : if provided, replay this numpy array row-by-row
    """

    def __init__(
        self,
        interval_sec:  float = 1.0,
        n_sensors:     int   = 3,
        inject_faults: bool  = True,
        replay_data:   Optional[np.ndarray] = None,
    ):
        self.interval_sec  = interval_sec
        self.n_sensors     = n_sensors
        self.inject_faults = inject_faults
        self._replay       = replay_data
        self._replay_idx   = 0
        self._running      = False
        self._fault_prob   = 0.02   # 2% chance per tick per sensor

    # ── Public API ──────────────────────────────────────────────────────────

    async def stream(self) -> AsyncGenerator[Dict, None]:
        """Async generator — yields one reading dict per tick."""
        self._running = True
        t = datetime.utcnow()

        while self._running:
            for sid in range(self.n_sensors):
                values = self._next_values(sensor_idx=sid)
                yield {
                    "timestamp": t.isoformat(),
                    "sensor_id": f"sensor_{sid+1:02d}",
                    "values":    values,
                    "source":    "replay" if self._replay is not None else "synthetic",
                }
            t += timedelta(seconds=self.interval_sec)
            await asyncio.sleep(self.interval_sec)

    def stop(self) -> None:
        self._running = False

    # ── Internal helpers ────────────────────────────────────────────────────

    def _next_values(self, sensor_idx: int) -> Dict[str, float]:
        if self._replay is not None:
            row = self._replay[self._replay_idx % len(self._replay)]
            self._replay_idx += 1
            values = {FEATURE_COLS[i]: float(row[i]) for i in range(min(len(FEATURE_COLS), len(row)))}
        else:
            values = self._synthetic_tick(sensor_idx)

        if self.inject_faults and random.random() < self._fault_prob:
            values = self._inject_fault(values)

        return values

    def _synthetic_tick(self, sensor_idx: int) -> Dict[str, float]:
        """Generates a Gaussian random walk with slow degradation trend."""
        degradation = 0.001 * self._replay_idx
        return {
            col: float(np.random.normal(loc=degradation, scale=0.05))
            for col in FEATURE_COLS
        }

    def _inject_fault(self, values: Dict[str, float]) -> Dict[str, float]:
        """Spike 2–4 sensors to simulate a fault event."""
        fault_sensors = random.sample(list(values.keys()), k=min(3, len(values)))
        for s in fault_sensors:
            values[s] *= random.uniform(3.0, 8.0)   # spike
        logger.debug("StreamSimulator: fault injected on %s", fault_sensors)
        return values
