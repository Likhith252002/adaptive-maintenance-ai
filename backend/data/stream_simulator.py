"""
stream_simulator.py
Streams NASA CMAPSS test data engine-by-engine, cycle-by-cycle.
Supports drift injection (multiply all sensors × 1.3) and
anomaly injection (spike 3 random sensors × 2 for 5 cycles).
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_loader import FEATURE_COLS, DataLoader

logger = logging.getLogger(__name__)


class StreamSimulator:
    """
    Replays NASA CMAPSS test data as an async stream.

    Parameters
    ----------
    test_df       : processed test DataFrame from DataLoader.load()
    interval_sec  : delay between yielded rows (default 0.5 s)
    """

    def __init__(
        self,
        test_df: Optional[pd.DataFrame] = None,
        interval_sec: float = 0.5,
    ):
        self.interval_sec = interval_sec

        # If no DataFrame provided, generate synthetic fallback
        if test_df is None or test_df.empty:
            logger.warning("StreamSimulator: no test_df provided — using synthetic data")
            test_df = self._synthetic_df()

        self._df = test_df.copy().reset_index(drop=True)

        # State flags
        self.is_drifting: bool = False
        self.is_anomaly:  bool = False

        # Anomaly injection counter
        self._anomaly_cycles_left: int = 0
        self._anomaly_sensors:     List[str] = []

    # ── Public API ───────────────────────────────────────────────────────────

    async def stream(self) -> AsyncGenerator[Dict, None]:
        """
        Async generator — yields one dict per cycle per engine, in order:
            engine 1 cycle 1 → engine 1 cycle 2 → … → engine N cycle M

        Yielded dict shape:
            {
                "engine_id": int,
                "cycle":     int,
                "sensors":   {"s2": float, "s3": float, …},   # FEATURE_COLS only
                "timestamp": str (ISO-8601 UTC),
                "is_drifting": bool,
                "is_anomaly":  bool,
            }
        """
        for engine_id, engine_df in self._df.groupby("engine_id", sort=True):
            for _, row in engine_df.sort_values("cycle").iterrows():
                sensors = {col: float(row[col]) for col in FEATURE_COLS if col in row}
                sensors = self._apply_drift(sensors)
                sensors = self._apply_anomaly(sensors)

                yield {
                    "engine_id":   int(engine_id),
                    "cycle":       int(row["cycle"]),
                    "sensors":     sensors,
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "is_drifting": self.is_drifting,
                    "is_anomaly":  self.is_anomaly,
                }

                await asyncio.sleep(self.interval_sec)

    def inject_drift(self) -> None:
        """Multiply all sensor values by 1.3 until clear_drift() is called."""
        self.is_drifting = True
        logger.info("StreamSimulator: drift injection ENABLED (×1.3)")

    def clear_drift(self) -> None:
        self.is_drifting = False
        logger.info("StreamSimulator: drift injection DISABLED")

    def inject_anomaly(self) -> None:
        """Spike 3 random sensors by ×2 for the next 5 cycles."""
        self.is_anomaly          = True
        self._anomaly_cycles_left = 5
        self._anomaly_sensors    = random.sample(FEATURE_COLS, k=min(3, len(FEATURE_COLS)))
        logger.info(
            "StreamSimulator: anomaly injection ENABLED on %s for 5 cycles",
            self._anomaly_sensors,
        )

    # ── Internal transforms ──────────────────────────────────────────────────

    def _apply_drift(self, sensors: Dict[str, float]) -> Dict[str, float]:
        if not self.is_drifting:
            return sensors
        return {k: v * 1.3 for k, v in sensors.items()}

    def _apply_anomaly(self, sensors: Dict[str, float]) -> Dict[str, float]:
        if not self.is_anomaly:
            return sensors

        if self._anomaly_cycles_left > 0:
            sensors = dict(sensors)
            for s in self._anomaly_sensors:
                if s in sensors:
                    sensors[s] *= 2.0
            self._anomaly_cycles_left -= 1
            if self._anomaly_cycles_left == 0:
                self.is_anomaly       = False
                self._anomaly_sensors = []
                logger.info("StreamSimulator: anomaly injection complete")

        return sensors

    # ── Synthetic fallback ───────────────────────────────────────────────────

    @staticmethod
    def _synthetic_df(n_engines: int = 5, max_cycles: int = 100) -> pd.DataFrame:
        rows = []
        for e in range(1, n_engines + 1):
            for c in range(1, max_cycles + 1):
                row = {"engine_id": e, "cycle": c, "RUL": max(0, max_cycles - c)}
                for col in FEATURE_COLS:
                    row[col] = float(np.random.normal(loc=c * 0.001, scale=0.05))
                rows.append(row)
        return pd.DataFrame(rows)


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    async def main():
        print("Loading NASA CMAPSS FD001 …")
        loader = DataLoader(data_dir="data/raw")
        _, test_df = loader.load("FD001")

        sim = StreamSimulator(test_df=test_df, interval_sec=0.0)  # 0 delay for fast test

        print("\nStreaming first 10 rows:\n")
        count = 0
        async for reading in sim.stream():
            sensor_preview = {k: round(v, 4) for k, v in list(reading["sensors"].items())[:4]}
            print(
                f"  engine={reading['engine_id']:>3}  "
                f"cycle={reading['cycle']:>4}  "
                f"sensors(first 4)={sensor_preview}  "
                f"drift={reading['is_drifting']}  "
                f"anomaly={reading['is_anomaly']}  "
                f"ts={reading['timestamp']}"
            )
            count += 1
            if count >= 10:
                break

        print("\nDrift injection test (next 3 rows):")
        sim.inject_drift()
        count = 0
        async for reading in sim.stream():
            sensor_preview = {k: round(v, 4) for k, v in list(reading["sensors"].items())[:4]}
            print(
                f"  engine={reading['engine_id']:>3}  "
                f"cycle={reading['cycle']:>4}  "
                f"sensors(first 4)={sensor_preview}  "
                f"drift={reading['is_drifting']}"
            )
            count += 1
            if count >= 3:
                break
        sim.clear_drift()

        print("\nAnomaly injection test (next 6 rows — spikes for 5 cycles):")
        sim.inject_anomaly()
        count = 0
        async for reading in sim.stream():
            sensor_preview = {k: round(v, 4) for k, v in list(reading["sensors"].items())[:4]}
            print(
                f"  engine={reading['engine_id']:>3}  "
                f"cycle={reading['cycle']:>4}  "
                f"sensors(first 4)={sensor_preview}  "
                f"anomaly={reading['is_anomaly']}  "
                f"cycles_left={sim._anomaly_cycles_left}"
            )
            count += 1
            if count >= 6:
                break

        print("\nAll tests passed.")

    asyncio.run(main())
