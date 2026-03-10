"""
websocket_manager.py
Thread-safe WebSocket connection registry with broadcast support.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Maintains a set of active WebSocket connections and provides
    broadcast / unicast helpers.
    """

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ── Connection management ──────────────────────────────────────────────

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    def count(self) -> int:
        return len(self._connections)

    # ── Messaging ──────────────────────────────────────────────────────────

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send JSON message to all connected clients."""
        if not self._connections:
            return
        payload = json.dumps(message, default=str)
        dead: List[WebSocket] = []
        async with self._lock:
            connections = list(self._connections)

        results = await asyncio.gather(
            *[self._send(ws, payload) for ws in connections],
            return_exceptions=True,
        )
        for ws, result in zip(connections, results):
            if isinstance(result, Exception):
                logger.debug("WebSocketManager: removing dead connection — %s", result)
                dead.append(ws)

        async with self._lock:
            for ws in dead:
                self._connections.discard(ws)

    async def send_to(self, ws: WebSocket, message: Dict[str, Any]) -> None:
        """Send JSON message to a single client."""
        payload = json.dumps(message, default=str)
        await self._send(ws, payload)

    # ── Private ────────────────────────────────────────────────────────────

    @staticmethod
    async def _send(ws: WebSocket, payload: str) -> None:
        await ws.send_text(payload)
