"""
orchestrator.py
LangGraph-based multi-agent orchestrator that wires MonitorAgent →
AlertAgent and MonitorAgent → RetrainingAgent into a single stateful graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph, END

from .monitor_agent    import MonitorAgent, SensorReading
from .retraining_agent import RetrainingAgent
from .alert_agent      import AlertAgent

logger = logging.getLogger(__name__)


# ── Shared state schema ────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    latest_reading:   SensorReading
    health_metrics:   Any
    alert_required:   bool
    retrain_required: bool
    active_alerts:    list
    model_updated:    bool
    retrain_summary:  dict


# ── Routing functions ──────────────────────────────────────────────────────

def route_after_monitor(state: AgentState) -> str:
    """Fan-out: if both alert and retrain needed, run alert first then retrain."""
    if state.get("alert_required") and state.get("retrain_required"):
        return "alert"
    if state.get("alert_required"):
        return "alert"
    if state.get("retrain_required"):
        return "retrain"
    return END


def route_after_alert(state: AgentState) -> str:
    return "retrain" if state.get("retrain_required") else END


# ── Orchestrator ───────────────────────────────────────────────────────────

class Orchestrator:
    """
    Builds and compiles a LangGraph StateGraph connecting all agents.

    Usage
    -----
    orch = Orchestrator(monitor_agent, alert_agent, retraining_agent)
    result = await orch.run({"latest_reading": reading})
    """

    def __init__(
        self,
        monitor_agent:    MonitorAgent,
        alert_agent:      AlertAgent,
        retraining_agent: RetrainingAgent,
    ):
        self.monitor    = monitor_agent
        self.alerter    = alert_agent
        self.retrainer  = retraining_agent
        self._graph     = self._build_graph()

    # ── Public API ──────────────────────────────────────────────────────────

    async def run(self, initial_state: Dict[str, Any]) -> AgentState:
        """Process one sensor reading through the full agent pipeline."""
        result = await self._graph.ainvoke(initial_state)
        logger.debug("Orchestrator cycle complete: alert=%s retrain=%s",
                     result.get("alert_required"), result.get("retrain_required"))
        return result

    def get_graph_png(self) -> bytes:
        """Return a PNG of the compiled graph (useful for debugging)."""
        return self._graph.get_graph().draw_mermaid_png()

    # ── Graph construction ──────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("monitor",  self.monitor.run)
        graph.add_node("alert",    self.alerter.run)
        graph.add_node("retrain",  self.retrainer.run)

        # Entry
        graph.set_entry_point("monitor")

        # Edges
        graph.add_conditional_edges("monitor", route_after_monitor, {
            "alert":   "alert",
            "retrain": "retrain",
            END:       END,
        })
        graph.add_conditional_edges("alert", route_after_alert, {
            "retrain": "retrain",
            END:       END,
        })
        graph.add_edge("retrain", END)

        return graph.compile()
