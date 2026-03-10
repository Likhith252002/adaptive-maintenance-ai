import { useEffect, useRef, useState, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const WS_URL  = import.meta.env.VITE_WS_URL  ?? `${API_URL.replace(/^http/, "ws")}/ws`;

const STATUS = { CONNECTING: "connecting", OPEN: "open", CLOSED: "closed", ERROR: "error" };

/**
 * useWebSocket
 * Custom hook that manages a persistent WebSocket connection to the backend,
 * with automatic reconnection and typed message dispatching.
 *
 * Returns:
 *   status       {string}     "connecting" | "open" | "closed" | "error"
 *   metrics      {Object}     latest health metrics keyed by sensor_id
 *   alerts       {Alert[]}    last 50 alerts received
 *   agentEvents  {Event[]}    last 100 agent activity events
 *   driftStatus  {Object}     drift state keyed by sensor_id
 *   sendMessage  {(msg)=>void} send raw message to server
 */
export function useWebSocket() {
  const ws             = useRef(null);
  const reconnectTimer = useRef(null);
  const [status,      setStatus]      = useState(STATUS.CONNECTING);
  const [metrics,     setMetrics]     = useState({});
  const [alerts,      setAlerts]      = useState([]);
  const [agentEvents, setAgentEvents] = useState([]);
  const [driftStatus, setDriftStatus] = useState({});

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;

    setStatus(STATUS.CONNECTING);
    ws.current = new WebSocket(WS_URL);

    ws.current.onopen = () => {
      setStatus(STATUS.OPEN);
      clearTimeout(reconnectTimer.current);
      // Keep-alive ping every 20 s
      reconnectTimer.current = setInterval(() => {
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send("ping");
        }
      }, 20_000);
    };

    ws.current.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
      } catch {
        // ignore non-JSON (e.g. "pong")
      }
    };

    ws.current.onerror = () => setStatus(STATUS.ERROR);

    ws.current.onclose = () => {
      setStatus(STATUS.CLOSED);
      clearInterval(reconnectTimer.current);
      // Reconnect after 3 s
      reconnectTimer.current = setTimeout(connect, 3_000);
    };
  }, []);

  const handleMessage = useCallback((msg) => {
    switch (msg.type) {
      case "metrics":
        setMetrics((prev) => ({
          ...prev,
          [msg.data.sensor_id]: msg.data,
        }));
        // Add to agent events
        setAgentEvents((prev) => [
          ...prev.slice(-99),
          {
            id:        `ev-${Date.now()}-${Math.random()}`,
            agent:     "monitor",
            message:   `${msg.data.sensor_id} — score=${msg.data.anomaly_score} RUL=${msg.data.rul_estimate}`,
            timestamp: msg.data.timestamp,
          },
        ]);
        // Update drift status
        setDriftStatus((prev) => ({
          ...prev,
          [msg.data.sensor_id]: {
            id:             msg.data.sensor_id,
            drift_detected: msg.data.drift_detected,
            drift_score:    msg.data.anomaly_score,
            last_checked:   msg.data.timestamp,
          },
        }));
        break;

      case "alert":
        setAlerts((prev) => {
          const next = [msg.data, ...prev].slice(0, 50);
          return next;
        });
        setAgentEvents((prev) => [
          ...prev.slice(-99),
          {
            id:        `ev-${Date.now()}`,
            agent:     "alert",
            message:   `[${msg.data.severity.toUpperCase()}] ${msg.data.title}`,
            timestamp: msg.data.timestamp,
          },
        ]);
        break;

      case "retrain":
        setAgentEvents((prev) => [
          ...prev.slice(-99),
          {
            id:        `ev-${Date.now()}`,
            agent:     "retrain",
            message:   msg.data.success
              ? `Retraining complete — cycle #${msg.data.retrain_cycle}, loss=${msg.data.new_loss}`
              : `Retraining failed: ${msg.data.reason}`,
            timestamp: new Date().toISOString(),
          },
        ]);
        break;

      default:
        break;
    }
  }, []);

  const sendMessage = useCallback((msg) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearInterval(reconnectTimer.current);
      clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, [connect]);

  return {
    status,
    metrics,
    alerts,
    agentEvents,
    driftStatus: Object.values(driftStatus),
    sendMessage,
  };
}
