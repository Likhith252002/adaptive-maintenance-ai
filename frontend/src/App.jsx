import { useState, useMemo } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import SensorChart      from "./components/SensorChart";
import RULGauge         from "./components/RULGauge";
import AgentActivityLog from "./components/AgentActivityLog";
import AlertPanel       from "./components/AlertPanel";
import DriftIndicator   from "./components/DriftIndicator";

const SENSOR_COLORS = ["#6366f1", "#06b6d4", "#f59e0b"];

function StatusBadge({ status }) {
  const styles = {
    open:       "bg-green-500/20 text-green-400",
    connecting: "bg-yellow-500/20 text-yellow-400 animate-pulse",
    closed:     "bg-slate-500/20 text-slate-400",
    error:      "bg-red-500/20 text-red-400",
  };
  const labels = { open: "● Live", connecting: "◌ Connecting…", closed: "○ Disconnected", error: "✕ Error" };
  return (
    <span className={`text-xs px-2 py-1 rounded-full font-medium ${styles[status] ?? styles.closed}`}>
      {labels[status] ?? status}
    </span>
  );
}

export default function App() {
  const { status, metrics, alerts, agentEvents, driftStatus } = useWebSocket();
  const [dismissedAlerts, setDismissedAlerts] = useState(new Set());

  // Build per-sensor history (last 120 ticks) from accumulated metrics
  const [sensorHistory, setSensorHistory] = useState({});
  useMemo(() => {
    Object.entries(metrics).forEach(([sid, m]) => {
      setSensorHistory((prev) => {
        const hist = prev[sid] ?? { scores: [], timestamps: [] };
        return {
          ...prev,
          [sid]: {
            scores:     [...hist.scores.slice(-119),     m.anomaly_score],
            timestamps: [...hist.timestamps.slice(-119), m.timestamp],
          },
        };
      });
    });
  }, [metrics]);

  const visibleAlerts = alerts.filter((a) => !dismissedAlerts.has(a.id));
  const sensorIds     = Object.keys(metrics).sort();

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans">
      {/* Header */}
      <header className="border-b border-slate-700/50 px-6 py-4 flex items-center justify-between sticky top-0 z-10 bg-slate-900/95 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center text-lg">⚙</div>
          <div>
            <h1 className="text-base font-bold text-white">Adaptive Maintenance AI</h1>
            <p className="text-xs text-slate-400">Real-time predictive maintenance dashboard</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={status} />
          {alerts.length > 0 && (
            <span className="px-2 py-1 rounded-full bg-red-500/20 text-red-400 text-xs font-bold">
              {visibleAlerts.length} alert{visibleAlerts.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>
      </header>

      <main className="p-6 space-y-6">
        {/* RUL Gauges */}
        <section>
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Sensor Health & RUL
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {sensorIds.length === 0
              ? [1, 2, 3].map((i) => (
                  <div key={i} className="bg-slate-800 rounded-xl p-4 h-40 animate-pulse" />
                ))
              : sensorIds.map((sid, idx) => (
                  <RULGauge
                    key={sid}
                    sensorId={sid}
                    rul={metrics[sid]?.rul_estimate ?? 0}
                    healthScore={metrics[sid]?.health_score ?? 1}
                    color={SENSOR_COLORS[idx % SENSOR_COLORS.length]}
                  />
                ))}
          </div>
        </section>

        {/* Sensor Charts */}
        <section>
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Anomaly Score — Live Stream
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {sensorIds.length === 0
              ? [1, 2, 3].map((i) => (
                  <div key={i} className="bg-slate-800 rounded-xl p-4 h-48 animate-pulse" />
                ))
              : sensorIds.map((sid, idx) => (
                  <SensorChart
                    key={sid}
                    sensorId={sid}
                    dataHistory={sensorHistory[sid]?.scores ?? []}
                    timestamps={sensorHistory[sid]?.timestamps ?? []}
                    color={SENSOR_COLORS[idx % SENSOR_COLORS.length]}
                  />
                ))}
          </div>
        </section>

        {/* Bottom row */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <AlertPanel
            alerts={visibleAlerts}
            onDismiss={(id) => setDismissedAlerts((prev) => new Set([...prev, id]))}
          />
          <DriftIndicator sensors={driftStatus} />
          <AgentActivityLog events={agentEvents} />
        </section>
      </main>
    </div>
  );
}
