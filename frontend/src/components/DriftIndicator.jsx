/**
 * DriftIndicator
 * Visual indicator for data drift status per sensor.
 *
 * Props:
 *   sensors  {Array<{id, drift_detected, drift_score, last_checked}>}
 *   onRefresh {() => void}
 */
export default function DriftIndicator({ sensors = [], onRefresh }) {
  const driftingCount = sensors.filter((s) => s.drift_detected).length;

  return (
    <div className="bg-slate-800 rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          Data Drift Monitor
        </h3>
        <div className="flex items-center gap-2">
          {driftingCount > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-orange-500/20 text-orange-400 text-xs font-bold">
              {driftingCount} drifting
            </span>
          )}
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="text-slate-400 hover:text-slate-200 text-xs transition-colors"
              title="Refresh drift check"
            >
              ↻
            </button>
          )}
        </div>
      </div>

      {sensors.length === 0 ? (
        <p className="text-slate-500 text-sm italic text-center py-4">
          Awaiting sensor data…
        </p>
      ) : (
        <div className="space-y-2">
          {sensors.map((sensor) => {
            const score = sensor.drift_score ?? 0;
            const drifting = sensor.drift_detected;
            const color = drifting
              ? score > 0.7 ? "#ef4444" : "#f59e0b"
              : "#22c55e";

            return (
              <div key={sensor.id} className="flex items-center gap-3">
                {/* Status dot */}
                <div
                  className="w-2.5 h-2.5 rounded-full shrink-0 animate-pulse"
                  style={{ backgroundColor: color }}
                />

                {/* Sensor label */}
                <span className="text-xs text-slate-300 w-24 shrink-0">{sensor.id}</span>

                {/* Score bar */}
                <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${score * 100}%`, backgroundColor: color }}
                  />
                </div>

                {/* Score value */}
                <span className="text-xs w-10 text-right" style={{ color }}>
                  {(score * 100).toFixed(0)}%
                </span>

                {/* Badge */}
                <span
                  className="text-xs px-1.5 py-0.5 rounded font-medium shrink-0"
                  style={{
                    backgroundColor: `${color}22`,
                    color,
                  }}
                >
                  {drifting ? "DRIFT" : "OK"}
                </span>
              </div>
            );
          })}
        </div>
      )}

      <p className="mt-3 text-xs text-slate-500">
        Detection: Page-Hinkley + Evidently DataDriftPreset
      </p>
    </div>
  );
}
