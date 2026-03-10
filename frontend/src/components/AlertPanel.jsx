const SEVERITY_STYLES = {
  low:      { border: "border-green-500",  bg: "bg-green-500/10",  text: "text-green-400",  badge: "bg-green-500/20"  },
  medium:   { border: "border-yellow-500", bg: "bg-yellow-500/10", text: "text-yellow-400", badge: "bg-yellow-500/20" },
  high:     { border: "border-orange-500", bg: "bg-orange-500/10", text: "text-orange-400", badge: "bg-orange-500/20" },
  critical: { border: "border-red-500",    bg: "bg-red-500/10",    text: "text-red-400",    badge: "bg-red-500/20"    },
};

const SEVERITY_ICONS = { low: "🟢", medium: "🟡", high: "🟠", critical: "🔴" };

/**
 * AlertPanel
 * Displays a list of active maintenance alerts sorted by severity.
 *
 * Props:
 *   alerts  {Array<Alert>}  alert objects from AlertAgent
 *   onDismiss {(id) => void} optional dismiss callback
 */
export default function AlertPanel({ alerts = [], onDismiss }) {
  const sorted = [...alerts].sort((a, b) => {
    const order = { critical: 0, high: 1, medium: 2, low: 3 };
    return (order[a.severity] ?? 4) - (order[b.severity] ?? 4);
  });

  return (
    <div className="bg-slate-800 rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          Active Alerts
        </h3>
        {alerts.length > 0 && (
          <span className="px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 text-xs font-bold">
            {alerts.length}
          </span>
        )}
      </div>

      {sorted.length === 0 ? (
        <div className="text-center py-8 text-slate-500">
          <p className="text-2xl mb-2">✅</p>
          <p className="text-sm">No active alerts</p>
        </div>
      ) : (
        <div className="space-y-3 overflow-y-auto max-h-96">
          {sorted.map((alert) => {
            const style = SEVERITY_STYLES[alert.severity] ?? SEVERITY_STYLES.low;
            return (
              <div
                key={alert.id}
                className={`border-l-4 rounded-r-lg p-3 ${style.border} ${style.bg}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded-full ${style.badge} ${style.text}`}>
                        {SEVERITY_ICONS[alert.severity]} {alert.severity}
                      </span>
                      <span className="text-xs text-slate-400">{alert.sensor_id}</span>
                      {alert.rul_estimate != null && (
                        <span className="text-xs text-slate-400">
                          RUL: {Math.round(alert.rul_estimate)} cycles
                        </span>
                      )}
                    </div>
                    <p className={`mt-1 text-sm font-semibold ${style.text}`}>{alert.title}</p>
                    <p className="mt-1 text-xs text-slate-400 leading-relaxed">{alert.message}</p>
                    <p className="mt-2 text-xs text-slate-300">
                      <span className="font-medium">Action:</span> {alert.recommended_action}
                    </p>
                  </div>
                  {onDismiss && (
                    <button
                      onClick={() => onDismiss(alert.id)}
                      className="text-slate-500 hover:text-slate-300 text-xs shrink-0"
                    >
                      ✕
                    </button>
                  )}
                </div>
                <p className="mt-2 text-xs text-slate-500">
                  {new Date(alert.timestamp).toLocaleString()}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
