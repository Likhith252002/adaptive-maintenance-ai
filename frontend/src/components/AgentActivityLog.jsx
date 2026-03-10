import { useEffect, useRef } from "react";

const AGENT_COLORS = {
  monitor:   "text-blue-400",
  alert:     "text-red-400",
  retrain:   "text-yellow-400",
  orchestrator: "text-purple-400",
  system:    "text-slate-400",
};

const AGENT_ICONS = {
  monitor:   "🔍",
  alert:     "🚨",
  retrain:   "🔄",
  orchestrator: "🧠",
  system:    "⚙️",
};

/**
 * AgentActivityLog
 * Auto-scrolling log of multi-agent activity events.
 *
 * Props:
 *   events  {Array<{id, agent, message, timestamp, level}>}
 *   maxRows {number} max visible rows before scroll
 */
export default function AgentActivityLog({ events = [], maxRows = 12 }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  return (
    <div className="bg-slate-800 rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          Agent Activity Log
        </h3>
        <span className="text-xs text-slate-400">{events.length} events</span>
      </div>

      <div
        className="overflow-y-auto font-mono text-xs space-y-1"
        style={{ maxHeight: maxRows * 22 }}
      >
        {events.length === 0 && (
          <p className="text-slate-500 italic">Waiting for agent activity…</p>
        )}

        {events.map((ev) => {
          const agentKey  = (ev.agent || "system").toLowerCase();
          const colorCls  = AGENT_COLORS[agentKey] ?? "text-slate-300";
          const icon      = AGENT_ICONS[agentKey]  ?? "•";
          const time      = new Date(ev.timestamp).toLocaleTimeString();

          return (
            <div key={ev.id} className="flex gap-2 items-start">
              <span className="text-slate-500 shrink-0">{time}</span>
              <span className={`shrink-0 ${colorCls}`}>{icon} [{agentKey}]</span>
              <span className="text-slate-300 break-all">{ev.message}</span>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
