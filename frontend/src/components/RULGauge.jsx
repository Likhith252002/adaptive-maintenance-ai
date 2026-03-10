/**
 * RULGauge
 * Semi-circular gauge showing Remaining Useful Life as a percentage of max RUL.
 *
 * Props:
 *   sensorId   {string}  sensor label
 *   rul        {number}  current RUL estimate (cycles)
 *   maxRul     {number}  maximum RUL (default 125)
 *   healthScore{number}  0–1 health score
 */
export default function RULGauge({ sensorId, rul = 0, maxRul = 125, healthScore = 1 }) {
  const pct     = Math.min(100, Math.round((rul / maxRul) * 100));
  const degrees = Math.round((pct / 100) * 180);

  const color =
    pct > 60 ? "#22c55e" :
    pct > 30 ? "#f59e0b" :
               "#ef4444";

  const needleStyle = {
    transform: `rotate(${degrees - 90}deg)`,
    transformOrigin: "bottom center",
    transition: "transform 0.6s ease-in-out",
  };

  return (
    <div className="bg-slate-800 rounded-xl p-4 shadow-lg flex flex-col items-center">
      <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider mb-3">
        {sensorId} — RUL
      </h3>

      {/* SVG gauge arc */}
      <svg viewBox="0 0 120 70" className="w-36">
        {/* Background arc */}
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke="#1e293b"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Foreground arc (progress) */}
        <path
          d="M 10 65 A 50 50 0 0 1 110 65"
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${(pct / 100) * 157} 157`}
        />
        {/* Centre text */}
        <text x="60" y="60" textAnchor="middle" fontSize="14" fontWeight="bold" fill={color}>
          {pct}%
        </text>
      </svg>

      <div className="mt-2 text-center">
        <p className="text-2xl font-bold" style={{ color }}>
          {Math.round(rul)}
        </p>
        <p className="text-xs text-slate-400">cycles remaining</p>
      </div>

      <div className="mt-3 w-full">
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Health</span>
          <span>{(healthScore * 100).toFixed(1)}%</span>
        </div>
        <div className="h-1.5 bg-slate-700 rounded-full">
          <div
            className="h-1.5 rounded-full transition-all duration-500"
            style={{ width: `${healthScore * 100}%`, backgroundColor: color }}
          />
        </div>
      </div>
    </div>
  );
}
