import { useEffect, useRef } from "react";
import {
  Chart,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";

Chart.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend, Filler);

const MAX_POINTS = 60;

/**
 * SensorChart
 * Scrolling line chart showing the last MAX_POINTS readings for a sensor.
 *
 * Props:
 *   sensorId    {string}   sensor label
 *   dataHistory {number[]} array of recent anomaly scores (0–1)
 *   timestamps  {string[]} matching ISO timestamp strings
 *   color       {string}   chart line color (hex/rgb)
 */
export default function SensorChart({ sensorId, dataHistory = [], timestamps = [], color = "#6366f1" }) {
  const data = {
    labels: timestamps.slice(-MAX_POINTS).map((t) => {
      const d = new Date(t);
      return `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
    }),
    datasets: [
      {
        label: `${sensorId} — Anomaly Score`,
        data: dataHistory.slice(-MAX_POINTS),
        borderColor: color,
        backgroundColor: `${color}22`,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: "index",
        intersect: false,
        callbacks: {
          label: (ctx) => ` Score: ${ctx.parsed.y.toFixed(4)}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { maxTicksLimit: 6, color: "#94a3b8", font: { size: 11 } },
        grid: { color: "#1e293b" },
      },
      y: {
        min: 0,
        max: 1,
        ticks: { color: "#94a3b8", font: { size: 11 } },
        grid: { color: "#1e293b" },
      },
    },
  };

  return (
    <div className="bg-slate-800 rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          {sensorId}
        </h3>
        <span className="text-xs text-slate-400">{dataHistory.length} readings</span>
      </div>
      <div style={{ height: 160 }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
}
