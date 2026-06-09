import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
);

function MetricCard({ title, value, accent, suffix = "" }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-slate-950/50 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm text-slate-400">{title}</p>
        <span className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.25em] ${accent}`}>
          Live
        </span>
      </div>
      <p className="mt-4 text-3xl font-semibold tracking-tight">
        {value}
        {suffix}
      </p>
    </div>
  );
}

const DAY_ABBREVS = {
  monday: "Mon", tuesday: "Tue", wednesday: "Wed",
  thursday: "Thu", friday: "Fri", saturday: "Sat", sunday: "Sun",
};

function convergenceBarColor(pct) {
  if (pct >= 76) return "from-emerald-400 to-emerald-300";
  if (pct >= 40) return "from-amber-400 to-amber-300";
  return "from-red-400 to-red-300";
}

export default function MetricsPanel({
  snapshot,
  rewardHistory,
  convergencePct,
  simDay,
  simTime,
  demandMode,
  activeDemandLevel,
}) {
  const isConverged = snapshot.stopped_reason === "converged";

  // Convergence threshold: 97% of the max reward seen (dashed reference line)
  const maxReward = rewardHistory.length > 0
    ? Math.max(...rewardHistory.map((r) => r.rl))
    : 0;
  const convergenceThreshold = maxReward > 0 ? Math.round(maxReward * 0.97) : null;

  const rewardData = {
    labels: rewardHistory.map((item) => `E${item.episode}`),
    datasets: [
      {
        label: "LightMind AI Reward",
        data: rewardHistory.map((item) => item.rl),
        borderColor: "#33d4ff",
        backgroundColor: "rgba(51, 212, 255, 0.14)",
        fill: true,
        tension: 0.35,
        pointRadius: 0,
      },
      ...(convergenceThreshold !== null
        ? [
            {
              label: "Convergence Zone",
              data: rewardHistory.map(() => convergenceThreshold),
              borderColor: "rgba(250, 204, 21, 0.55)",
              borderWidth: 1.5,
              borderDash: [6, 4],
              pointRadius: 0,
              fill: false,
              tension: 0,
            },
          ]
        : []),
    ],
  };

  const rewardOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#cbd5e1", boxWidth: 12 } },
    },
    scales: {
      x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148, 163, 184, 0.08)" } },
      y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148, 163, 184, 0.08)" } },
    },
  };

  // Demand row label
  let demandLabel;
  if (demandMode === "schedule" && activeDemandLevel) {
    const dayAbbrev = DAY_ABBREVS[simDay?.toLowerCase()] ?? simDay;
    demandLabel = (
      <span>
        Schedule:{" "}
        <span className="text-slate-200">{activeDemandLevel}</span>
        {dayAbbrev && simTime && (
          <span className="ml-1 text-slate-400">({dayAbbrev} {simTime})</span>
        )}
      </span>
    );
  } else if (activeDemandLevel) {
    demandLabel = (
      <span>
        Auto →{" "}
        <span className="text-slate-200">{activeDemandLevel}</span>
        <span className="ml-1 text-slate-500">(ep {snapshot.episode})</span>
      </span>
    );
  } else {
    demandLabel = <span className="text-slate-500">—</span>;
  }

  return (
    <div className="glass-panel flex flex-col gap-5 p-5">
      {/* Convergence progress */}
      <div className="rounded-3xl border border-white/10 bg-slate-950/50 p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">
              Convergence Progress
            </p>
            <h3 className="mt-2 text-xl font-semibold">Training Telemetry</h3>
          </div>
          <p className="text-sm text-slate-300">{convergencePct}% converged</p>
        </div>

        {isConverged ? (
          <div className="mt-4 flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-400/15 px-4 py-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <p className="text-sm font-semibold text-emerald-200">
              Converged at episode {snapshot.final_episode}
            </p>
          </div>
        ) : (
          <div className="mt-4 h-3 overflow-hidden rounded-full bg-white/10">
            <div
              className={`h-full rounded-full bg-gradient-to-r transition-all ${convergenceBarColor(convergencePct)}`}
              style={{ width: `${convergencePct}%` }}
            />
          </div>
        )}

        <div className="mt-3 flex items-center justify-between text-sm">
          <span className="text-slate-400">Demand</span>
          <span className="text-xs text-slate-300">{demandLabel}</span>
        </div>
      </div>

      {/* Live metric cards — raw values only */}
      <div className="grid gap-4 md:grid-cols-2">
        <MetricCard
          title="Reward"
          value={snapshot.rl.reward}
          accent="border-cyan-400/20 bg-cyan-400/10 text-cyan-100"
        />
        <MetricCard
          title="Waiting Time"
          value={snapshot.rl.waiting_time}
          suffix="s"
          accent="border-emerald-400/20 bg-emerald-400/10 text-emerald-100"
        />
        <MetricCard
          title="Queue Length"
          value={snapshot.rl.queue_length}
          suffix=" veh"
          accent="border-amber-400/20 bg-amber-400/10 text-amber-100"
        />
        <MetricCard
          title="Throughput"
          value={snapshot.rl.throughput}
          suffix="/h"
          accent="border-red-400/20 bg-red-400/10 text-red-100"
        />
      </div>

      {/* Reward curve — single line + convergence threshold */}
      <div className="rounded-3xl border border-white/10 bg-slate-950/50 p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Reward Chart</p>
            <h4 className="mt-2 text-lg font-semibold">
              Reward Curve — LightMind AI Training
            </h4>
          </div>
          <div className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-400">
            Live stream
          </div>
        </div>
        <div className="h-72">
          <Line data={rewardData} options={rewardOptions} />
        </div>
      </div>
    </div>
  );
}
