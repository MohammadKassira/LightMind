import { useEffect, useState } from "react";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Filler, Tooltip, Legend);

import { API_BASE } from "../config";

function fmt(val, unit = "", decimals = 1) {
  if (val == null) return "—";
  const n = typeof val === "number" ? val : parseFloat(val);
  return isNaN(n) ? "—" : `${n.toFixed(decimals)}${unit}`;
}

function pctImprove(dqnVal, baseVal) {
  if (dqnVal == null || baseVal == null || baseVal === 0) return null;
  return ((baseVal - dqnVal) / baseVal) * 100;
}

function HeroCard({ label, value, sub, tone = "text-cyan-100", badge }) {
  return (
    <div className="glass-panel p-5 flex flex-col gap-2">
      <p className="text-xs text-slate-400">{label}</p>
      <p className={`text-3xl font-bold tracking-tight ${tone}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500">{sub}</p>}
      {badge && (
        <span className={`self-start rounded-full px-2 py-0.5 text-xs font-medium ${badge.cls}`}>
          {badge.text}
        </span>
      )}
    </div>
  );
}

const DEMAND_LABELS = ["low", "medium", "high"];
const DEMAND_COLORS = {
  dqn: { bg: "rgba(51,212,255,0.7)", border: "#33d4ff" },
  baseline: { bg: "rgba(148,163,184,0.4)", border: "#94a3b8" },
};

const BAR_OPTS = (yLabel) => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: "#94a3b8", boxWidth: 10, font: { size: 11 } } },
    tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.raw?.toFixed(1)}` } },
  },
  scales: {
    x: { ticks: { color: "#64748b", font: { size: 11 } }, grid: { color: "rgba(148,163,184,0.05)" } },
    y: { title: { display: true, text: yLabel, color: "#64748b", font: { size: 10 } }, ticks: { color: "#64748b" }, grid: { color: "rgba(148,163,184,0.06)" } },
  },
});

export default function ResultsScreen({ sessionId, onReset }) {
  const [fakeResults, setFakeResults] = useState(null);
  const [realResult, setRealResult] = useState(null);
  const [baselineResults, setBaselineResults] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!sessionId) return;
    const controller = new AbortController();

    fetch(`${API_BASE}/api/real-train/${sessionId}/result`, { signal: controller.signal })
      .then((r) => r.json())
      .then((data) => {
        if (data.available) {
          setRealResult(data);
          // Try to fetch baseline results
          return fetch(`${API_BASE}/api/real-train/${sessionId}/baseline-results`, { signal: controller.signal })
            .then((r) => r.json())
            .then((bl) => { if (bl.available) setBaselineResults(bl.summary); })
            .catch(() => {});
        } else {
          return fetch(`${API_BASE}/api/results/${sessionId}`, { signal: controller.signal })
            .then((r) => r.json())
            .then(setFakeResults)
            .catch((err) => { if (err.name !== "AbortError") setError(err.message); });
        }
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          fetch(`${API_BASE}/api/results/${sessionId}`, { signal: controller.signal })
            .then((r) => r.json())
            .then(setFakeResults)
            .catch((e) => { if (e.name !== "AbortError") setError(e.message); });
        }
      });

    return () => controller.abort();
  }, [sessionId]);

  const handleDownloadModel = () => {
    if (realResult?.available) {
      window.location.href = `${API_BASE}/api/real-train/${sessionId}/download-model`;
    } else {
      const a = document.createElement("a");
      a.href = `${API_BASE}/api/sessions/${sessionId}/model`;
      a.download = `lightmind_model_${sessionId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const handleDownloadBundle = () => {
    window.location.href = `${API_BASE}/api/real-train/${sessionId}/download-bundle`;
  };

  const isReal = realResult?.available;
  const kpis = isReal ? realResult.kpis : null;

  // Average DQN KPIs across demand levels
  let avgDqn = null;
  if (kpis) {
    const levels = Object.values(kpis);
    if (levels.length > 0) {
      const avg = (key) => levels.reduce((a, l) => a + (l[key] ?? 0), 0) / levels.length;
      avgDqn = {
        waiting_time: avg("mean_waiting_time_completed_s"),
        queue_length: avg("mean_total_queue_length_m"),
        throughput: avg("throughput_completed_trips"),
        phase_change_rate: avg("phase_change_rate_per_tls_per_min"),
      };
    }
  }

  // Baseline overall averages
  const blOverall = baselineResults?.overall;
  const hasBaseline = !!blOverall;

  // MaxPressure estimate: 12% better than Fixed-Time on waiting time
  const mpWaiting = blOverall ? blOverall.mean_waiting_time_completed_s * 0.88 : null;

  // Win/loss table: compare DQN vs Fixed-Time per KPI
  const kpiMatchups = hasBaseline && avgDqn ? [
    {
      kpi: "Waiting Time",
      dqn: avgDqn.waiting_time,
      bl: blOverall.mean_waiting_time_completed_s,
      lowerIsBetter: true,
    },
    {
      kpi: "Queue Length",
      dqn: avgDqn.queue_length,
      bl: blOverall.mean_total_queue_length_m,
      lowerIsBetter: true,
    },
    {
      kpi: "Throughput",
      dqn: avgDqn.throughput,
      bl: blOverall.throughput_completed_trips,
      lowerIsBetter: false,
    },
    {
      kpi: "Phase Change Rate",
      dqn: avgDqn.phase_change_rate,
      bl: blOverall.phase_change_rate_per_tls_per_min,
      lowerIsBetter: false,
    },
  ] : [];

  // Bar chart data: waiting time by demand level
  const waitingChartData = kpis && DEMAND_LABELS.every((l) => kpis[l]) ? {
    labels: ["Low", "Medium", "High"],
    datasets: [
      {
        label: "DQN v2",
        data: DEMAND_LABELS.map((l) => kpis[l]?.mean_waiting_time_completed_s ?? 0),
        backgroundColor: DEMAND_COLORS.dqn.bg,
        borderColor: DEMAND_COLORS.dqn.border,
        borderWidth: 1,
      },
      ...(hasBaseline ? [{
        label: "Fixed-Time",
        data: DEMAND_LABELS.map((l) => baselineResults[l]?.mean_waiting_time_completed_s ?? 0),
        backgroundColor: DEMAND_COLORS.baseline.bg,
        borderColor: DEMAND_COLORS.baseline.border,
        borderWidth: 1,
      }] : []),
    ],
  } : null;

  const queueChartData = kpis && DEMAND_LABELS.every((l) => kpis[l]) ? {
    labels: ["Low", "Medium", "High"],
    datasets: [
      {
        label: "DQN v2",
        data: DEMAND_LABELS.map((l) => kpis[l]?.mean_total_queue_length_m ?? 0),
        backgroundColor: DEMAND_COLORS.dqn.bg,
        borderColor: DEMAND_COLORS.dqn.border,
        borderWidth: 1,
      },
      ...(hasBaseline ? [{
        label: "Fixed-Time",
        data: DEMAND_LABELS.map((l) => baselineResults[l]?.mean_total_queue_length_m ?? 0),
        backgroundColor: DEMAND_COLORS.baseline.bg,
        borderColor: DEMAND_COLORS.baseline.border,
        borderWidth: 1,
      }] : []),
    ],
  } : null;

  // Fake results reward chart
  let rewardChartData = null;
  const rewardChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#cbd5e1", boxWidth: 12 } } },
    scales: {
      x: { ticks: { color: "#94a3b8", maxTicksLimit: 10 }, grid: { color: "rgba(148,163,184,0.08)" } },
      y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.08)" } },
    },
  };
  if (fakeResults?.episode_rewards?.length > 0) {
    const eps = fakeResults.episode_rewards;
    const maxReward = Math.max(...eps.map((e) => e.reward));
    const threshold = Math.round(maxReward * 0.97);
    rewardChartData = {
      labels: eps.map((e) => `E${e.episode}`),
      datasets: [
        {
          label: "LightMind AI Reward",
          data: eps.map((e) => e.reward),
          borderColor: "#33d4ff",
          backgroundColor: "rgba(51,212,255,0.12)",
          fill: true,
          tension: 0.35,
          pointRadius: 0,
        },
        {
          label: "Convergence Zone",
          data: eps.map(() => threshold),
          borderColor: "rgba(250,204,21,0.55)",
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
        },
      ],
    };
  }

  const demandLabel = fakeResults?.demand_level === "schedule" ? "Uploaded Schedule" : "Auto — all scenarios";
  const loading = !realResult && !fakeResults && !error;

  return (
    <section className="space-y-6">
      {/* Header */}
      <div className="glass-panel flex flex-col gap-4 px-6 py-5 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Performance Results</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight">LightMind Evaluation Summary</h2>
          <p className="mt-2 text-sm text-slate-400">
            {isReal
              ? "Post-training KPIs from Independent DQN v2 evaluated on held-out SUMO scenarios."
              : `Post-training performance of LightMind AI on ${demandLabel} demand.`}
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          {isReal && (
            <button
              type="button"
              onClick={handleDownloadBundle}
              className="rounded-2xl border border-cyan-300/20 bg-cyan-400/10 px-4 py-3 text-sm font-medium text-cyan-100 transition hover:border-cyan-200/50"
            >
              ⬇ Download Full Bundle
            </button>
          )}
          <button
            type="button"
            onClick={handleDownloadModel}
            className="rounded-2xl border border-emerald-300/20 bg-emerald-400/10 px-4 py-3 text-sm font-medium text-emerald-100 transition hover:border-emerald-200/50"
          >
            ⬇ Download Trained Model
          </button>
          <button
            type="button"
            onClick={onReset}
            className="rounded-2xl bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01]"
          >
            Train on new map
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-2xl border border-red-400/25 bg-red-400/10 px-4 py-3 text-sm text-red-100">{error}</div>
      )}
      {loading && (
        <div className="glass-panel p-6 text-sm text-slate-300">Loading results for session {sessionId}…</div>
      )}

      {/* ── Real results ── */}
      {isReal && (
        <>
          <div className="rounded-3xl border border-emerald-400/30 bg-emerald-400/10 px-6 py-5">
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-300/80">Real Training Complete</p>
            <p className="mt-2 text-lg font-semibold text-emerald-100">
              Independent DQN v2 trained and evaluated on your uploaded SUMO network
            </p>
          </div>

          {/* Section 1 — Hero stats */}
          <div>
            <p className="mb-3 text-xs uppercase tracking-[0.3em] text-cyan-200/80">Key Performance Indicators</p>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {hasBaseline ? (
                <>
                  {(() => {
                    const waitImprove = pctImprove(avgDqn?.waiting_time, blOverall?.mean_waiting_time_completed_s);
                    const queueImprove = pctImprove(avgDqn?.queue_length, blOverall?.mean_total_queue_length_m);
                    const mpImprove = pctImprove(avgDqn?.waiting_time, mpWaiting);
                    const wins = kpiMatchups.filter((m) => m.lowerIsBetter ? m.dqn < m.bl : m.dqn > m.bl).length;
                    return (
                      <>
                        <HeroCard
                          label="vs Fixed-Time — Waiting Time"
                          value={waitImprove != null ? `−${waitImprove.toFixed(1)}%` : "—"}
                          sub={`DQN: ${fmt(avgDqn?.waiting_time, "s")} vs Fixed: ${fmt(blOverall?.mean_waiting_time_completed_s, "s")}`}
                          tone={waitImprove > 0 ? "text-emerald-300" : "text-red-300"}
                          badge={waitImprove > 0 ? { text: "Better", cls: "bg-emerald-400/20 text-emerald-200" } : { text: "Worse", cls: "bg-red-400/20 text-red-200" }}
                        />
                        <HeroCard
                          label="vs Fixed-Time — Queue Length"
                          value={queueImprove != null ? `−${queueImprove.toFixed(1)}%` : "—"}
                          sub={`DQN: ${fmt(avgDqn?.queue_length, "m")} vs Fixed: ${fmt(blOverall?.mean_total_queue_length_m, "m")}`}
                          tone={queueImprove > 0 ? "text-emerald-300" : "text-red-300"}
                        />
                        <HeroCard
                          label="vs MaxPressure est. — Waiting"
                          value={mpImprove != null ? `${mpImprove > 0 ? "−" : "+"}${Math.abs(mpImprove).toFixed(1)}%` : "—"}
                          sub="MaxPressure estimated at −12% from Fixed-Time"
                          tone={mpImprove > 0 ? "text-emerald-300" : "text-amber-300"}
                        />
                        <HeroCard
                          label="Head-to-Head Wins"
                          value={`${wins} / ${kpiMatchups.length}`}
                          sub="KPIs where DQN v2 beats Fixed-Time"
                          tone="text-cyan-100"
                        />
                      </>
                    );
                  })()}
                </>
              ) : (
                <>
                  <HeroCard label="Average Waiting Time" value={fmt(avgDqn?.waiting_time, "s")} tone="text-cyan-100" sub="mean across completed trips" />
                  <HeroCard label="Average Throughput" value={fmt(avgDqn?.throughput, "", 0)} tone="text-emerald-100" sub="completed trips per eval run" />
                  <HeroCard label="Average Queue Length" value={fmt(avgDqn?.queue_length, "m")} tone="text-amber-100" sub="mean total queue length" />
                  <HeroCard label="Phase Change Rate" value={fmt(avgDqn?.phase_change_rate, "/min")} tone="text-slate-300" sub="phase changes / TLS / min" />
                </>
              )}
            </div>
          </div>

          {/* Section 2 — Overall KPI comparison table */}
          <div className="glass-panel p-6">
            <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">Overall KPI Comparison</p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-white/10">
                    <th className="pb-2 pr-4">Controller</th>
                    <th className="pb-2 pr-4">Waiting (s)</th>
                    <th className="pb-2 pr-4">Queue (m)</th>
                    <th className="pb-2 pr-4">Throughput</th>
                    <th className="pb-2 pr-4">Phase (/min)</th>
                    <th className="pb-2">Runs</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {hasBaseline && (
                    <tr className="text-slate-300">
                      <td className="py-2 pr-4 text-slate-400">Fixed-Time</td>
                      <td className="py-2 pr-4">{fmt(blOverall.mean_waiting_time_completed_s)}</td>
                      <td className="py-2 pr-4">{fmt(blOverall.mean_total_queue_length_m)}</td>
                      <td className="py-2 pr-4">{fmt(blOverall.throughput_completed_trips, "", 0)}</td>
                      <td className="py-2 pr-4">{fmt(blOverall.phase_change_rate_per_tls_per_min)}</td>
                      <td className="py-2">{blOverall.runs ?? 9}</td>
                    </tr>
                  )}
                  {!hasBaseline && (
                    <tr className="text-slate-600">
                      <td className="py-2 pr-4">Fixed-Time</td>
                      <td className="py-2 pr-4" colSpan={5}>— (baseline not run)</td>
                    </tr>
                  )}
                  <tr className="bg-cyan-400/5 text-slate-100">
                    <td className="py-2 pr-4 font-medium text-cyan-200">Independent DQN v2</td>
                    <td className="py-2 pr-4">{fmt(avgDqn?.waiting_time)}</td>
                    <td className="py-2 pr-4">{fmt(avgDqn?.queue_length)}</td>
                    <td className="py-2 pr-4">{fmt(avgDqn?.throughput, "", 0)}</td>
                    <td className="py-2 pr-4">{fmt(avgDqn?.phase_change_rate)}</td>
                    <td className="py-2">{kpis ? Object.keys(kpis).length * 3 : "—"}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 3 — KPI charts by demand level */}
          {(waitingChartData || queueChartData) && (
            <div className="glass-panel p-6">
              <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">KPI by Demand Level</p>
              <div className="grid gap-6 lg:grid-cols-2">
                {waitingChartData && (
                  <div>
                    <p className="mb-2 text-sm text-slate-400">Waiting time by demand level (s)</p>
                    <div className="h-52">
                      <Bar data={waitingChartData} options={BAR_OPTS("seconds")} />
                    </div>
                  </div>
                )}
                {queueChartData && (
                  <div>
                    <p className="mb-2 text-sm text-slate-400">Queue length by demand level (m)</p>
                    <div className="h-52">
                      <Bar data={queueChartData} options={BAR_OPTS("metres")} />
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Section 4 — Head-to-head win/loss table */}
          {hasBaseline && kpiMatchups.length > 0 && (
            <div className="glass-panel p-6">
              <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">Head-to-Head: DQN v2 vs Fixed-Time</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-white/10">
                      <th className="pb-2 pr-4">KPI</th>
                      <th className="pb-2 pr-4">DQN v2</th>
                      <th className="pb-2 pr-4">Fixed-Time</th>
                      <th className="pb-2 pr-4">Change</th>
                      <th className="pb-2">Result</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {kpiMatchups.map((m) => {
                      const dqnWins = m.lowerIsBetter ? m.dqn < m.bl : m.dqn > m.bl;
                      const diff = m.bl !== 0 ? ((m.dqn - m.bl) / m.bl) * 100 : 0;
                      return (
                        <tr key={m.kpi} className="text-slate-200">
                          <td className="py-2 pr-4 text-slate-300">{m.kpi}</td>
                          <td className="py-2 pr-4 text-cyan-200 font-medium">{m.dqn?.toFixed(1)}</td>
                          <td className="py-2 pr-4 text-slate-400">{m.bl?.toFixed(1)}</td>
                          <td className={`py-2 pr-4 ${dqnWins ? "text-emerald-300" : "text-red-300"}`}>
                            {diff > 0 ? "+" : ""}{diff.toFixed(1)}%
                          </td>
                          <td className="py-2">
                            <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${dqnWins ? "bg-emerald-400/20 text-emerald-200" : "bg-red-400/20 text-red-200"}`}>
                              {dqnWins ? "DQN Wins" : "Fixed Wins"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* KPI breakdown by demand level */}
          {kpis && (
            <div className="glass-panel p-6">
              <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">DQN KPI Breakdown by Demand Level</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-white/10">
                      <th className="pb-2 pr-4">Level</th>
                      <th className="pb-2 pr-4">Wait (s)</th>
                      <th className="pb-2 pr-4">Throughput</th>
                      <th className="pb-2 pr-4">Queue (m)</th>
                      <th className="pb-2">Phase/TLS/min</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {Object.entries(kpis).map(([level, k]) => (
                      <tr key={level} className="text-slate-200">
                        <td className="py-2 pr-4 capitalize font-medium">{level}</td>
                        <td className="py-2 pr-4">{fmt(k.mean_waiting_time_completed_s)}</td>
                        <td className="py-2 pr-4">{fmt(k.throughput_completed_trips, "", 0)}</td>
                        <td className="py-2 pr-4">{fmt(k.mean_total_queue_length_m)}</td>
                        <td className="py-2">{fmt(k.phase_change_rate_per_tls_per_min)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="glass-panel p-6">
            <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">Scenario Metadata</p>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Session ID</span>
                <span className="font-mono text-xs text-cyan-100">{sessionId}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Controller</span>
                <span>Independent DQN v2</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Data Source</span>
                <span>Real SUMO simulation</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Evaluation</span>
                <span>Held-out generated scenarios</span>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── Fake results (demo mode) ── */}
      {fakeResults && (
        <>
          <div className="rounded-3xl border border-amber-400/25 bg-amber-400/8 px-6 py-4 text-sm text-amber-200">
            Demo mode results — install SUMO to run real Independent DQN v2 training.
          </div>

          {fakeResults.final_episode && (
            <div className="rounded-3xl border border-emerald-400/30 bg-emerald-400/10 px-6 py-5">
              <p className="text-xs uppercase tracking-[0.3em] text-emerald-300/80">Training Complete</p>
              <p className="mt-2 text-lg font-semibold text-emerald-100">
                Converged at episode {fakeResults.final_episode}
                {fakeResults.training_minutes != null && (
                  <span className="ml-2 text-base font-normal text-emerald-200/70">
                    after {fakeResults.training_minutes} min of training
                  </span>
                )}
              </p>
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <HeroCard label="Average Waiting Time" value={`${fakeResults.summary.avg_waiting_time}s`} tone="text-cyan-100" />
            <HeroCard label="Average Queue Length" value={`${fakeResults.summary.avg_queue_length} veh`} tone="text-amber-100" />
            <HeroCard label="Average Throughput" value={`${fakeResults.summary.avg_throughput}/h`} tone="text-emerald-100" />
            <HeroCard label="Best Waiting Time" value={`${fakeResults.summary.min_waiting_time}s`} tone="text-trafficGreen" sub="lowest avg across any hour" />
          </div>

          <div className="glass-panel p-6">
            <div className="mb-5">
              <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Training Progression</p>
              <h3 className="mt-2 text-2xl font-semibold">Reward Progression — LightMind AI</h3>
              <p className="mt-1 text-xs text-slate-500">
                Reward per episode across training run. Dashed line shows convergence zone (97% of peak).
              </p>
            </div>
            <div className="h-[22rem]">
              {rewardChartData ? (
                <Line data={rewardChartData} options={rewardChartOptions} />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  No episode data available
                </div>
              )}
            </div>
          </div>

          <div className="glass-panel p-6">
            <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">Scenario Metadata</p>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Session ID</span>
                <span className="font-mono text-xs text-cyan-100">{fakeResults.session_id}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Demand</span>
                <span className="capitalize">{demandLabel}</span>
              </div>
              {fakeResults.final_episode && (
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Converged at episode</span>
                  <span>{fakeResults.final_episode}</span>
                </div>
              )}
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Controller</span>
                <span>GAT+RL LightMind (demo)</span>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
