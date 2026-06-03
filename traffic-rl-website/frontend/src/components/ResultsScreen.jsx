import { useEffect, useState } from "react";
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

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

const API_BASE = "http://localhost:8000";

function SummaryCard({ label, value, tone, sub }) {
  return (
    <div className="glass-panel p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className={`mt-3 text-3xl font-semibold tracking-tight ${tone}`}>{value}</p>
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

function fmt(val, unit = "") {
  if (val == null) return "—";
  const n = typeof val === "number" ? val : parseFloat(val);
  return isNaN(n) ? "—" : `${n.toFixed(1)}${unit}`;
}

export default function ResultsScreen({ sessionId, onReset }) {
  const [fakeResults, setFakeResults] = useState(null);
  const [realResult, setRealResult] = useState(null);   // { available, result, kpis }
  const [error, setError] = useState("");

  useEffect(() => {
    if (!sessionId) return;
    const controller = new AbortController();

    // Check for real training result first
    fetch(`${API_BASE}/api/real-train/${sessionId}/result`, { signal: controller.signal })
      .then((r) => r.json())
      .then((data) => {
        if (data.available) {
          setRealResult(data);
        } else {
          // Fall back to fake results
          return fetch(`${API_BASE}/api/results/${sessionId}`, { signal: controller.signal })
            .then((r) => r.json())
            .then(setFakeResults)
            .catch((err) => { if (err.name !== "AbortError") setError(err.message); });
        }
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          // Real-train endpoint not reachable — fall back to fake
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

  // ── Reward chart (fake results only) ──────────────────────────────────────
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

  const isReal = realResult?.available;
  const kpis = isReal ? realResult.kpis : null;

  // Flatten kpi_summary_by_level (averaged across levels) for display
  let avgKpis = null;
  if (kpis) {
    const levels = Object.values(kpis);
    if (levels.length > 0) {
      const sum = (key) => levels.reduce((acc, l) => acc + (l[key] ?? 0), 0) / levels.length;
      avgKpis = {
        mean_waiting_time_completed_s: sum("mean_waiting_time_completed_s"),
        throughput_completed_trips: sum("throughput_completed_trips"),
        mean_total_queue_length_m: sum("mean_total_queue_length_m"),
        phase_change_rate_per_tls_per_min: sum("phase_change_rate_per_tls_per_min"),
      };
    }
  }

  const demandLabel = fakeResults?.demand_level === "schedule"
    ? "Uploaded Schedule"
    : "Auto — all scenarios";

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
        <div className="rounded-2xl border border-red-400/25 bg-red-400/10 px-4 py-3 text-sm text-red-100">
          {error}
        </div>
      )}

      {loading && (
        <div className="glass-panel p-6 text-sm text-slate-300">
          Loading results for session {sessionId}…
        </div>
      )}

      {/* ── Real results ── */}
      {isReal && (
        <>
          <div className="rounded-3xl border border-emerald-400/30 bg-emerald-400/10 px-6 py-5">
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-300/80">Real Training Complete</p>
            <p className="mt-2 text-lg font-semibold text-emerald-100">
              Independent DQN v2 trained and evaluated on your uploaded map
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <SummaryCard
              label="Average Waiting Time"
              value={fmt(avgKpis?.mean_waiting_time_completed_s, "s")}
              tone="text-cyan-100"
              sub="mean across completed trips"
            />
            <SummaryCard
              label="Average Throughput"
              value={fmt(avgKpis?.throughput_completed_trips)}
              tone="text-emerald-100"
              sub="completed trips per eval run"
            />
            <SummaryCard
              label="Average Queue Length"
              value={fmt(avgKpis?.mean_total_queue_length_m, "m")}
              tone="text-amber-100"
              sub="mean total queue length"
            />
            <SummaryCard
              label="Phase Change Rate"
              value={fmt(avgKpis?.phase_change_rate_per_tls_per_min)}
              tone="text-trafficGreen"
              sub="phase changes / TLS / min"
            />
          </div>

          {/* KPI breakdown by demand level */}
          {kpis && (
            <div className="glass-panel p-6">
              <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">KPI Breakdown by Demand Level</p>
              <div className="mt-4 overflow-x-auto">
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
                        <td className="py-2 pr-4">{fmt(k.throughput_completed_trips)}</td>
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
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Scenario Metadata</p>
            <div className="mt-4 space-y-3 text-sm">
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
            <SummaryCard label="Average Waiting Time" value={`${fakeResults.summary.avg_waiting_time}s`} tone="text-cyan-100" />
            <SummaryCard label="Average Queue Length" value={`${fakeResults.summary.avg_queue_length} veh`} tone="text-amber-100" />
            <SummaryCard label="Average Throughput" value={`${fakeResults.summary.avg_throughput}/h`} tone="text-emerald-100" />
            <SummaryCard label="Best Waiting Time Achieved" value={`${fakeResults.summary.min_waiting_time}s`} tone="text-trafficGreen" sub="lowest avg across any hour" />
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
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Scenario Metadata</p>
            <div className="mt-4 space-y-3 text-sm">
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
