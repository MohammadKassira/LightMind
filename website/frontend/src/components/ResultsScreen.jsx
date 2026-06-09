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

import { API_BASE } from "../config";

function fmt(val, unit = "", decimals = 1) {
  if (val == null) return "—";
  const n = typeof val === "number" ? val : parseFloat(val);
  return isNaN(n) ? "—" : `${n.toFixed(decimals)}${unit}`;
}

function pctImprove(gatVal, baseVal) {
  if (gatVal == null || baseVal == null || baseVal === 0) return null;
  return ((baseVal - gatVal) / baseVal) * 100;
}

function avgArr(arr) {
  if (!arr?.length) return null;
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function rollingMean(arr, window = 20) {
  return arr.map((_, i) => {
    const slice = arr.slice(Math.max(0, i - window + 1), i + 1);
    return slice.reduce((s, v) => s + v, 0) / slice.length;
  });
}

function VerdictBadge({ verdict, onDeploy, onRetrain }) {
  if (!verdict || verdict.passed === null || verdict.passed === undefined) return null;
  const pass = verdict.passed;
  return (
    <div className={`flex flex-col gap-4 rounded-2xl border p-5 sm:flex-row sm:items-center sm:justify-between ${
      pass
        ? "border-emerald-400/30 bg-emerald-400/10"
        : "border-red-400/30 bg-red-400/10"
    }`}>
      <div className={`flex items-center gap-3 ${pass ? "text-emerald-200" : "text-red-200"}`}>
        <span className="text-2xl font-bold leading-none">{pass ? "✓" : "✗"}</span>
        <div>
          <p className="font-semibold text-sm">{pass ? "PASS" : "FAIL"}</p>
          <p className="text-xs opacity-75">
            {verdict.wait_improvement_pct > 0 ? "+" : ""}
            {verdict.wait_improvement_pct}% wait vs fixed-time
            {" · "}threshold {verdict.threshold_pct}%
          </p>
        </div>
      </div>
      {pass && onDeploy && (
        <button
          type="button"
          onClick={onDeploy}
          className="rounded-2xl bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01]"
        >
          Deploy Model →
        </button>
      )}
      {!pass && onRetrain && (
        <button
          type="button"
          onClick={onRetrain}
          className="rounded-2xl border border-amber-400/40 bg-amber-400/10 px-5 py-2.5 text-sm font-semibold text-amber-200 transition hover:border-amber-300/60"
        >
          Retrain Model →
        </button>
      )}
    </div>
  );
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

function EvalComparisonTable({ gat, fixedTime }) {
  if (!gat || !fixedTime) return null;
  const waitImprove = fixedTime.mean_waiting_time > 0
    ? ((fixedTime.mean_waiting_time - gat.mean_waiting_time) / fixedTime.mean_waiting_time) * 100
    : null;
  const tputImprove = fixedTime.mean_throughput > 0
    ? ((gat.mean_throughput - fixedTime.mean_throughput) / fixedTime.mean_throughput) * 100
    : null;
  return (
    <div className="glass-panel p-6">
      <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">
        SUMO-GUI Evaluation — GAT vs Fixed-Time (5 episodes each)
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-400 border-b border-white/10">
              <th className="pb-2 pr-4">Controller</th>
              <th className="pb-2 pr-4">Mean Wait (s)</th>
              <th className="pb-2">Mean Throughput</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            <tr className="text-slate-300">
              <td className="py-2 pr-4 text-slate-400">Fixed-Time (net.xml program)</td>
              <td className="py-2 pr-4">{fixedTime.mean_waiting_time?.toFixed(2) ?? "—"}</td>
              <td className="py-2">{fixedTime.mean_throughput?.toFixed(0) ?? "—"}</td>
            </tr>
            <tr className="bg-cyan-400/5 text-slate-100">
              <td className="py-2 pr-4 font-medium text-cyan-200">GAT model (LightMind)</td>
              <td className={`py-2 pr-4 font-medium ${waitImprove != null ? (waitImprove > 0 ? "text-emerald-300" : "text-red-300") : ""}`}>
                {gat.mean_waiting_time?.toFixed(2) ?? "—"}
                {waitImprove != null && (
                  <span className="ml-2 text-xs">
                    ({waitImprove > 0 ? "−" : "+"}{Math.abs(waitImprove).toFixed(1)}%)
                  </span>
                )}
              </td>
              <td className={`py-2 font-medium ${tputImprove != null ? (tputImprove > 0 ? "text-emerald-300" : "text-red-300") : ""}`}>
                {gat.mean_throughput?.toFixed(0) ?? "—"}
                {tputImprove != null && (
                  <span className="ml-2 text-xs">
                    ({tputImprove > 0 ? "+" : ""}{tputImprove.toFixed(1)}%)
                  </span>
                )}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function ResultsScreen({ sessionId, onReset, onDeploy, onRetrain }) {
  const [realResult, setRealResult] = useState(null);
  const [error, setError] = useState("");
  const [evalStatus, setEvalStatus] = useState({ phase: "not_started" });
  const [evalComparison, setEvalComparison] = useState(null);

  useEffect(() => {
    if (!sessionId) return;
    const controller = new AbortController();

    fetch(`${API_BASE}/api/real-train/${sessionId}/result`, { signal: controller.signal })
      .then((r) => r.json())
      .then((data) => {
        if (data.available) {
          setRealResult(data);
        } else {
          setError("Training did not complete — no results available.");
        }
      })
      .catch((err) => {
        if (err.name !== "AbortError") setError("Failed to load results. Please try again.");
      });

    return () => controller.abort();
  }, [sessionId]);

  // Poll eval status every 3 s once started
  useEffect(() => {
    if (!sessionId || evalStatus.phase === "not_started" || evalStatus.phase === "done" || evalStatus.phase === "error") return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/real-train/${sessionId}/eval-status`);
        const data = await res.json();
        setEvalStatus(data);
        if (data.phase === "done") {
          clearInterval(interval);
          const cmp = await fetch(`${API_BASE}/api/real-train/${sessionId}/eval-comparison`);
          setEvalComparison(await cmp.json());
        }
      } catch (_) {}
    }, 3000);
    return () => clearInterval(interval);
  }, [sessionId, evalStatus.phase]);

  const handleStartEval = async () => {
    setEvalStatus({ phase: "fixed_time", episode1_running: true });
    await fetch(`${API_BASE}/api/real-train/${sessionId}/start-eval`, { method: "POST" });
  };

  const handleDownloadModel = () => {
    if (realResult?.available) {
      window.location.href = `${API_BASE}/api/real-train/${sessionId}/download-model`;
    }
  };

  const handleDownloadBundle = () => {
    window.location.href = `${API_BASE}/api/real-train/${sessionId}/download-bundle`;
  };

  // ── KPIs — prefer GUI eval data (evalComparison), fall back to job_result eval_metrics ──
  // Post-training headless eval is skipped by the web pipeline (--eval-episodes 0),
  // so evalMetrics may be empty. Once the user runs GUI eval the comparison data fills in.
  const evalMetrics  = realResult?.eval_metrics ?? {};
  const guiGat       = evalComparison?.gat ?? {};         // richer source: GUI eval
  const blData       = realResult?.baseline_metrics;
  const blMetrics    = blData?.available ? (blData.metrics ?? {}) : {};
  const hasBaseline  = blData?.available === true;

  const avgWait  = avgArr(guiGat.avg_waiting_time)  ?? guiGat.mean_waiting_time
                ?? avgArr(evalMetrics.avg_waiting_time)  ?? evalMetrics.mean_waiting_time  ?? null;
  const avgQueue = avgArr(evalMetrics.avg_queue_length)  ?? null;
  const avgTput  = avgArr(guiGat.throughput)        ?? guiGat.mean_throughput
                ?? avgArr(evalMetrics.throughput)        ?? null;

  const blWait   = avgArr(evalComparison?.fixed_time?.avg_waiting_time) ?? evalComparison?.fixed_time?.mean_waiting_time
                ?? avgArr(blMetrics.avg_waiting_time)   ?? blMetrics.mean_waiting_time  ?? null;
  const blQueue  = null;  // not tracked in fixed-time eval
  const blTput   = avgArr(evalComparison?.fixed_time?.throughput) ?? evalComparison?.fixed_time?.mean_throughput
                ?? avgArr(blMetrics.throughput)         ?? null;

  const episodeCount = realResult?.training_metrics?.episode_returns?.length ?? 0;

  // ── Reward curve ──
  const trainingReturns = realResult?.training_metrics?.episode_returns ?? [];
  const smoothed = trainingReturns.length > 1 ? rollingMean(trainingReturns) : [];

  // Downsample raw scatter to max 500 points
  const rawStep = Math.max(1, Math.floor(trainingReturns.length / 500));
  const rawScatter = trainingReturns
    .filter((_, i) => i % rawStep === 0)
    .map((v, i) => ({ x: i * rawStep + 1, y: v }));

  const rewardChartData = smoothed.length > 1 ? {
    labels: smoothed.map((_, i) => i + 1),
    datasets: [
      {
        type: "line",
        label: "20-ep rolling mean",
        data: smoothed,
        borderColor: "#33d4ff",
        backgroundColor: "rgba(51,212,255,0.08)",
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
        order: 1,
      },
      {
        type: "scatter",
        label: "Episode reward",
        data: rawScatter,
        backgroundColor: "rgba(51,212,255,0.18)",
        borderColor: "transparent",
        pointRadius: 2,
        order: 2,
      },
    ],
  } : null;

  const rewardChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#94a3b8", boxWidth: 10, font: { size: 11 } } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.raw?.y ?? ctx.raw).toFixed(2)}` } },
    },
    scales: {
      x: { ticks: { color: "#64748b", maxTicksLimit: 10 }, grid: { color: "rgba(148,163,184,0.06)" } },
      y: { ticks: { color: "#64748b" }, grid: { color: "rgba(148,163,184,0.06)" } },
    },
  };

  const loading = !realResult && !error;

  // ── Improvement badges ──
  const waitImprove  = hasBaseline ? pctImprove(avgWait, blWait) : null;
  const queueImprove = hasBaseline ? pctImprove(avgQueue, blQueue) : null;
  const tputImprove  = hasBaseline ? pctImprove(blTput, avgTput) : null; // higher throughput = better

  return (
    <section className="space-y-6">
      {/* Header */}
      <div className="glass-panel flex flex-col gap-4 px-6 py-5 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Performance Results</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight">LightMind Evaluation Summary</h2>
          <p className="mt-2 text-sm text-slate-400">
            GAT model trained and evaluated on your uploaded SUMO network.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          {realResult?.available && (
            <button
              type="button"
              onClick={handleDownloadBundle}
              className="rounded-2xl border border-cyan-300/20 bg-cyan-400/10 px-4 py-3 text-sm font-medium text-cyan-100 transition hover:border-cyan-200/50"
            >
              ⬇ Download Full Bundle
            </button>
          )}
          {realResult?.available && (
            <button
              type="button"
              onClick={handleDownloadModel}
              className="rounded-2xl border border-emerald-300/20 bg-emerald-400/10 px-4 py-3 text-sm font-medium text-emerald-100 transition hover:border-emerald-200/50"
            >
              ⬇ Download Trained Model
            </button>
          )}
          {realResult?.available && onDeploy && evalComparison?.verdict?.passed !== false && (
            <button
              type="button"
              onClick={onDeploy}
              className="rounded-2xl bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01]"
            >
              Deploy Model
            </button>
          )}
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
        <div className="rounded-2xl border border-red-400/25 bg-red-400/10 px-5 py-4 text-sm text-red-100">
          {error}
        </div>
      )}

      {loading && (
        <div className="glass-panel p-6 text-sm text-slate-300">
          Loading results for session {sessionId}…
        </div>
      )}

      {realResult?.available && (
        <>
          {/* Training complete banner */}
          <div className="rounded-3xl border border-emerald-400/30 bg-emerald-400/10 px-6 py-5">
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-300/80">Training Complete</p>
            <p className="mt-2 text-lg font-semibold text-emerald-100">
              GAT model trained on your uploaded SUMO network
              {episodeCount > 0 && (
                <span className="ml-2 text-base font-normal text-emerald-200/70">
                  — {episodeCount.toLocaleString()} episodes
                </span>
              )}
            </p>
          </div>

          {/* SUMO-GUI Evaluation Section */}
          <div className="glass-panel p-6 space-y-4">
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Live SUMO-GUI Evaluation</p>

            {evalStatus.phase === "not_started" && (
              <div className="flex items-center gap-4">
                <button
                  type="button"
                  onClick={handleStartEval}
                  className="rounded-2xl bg-gradient-to-r from-cyan-500 via-sky-500 to-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01]"
                >
                  ▶ Start Evaluation
                </button>
                <p className="text-xs text-slate-400">
                  Watch fixed-time and GAT controllers run live in SUMO-GUI, then compare results.
                </p>
              </div>
            )}

            {(evalStatus.phase === "fixed_time" || evalStatus.phase === "gat") && (
              <p className="text-sm font-medium text-cyan-300">
                {evalStatus.phase === "fixed_time"
                  ? "Phase 1/2 — Running Fixed-Time Baseline…"
                  : "Phase 2/2 — Running GAT Model…"}
              </p>
            )}

            {evalStatus.episode1_running && (
              <div className="w-full rounded-xl overflow-hidden border border-cyan-800/60">
                <p className="text-xs text-cyan-400 px-3 py-1 bg-slate-900">
                  {evalStatus.phase === "fixed_time"
                    ? "Fixed-Time — Episode 1 (Live)"
                    : "GAT Model — Episode 1 (Live)"}
                </p>
                <iframe
                  src={`${API_BASE}/novnc/vnc.html?autoconnect=1&resize=scale&view_only=1`}
                  width="100%"
                  height="600px"
                  className="bg-slate-950"
                  title="SUMO-GUI Live View"
                />
              </div>
            )}

            {(evalStatus.phase === "fixed_time" || evalStatus.phase === "gat") && (
              <p className="text-xs text-slate-500">
                Episodes 2–5 running headless in parallel…
              </p>
            )}

            {evalStatus.phase === "error" && (
              <div className="rounded-2xl border border-red-400/25 bg-red-400/10 px-4 py-3 text-sm text-red-200">
                Evaluation failed: {evalStatus.error ?? "unknown error"}
              </div>
            )}

            {evalStatus.phase === "done" && evalComparison?.available && (
              <>
                <EvalComparisonTable gat={evalComparison.gat} fixedTime={evalComparison.fixed_time} />
                <VerdictBadge
                  verdict={evalComparison.verdict}
                  onDeploy={onDeploy}
                  onRetrain={onRetrain}
                />
              </>
            )}
          </div>

          {/* Section 1 — KPI hero cards */}
          <div>
            <p className="mb-3 text-xs uppercase tracking-[0.3em] text-cyan-200/80">
              Evaluation KPIs{hasBaseline ? " — GAT vs Fixed-Time" : ""}
            </p>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <HeroCard
                label="Avg Waiting Time"
                value={fmt(avgWait, "s")}
                sub={hasBaseline && blWait != null ? `Fixed-Time baseline: ${fmt(blWait, "s")}` : "mean across eval episodes"}
                tone="text-cyan-100"
                badge={waitImprove != null
                  ? waitImprove > 0
                    ? { text: `−${waitImprove.toFixed(1)}% vs Fixed`, cls: "bg-emerald-400/20 text-emerald-200" }
                    : { text: `+${Math.abs(waitImprove).toFixed(1)}% vs Fixed`, cls: "bg-red-400/20 text-red-200" }
                  : undefined}
              />
              <HeroCard
                label="Avg Queue Length"
                value={fmt(avgQueue, "m")}
                sub={hasBaseline && blQueue != null ? `Fixed-Time baseline: ${fmt(blQueue, "m")}` : "mean total queue"}
                tone="text-amber-100"
                badge={queueImprove != null
                  ? queueImprove > 0
                    ? { text: `−${queueImprove.toFixed(1)}% vs Fixed`, cls: "bg-emerald-400/20 text-emerald-200" }
                    : { text: `+${Math.abs(queueImprove).toFixed(1)}% vs Fixed`, cls: "bg-red-400/20 text-red-200" }
                  : undefined}
              />
              <HeroCard
                label="Avg Throughput"
                value={fmt(avgTput, " trips", 0)}
                sub={hasBaseline && blTput != null ? `Fixed-Time baseline: ${fmt(blTput, " trips", 0)}` : "completed trips per eval run"}
                tone="text-emerald-100"
                badge={tputImprove != null
                  ? tputImprove > 0
                    ? { text: `+${tputImprove.toFixed(1)}% vs Fixed`, cls: "bg-emerald-400/20 text-emerald-200" }
                    : { text: `${tputImprove.toFixed(1)}% vs Fixed`, cls: "bg-red-400/20 text-red-200" }
                  : undefined}
              />
              <HeroCard
                label="Training Episodes"
                value={episodeCount > 0 ? episodeCount.toLocaleString() : "—"}
                sub="total GAT training episodes"
                tone="text-slate-200"
              />
            </div>
          </div>

          {/* Section 2 — Fixed-time baseline comparison */}
          <div className="glass-panel p-6">
            <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">
              vs Fixed-Time Baseline
            </p>
            {hasBaseline ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-white/10">
                      <th className="pb-2 pr-4">Controller</th>
                      <th className="pb-2 pr-4">Avg Wait (s)</th>
                      <th className="pb-2 pr-4">Avg Queue (m)</th>
                      <th className="pb-2">Avg Throughput</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    <tr className="text-slate-300">
                      <td className="py-2 pr-4 text-slate-400">Fixed-Time (net.xml program)</td>
                      <td className="py-2 pr-4">{fmt(blWait)}</td>
                      <td className="py-2 pr-4">{fmt(blQueue)}</td>
                      <td className="py-2">{fmt(blTput, "", 0)}</td>
                    </tr>
                    <tr className="bg-cyan-400/5 text-slate-100">
                      <td className="py-2 pr-4 font-medium text-cyan-200">GAT model (LightMind)</td>
                      <td className={`py-2 pr-4 font-medium ${waitImprove != null ? (waitImprove > 0 ? "text-emerald-300" : "text-red-300") : ""}`}>
                        {fmt(avgWait)}
                      </td>
                      <td className={`py-2 pr-4 font-medium ${queueImprove != null ? (queueImprove > 0 ? "text-emerald-300" : "text-red-300") : ""}`}>
                        {fmt(avgQueue)}
                      </td>
                      <td className={`py-2 font-medium ${tputImprove != null ? (tputImprove > 0 ? "text-emerald-300" : "text-red-300") : ""}`}>
                        {fmt(avgTput, "", 0)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="rounded-2xl border border-amber-400/20 bg-amber-400/8 px-4 py-3 text-sm text-amber-200">
                Fixed-time baseline unavailable — it may still be running or failed. Check the training log.
              </div>
            )}
          </div>

          {/* Section 3 — Reward curve */}
          <div className="glass-panel p-6">
            <div className="mb-5">
              <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Training Reward Curve</p>
              <p className="mt-1 text-xs text-slate-500">
                Cyan line: 20-episode rolling mean. Faint dots: raw episode rewards.
              </p>
            </div>
            <div className="h-[22rem]">
              {rewardChartData ? (
                <Line data={rewardChartData} options={rewardChartOptions} />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  No training reward data available
                </div>
              )}
            </div>
          </div>

          {/* Section 4 — Metadata */}
          <div className="glass-panel p-6">
            <p className="mb-4 text-xs uppercase tracking-[0.3em] text-cyan-200/80">Session Metadata</p>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Session ID</span>
                <span className="font-mono text-xs text-cyan-100">{sessionId}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Controller</span>
                <span>GAT model (LightMind)</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Training episodes</span>
                <span>{episodeCount > 0 ? episodeCount.toLocaleString() : "—"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Evaluation</span>
                <span>5 greedy episodes, seeds [0, 100, 200, 300, 400]</span>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
