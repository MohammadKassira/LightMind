import { useState } from "react";

import { API_BASE } from "../config";

const GREEN_DURATIONS = [30, 45, 60, 90, 120];

export default function ModelSelect({ sessionId, demandData, onBack, onTrainingStarted }) {
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState("");
  const [runBaseline, setRunBaseline] = useState(false);
  const [greenDuration, setGreenDuration] = useState(60);

  const handleStart = async () => {
    setError("");
    setIsStarting(true);

    const sequence = demandData?.sequence || null;

    try {
      const response = await fetch(`${API_BASE}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          baseline: runBaseline ? "fixed_time" : "none",
          demand_level: sequence ? "schedule" : "auto",
          custom_demand: null,
          demand_schedule: sequence,
          run_baseline: runBaseline,
          fixed_time_green_duration: greenDuration,
        }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Failed to start training");
      }

      await response.json();
      onTrainingStarted({
        runBaseline,
        greenDuration,
        demandSequence: sequence,
      });
    } catch (trainingError) {
      setError(trainingError.message);
    } finally {
      setIsStarting(false);
    }
  };

  // Demand summary for Run Summary box
  const demandSummaryLine = (() => {
    if (!demandData) return "Auto — equal Low / Medium / High cycling";
    if (demandData.mixText) return `Custom schedule (${demandData.periods?.length ?? "?"} periods) — ${demandData.mixText}`;
    return `Custom schedule (${demandData.sequence?.length ?? "?"} episodes)`;
  })();

  return (
    <section className="grid gap-6 xl:grid-cols-[1.4fr_0.9fr]">
      {/* Left: Controller info + baseline */}
      <div className="glass-panel flex flex-col gap-6 p-7">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Primary Controller</p>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight">GAT+RL LightMind Controller</h2>
            <p className="mt-3 max-w-2xl text-sm text-slate-400">
              Graph attention + reinforcement learning adaptive signal optimization.
              Trains across all demand conditions to learn a robust control policy.
            </p>
          </div>
          <div className="shrink-0 rounded-2xl border border-cyan-300/25 bg-cyan-400/10 px-4 py-3 text-right">
            <p className="text-xs uppercase tracking-[0.25em] text-cyan-100/70">Session</p>
            <p className="mt-1 font-mono text-xs text-cyan-100">{sessionId}</p>
          </div>
        </div>

        {/* Demand summary — read-only, set on page 1 */}
        {demandData ? (
          <div className="rounded-2xl border border-emerald-400/25 bg-emerald-400/8 px-4 py-3">
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-300/70 mb-1">Demand Schedule</p>
            <p className="text-sm font-medium text-emerald-100">{demandData.previewText}</p>
            {demandData.mixText && (
              <p className="mt-1 text-xs text-slate-400">Effective mix: {demandData.mixText}</p>
            )}
          </div>
        ) : (
          <div className="rounded-2xl border border-white/10 bg-slate-950/30 px-4 py-3">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-500 mb-1">Demand Schedule</p>
            <p className="text-sm text-slate-400">Auto — equal Low / Medium / High cycling</p>
            <p className="mt-1 text-xs text-slate-500">To customize, upload a demand schedule on the previous page.</p>
          </div>
        )}

        {/* Baseline comparison */}
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Baseline Comparison (Optional)</p>
          <h3 className="mt-2 text-xl font-semibold">Fixed-Time Baseline</h3>

          <div className="mt-4 space-y-3">
            {/* Toggle row */}
            <div className="flex items-start gap-4 rounded-2xl border border-white/10 bg-slate-950/30 px-4 py-4">
              <button
                type="button"
                aria-pressed={runBaseline}
                onClick={() => setRunBaseline((v) => !v)}
                className={`relative mt-0.5 h-6 w-11 shrink-0 rounded-full transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
                  runBaseline ? "bg-cyan-500" : "bg-slate-700"
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform duration-200 ${
                    runBaseline ? "translate-x-5" : "translate-x-0"
                  }`}
                />
              </button>
              <div>
                <p className="text-sm font-medium text-slate-200">Run Fixed-Time baseline for comparison</p>
                <p className="mt-1 text-xs text-slate-400">
                  {runBaseline
                    ? "Runs 9 evaluation episodes (3 seeds × Low/Medium/High demand) using fixed signal timing. Takes ~15–20 minutes. Enables performance comparison charts."
                    : "Training only — results will show DQN model performance without comparison."}
                </p>
              </div>
            </div>

            {/* Green phase duration — only visible when baseline is ON */}
            {runBaseline && (
              <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/5 px-4 py-4 space-y-3">
                <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/70">Fixed-Time Cycle Length</p>
                <div className="flex flex-wrap gap-2">
                  {GREEN_DURATIONS.map((d) => (
                    <button
                      key={d}
                      type="button"
                      onClick={() => setGreenDuration(d)}
                      className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                        greenDuration === d
                          ? "bg-cyan-500 text-slate-950"
                          : "border border-white/15 bg-slate-950/50 text-slate-300 hover:border-white/30"
                      }`}
                    >
                      {d}s
                    </button>
                  ))}
                </div>
                <p className="text-xs text-slate-500">
                  How many seconds each green phase lasts before switching. Shorter = more frequent changes. Longer = fewer changes but more waiting.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right: Run summary + start */}
      <div className="glass-panel flex flex-col gap-6 p-7">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Run Configuration</p>
          <h3 className="mt-2 text-2xl font-semibold">Ready to Train</h3>
        </div>

        <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-5">
          <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Run Summary</p>
          <div className="mt-4 space-y-4 text-sm">
            <div className="flex items-start justify-between gap-3">
              <span className="shrink-0 text-slate-400">Controller</span>
              <span className="text-right text-cyan-100">GAT+RL LightMind Controller</span>
            </div>
            <div className="flex items-start justify-between gap-3">
              <span className="shrink-0 text-slate-400">Demand</span>
              <span className="max-w-[60%] text-right text-cyan-200 text-xs leading-snug">
                {demandSummaryLine}
              </span>
            </div>
            <div className="flex items-start justify-between gap-3">
              <span className="shrink-0 text-slate-400">Baseline</span>
              <span className={`text-right ${runBaseline ? "text-amber-200" : "text-slate-500"}`}>
                {runBaseline ? `Fixed-Time ${greenDuration}s green phase (9 runs)` : "None"}
              </span>
            </div>
            <div className="flex items-start justify-between gap-3">
              <span className="shrink-0 text-slate-400">Episodes</span>
              <span className="text-right text-slate-300">
                Until convergence
                <span className="block text-xs text-slate-500">(est. 80–200 based on network)</span>
              </span>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-white/8 bg-slate-950/30 p-4">
          <p className="text-xs text-slate-500">
            Training stops automatically when the reward curve converges. Larger networks require more warmup.
            {runBaseline && ` Baseline runs start after DQN training completes (${greenDuration}s green phases).`}
          </p>
        </div>

        {error && (
          <div className="rounded-2xl border border-red-400/25 bg-red-400/10 px-4 py-3 text-sm text-red-100">
            {error}
          </div>
        )}

        <div className="mt-auto flex flex-col gap-3">
          <button
            type="button"
            onClick={handleStart}
            disabled={isStarting}
            className="rounded-2xl bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isStarting ? "Starting Training…" : "Start Training"}
          </button>
          <button
            type="button"
            onClick={onBack}
            className="rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-medium text-slate-200 transition hover:border-white/20"
          >
            Back
          </button>
        </div>
      </div>
    </section>
  );
}
