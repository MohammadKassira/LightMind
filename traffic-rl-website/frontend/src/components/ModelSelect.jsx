import { useState } from "react";

const API_BASE = "http://localhost:8000";

export default function ModelSelect({ sessionId, onBack, onTrainingStarted }) {
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState("");

  const handleStart = async () => {
    setError("");
    setIsStarting(true);

    try {
      const response = await fetch(`${API_BASE}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          baseline: "fixed_time",
          demand_level: "auto",
          custom_demand: null,
          demand_schedule: null,
        }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Failed to start training");
      }

      await response.json();
      onTrainingStarted();
    } catch (trainingError) {
      setError(trainingError.message);
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <section className="grid gap-6 xl:grid-cols-[1.4fr_0.9fr]">
      {/* Left: Controller info + demand configuration */}
      <div className="glass-panel flex flex-col gap-6 p-7">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">
              Primary Controller
            </p>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight">
              GAT+RL LightMind Controller
            </h2>
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

        {/* Demand configuration — auto only */}
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">
            Demand Configuration
          </p>
          <h3 className="mt-2 text-xl font-semibold">Traffic Demand</h3>

          <div className="mt-4 rounded-2xl border border-cyan-400/20 bg-cyan-400/8 p-4">
            <p className="text-sm font-medium text-cyan-100">
              Auto: AI will simulate all demand scenarios
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Cycles through Low / Medium / High demand every episode so the model
              learns under all conditions
            </p>
          </div>
        </div>
      </div>

      {/* Right: Run summary + start */}
      <div className="glass-panel flex flex-col gap-6 p-7">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">
            Run Configuration
          </p>
          <h3 className="mt-2 text-2xl font-semibold">Ready to Train</h3>
        </div>

        <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-5">
          <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Run Summary</p>
          <div className="mt-4 space-y-4 text-sm">
            <div className="flex items-start justify-between gap-3">
              <span className="text-slate-400">Controller</span>
              <span className="text-right text-cyan-100">GAT+RL LightMind Controller</span>
            </div>
            <div className="flex items-start justify-between gap-3">
              <span className="shrink-0 text-slate-400">Demand</span>
              <span className="text-right text-cyan-200">Auto — all scenarios</span>
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
            Training stops automatically when the reward curve converges — no manual
            episode cap. Larger networks require more warmup.
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
