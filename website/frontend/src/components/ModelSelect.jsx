import { useState } from "react";

export default function ModelSelect({ sessionId, onBack, onTrainingStarted }) {
  const [passThreshold, setPassThreshold] = useState(25);

  const handleStart = () => {
    onTrainingStarted({ passThresholdPct: passThreshold });
  };

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

        {/* Demand schedule — always auto-cycling */}
        <div className="rounded-2xl border border-white/10 bg-slate-950/30 px-4 py-3">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-500 mb-1">Demand Schedule</p>
          <p className="text-sm text-slate-400">Auto — equal Low / Medium / High cycling</p>
        </div>

        {/* Baseline comparison */}
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Baseline Comparison</p>
          <h3 className="mt-2 text-xl font-semibold">Fixed-Time Baseline</h3>
          <div className="mt-4 rounded-2xl border border-emerald-400/20 bg-emerald-400/5 px-4 py-4">
            <p className="text-sm font-medium text-emerald-100">Always included — no setup needed</p>
            <p className="mt-1 text-xs text-slate-400">
              After GAT training, 5 greedy evaluation episodes run automatically using the network's built-in fixed-time TL program (seeds 0, 100, 200, 300, 400). Results are compared side-by-side on the results page.
            </p>
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
                Auto — equal Low / Medium / High cycling
              </span>
            </div>
            <div className="flex items-start justify-between gap-3">
              <span className="shrink-0 text-slate-400">Baseline</span>
              <span className="text-right text-emerald-200 text-xs">Fixed-Time (5 episodes, built-in program)</span>
            </div>
            <div className="flex items-start justify-between gap-3">
              <span className="shrink-0 text-slate-400">Episodes</span>
              <span className="text-right text-slate-300">
                Until convergence
                <span className="block text-xs text-slate-500">(est. 80–200 based on network)</span>
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="shrink-0 text-slate-400">Pass threshold</span>
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  min={1}
                  max={99}
                  value={passThreshold}
                  onChange={e => setPassThreshold(Math.max(1, Math.min(99, Number(e.target.value))))}
                  className="w-14 rounded-lg bg-white/5 border border-white/10 px-2 py-1 text-sm text-right text-cyan-100 focus:outline-none focus:border-cyan-400/50"
                />
                <span className="text-slate-400 text-xs">% wait reduction</span>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-white/8 bg-slate-950/30 p-4">
          <p className="text-xs text-slate-500">
            Training runs for a fixed number of episodes scaled to your network size. After training, 5 fixed-time baseline episodes run automatically for comparison.
          </p>
        </div>

        <div className="mt-auto flex flex-col gap-3">
          <button
            type="button"
            onClick={handleStart}
            className="rounded-2xl bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01]"
          >
            Start Training
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
