import { useEffect, useState } from "react";
import { API_BASE } from "../config";

const MODE_CONFIG = {
  continue: {
    title:       "Resume Training",
    description: "Training stopped before reaching the pass threshold — either manually or by hitting the episode limit. Continue from the last checkpoint with the same epsilon schedule. The model picks up exploration exactly where it left off.",
    cta:         "Resume Training",
    color:       "amber",
  },
  explore: {
    title:       "Re-explore",
    description: "The model converged early but didn't reach the pass threshold. The policy is stuck in a local optimum. Resets epsilon to 0.5 (mid-range exploration) and runs a fresh episode budget from the current weights. The existing replay buffer is kept — old transitions provide diversity while new exploration fills it.",
    cta:         "Re-explore (ε = 0.5)",
    color:       "cyan",
  },
};

export default function RetrainingScreen({ sessionId, onBack, onReset, onStartTraining }) {
  const [stopReason, setStopReason] = useState(null);
  const [loading,    setLoading]    = useState(true);
  const [starting,   setStarting]   = useState(false);
  const [error,      setError]      = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/real-train/${sessionId}/result`)
      .then(r => r.json())
      .then(data => {
        const reason = data?.training_metrics?.stop_reason ?? "completed";
        setStopReason(reason);
      })
      .catch(() => setStopReason("completed"))
      .finally(() => setLoading(false));
  }, [sessionId]);

  const mode = stopReason === "converged" ? "explore" : "continue";
  const cfg  = MODE_CONFIG[mode];

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/real-train/${sessionId}/retrain`,
        {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ mode }),
        }
      );
      const data = await res.json();
      if (data.started) {
        onStartTraining();
      } else {
        setError("Could not start retraining — a job may already be running.");
      }
    } catch {
      setError("Request failed. Check the backend is running.");
    } finally {
      setStarting(false);
    }
  };

  const borderColor = cfg?.color === "cyan"
    ? "border-cyan-400/30 bg-cyan-400/8"
    : "border-amber-400/30 bg-amber-400/8";
  const titleColor  = cfg?.color === "cyan" ? "text-cyan-200" : "text-amber-200";
  const btnClass    = cfg?.color === "cyan"
    ? "bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 text-slate-950"
    : "bg-gradient-to-r from-amber-400 via-orange-400 to-yellow-400 text-slate-950";

  return (
    <section className="space-y-6">
      {/* Header */}
      <div className="glass-panel flex flex-col gap-4 px-6 py-5 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-amber-200/80">Model Improvement</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight">Retraining Pipeline</h2>
          <p className="mt-1 text-sm text-slate-400">
            The GAT model did not meet the pass threshold.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={onBack}
            className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300 transition hover:border-white/20"
          >
            ← Results
          </button>
          <button
            type="button"
            onClick={onReset}
            className="rounded-2xl bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:scale-[1.01]"
          >
            New map
          </button>
        </div>
      </div>

      {/* Mode card */}
      {loading ? (
        <div className="glass-panel p-6 text-sm text-slate-400">Detecting training stop reason…</div>
      ) : (
        <div className={`rounded-2xl border p-6 space-y-4 ${borderColor}`}>
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-slate-400 mb-1">Recommended Strategy</p>
            <h3 className={`text-xl font-semibold ${titleColor}`}>{cfg.title}</h3>
            <p className="mt-2 text-sm text-slate-300 max-w-2xl">{cfg.description}</p>
          </div>

          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span>Stop reason:</span>
            <code className="rounded bg-white/5 px-2 py-0.5 font-mono text-slate-300">{stopReason}</code>
            <span>→</span>
            <code className="rounded bg-white/5 px-2 py-0.5 font-mono text-slate-300">mode={mode}</code>
          </div>

          {error && (
            <p className="text-sm text-red-300">{error}</p>
          )}

          <button
            type="button"
            onClick={handleStart}
            disabled={starting}
            className={`rounded-2xl px-6 py-3 text-sm font-semibold shadow-glow transition hover:scale-[1.01] disabled:opacity-50 ${btnClass}`}
          >
            {starting ? "Starting…" : cfg.cta}
          </button>
        </div>
      )}

      {/* Info box */}
      <div className="rounded-2xl border border-white/8 bg-slate-950/30 px-5 py-4 text-xs text-slate-500 space-y-1">
        <p>After retraining completes, you will be taken back to the results page to re-run the evaluation.</p>
        <p className="font-mono text-slate-600">session: {sessionId}</p>
      </div>
    </section>
  );
}
