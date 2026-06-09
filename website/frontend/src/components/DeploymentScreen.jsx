import { useEffect, useRef, useState } from "react";
import { API_BASE } from "../config";

const TIER_LABEL  = ["GAT Model", "MaxPressure", "Fixed-Time"];
const TIER_COLOR  = ["text-emerald-300", "text-amber-300", "text-red-300"];
const TIER_BG     = ["bg-emerald-400/15 border-emerald-400/30", "bg-amber-400/15 border-amber-400/30", "bg-red-400/15 border-red-400/30"];
const TIER_DOT    = ["bg-emerald-400", "bg-amber-400", "bg-red-400"];

function fmt(v, unit = "", d = 1) {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : parseFloat(v);
  return isNaN(n) ? "—" : `${n.toFixed(d)}${unit}`;
}

function WatchdogBadge({ tier }) {
  return (
    <div className={`flex items-center gap-2 rounded-xl border px-4 py-2 text-sm font-medium ${TIER_BG[tier] ?? TIER_BG[2]}`}>
      <span className={`h-2.5 w-2.5 rounded-full ${TIER_DOT[tier] ?? TIER_DOT[2]}`} />
      <span className={TIER_COLOR[tier] ?? TIER_COLOR[2]}>
        {TIER_LABEL[tier] ?? "Unknown"} active
      </span>
    </div>
  );
}

function KpiCard({ label, value }) {
  return (
    <div className="glass-panel flex flex-col gap-1 px-5 py-4">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="text-2xl font-bold tracking-tight text-cyan-100">{value}</p>
    </div>
  );
}

function NodeCard({ nodeId, phase, queue, tier }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 space-y-1">
      <p className="text-xs font-mono text-cyan-300 truncate">{nodeId}</p>
      <div className="flex items-center justify-between text-xs text-slate-300">
        <span>Phase <span className="font-semibold text-white">{phase ?? "—"}</span></span>
        <span>Queue <span className="font-semibold text-white">{fmt(queue, " veh", 0)}</span></span>
      </div>
      <span className={`text-xs ${TIER_COLOR[tier ?? 0]}`}>{TIER_LABEL[tier ?? 0]}</span>
    </div>
  );
}

export default function DeploymentScreen({ sessionId, onBack, onReset }) {
  const [status, setStatus] = useState({ status: "stopped", last: null, log: [] });
  const [starting, setStarting] = useState(false);
  const logEndRef = useRef(null);

  // Stop both processes when the user closes the tab or navigates away
  useEffect(() => {
    const stopUrl = `${API_BASE}/api/deployment/${sessionId}/stop`;
    const onUnload = () => {
      // keepalive lets the request outlive the page
      fetch(stopUrl, { method: "POST", keepalive: true }).catch(() => {});
    };
    window.addEventListener("beforeunload", onUnload);
    return () => {
      window.removeEventListener("beforeunload", onUnload);
      // Also fire on SPA navigation away (component unmount)
      fetch(stopUrl, { method: "POST" }).catch(() => {});
    };
  }, [sessionId]);

  // Poll status every 3 s while running
  useEffect(() => {
    if (status.status === "stopped" || status.status === "done" || status.status === "error") return;
    const interval = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/deployment/${sessionId}/status`);
        const d = await r.json();
        setStatus(d);
      } catch (_) {}
    }, 3000);
    return () => clearInterval(interval);
  }, [sessionId, status.status]);

  // Auto-scroll inference log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [status.log?.length]);

  const handleStart = async () => {
    setStarting(true);
    try {
      await fetch(`${API_BASE}/api/deployment/${sessionId}/start`, { method: "POST" });
      setStatus((s) => ({ ...s, status: "starting" }));
    } finally {
      setStarting(false);
    }
  };

  const handleStop = async () => {
    await fetch(`${API_BASE}/api/deployment/${sessionId}/stop`, { method: "POST" });
    setStatus((s) => ({ ...s, status: "stopped" }));
  };

  const last = status.last;
  const tier = last?.fallback_tier ?? 0;
  const isLive = status.status === "running";
  const isRestarting = status.status === "restarting";
  const log = status.log ?? [];

  return (
    <section className="space-y-6">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="glass-panel flex flex-col gap-4 px-6 py-5 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Live Deployment</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight">LightMind — Production</h2>
          <p className="mt-1 text-sm text-slate-400">
            Inference server + SUMO demo running in separate processes, communicating via HTTP.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {isLive && <WatchdogBadge tier={tier} />}
          {isRestarting && (
            <div className="flex items-center gap-2 rounded-xl border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-300">
              <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
              Episode {status.episode} — restarting…
            </div>
          )}
          {isLive && status.episode > 0 && (
            <span className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-400">
              Episode {status.episode}
            </span>
          )}
          {(status.status === "stopped" || status.status === "done") && (
            <button
              onClick={handleStart}
              disabled={starting}
              className="rounded-2xl bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01] disabled:opacity-50"
            >
              {starting ? "Starting…" : "▶ Deploy Model"}
            </button>
          )}
          {(status.status === "starting" || isLive || isRestarting) && (
            <button
              onClick={handleStop}
              className="rounded-2xl border border-red-400/30 bg-red-400/10 px-5 py-3 text-sm font-medium text-red-200 transition hover:border-red-300/60"
            >
              ■ Stop
            </button>
          )}
          <button
            onClick={onBack}
            className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300 transition hover:border-white/20"
          >
            ← Results
          </button>
          <button
            onClick={onReset}
            className="rounded-2xl bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:scale-[1.01]"
          >
            New map
          </button>
        </div>
      </div>

      {status.status === "error" && (
        <div className="rounded-2xl border border-red-400/25 bg-red-400/10 px-5 py-4 text-sm text-red-100">
          {status.error ?? "Deployment error"}
        </div>
      )}

      {/* ── KPI strip ──────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <KpiCard label="Avg Waiting Time" value={fmt(last?.waiting_time, " s")} />
        <KpiCard label="Throughput (arrived)" value={fmt(last?.throughput, "", 0)} />
        <KpiCard label="Vehicles in Network" value={fmt(last?.vehicles, "", 0)} />
        <KpiCard label="Queue Length" value={fmt(last?.queue_length, " veh", 0)} />
      </div>

      {/* ── SUMO-GUI canvas + per-node cards ───────────────────────────── */}
      <div className="grid gap-6 xl:grid-cols-3">

        {/* noVNC live view */}
        <div className="xl:col-span-2 glass-panel p-4 space-y-2">
          <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">
            Live Simulation
            {isLive && (
              <span className="ml-3 inline-flex items-center gap-1.5 text-emerald-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                live
              </span>
            )}
          </p>
          {(status.status === "stopped" && !last) ? (
            <div className="flex h-64 items-center justify-center text-sm text-slate-500">
              Click "Deploy Model" to start the simulation
            </div>
          ) : (
            <div className="overflow-hidden rounded-xl border border-cyan-800/60">
              <iframe
                src={`${API_BASE}/novnc/vnc.html?autoconnect=1&resize=scale&view_only=1`}
                width="100%"
                height="520px"
                className="bg-slate-950"
                title="SUMO Live View"
              />
            </div>
          )}
        </div>

        {/* Per-intersection cards */}
        <div className="glass-panel p-4 space-y-3">
          <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Intersections</p>
          {last?.per_node && Object.keys(last.per_node).length > 0 ? (
            <div className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
              {Object.entries(last.per_node).map(([nid, nd]) => (
                <NodeCard
                  key={nid}
                  nodeId={nid}
                  phase={nd.phase}
                  queue={nd.queue}
                  tier={tier}
                />
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-500">No data yet</p>
          )}
        </div>
      </div>

      {/* ── Inference log + raw obs ─────────────────────────────────────── */}
      <div className="grid gap-6 xl:grid-cols-2">

        {/* Inference log */}
        <div className="glass-panel p-5 space-y-3">
          <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Inference Log</p>
          <div className="max-h-64 overflow-y-auto rounded-xl bg-slate-950/60 p-3 font-mono text-xs">
            {log.length === 0 ? (
              <span className="text-slate-600">Waiting for first decision cycle…</span>
            ) : (
              log.slice().reverse().map((entry, i) => (
                <div key={i} className="flex items-center gap-3 py-0.5 border-b border-white/5">
                  <span className="text-slate-500 w-12">t={fmt(entry.sim_time, "", 0)}</span>
                  <span className="text-slate-400 w-20">{fmt(entry.latency_ms, " ms")}</span>
                  <span className={`${TIER_COLOR[entry.fallback_tier ?? 0]} w-28`}>
                    {TIER_LABEL[entry.fallback_tier ?? 0]}
                  </span>
                  <span className="text-slate-500">{fmt(entry.waiting_time, "s wait")}</span>
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </div>

        {/* Raw obs panel */}
        <div className="glass-panel p-5 space-y-3">
          <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">
            Raw Observation (last cycle, first 20 dims per node)
          </p>
          <div className="max-h-64 overflow-y-auto rounded-xl bg-slate-950/60 p-3 font-mono text-xs">
            {last?.raw_obs && Object.keys(last.raw_obs).length > 0 ? (
              Object.entries(last.raw_obs).map(([nid, obs]) => (
                <div key={nid} className="mb-2">
                  <span className="text-cyan-400">{nid}: </span>
                  <span className="text-slate-400">
                    [{obs.map((v) => v.toFixed(2)).join(", ")}]
                  </span>
                </div>
              ))
            ) : (
              <span className="text-slate-600">No observations received yet</span>
            )}
          </div>
        </div>
      </div>

      {/* Architecture note for presentation */}
      <div className="rounded-2xl border border-cyan-400/15 bg-cyan-400/5 px-6 py-4 text-xs text-slate-400 space-y-1">
        <p className="text-cyan-300 font-medium text-sm">Architecture</p>
        <p>
          <span className="text-slate-300">Inference server</span> (port 8001) — loads checkpoint once,
          runs encoder → GAT → PhaseHead on every cycle. GAT message passing uses real
          neighbour observations from all intersections simultaneously.
        </p>
        <p>
          <span className="text-slate-300">Demo client</span> — reads lane counts from SUMO via TraCI,
          formats them into the training observation vector, POSTs to the inference server,
          applies returned phases back to SUMO. Replace TraCI calls with camera output for
          production.
        </p>
      </div>
    </section>
  );
}
