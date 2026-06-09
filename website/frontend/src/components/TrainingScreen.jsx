import { useEffect, useRef, useState } from "react";
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
import MetricsPanel from "./MetricsPanel";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

import { API_BASE, WS_BASE } from "../config";

const DEFAULT_SNAPSHOT = {
  episode: 0,
  max_episodes: 500,
  sim_day: "Monday",
  sim_minutes: 0,
  sim_time: "00:00",
  demand_mode: "auto",
  active_demand_level: null,
  convergence_pct: 0,
  convergence_streak: 0,
  stopped_reason: null,
  final_episode: null,
  cars: [],
  lights: [],
  rl: { reward: 0, waiting_time: 0, queue_length: 0, throughput: 0 },
  baseline: { reward: 0, waiting_time: 0, queue_length: 0, throughput: 0 },
};

// Must stay in sync with backend STAGE_ORDER in routers/real_train.py
const STAGE_ORDER = [
  "initialization",
  "random_route_generation",
  "scenario_manifest_creation",
  "independent_dqn_training",
  "evaluation",
  "complete",
];

const STAGE_LABELS = {
  initialization:              "Initializing job",
  random_route_generation:     "Generating traffic demand",
  scenario_manifest_creation:  "Building training config",
  independent_dqn_training:    "Training GAT model",
  evaluation:                  "Running fixed-time baseline",
};

function logColor(msg) {
  if (msg.startsWith("[Ep")) return "text-cyan-300 font-mono break-all";
  if (msg.startsWith("⚠️") || msg.startsWith("🔴") || msg.toLowerCase().includes("fail")) return "text-red-300";
  if (msg.startsWith("✅") || msg.startsWith("🟢") || msg.toLowerCase().includes("complet")) return "text-emerald-300";
  if (msg.startsWith("🚗") || msg.startsWith("🔁")) return "text-cyan-300";
  if (msg.startsWith("📈") || msg.startsWith("📉")) return "text-amber-300";
  return "text-slate-400";
}

function fmtElapsed(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

// ─── KPI helpers ─────────────────────────────────────────────────────────────

function fmtKpi(val, decimals, unit, loading) {
  if (val == null) return loading ? "Loading…" : "—";
  const n = typeof val === "number" ? val : parseFloat(val);
  return isNaN(n) ? "—" : `${n.toFixed(decimals)}${unit}`;
}

function KpiCard({ label, value, tone = "text-cyan-100" }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-slate-950/50 px-4 py-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 text-xl font-semibold ${tone}`}>{value}</p>
    </div>
  );
}

function fmt(val, unit = "") {
  if (val == null) return "—";
  const n = typeof val === "number" ? val : parseFloat(val);
  return isNaN(n) ? "—" : `${n.toFixed(1)}${unit}`;
}

// ─── Stop confirmation banner ────────────────────────────────────────────────

function StopConfirmBanner({ onConfirm, onCancel, currentEpisode, totalEpisodes }) {
  const [acknowledged, setAcknowledged] = useState(false);
  const pct = totalEpisodes > 0 ? Math.round((currentEpisode / totalEpisodes) * 100) : 0;

  return (
    <div className="rounded-3xl border-2 border-red-500/50 bg-red-950/60 p-6 space-y-5">
      {/* Header */}
      <div className="flex items-start gap-3">
        <span className="text-2xl mt-0.5">⚠️</span>
        <div>
          <p className="text-lg font-bold text-red-300 tracking-tight">Stop training early?</p>
          <p className="mt-1 text-sm text-slate-300">
            Training is only <span className="font-semibold text-red-300">{pct}% complete</span> ({currentEpisode} / {totalEpisodes} episodes).
            Stopping now will produce an <span className="font-semibold text-red-300">undertrained model</span>.
          </p>
        </div>
      </div>

      {/* Risk list */}
      <ul className="space-y-2 text-sm text-slate-300 border border-red-400/20 rounded-2xl bg-black/30 px-5 py-4">
        <li className="flex items-start gap-2"><span className="text-red-400 mt-0.5 shrink-0">✗</span> The agent has not finished learning — it may perform worse than a simple fixed-time controller.</li>
        <li className="flex items-start gap-2"><span className="text-red-400 mt-0.5 shrink-0">✗</span> Evaluation results will reflect the model in its current, incomplete state and may be misleading.</li>
        <li className="flex items-start gap-2"><span className="text-amber-400 mt-0.5 shrink-0">→</span> The model will be saved at its current weights and evaluated against the fixed-time baseline.</li>
        <li className="flex items-start gap-2"><span className="text-emerald-400 mt-0.5 shrink-0">✓</span> You can resume training later from this checkpoint to continue where you left off.</li>
      </ul>

      {/* Acknowledgement checkbox */}
      <label className="flex items-center gap-3 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(e) => setAcknowledged(e.target.checked)}
          className="h-4 w-4 rounded border-red-400/50 bg-slate-900 accent-red-500 cursor-pointer"
        />
        <span className="text-sm text-slate-300">
          I understand the model will be undertrained and results may be unreliable.
        </span>
      </label>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          type="button"
          onClick={onConfirm}
          disabled={!acknowledged}
          className="rounded-xl bg-red-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Stop training at my own risk
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-xl border border-white/15 bg-white/5 px-5 py-2.5 text-sm font-medium text-slate-200 transition hover:border-white/30 hover:bg-white/10"
        >
          Keep training
        </button>
      </div>
    </div>
  );
}

// ─── Real training view ──────────────────────────────────────────────────────

function RealTrainingView({ sessionId, netAbsPath, passThresholdPct = 25, onComplete, onReset }) {
  const [status, setStatus] = useState({
    stage: "initializing",
    label: "Initializing job…",
    progress_pct: 0,
    completed_stages: [],
  });
  const [failed, setFailed] = useState(null);
  const [progress, setProgress] = useState({ current_episode: 0, phase: "waiting", available: false });
  const [activityFeed, setActivityFeed] = useState([]);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [liveKpis, setLiveKpis] = useState(null);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [stopRequested, setStopRequested] = useState(false);
  const [showStopConfirm, setShowStopConfirm] = useState(false);

  const statusTimerRef = useRef(null);
  const progressTimerRef = useRef(null);
  const clockTimerRef = useRef(null);
  const metricsTimerRef = useRef(null);
  const onCompleteRef = useRef(onComplete);
  const prevEpisodeRef = useRef(0);
  const prevLogCountRef = useRef(0);
  const latestKpisRef = useRef(null);
  const feedRef = useRef(null);
  const dqnCompleteRef = useRef(false);

  useEffect(() => { onCompleteRef.current = onComplete; });

  // Elapsed-time clock
  useEffect(() => {
    const key = `lm_train_start_${sessionId}`;
    let stored = localStorage.getItem(key);
    if (!stored) {
      stored = String(Date.now());
      localStorage.setItem(key, stored);
    }
    const startMs = parseInt(stored, 10);
    clockTimerRef.current = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startMs) / 1000));
    }, 1000);
    return () => clearInterval(clockTimerRef.current);
  }, [sessionId]);

  // Seed activity feed with network stats
  useEffect(() => {
    fetch(`${API_BASE}/api/sessions/${sessionId}/network`)
      .then((r) => r.json())
      .then((data) => {
        if (!data.available) return;
        const { tl_count, edge_count } = data.stats ?? {};
        const msg = tl_count != null
          ? `Training started — ${tl_count} signals · ${edge_count} roads loaded`
          : "Training started — network loaded";
        setActivityFeed([{ id: "net-init", time: new Date().toLocaleTimeString(), msg }]);
      })
      .catch(() => {
        setActivityFeed([{ id: "init", time: new Date().toLocaleTimeString(), msg: "Training started" }]);
      });
  }, [sessionId]);

  // Auto-scroll feed
  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [activityFeed]);

  // Poll latest episode KPIs every 5 seconds from training_metrics.jsonl
  useEffect(() => {
    const fetchKpis = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/real-train/${sessionId}/latest-episode-kpis`);
        const data = await res.json();
        if (data.available) {
          setLiveKpis(data);
          latestKpisRef.current = data;
          if (Array.isArray(data.history) && data.history.length > 0) {
            setRewardHistory(data.history.map((h) => h.episode_return ?? 0));
          }
        }
      } catch {}
    };
    fetchKpis();
    metricsTimerRef.current = setInterval(fetchKpis, 5000);
    return () => clearInterval(metricsTimerRef.current);
  }, [sessionId]);

  // Start job + poll status (3s) + progress/logs (10s)
  useEffect(() => {
    fetch(`${API_BASE}/api/real-train/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, net_path: netAbsPath, pass_threshold_pct: passThresholdPct }),
    }).catch(() => {});

    statusTimerRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/real-train/${sessionId}/status`);
        const data = await res.json();
        setStatus(data);
        if (data.status === "failed") {
          setFailed(data.error || data.details?.error || "Training failed.");
          clearInterval(statusTimerRef.current);
          clearInterval(progressTimerRef.current);
        } else if (data.stage === "complete" && data.status === "passed") {
          if (!dqnCompleteRef.current) {
            dqnCompleteRef.current = true;
            clearInterval(statusTimerRef.current);
            clearInterval(progressTimerRef.current);
            onCompleteRef.current();
          }
        }
      } catch {}
    }, 3000);

    const pollProgress = async () => {
      try {
        const [progRes, logsRes] = await Promise.all([
          fetch(`${API_BASE}/api/real-train/${sessionId}/training-progress`),
          fetch(`${API_BASE}/api/real-train/${sessionId}/logs`),
        ]);
        const prog = await progRes.json();
        const logsData = await logsRes.json();
        setProgress(prog);

        const newEntries = [];
        const logs = logsData.logs ?? [];
        const alreadyShown = prevLogCountRef.current;
        for (let i = alreadyShown; i < logs.length; i++) {
          newEntries.push({ id: `log-${i}`, time: new Date(logs[i].time + "Z").toLocaleTimeString(), msg: logs[i].message });
        }
        prevLogCountRef.current = logs.length;

        const curEp = prog.current_episode ?? 0;
        prevEpisodeRef.current = curEp;
        if (newEntries.length > 0) setActivityFeed((prev) => [...prev, ...newEntries].slice(-200));
      } catch {}
    };

    pollProgress();
    progressTimerRef.current = setInterval(pollProgress, 10000);

    return () => {
      clearInterval(statusTimerRef.current);
      clearInterval(progressTimerRef.current);
    };
  }, [sessionId, netAbsPath]);


  const completedSet = new Set((status.completed_stages ?? []).map((s) => s.stage));
  const current_episode = liveKpis?.episode ?? progress.current_episode ?? 0;
  const total_episodes = progress.total_episodes ?? 1000;

  // Pre-training stages cap at 5%; training fills 5→95% by episode ratio; evaluation/complete = 100%
  const combinedProgress = (() => {
    const stage = status?.stage;
    if (stage === "complete" || stage === "evaluation") return 100;
    if (stage === "independent_dqn_training") {
      if (current_episode <= 0 || total_episodes <= 0) return 5;
      return Math.min(Math.round(5 + (current_episode / total_episodes) * 90), 95);
    }
    // initialization / random_route_generation / scenario_manifest_creation → 1–4%
    const preStages = ["initialization", "random_route_generation", "scenario_manifest_creation"];
    const idx = preStages.indexOf(stage);
    if (idx >= 0) return idx + 1; // 1%, 2%, 3%
    return 0;
  })();

  // Reward chart data
  const rewardChartData = rewardHistory.length > 1 ? {
    labels: rewardHistory.map((_, i) => `E${i + 1}`),
    datasets: [{
      label: "Episode Reward",
      data: rewardHistory.map((r) => typeof r === "number" ? r : r?.reward ?? r),
      borderColor: "#33d4ff",
      backgroundColor: "rgba(51,212,255,0.08)",
      fill: true,
      tension: 0.35,
      pointRadius: 0,
      borderWidth: 1.5,
    }],
  } : null;

  const rewardChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: { ticks: { color: "#64748b", font: { size: 10 } }, grid: { color: "rgba(148,163,184,0.06)" } },
    },
  };

  return (
    <section className="space-y-5">
      {/* Header */}
      <div className="glass-panel flex flex-col gap-4 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Real Training Session</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight">LightMind Real Training</h2>
          <p className="mt-2 text-sm text-slate-400">
            Training GAT model on your uploaded SUMO network with real simulation.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-4 py-2 text-sm font-medium text-emerald-100">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-trafficGreen shadow-[0_0_16px_rgba(34,197,94,0.9)]" />
            {status.stage === "evaluation" ? "Running baseline…" : "Running"}
          </div>
          <div className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-100">
            {combinedProgress}% pipeline
          </div>
          <div className="rounded-full border border-slate-400/20 bg-slate-400/10 px-4 py-2 text-sm text-slate-300">
            {fmtElapsed(elapsedSec)} elapsed
          </div>
          {!stopRequested && status.stage === "independent_dqn_training" && (
            <button
              type="button"
              onClick={() => setShowStopConfirm(true)}
              className="rounded-2xl border border-red-400/30 bg-red-400/10 px-4 py-2 text-sm text-red-300 transition hover:border-red-400/60 hover:bg-red-400/20"
            >
              Stop Training
            </button>
          )}
          {stopRequested && status.stage === "independent_dqn_training" && (
            <div className="rounded-full border border-amber-400/30 bg-amber-400/10 px-4 py-2 text-sm text-amber-300">
              Stopping after this episode…
            </div>
          )}
          {stopRequested && status.stage === "complete" && (
            <button
              type="button"
              onClick={() => {
                setStopRequested(false);
                fetch(`${API_BASE}/api/real-train/${sessionId}/resume`, { method: "POST" }).catch(() => {});
              }}
              className="rounded-2xl border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm text-emerald-300 transition hover:border-emerald-400/60 hover:bg-emerald-400/20"
            >
              Continue Training
            </button>
          )}
          <button
            type="button"
            onClick={onReset}
            className="rounded-2xl border border-white/15 bg-white/5 px-4 py-2 text-sm text-slate-300 transition hover:border-white/30 hover:bg-white/10"
          >
            ← Start New Map
          </button>
        </div>
      </div>

      {showStopConfirm && <StopConfirmBanner
        onConfirm={() => {
          setShowStopConfirm(false);
          setStopRequested(true);
          fetch(`${API_BASE}/api/real-train/${sessionId}/stop`, { method: "POST" }).catch(() => {});
        }}
        onCancel={() => setShowStopConfirm(false)}
        currentEpisode={current_episode}
        totalEpisodes={total_episodes}
      />}

      {status.stage === "evaluation" ? (
        <div className="rounded-3xl border border-amber-400/30 bg-amber-400/10 px-6 py-4 text-sm text-amber-200">
          Running fixed-time baseline evaluation (5 episodes with built-in TL program)…
        </div>
      ) : (
        <div className="rounded-3xl border border-amber-400/30 bg-amber-400/10 px-6 py-4 text-sm text-amber-200">
          GAT model training is running on your network. SUMO is simulating traffic headlessly. Do not close this tab. Training takes 30–240 minutes depending on map size.
        </div>
      )}

      {failed && (
        <div className="rounded-3xl border border-red-400/30 bg-red-400/10 px-6 py-4 text-sm text-red-200">
          Training failed: {failed}
        </div>
      )}

      {/* Episode progress */}
      {current_episode > 0 && (
        <div className="glass-panel px-6 py-5 space-y-3">
          <div className="flex items-baseline justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Episode Progress</p>
              <p className="mt-2 text-4xl font-bold tracking-tight">
                Episode <span className="text-cyan-300">{current_episode}</span>
                <span className="ml-3 text-lg font-normal text-slate-400">
                  / {total_episodes} · {fmtElapsed(elapsedSec)} elapsed
                </span>
              </p>
            </div>
            <div className="text-right text-sm text-slate-400">
              <p className="text-cyan-300 font-medium">
                {total_episodes > 0 ? Math.round((current_episode / total_episodes) * 100) : 0}%
              </p>
            </div>
          </div>
          <div className="h-2.5 w-full rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all duration-1000"
              style={{ width: `${total_episodes > 0 ? Math.min((current_episode / total_episodes) * 100, 100) : 0}%` }}
            />
          </div>
          <p className="text-xs text-slate-500">{total_episodes} total episodes</p>
        </div>
      )}

      {/* Live episode KPI cards */}
      {current_episode > 0 && (
        <div className="glass-panel px-6 py-5">
          <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80 mb-4">
            Live Episode Metrics — Episode {current_episode}
          </p>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <KpiCard
              label="Avg Wait Time"
              value={fmtKpi(liveKpis?.wait, 1, "s", !liveKpis)}
              tone="text-cyan-100"
            />
            <KpiCard
              label="Queue Length"
              value={fmtKpi(liveKpis?.q, 0, "m", !liveKpis)}
              tone="text-amber-100"
            />
            <KpiCard
              label="Throughput"
              value={fmtKpi(liveKpis?.tput, 0, " trips", !liveKpis)}
              tone="text-emerald-100"
            />
            <KpiCard
              label="Loss"
              value={liveKpis?.loss != null ? fmtKpi(liveKpis.loss, 4, "", false) : (liveKpis ? "warming up" : "—")}
              tone="text-slate-300"
            />
          </div>

          {/* Running reward chart */}
          {rewardChartData && (
            <div className="mt-4">
              <p className="mb-2 text-xs text-slate-500">Reward trend</p>
              <div className="h-20">
                <Line data={rewardChartData} options={rewardChartOptions} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Activity feed + pipeline checklist */}
      <div className="grid gap-5 xl:grid-cols-[1.6fr_0.9fr]">
        {/* Activity feed */}
        <div className="glass-panel p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Training Activity</p>
            <span className="text-xs text-slate-500">{activityFeed.length} events</span>
          </div>
          <div
            ref={feedRef}
            className="max-h-72 space-y-1 overflow-y-auto rounded-2xl border border-white/6 bg-slate-950/70 p-3 font-mono text-xs"
          >
            {activityFeed.length === 0 ? (
              <p className="text-slate-600">Waiting for training to start…</p>
            ) : (
              activityFeed.map((entry) => (
                <div key={entry.id} className="mb-1 last:mb-0 flex gap-2 leading-relaxed">
                  <span className="shrink-0 text-slate-600">{entry.time}</span>
                  <span className={`whitespace-pre-wrap ${logColor(entry.msg)}`}>{entry.msg}</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Pipeline stage checklist */}
        <div className="glass-panel p-6 space-y-4">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Current Stage</p>
            <p className="mt-2 text-xl font-semibold">{status.label ?? status.stage}</p>
          </div>
          <div className="h-2.5 w-full rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all duration-700"
              style={{ width: `${combinedProgress}%` }}
            />
          </div>
          <p className="text-xs text-slate-500">
            {combinedProgress}% — session {sessionId.slice(0, 12)}…
          </p>
          <div className="mt-4 space-y-2">
            {STAGE_ORDER.map((s) => {
              const isDone = s === "complete"
                ? status.stage === "complete" && status.status === "passed"
                : completedSet.has(s) || status.stage === "complete";
              const isCurrent = status.stage === s;
              const label = s === "complete" ? "Evaluating" : (STAGE_LABELS[s] ?? s.replace(/_/g, " "));
              return (
                <div
                  key={s}
                  className={`flex items-center gap-3 text-sm ${
                    isDone ? "text-emerald-300" : isCurrent ? "text-cyan-200" : "text-slate-600"
                  }`}
                >
                  <span className="text-base">{isDone ? "✓" : isCurrent ? "›" : "○"}</span>
                  <span>{label}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

// ─── Fake (demo) training view ───────────────────────────────────────────────

function FakeTrainingView({ sessionId, onComplete, onReset }) {
  const [snapshot, setSnapshot] = useState(DEFAULT_SNAPSHOT);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [activityFeed, setActivityFeed] = useState([]);
  const [, setConnectionState] = useState("connecting");
  const [, setMapCenter] = useState(null); // retained to avoid refactoring WS payload handler
  const socketRef = useRef(null);
  const feedRef = useRef(null);

  useEffect(() => {
    if (!sessionId) {
      setSnapshot(DEFAULT_SNAPSHOT);
      setRewardHistory([]);
      setActivityFeed([]);
      setConnectionState("idle");
      return undefined;
    }

    let isCurrentSession = true;
    setSnapshot(DEFAULT_SNAPSHOT);
    setRewardHistory([]);
    setActivityFeed([]);
    setConnectionState("connecting");

    const socket = new WebSocket(`${WS_BASE}/ws/live/${sessionId}`);
    socketRef.current = socket;

    socket.onopen = () => { if (isCurrentSession) setConnectionState("live"); };

    socket.onmessage = (event) => {
      if (!isCurrentSession) return;
      const payload = JSON.parse(event.data);
      setSnapshot(payload);
      if (payload.map_center) setMapCenter([payload.map_center.lat, payload.map_center.lng]);
      setRewardHistory((cur) => [...cur.slice(-49), { episode: payload.episode, rl: payload.rl.reward }]);
      const logs = payload.activity_logs ?? [];
      if (logs.length > 0) {
        const ts = `${(payload.sim_day ?? "").slice(0, 3)} ${payload.sim_time ?? "00:00"}`;
        setActivityFeed((prev) =>
          [...prev, ...logs.map((msg, i) => ({ id: `${payload.episode}-${i}`, ts, msg }))].slice(-50),
        );
      }
    };

    socket.onerror = () => { if (isCurrentSession) setConnectionState("error"); };
    socket.onclose = () => { if (isCurrentSession) { setConnectionState("closed"); onComplete(); } };

    return () => {
      isCurrentSession = false;
      socket.close();
      if (socketRef.current === socket) socketRef.current = null;
    };
  }, [sessionId, onComplete]);

  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [activityFeed]);

  const convergencePct = snapshot.convergence_pct ?? 0;
  const isConverged = snapshot.stopped_reason === "converged";
  const episodeLabel = isConverged
    ? `Episode ${snapshot.final_episode} / converged`
    : `Episode ${snapshot.episode} / converging…`;

  return (
    <section className="space-y-5">
      <div className="glass-panel flex flex-col gap-4 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">Live Training Session</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight">LightMind Live Training</h2>
          <p className="mt-2 text-sm text-slate-400">
            Training GAT+RL LightMind Controller across all demand scenarios until convergence.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {isConverged ? (
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/40 bg-emerald-400/15 px-4 py-2 text-sm font-medium text-emerald-100">
              <span className="h-2.5 w-2.5 rounded-full bg-trafficGreen shadow-[0_0_10px_rgba(34,197,94,0.8)]" />
              Converged
            </div>
          ) : (
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-4 py-2 text-sm font-medium text-emerald-100">
              <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-trafficGreen shadow-[0_0_16px_rgba(34,197,94,0.9)]" />
              Live
            </div>
          )}
          <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300">
            {episodeLabel}
          </div>
          <div className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-100">
            {convergencePct}% converged
          </div>
          <button
            type="button"
            onClick={onReset}
            className="rounded-2xl border border-white/15 bg-white/5 px-4 py-2 text-sm text-slate-300 transition hover:border-white/30 hover:bg-white/10"
          >
            ← Start New Map
          </button>
        </div>
      </div>

      <div className="rounded-3xl border border-amber-400/30 bg-amber-400/10 px-6 py-4 text-sm text-amber-200">
        SUMO not installed — running in demo mode with simulated training data.
      </div>

      {isConverged && (
        <div className="rounded-3xl border border-emerald-400/30 bg-emerald-400/10 px-6 py-5">
          <p className="text-xs uppercase tracking-[0.3em] text-emerald-300/80">Training Complete</p>
          <p className="mt-2 text-lg font-semibold text-emerald-100">
            Model converged at episode {snapshot.final_episode} — ready for deployment
          </p>
        </div>
      )}

      <div className="grid gap-5 xl:grid-cols-[1.4fr_0.9fr]">
        <div className="flex flex-col gap-4">
          <div className="glass-panel p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Network Activity Feed</p>
              <span className="text-xs text-slate-500">{activityFeed.length} events</span>
            </div>
            <div
              ref={feedRef}
              className="max-h-72 space-y-1 overflow-y-auto rounded-2xl border border-white/6 bg-slate-950/70 p-3 font-mono text-xs"
            >
              {activityFeed.length === 0 ? (
                <p className="text-slate-600">Waiting for first episode…</p>
              ) : (
                activityFeed.map((entry) => (
                  <div key={entry.id} className="flex gap-2 leading-relaxed">
                    <span className="shrink-0 text-slate-600">{entry.ts}</span>
                    <span className={logColor(entry.msg)}>{entry.msg}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
        <MetricsPanel
          snapshot={snapshot}
          rewardHistory={rewardHistory}
          convergencePct={convergencePct}
          simDay={snapshot.sim_day}
          simTime={snapshot.sim_time}
          demandMode={snapshot.demand_mode}
          activeDemandLevel={snapshot.active_demand_level}
        />
      </div>
    </section>
  );
}

// ─── Root — routes to real vs demo view ─────────────────────────────────────

export default function TrainingScreen({ sessionId, netAbsPath, runBaseline, greenDuration, passThresholdPct = 25, onComplete, onReset }) {
  const [mode, setMode] = useState(null);

  useEffect(() => {
    if (!sessionId) return;
    fetch(`${API_BASE}/api/system/status`)
      .then((r) => r.json())
      .then((data) => setMode(data.sumo_available ? "real" : "fake"))
      .catch(() => setMode("fake"));
  }, [sessionId]);

  if (!mode) {
    return (
      <div className="glass-panel p-8 text-sm text-slate-400">
        Checking system capabilities…
      </div>
    );
  }

  if (mode === "real") {
    return (
      <RealTrainingView
        sessionId={sessionId}
        netAbsPath={netAbsPath}
        passThresholdPct={passThresholdPct}
        runBaseline={runBaseline}
        greenDuration={greenDuration}
        onComplete={onComplete}
        onReset={onReset}
      />
    );
  }

  return <FakeTrainingView sessionId={sessionId} onComplete={onComplete} onReset={onReset} />;
}
