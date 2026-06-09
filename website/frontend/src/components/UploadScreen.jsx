import { useRef, useState } from "react";

import { API_BASE } from "../config";

export default function UploadScreen({
  sessionId,
  uploadedFiles,
  onSessionCreated,
  onNext,
}) {
  const [isUploadingNet, setIsUploadingNet] = useState(false);
  const [error, setError] = useState("");
  const [networkStatus, setNetworkStatus] = useState(null);

  const netInputRef = useRef(null);

  // ── Network upload ──────────────────────────────────────────────────────────

  const handleNetUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setIsUploadingNet(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/api/upload/net`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Failed to upload network file");
      }

      const payload = await response.json();
      onSessionCreated(payload.session_id, file.name, payload.net_absolute_path || "");

      if (payload.network_summary) {
        const { tl_count, edge_count } = payload.network_summary.stats || {};
        setNetworkStatus({
          state: "ok",
          message: `Network loaded — ${tl_count ?? 0} traffic signals · ${edge_count ?? 0} road edges`,
        });
      } else {
        setNetworkStatus({
          state: "warn",
          message: "Network file accepted. Could not extract coordinate data — map overlay may be limited.",
        });
      }
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setIsUploadingNet(false);
      event.target.value = "";
    }
  };

  return (
    <section className="mx-auto flex max-w-5xl flex-col gap-8">
      {/* Hero banner */}
      <div className="glass-panel relative overflow-hidden px-6 py-10 sm:px-10">
        <div className="absolute inset-y-0 right-0 hidden w-1/2 bg-[radial-gradient(circle_at_top_right,rgba(51,212,255,0.22),transparent_55%)] lg:block" />
        <div className="relative flex flex-col items-start gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-2xl space-y-5">
            <img src="/logo.jpg" alt="LightMind logo" className="h-20 w-20 rounded-[1.75rem] border border-cyan-300/30 object-cover shadow-glow" />
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.45em] text-cyan-200/80">Smart City Control Center</p>
              <h2 className="text-4xl font-semibold tracking-tight sm:text-5xl">LightMind</h2>
              <p className="max-w-2xl text-lg text-slate-300">AI traffic signal control for adaptive, smarter cities.</p>
            </div>
          </div>

          <div className="grid w-full max-w-md gap-3 rounded-3xl border border-white/10 bg-slate-950/40 p-5">
            <div className="flex items-center justify-between rounded-2xl border border-emerald-400/15 bg-emerald-400/5 px-4 py-3">
              <span className="text-sm text-slate-300">Active Session</span>
              <span className="font-mono text-xs text-emerald-200">{sessionId || "Pending"}</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="h-2.5 w-2.5 rounded-full bg-trafficGreen shadow-[0_0_18px_rgba(34,197,94,0.9)]" />
              Fake live training data enabled for the MVP build.
            </div>
          </div>
        </div>
      </div>

      {/* Network upload */}
      <div className="glass-panel flex flex-col gap-4 p-6">
        <div className="space-y-1">
          <div className="inline-flex rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.25em] text-cyan-100">
            Required
          </div>
          <p className="text-sm text-slate-300">
            Upload your SUMO network file (.net.xml). Prepare it using SUMO's netedit tool with your traffic lights already positioned correctly.
          </p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4 text-sm text-slate-400">
          {uploadedFiles.net || (isUploadingNet ? "Uploading…" : "No file uploaded yet")}
        </div>
        <input ref={netInputRef} type="file" accept=".net.xml,.xml" className="hidden" onChange={handleNetUpload} disabled={isUploadingNet} />
        <button
          type="button"
          disabled={isUploadingNet}
          onClick={() => netInputRef.current?.click()}
          className="inline-flex items-center justify-center rounded-2xl border border-cyan-300/30 bg-cyan-400/10 px-4 py-3 text-sm font-medium text-cyan-100 transition hover:border-cyan-200/60 hover:bg-cyan-400/15 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/5 disabled:text-slate-500"
        >
          {isUploadingNet ? "Uploading network…" : "Upload SUMO Network File (.net.xml)"}
        </button>

        {networkStatus && (
          <div className={`rounded-2xl px-4 py-3 text-sm ${
            networkStatus.state === "ok" ? "border border-emerald-400/30 bg-emerald-400/8 text-emerald-200" : "border border-amber-400/30 bg-amber-400/8 text-amber-200"
          }`}>
            {networkStatus.state === "ok" ? "✓ " : "⚠ "}{networkStatus.message}
          </div>
        )}
      </div>

      {error && (
        <div className="glass-panel border-red-400/30 px-5 py-4 text-sm text-red-200">{error}</div>
      )}

      <div className="flex flex-col items-stretch gap-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-slate-400">
          SUMO network file upload is required before continuing.
        </p>
        <button
          type="button"
          onClick={onNext}
          disabled={!sessionId}
          className="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 px-6 py-3 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-40"
        >
          Continue to Model Setup →
        </button>
      </div>
    </section>
  );
}
