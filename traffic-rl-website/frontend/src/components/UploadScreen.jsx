import { useRef, useState } from "react";
import logo from "../assets/lightmind-logo.png";

const API_BASE = "http://localhost:8000";

function UploadCard({
  title,
  description,
  filename,
  disabled,
  accentClass,
  buttonLabel,
  accept,
  inputRef,
  onSelect,
}) {
  return (
    <div className="glass-panel flex flex-col gap-4 p-6">
      <div className="space-y-2">
        <div className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.25em] ${accentClass}`}>
          {title}
        </div>
        <p className="text-sm text-slate-300">{description}</p>
      </div>
      <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4 text-sm text-slate-400">
        {filename || "No file uploaded yet"}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={onSelect}
        disabled={disabled}
      />
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        className="inline-flex items-center justify-center rounded-2xl border border-cyan-300/30 bg-cyan-400/10 px-4 py-3 text-sm font-medium text-cyan-100 transition hover:border-cyan-200/60 hover:bg-cyan-400/15 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/5 disabled:text-slate-500"
      >
        {buttonLabel}
      </button>
    </div>
  );
}

export default function UploadScreen({
  sessionId,
  uploadedFiles,
  onSessionCreated,
  onDemandUploaded,
  onNext,
}) {
  const [isUploadingOsm, setIsUploadingOsm] = useState(false);
  const [isUploadingDemand, setIsUploadingDemand] = useState(false);
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  // { state: "ok"|"warn"|"error", message: string }
  const [sumoStatus, setSumoStatus] = useState(null);
  const osmInputRef = useRef(null);
  const demandInputRef = useRef(null);

  const handleOsmUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setInfo("");
    setIsUploadingOsm(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/api/upload/osm`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Failed to upload OSM file");
      }

      const payload = await response.json();
      onSessionCreated(payload.session_id, file.name, payload.osm_absolute_path || "");
      setInfo(payload.message);

      if (payload.sumo_converted && payload.network_summary) {
        const { tl_count, edge_count } = payload.network_summary.stats;
        setSumoStatus({ state: "ok", message: `Network converted — ${tl_count} signals and ${edge_count} roads detected` });
      } else if (payload.sumo_error) {
        setSumoStatus({ state: "error", message: `Conversion failed — ${payload.sumo_error}` });
      } else {
        setSumoStatus({ state: "warn", message: "SUMO not installed — using map overlay only. Install SUMO for real network data." });
      }
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setIsUploadingOsm(false);
      event.target.value = "";
    }
  };

  const handleDemandUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !sessionId) return;

    setError("");
    setInfo("");
    setIsUploadingDemand(true);

    try {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/api/upload/demand`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Failed to upload demand file");
      }

      await response.json();
      onDemandUploaded(file.name);
      setInfo("Demand file uploaded successfully");
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setIsUploadingDemand(false);
      event.target.value = "";
    }
  };

  return (
    <section className="mx-auto flex max-w-5xl flex-col gap-8">
      <div className="glass-panel relative overflow-hidden px-6 py-10 sm:px-10">
        <div className="absolute inset-y-0 right-0 hidden w-1/2 bg-[radial-gradient(circle_at_top_right,rgba(51,212,255,0.22),transparent_55%)] lg:block" />
        <div className="relative flex flex-col items-start gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-2xl space-y-5">
            <img
              src={logo}
              alt="LightMind logo"
              className="h-20 w-20 rounded-[1.75rem] border border-cyan-300/30 object-cover shadow-glow"
            />
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.45em] text-cyan-200/80">
                Smart City Control Center
              </p>
              <h2 className="text-4xl font-semibold tracking-tight sm:text-5xl">
                LightMind
              </h2>
              <p className="max-w-2xl text-lg text-slate-300">
                AI traffic signal control for adaptive, smarter cities.
              </p>
              <p className="max-w-2xl text-sm text-slate-400">
                Upload your OpenStreetMap file. Traffic demand data is optional.
              </p>
            </div>
          </div>

          <div className="grid w-full max-w-md gap-3 rounded-3xl border border-white/10 bg-slate-950/40 p-5">
            <div className="flex items-center justify-between rounded-2xl border border-emerald-400/15 bg-emerald-400/5 px-4 py-3">
              <span className="text-sm text-slate-300">Active Session</span>
              <span className="font-mono text-xs text-emerald-200">
                {sessionId || "Pending"}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="h-2.5 w-2.5 rounded-full bg-trafficGreen shadow-[0_0_18px_rgba(34,197,94,0.9)]" />
              Fake live training data enabled for the MVP build.
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <UploadCard
          title="Required"
          description="Import the road network `.osm` file that defines the simulation map and junction layout."
          filename={uploadedFiles.osm || (isUploadingOsm ? "Uploading..." : "")}
          disabled={isUploadingOsm}
          accentClass="border-cyan-300/30 bg-cyan-400/10 text-cyan-100"
          buttonLabel={isUploadingOsm ? "Uploading OSM..." : "Upload OSM Map"}
          accept=".osm"
          inputRef={osmInputRef}
          onSelect={handleOsmUpload}
        />
        <UploadCard
          title="Optional"
          description="Attach demand data in `.xlsx` or `.csv` format to emulate scenario-specific traffic pressure."
          filename={uploadedFiles.demand || (isUploadingDemand ? "Uploading..." : "")}
          disabled={!sessionId || isUploadingDemand}
          accentClass="border-amber-300/30 bg-amber-400/10 text-amber-100"
          buttonLabel={
            !sessionId
              ? "Upload OSM First"
              : isUploadingDemand
                ? "Uploading Demand..."
                : "Upload Demand File"
          }
          accept=".xlsx,.csv"
          inputRef={demandInputRef}
          onSelect={handleDemandUpload}
        />
      </div>

      {sumoStatus && (
        <div
          className={`glass-panel px-5 py-4 text-sm ${
            sumoStatus.state === "ok"
              ? "border-emerald-400/30 text-emerald-200"
              : sumoStatus.state === "warn"
                ? "border-amber-400/30 text-amber-200"
                : "border-red-400/30 text-red-200"
          }`}
        >
          {sumoStatus.state === "ok" ? "✓ " : sumoStatus.state === "warn" ? "⚠ " : "✗ "}
          {sumoStatus.message}
        </div>
      )}

      {(error || info) && (
        <div
          className={`glass-panel px-5 py-4 text-sm ${
            error ? "border-red-400/30 text-red-200" : "border-emerald-400/20 text-emerald-200"
          }`}
        >
          {error || info}
        </div>
      )}

      <div className="flex flex-col items-stretch gap-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-slate-400">
          OSM upload is required before continuing to model selection.
        </p>
        <button
          type="button"
          onClick={onNext}
          disabled={!sessionId}
          className="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 px-6 py-3 text-sm font-semibold text-slate-950 shadow-glow transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-40"
        >
          Continue to Model Setup
        </button>
      </div>
    </section>
  );
}
