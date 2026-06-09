import { useEffect, useRef, useState } from "react";
import DeploymentScreen from "./components/DeploymentScreen";
import ModelSelect from "./components/ModelSelect";
import ResultsScreen from "./components/ResultsScreen";
import RetrainingScreen from "./components/RetrainingScreen";
import TrainingScreen from "./components/TrainingScreen";
import UploadScreen from "./components/UploadScreen";

import { API_BASE } from "./config";

const STEPS = { upload: 1, model: 2, training: 3, results: 4, deployment: 5, retraining: 6 };
const PAGE_TO_STEP = { 1: "upload", 2: "model", 3: "training", 4: "results", 5: "deployment", 6: "retraining" };

function parseHash() {
  const raw = window.location.hash.replace(/^#/, "");
  const p = Object.fromEntries(new URLSearchParams(raw));
  return { page: p.page ? parseInt(p.page, 10) : null, session: p.session || null };
}

function clearStorage() {
  ["lm_step", "lm_session", "lm_netPath", "lm_netFile"].forEach((k) => localStorage.removeItem(k));
  window.location.hash = "";
}

function saveState(step, sessionId, netAbsPath, netFile) {
  // Don't pollute the URL when there's no meaningful state to persist
  if (step === "upload" && !sessionId) {
    window.location.hash = "";
    return;
  }
  const page = STEPS[step] || 1;
  window.location.hash = sessionId ? `page=${page}&session=${sessionId}` : `page=${page}`;
  localStorage.setItem("lm_step", step);
  localStorage.setItem("lm_session", sessionId || "");
  localStorage.setItem("lm_netPath", netAbsPath || "");
  localStorage.setItem("lm_netFile", netFile || "");
}

export default function App() {
  const [step, setStep] = useState("upload");
  const [sessionId, setSessionId] = useState("");
  const [netAbsPath, setNetAbsPath] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState({ net: "", demand: "" });
  const [trainingSessionKey, setTrainingSessionKey] = useState(0);
  const [restored, setRestored] = useState(false);
  const [demandData, setDemandData] = useState(null); // parsed from UploadScreen
  const [trainingOptions, setTrainingOptions] = useState({ runBaseline: false, greenDuration: 60, demandSequence: null });

  // On mount: restore ONLY when the URL hash explicitly contains page=N (i.e. a same-tab refresh).
  // A clean URL (no hash) or ?fresh=1 always starts fresh and clears any saved state.
  useEffect(() => {
    const hash = parseHash();
    const isFresh = new URLSearchParams(window.location.search).get("fresh") === "1";
    const hasHash = Boolean(hash.page);

    if (isFresh || !hasHash) {
      clearStorage();
      setRestored(true);
      return;
    }

    // Hash is present — this is a genuine page refresh. Restore from it.
    const targetSession = hash.session;
    const targetStep = PAGE_TO_STEP[hash.page] || "upload";

    const savedNetPath = localStorage.getItem("lm_netPath") || "";
    const savedNetFile = localStorage.getItem("lm_netFile") || "";

    if (!targetSession || targetStep === "upload") {
      setRestored(true);
      return;
    }

    const restore = (s) => {
      setSessionId(targetSession);
      setNetAbsPath(savedNetPath);
      setUploadedFiles({ net: savedNetFile, demand: "" });
      setStep(s);
    };

    const validate = async () => {
      try {
        if (targetStep === "training" || targetStep === "results") {
          const r = await fetch(`${API_BASE}/api/real-train/${targetSession}/status`);
          if (!r.ok) throw new Error("not found");
          const data = await r.json();
          // If pipeline finished, land on results; otherwise resume training
          restore(data.stage === "complete" ? "results" : "training");
        } else {
          // model step — trust localStorage
          restore(targetStep);
        }
      } catch {
        // Session gone or unreachable — start fresh
      } finally {
        setRestored(true);
      }
    };

    validate();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist step + session whenever they change (skips initial render before restoration)
  const restoredRef = useRef(false);
  useEffect(() => {
    if (!restored) return;
    restoredRef.current = true;
    saveState(step, sessionId, netAbsPath, uploadedFiles.net);
  }, [step, sessionId, netAbsPath, uploadedFiles.net, restored]);

  const resetApp = () => {
    clearStorage();
    setStep("upload");
    setSessionId("");
    setNetAbsPath("");
    setUploadedFiles({ net: "", demand: "" });
    setDemandData(null);
    setTrainingSessionKey((k) => k + 1);
  };

  const resetForNewMap = (newSessionId, netFilename, absPath) => {
    setSessionId(newSessionId);
    setNetAbsPath(absPath || "");
    setUploadedFiles({ net: netFilename, demand: "" });
    setDemandData(null);
    setStep("upload");
    setTrainingSessionKey((k) => k + 1);
  };

  if (!restored) {
    return (
      <div className="min-h-screen bg-dashboard text-slate-100 flex items-center justify-center">
        <p className="text-sm text-slate-400">Restoring session…</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dashboard text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8">
        <header className="glass-panel mb-6 flex items-center justify-between px-5 py-4">
          <div className="flex items-center gap-4">
            <img
              src="/logo.jpg"
              alt="LightMind logo"
              className="h-12 w-12 rounded-2xl border border-cyan-400/30 object-cover shadow-glow"
            />
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/80">AI Traffic Control</p>
              <h1 className="text-2xl font-semibold tracking-tight">LightMind</h1>
            </div>
          </div>
          <div className="hidden items-center gap-3 md:flex">
            {Object.entries(STEPS).map(([key, number]) => (
              <div
                key={key}
                className={`flex h-10 w-10 items-center justify-center rounded-full border text-sm font-medium transition ${
                  step === key
                    ? "border-cyan-300 bg-cyan-400/20 text-cyan-100 shadow-glow"
                    : "border-white/10 bg-white/5 text-slate-400"
                }`}
              >
                {number}
              </div>
            ))}
          </div>
        </header>

        <main className="flex-1">
          {step === "upload" && (
            <UploadScreen
              sessionId={sessionId}
              uploadedFiles={uploadedFiles}
              onSessionCreated={resetForNewMap}
              onDemandParsed={setDemandData}
              onNext={() => setStep("model")}
            />
          )}


          {step === "model" && (
            <ModelSelect
              key={`model-${sessionId}`}
              sessionId={sessionId}
              demandData={demandData}
              onBack={() => setStep("upload")}
              onTrainingStarted={(opts) => { setTrainingOptions(opts || {}); setStep("training"); }}
            />
          )}

          {step === "training" && (
            <TrainingScreen
              key={`training-${sessionId}-${trainingSessionKey}`}
              sessionId={sessionId}
              netAbsPath={netAbsPath}
              runBaseline={trainingOptions.runBaseline || false}
              greenDuration={trainingOptions.greenDuration || 60}
              demandSequence={trainingOptions.demandSequence || null}
              passThresholdPct={trainingOptions.passThresholdPct ?? 25}
              onComplete={() => setStep("results")}
              onReset={resetApp}
            />
          )}

          {step === "results" && (
            <ResultsScreen
              key={`results-${sessionId}-${trainingSessionKey}`}
              sessionId={sessionId}
              onReset={resetApp}
              onDeploy={() => setStep("deployment")}
              onRetrain={() => setStep("retraining")}
            />
          )}

          {step === "deployment" && (
            <DeploymentScreen
              key={`deployment-${sessionId}`}
              sessionId={sessionId}
              onBack={() => setStep("results")}
              onReset={resetApp}
            />
          )}

          {step === "retraining" && (
            <RetrainingScreen
              key={`retraining-${sessionId}`}
              sessionId={sessionId}
              onBack={() => setStep("results")}
              onReset={resetApp}
              onStartTraining={() => setStep("training")}
            />
          )}
        </main>
      </div>
    </div>
  );
}
