// VITE_API_URL="" (empty string) means same-origin — used in HF Spaces single-container mode.
// Undefined (not set at build time) falls back to local dev default.
const _envUrl = import.meta.env.VITE_API_URL;
export const API_BASE = (_envUrl !== undefined && _envUrl !== null) ? _envUrl : "http://localhost:7860";

// When API_BASE is empty (same-origin), derive the WebSocket URL from the page's location at runtime.
export const WS_BASE = API_BASE
  ? API_BASE.replace(/^http/, "ws")
  : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;
