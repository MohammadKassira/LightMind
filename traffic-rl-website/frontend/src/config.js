export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Derive WebSocket base from API_BASE (http→ws, https→wss)
export const WS_BASE = API_BASE.replace(/^http/, "ws");
