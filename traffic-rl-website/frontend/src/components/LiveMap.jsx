import { useEffect, useRef, useState } from "react";
import L from "leaflet";

const API_BASE = "http://localhost:8000";
const DEFAULT_CENTER = [33.8938, 35.5018];
const DEMAND_RADIUS = { Low: 4, Medium: 5, High: 7 };
const DEMAND_COLOR = { Low: "#60a5fa", Medium: "#33d4ff", High: "#f97316" };

function lightColor(state) {
  if (state === "green") return "#22c55e";
  if (state === "yellow") return "#facc15";
  return "#ef4444";
}

export default function LiveMap({ sessionId, cars, lights, mapCenter, activeDemandLevel, headlessLabel }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const staticLayersRef = useRef([]);   // roads + base TL markers — added once, never cleared
  const dynamicLayersRef = useRef([]);  // vehicles + active TL overlays — cleared each update
  const networkDataRef = useRef(null);
  const [networkStats, setNetworkStats] = useState(null);

  // Create / destroy the Leaflet map instance when sessionId changes
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Destroy the previous map completely before creating a new one
    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }
    staticLayersRef.current = [];
    dynamicLayersRef.current = [];
    networkDataRef.current = null;
    setNetworkStats(null);

    if (!sessionId) return;

    const map = L.map(el, { center: DEFAULT_CENTER, zoom: 16, scrollWheelZoom: true });
    mapRef.current = map;

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap © CARTO",
      subdomains: ["a", "b", "c", "d"],
      maxZoom: 19,
    }).addTo(map);

    let cancelled = false;

    fetch(`${API_BASE}/api/sessions/${sessionId}/network`)
      .then((r) => r.json())
      .then((data) => {
        if (cancelled || !data.available || !mapRef.current) return;
        networkDataRef.current = data;
        setNetworkStats(data.stats);

        const m = mapRef.current;

        // Fit the map to the traffic light bounds so any city loads correctly
        if (data.traffic_lights?.length > 0) {
          const bounds = L.latLngBounds(data.traffic_lights.map((tl) => [tl.lat, tl.lon]));
          m.fitBounds(bounds, { padding: [30, 30] });
        }

        // Road polylines (static)
        for (const seg of data.road_segments ?? []) {
          const poly = L.polyline(seg.coords.map((c) => [c.lat, c.lon]), {
            color: "#33d4ff", weight: 2, opacity: 0.45,
          }).addTo(m);
          staticLayersRef.current.push(poly);
        }

        // Base traffic light positions (static, always visible)
        for (const tl of data.traffic_lights ?? []) {
          const marker = L.circleMarker([tl.lat, tl.lon], {
            radius: 5, color: "#22c55e", fillColor: "#22c55e", fillOpacity: 0.7, weight: 1,
          })
            .bindTooltip(`${tl.id} | signal`, { direction: "top", opacity: 0.9 })
            .addTo(m);
          staticLayersRef.current.push(marker);
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      if (mapRef.current === map) {
        map.remove();
        mapRef.current = null;
      }
    };
  }, [sessionId]);

  // Refresh dynamic layers (active TL state overlays + vehicles) on every data update
  useEffect(() => {
    const m = mapRef.current;
    if (!m) return;

    for (const layer of dynamicLayersRef.current) m.removeLayer(layer);
    dynamicLayersRef.current = [];

    const carRadius = DEMAND_RADIUS[activeDemandLevel] ?? 5;
    const carColor = DEMAND_COLOR[activeDemandLevel] ?? "#33d4ff";

    // Active TL state overlays on top of base markers
    for (const light of lights) {
      const col = lightColor(light.state);
      const marker = L.circleMarker([light.lat, light.lng], {
        radius: 8, color: col, fillColor: col, fillOpacity: 0.92, weight: 2,
      })
        .bindTooltip(`${light.id} | ${light.state}`, { direction: "top", opacity: 0.95 })
        .addTo(m);
      dynamicLayersRef.current.push(marker);
    }

    // Vehicles
    for (const car of cars) {
      const stopped = car.speed < 0.5;
      const col = stopped ? "#94a3b8" : carColor;
      const marker = L.circleMarker([car.lat, car.lng], {
        radius: stopped ? 4 : carRadius, color: col, fillColor: col, fillOpacity: 0.85, weight: 1.5,
      })
        .bindTooltip(`${car.id} | ${car.speed} m/s`, { direction: "top", opacity: 0.95 })
        .addTo(m);
      dynamicLayersRef.current.push(marker);
    }

    // Pan the map when an explicit center is provided
    if (mapCenter) {
      m.setView(mapCenter, m.getZoom(), { animate: true });
    } else if (cars.length > 0) {
      m.setView([cars[0].lat, cars[0].lng], m.getZoom(), { animate: true });
    }
  }, [cars, lights, mapCenter, activeDemandLevel]);

  return (
    <div className="glass-panel overflow-hidden p-4" data-session-id={sessionId}>
      <div className="mb-4 flex items-center justify-between gap-3 px-2">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-cyan-200/80">Live Mobility Map</p>
          <h3 className="mt-2 text-xl font-semibold">Network Activity</h3>
        </div>
        <div className="flex items-center gap-2">
          {networkStats && (
            <div className="rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-200">
              Network: {networkStats.tl_count} signals · {networkStats.edge_count} roads
            </div>
          )}
          <div className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-xs uppercase tracking-[0.25em] text-cyan-100">
            Live Network Zone
          </div>
        </div>
      </div>

      <div className="relative overflow-hidden rounded-[1.75rem] border border-cyan-400/15 shadow-glow">
        <div className="absolute inset-0 z-[400] bg-[radial-gradient(circle_at_top_left,rgba(51,212,255,0.14),transparent_35%),linear-gradient(180deg,transparent,rgba(2,6,23,0.18))] pointer-events-none" />
        {headlessLabel && (
          <div className="absolute bottom-4 left-1/2 z-[500] -translate-x-1/2 rounded-full border border-amber-400/30 bg-amber-400/15 px-4 py-2 text-xs text-amber-200 backdrop-blur-sm pointer-events-none whitespace-nowrap">
            {headlessLabel}
          </div>
        )}
        <div ref={containerRef} className="h-[30rem] w-full" />
      </div>

      <div className="mt-4 flex flex-wrap gap-3 px-1 text-xs text-slate-400">
        {cars.length > 0 && (
          <>
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/15 bg-cyan-400/10 px-3 py-2 text-cyan-100">
              <span className="h-2.5 w-2.5 rounded-full bg-cyan-300" />
              Moving vehicles
            </div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2">
              <span className="h-2.5 w-2.5 rounded-full bg-slate-400" />
              Stopped vehicles
            </div>
          </>
        )}
        <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/15 bg-emerald-400/10 px-3 py-2 text-emerald-100">
          <span className="h-2.5 w-2.5 rounded-full bg-trafficGreen" />
          Traffic lights
        </div>
        {networkStats && (
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/15 bg-cyan-400/5 px-3 py-2 text-slate-400">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-400/50" />
            SUMO road overlay
          </div>
        )}
        {activeDemandLevel && (
          <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 ${
            activeDemandLevel === "High"
              ? "border-orange-400/30 bg-orange-400/10 text-orange-200"
              : activeDemandLevel === "Low"
                ? "border-blue-400/20 bg-blue-400/5 text-blue-200"
                : "border-cyan-400/20 bg-cyan-400/10 text-cyan-200"
          }`}>
            <span className={`h-2.5 w-2.5 rounded-full ${
              activeDemandLevel === "High" ? "bg-orange-400" : activeDemandLevel === "Low" ? "bg-blue-400" : "bg-cyan-300"
            }`} />
            {activeDemandLevel} demand
          </div>
        )}
      </div>
    </div>
  );
}
