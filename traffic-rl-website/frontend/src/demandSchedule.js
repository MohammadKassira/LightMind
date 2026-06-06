import * as XLSX from "xlsx";


export function timeToMinutes(timeStr) {
  if (timeStr === null || timeStr === undefined) return 0;
  const str = String(timeStr).trim();
  // Excel stores time as fraction of a day (e.g. 07:00 → 0.29166…)
  const num = parseFloat(str);
  if (!isNaN(num) && num >= 0 && num < 1) {
    return Math.round(num * 1440);
  }
  const isPM = /pm/i.test(str);
  const isAM = /am/i.test(str);
  const clean = str.replace(/[ap]m/gi, "").trim();
  const parts = clean.split(":").map((p) => parseInt(p, 10));
  let h = parts[0] || 0;
  const m = parts[1] || 0;
  if (isPM && h < 12) h += 12;
  if (isAM && h === 12) h = 0;
  return h * 60 + m;
}

export function minutesToTimeStr(minutes) {
  const h = Math.floor(minutes / 60) % 24;
  const m = minutes % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export function detectOverlaps(schedule) {
  const warnings = [];
  const byDay = {};
  for (const row of schedule) {
    const key = row.day.toLowerCase();
    if (!byDay[key]) byDay[key] = [];
    byDay[key].push(row);
  }
  for (const rows of Object.values(byDay)) {
    const sorted = [...rows].sort((a, b) => a.startMinutes - b.startMinutes);
    for (let i = 0; i < sorted.length - 1; i++) {
      if (sorted[i].endMinutes > sorted[i + 1].startMinutes) {
        warnings.push(
          `${sorted[i].day}: ${minutesToTimeStr(sorted[i].startMinutes)}–${minutesToTimeStr(sorted[i].endMinutes)} overlaps with ${minutesToTimeStr(sorted[i + 1].startMinutes)}–${minutesToTimeStr(sorted[i + 1].endMinutes)}`
        );
      }
    }
  }
  return warnings;
}

export async function extractRawTable(file) {
  const arrayBuffer = await file.arrayBuffer();
  const data = new Uint8Array(arrayBuffer);
  const workbook = XLSX.read(data, { type: "array" });
  const sheet = workbook.Sheets[workbook.SheetNames[0]];
  const rawRows = XLSX.utils.sheet_to_json(sheet, { header: 1, raw: false, defval: "" });
  return rawRows
    .filter((row) => row.some((cell) => String(cell).trim()))
    .map((row) => row.map((cell) => String(cell ?? "").trim()).join("\t"))
    .join("\n");
}

export function normalizeAiRows(apiRows) {
  return apiRows.map((row) => ({
    day: row.day,
    startTime: row.startTime,
    endTime: row.endTime,
    level: row.level,
    startMinutes: timeToMinutes(row.startTime),
    endMinutes: timeToMinutes(row.endTime),
  }));
}

export function getDemandForTime(schedule, dayName, currentMinutes) {
  const match = schedule.find(
    (row) =>
      row.day.toLowerCase() === dayName.toLowerCase() &&
      currentMinutes >= row.startMinutes &&
      currentMinutes < row.endMinutes
  );
  return match ? match.level : "Low";
}

// ── Period-based demand schedule (new 7-day template format) ─────────────────

function _normalizeLevel(raw) {
  if (!raw) return null;
  const s = String(raw).trim().toLowerCase();
  if (s === "high") return "High";
  if (s === "medium") return "Medium";
  if (s === "low") return "Low";
  return null;
}

/**
 * Parse rows from the 7-day demand template.
 * Columns: Day | Period | Demand_Level | Weight | Description
 * Also handles old format: Period_Name | Demand_Level | Weight | Description
 *
 * Returns { periods, warnings } where periods = [{name, level, weight}]
 * or null if headers not found.
 */
export function parsePeriodSchedule(rows) {
  if (!rows || rows.length < 2) return null;

  const header = rows[0].map((h) => String(h || "").trim().toLowerCase());
  const levelCol = header.findIndex((h) => h === "demand_level" || h.includes("demand") || (h.includes("level") && !h.includes("demand")));
  const weightCol = header.findIndex((h) => h === "weight" || h.includes("weight"));
  // "Period" column for period name (prefer exact "period" match over "period_name")
  const periodCol = header.findIndex((h) => h === "period" || h.includes("period") || h.includes("name"));

  if (levelCol === -1 || weightCol === -1) return null;

  const periods = [];
  const warnings = [];

  for (let i = 1; i < rows.length; i++) {
    const row = rows[i];
    const rawLevel = String(row[levelCol] || "").trim();
    const rawWeight = String(row[weightCol] || "").trim();

    // Skip completely empty rows and note/header rows
    if (!rawLevel && !rawWeight) continue;
    // Skip rows with no level (e.g. unfilled template rows — not an error)
    if (!rawLevel) continue;

    const level = _normalizeLevel(rawLevel);
    if (!level) {
      warnings.push(`Row ${i + 1}: unrecognised Demand_Level "${rawLevel}" — must be Low, Medium, or High`);
      continue;
    }

    const weight = parseInt(rawWeight, 10);
    if (!weight || weight < 1) {
      warnings.push(`Row ${i + 1}: invalid Weight "${rawWeight}" — must be a positive integer`);
      continue;
    }

    const name = periodCol >= 0 ? String(row[periodCol] || "").trim() || `Period ${i}` : `Period ${i}`;
    periods.push({ name, level, weight });
  }

  return periods.length > 0 ? { periods, warnings } : null;
}

/**
 * Convert parsed periods into the demandData shape used by the rest of the app.
 * Returns {sequence, periods (with pct), mixText, periodText, previewText}
 */
export function buildDemandData(periods) {
  if (!periods || periods.length === 0) return null;

  const totalWeight = periods.reduce((s, p) => s + p.weight, 0);
  if (totalWeight === 0) return null;

  // Build episode sequence: repeat each level `weight` times
  const sequence = [];
  for (const p of periods) {
    for (let i = 0; i < p.weight; i++) sequence.push(p.level);
  }

  const periodPcts = periods.map((p) => ({ ...p, pct: (p.weight / totalWeight) * 100 }));

  // Aggregate by demand level for the mix summary
  const mixByLevel = { High: 0, Medium: 0, Low: 0 };
  for (const p of periodPcts) mixByLevel[p.level] = (mixByLevel[p.level] || 0) + p.pct;

  const mixText = Object.entries(mixByLevel)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => `${v.toFixed(1)}% ${k}`)
    .join(" · ");

  const periodText = periodPcts
    .slice(0, 4)
    .map((p) => `${p.pct.toFixed(1)}% ${p.level} (${p.name})`)
    .join(" · ") + (periodPcts.length > 4 ? " · …" : "");

  const previewText = `✓ ${periods.length} demand period${periods.length !== 1 ? "s" : ""} loaded`;

  return { sequence, periods: periodPcts, mixText, periodText, previewText };
}
