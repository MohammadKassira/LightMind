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
