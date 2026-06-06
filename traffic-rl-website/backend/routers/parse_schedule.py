from __future__ import annotations

import json
import os

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["parse"])

SYSTEM_PROMPT = """You are a data normalizer for a traffic simulation system.
The user will give you a raw table extracted from an Excel file.
Your job is to interpret it and return ONLY a JSON array in this exact format:
[
  { "day": "Monday", "startTime": "07:00", "endTime": "11:00", "level": "High" },
  ...
]
Rules:
- day must be a full English weekday name (Monday–Sunday)
- startTime and endTime must be HH:MM 24h format
- level must be exactly: Low, Medium, or High
- If volume/flow numbers are given instead of levels, convert: 0–33% of max = Low, 34–66% = Medium, 67–100% = High
- If times are ambiguous (e.g. "morning"), use your best judgment: morning = 06:00–12:00, afternoon = 12:00–18:00, evening = 18:00–22:00, night = 22:00–06:00
- If days are missing, apply the row to all weekdays
- Return ONLY the JSON array, no explanation, no markdown"""

_client: anthropic.Anthropic | None = None

VALID_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
VALID_LEVELS = {"Low", "Medium", "High"}


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="ANTHROPIC_API_KEY not configured on server.",
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


class ParseRequest(BaseModel):
    raw_table: str
    system_hint: str | None = None  # "periods" → return [{period_name, demand_level, weight}]; any other string → simple array


SIMPLE_ARRAY_SYSTEM = """You are a data normalizer for a traffic simulation system.
The user will give you a raw table extracted from an Excel file.
The output array should be a simple ordered list of demand levels for each episode, one per row.
Return ONLY a JSON array of strings like: ["High", "Low", "Medium", "High"]
Rules:
- Each string must be exactly one of: Low, Medium, or High
- Order rows by episode/time sequence
- If volume/flow numbers are given instead of levels, convert: 0–33% of max = Low, 34–66% = Medium, 67–100% = High
- Return ONLY the JSON array, no explanation, no markdown"""

PERIODS_SYSTEM = """You are a data normalizer for a traffic simulation system.
The user has uploaded a weekly traffic demand schedule.
It has columns: Day, Period, Demand_Level, Weight, Description (some may be in a different order or named slightly differently).
Convert any valid rows into a JSON array of objects, one per row that has a non-empty Demand_Level.
Each object must have exactly these fields:
  - period_name: the Period or name value from that row (string)
  - demand_level: exactly "Low", "Medium", or "High"
  - weight: positive integer from the Weight column (default 1 if missing or zero)
  - description: the Description value or empty string

Rules:
- Skip rows where Demand_Level is empty or missing
- Demand_Level must be exactly Low, Medium, or High (case-insensitive input, normalize to title case)
- Infer demand_level if numeric: 0–33% of max = Low, 34–66% = Medium, 67–100% = High
- Infer from text if ambiguous: peak/rush/busy/morning rush/evening rush = High, normal/moderate/midday = Medium, quiet/night/off-peak/low = Low
- Infer weight from any numeric column if Weight is absent; default to 3 for High/Medium and 5 for Low
- Day column is for user reference only — do not group or filter by it

Return ONLY the JSON array, no explanation, no markdown."""


@router.post("/parse-schedule")
def parse_schedule(payload: ParseRequest) -> dict:
    if not payload.raw_table.strip():
        raise HTTPException(status_code=422, detail="raw_table is empty.")

    if payload.system_hint == "periods":
        system_text = PERIODS_SYSTEM
    elif payload.system_hint:
        system_text = SIMPLE_ARRAY_SYSTEM
    else:
        system_text = SYSTEM_PROMPT

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": payload.raw_table}],
        )
    except anthropic.AuthenticationError as exc:
        raise HTTPException(status_code=503, detail="Invalid Anthropic API key.") from exc
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc}") from exc

    raw_text = message.content[0].text.strip()

    try:
        rows = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not interpret your file. Please check the format or use the preset demand levels.",
        ) from exc

    if not isinstance(rows, list) or len(rows) == 0:
        raise HTTPException(
            status_code=422,
            detail="Could not interpret your file. Please check the format or use the preset demand levels.",
        )

    # Periods mode: return [{period_name, demand_level, weight, description}]
    if payload.system_hint == "periods":
        valid = {"Low", "Medium", "High"}
        periods = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            lv = str(item.get("demand_level", item.get("level", ""))).strip().capitalize()
            if lv not in valid:
                continue
            w = item.get("weight", 3)
            try:
                w = max(1, int(w))
            except (TypeError, ValueError):
                w = 3
            periods.append({
                "period_name": str(item.get("period_name", item.get("name", "Period"))).strip(),
                "demand_level": lv,
                "weight": w,
                "description": str(item.get("description", "")).strip(),
            })
        return {"periods": periods}

    # Simple array mode
    if payload.system_hint:
        valid = {"Low", "Medium", "High"}
        schedule = []
        for item in rows:
            if isinstance(item, str) and item in valid:
                schedule.append(item)
            elif isinstance(item, dict):
                lv = str(item.get("level", item.get("Demand_Level", ""))).strip()
                if lv in valid:
                    schedule.append(lv)
        return {"schedule": schedule}

    # Structured schedule mode: return rows with day/time
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        day = str(row.get("day", "")).strip()
        start_time = str(row.get("startTime", "")).strip()
        end_time = str(row.get("endTime", "")).strip()
        level = str(row.get("level", "")).strip()
        if day.lower() not in VALID_DAYS or level not in VALID_LEVELS:
            continue
        cleaned.append({"day": day, "startTime": start_time, "endTime": end_time, "level": level})

    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Could not interpret your file. Please check the format or use the preset demand levels.",
        )

    return {"rows": cleaned}
