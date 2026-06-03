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


@router.post("/parse-schedule")
def parse_schedule(payload: ParseRequest) -> dict:
    if not payload.raw_table.strip():
        raise HTTPException(status_code=422, detail="raw_table is empty.")

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
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

    cleaned = []
    for row in rows:
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
