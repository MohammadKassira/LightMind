from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .exceptions import WebIntegrationError


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def csv_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def require_exists(path: Path, *, stage: str) -> None:
    if not path.exists():
        raise WebIntegrationError(stage, f"Required path missing: {path}")


def run_command(
    cmd: list[str],
    *,
    stage: str,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    if stdout_path is None and stderr_path is None:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=merged_env)
        if proc.returncode != 0:
            raise WebIntegrationError(
                stage,
                (
                    f"Command failed rc={proc.returncode}: {' '.join(cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                ),
            )
        return proc

    ensure_dir((stdout_path or stderr_path or Path(".")).parent)
    with (
        (stdout_path.open("w", encoding="utf-8") if stdout_path else open(os.devnull, "w", encoding="utf-8")) as out,
        (stderr_path.open("w", encoding="utf-8") if stderr_path else open(os.devnull, "w", encoding="utf-8")) as err,
    ):
        proc = subprocess.run(cmd, stdout=out, stderr=err, check=False, env=merged_env)
    if proc.returncode != 0:
        stdout_hint = f" stdout={stdout_path}" if stdout_path else ""
        stderr_hint = f" stderr={stderr_path}" if stderr_path else ""
        raise WebIntegrationError(
            stage,
            f"Command failed rc={proc.returncode}: {' '.join(cmd)}{stdout_hint}{stderr_hint}",
        )
    return proc


def resolve_random_trips_path() -> Path:
    sumo_home = os.environ.get("SUMO_HOME", "")
    candidates: list[Path] = []
    if sumo_home:
        candidates.append(Path(sumo_home) / "tools" / "randomTrips.py")
    candidates.extend(
        [
            Path("/usr/share/sumo/tools/randomTrips.py"),
            Path("/usr/local/share/sumo/tools/randomTrips.py"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise WebIntegrationError(
        "random_route_generation",
        "Unable to find randomTrips.py. Set SUMO_HOME to a valid SUMO installation.",
    )
