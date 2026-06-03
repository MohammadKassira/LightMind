from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .utils import now_iso


def package_job_outputs(*, job_dir: Path, package_dir: Path, package_name: str) -> dict[str, Any]:
    package_dir.mkdir(parents=True, exist_ok=True)
    archive_base = package_dir / package_name
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(job_dir)))
    return {
        "packaged_at_utc": now_iso(),
        "job_dir": str(job_dir),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size if archive_path.exists() else 0,
    }
