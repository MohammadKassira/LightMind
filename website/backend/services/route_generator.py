"""Route generation service for capacity-based demand creation.

Wraps the model/scripts/generate_capacity_based_routes.py script
to generate 4 stochastic route files (light/medium/dense/heavy)
based on network capacity.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ML_PROJECT_ROOT env var is set in Docker (ML_PROJECT_ROOT=/app).
# Falls back to 4 parent levels for local dev (ml_project/website/backend/services/).
_ml_project_root = Path(os.environ.get(
    "ML_PROJECT_ROOT",
    str(Path(__file__).parent.parent.parent.parent),
))


def generate_routes_for_network(
    net_path: Path,
    source_rou_path: Path | None,
    output_dir: Path,
    network_name: str,
    begin_time: int = 0,
    time_window: int = 3600,
) -> dict:
    """Generate 4 capacity-based route files for a network.

    Args:
        net_path: Path to .net.xml file
        source_rou_path: Path to source .rou.xml (optional)
        output_dir: Where to save generated route files
        network_name: Network name for file naming
        begin_time: SUMO simulation start time
        time_window: Source route file time window

    Returns:
        {
            "status": "success" | "error",
            "jam_capacity": float,
            "route_files": {
                "light": Path,
                "medium": Path,
                "dense": Path,
                "heavy": Path,
            },
            "metadata": {
                "num_od_pairs": int,
                "avg_trip_time": float,
            },
            "error": str (if status == "error"),
        }
    """
    result = {"status": "error", "error": ""}

    # Validate inputs
    if not net_path.exists():
        result["error"] = f"Network file not found: {net_path}"
        return result

    if source_rou_path and not source_rou_path.exists():
        result["error"] = f"Source route file not found: {source_rou_path}"
        return result

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,
        "model/scripts/generate_capacity_based_routes.py",
        "--net", str(net_path.resolve()),
        "--output", str(output_dir.resolve()),
        "--name", network_name,
        "--begin-time", str(begin_time),
        "--time-window", str(time_window),
    ]

    if source_rou_path:
        cmd.extend(["--source", str(source_rou_path.resolve())])

    # Run generator
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_ml_project_root,
        )

        if process.returncode != 0:
            result["error"] = f"Route generation failed:\n{process.stderr}"
            return result

        # Verify output files exist
        variants = ["light", "medium", "dense", "heavy"]
        route_files = {}
        for variant in variants:
            route_file = output_dir / f"{network_name}_{variant}_stochastic.rou.xml"
            if not route_file.exists():
                result["error"] = f"Route file not created: {route_file}"
                return result
            route_files[variant] = route_file

        # Parse output for metadata
        metadata = _parse_generator_output(process.stdout)

        result["status"] = "success"
        result["route_files"] = route_files
        result["metadata"] = metadata
        result["jam_capacity"] = metadata.get("jam_capacity", 0.0)

    except subprocess.TimeoutExpired:
        result["error"] = "Route generation timed out (60s)"
    except Exception as e:
        result["error"] = f"Route generation error: {str(e)}"

    return result


def _parse_generator_output(output: str) -> dict:
    """Extract metadata from generator stdout.

    Example output:
        Jam capacity: 385 vehicles
        Estimated avg trip time: 147 seconds
    """
    metadata = {"jam_capacity": 0.0, "avg_trip_time": 0.0, "num_od_pairs": 0}

    for line in output.split("\n"):
        if "Jam capacity:" in line:
            try:
                capacity = float(line.split(":")[1].strip().split()[0])
                metadata["jam_capacity"] = capacity
            except (IndexError, ValueError):
                pass

        if "Estimated avg trip time:" in line or "Using provided avg trip time:" in line:
            try:
                trip_time = float(line.split(":")[1].strip().split()[0])
                metadata["avg_trip_time"] = trip_time
            except (IndexError, ValueError):
                pass

        if "unique OD pairs" in line or "synthetic OD pairs" in line:
            m = re.search(r"(\d+)", line)
            if m:
                metadata["num_od_pairs"] = int(m.group(1))

    return metadata
