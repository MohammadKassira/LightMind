"""Generate random vehicle demand for SUMO networks that lack route files.

Uses SUMO's bundled randomTrips.py tool. Requires SUMO_HOME to be set.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _random_trips_script() -> Path:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError(
            "SUMO_HOME environment variable is not set. "
            "Point it to your SUMO installation directory."
        )
    script = Path(sumo_home) / "tools" / "randomTrips.py"
    if not script.exists():
        raise RuntimeError(f"randomTrips.py not found at {script}")
    return script


def generate_demand(
    net_file: str | Path,
    output_rou: str | Path,
    period: float = 2.0,
    duration: int = 3600,
    seed: int = 42,
) -> Path:
    """Run SUMO's randomTrips.py to generate a route file for net_file.

    Args:
        net_file:   Path to SUMO .net.xml
        output_rou: Output .rou.xml path (will be created/overwritten)
        period:     Average seconds between vehicle insertions (lower = more traffic)
        duration:   Simulation horizon in seconds (end time for trip generation)
        seed:       Random seed for reproducibility

    Returns:
        Path to the generated .rou.xml file.
    """
    script = _random_trips_script()
    net_file = Path(net_file)
    output_rou = Path(output_rou)

    with tempfile.NamedTemporaryFile(suffix=".trips.xml", delete=False) as tf:
        trips_xml = Path(tf.name)

    try:
        cmd = [
            sys.executable,
            str(script),
            "-n", str(net_file),
            "-o", str(trips_xml),
            "-r", str(output_rou),
            "-p", str(period),
            "--end", str(duration),
            "--seed", str(seed),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"randomTrips.py failed (exit {result.returncode}):\n"
                f"{result.stderr}"
            )
    finally:
        if trips_xml.exists():
            trips_xml.unlink()

    return output_rou
