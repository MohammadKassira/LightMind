"""Generate flow-based stochastic route files for grid4x4.

Reads grid4x4_dense.rou.xml, extracts OD pair frequencies, and writes
<flow probability="..."> route files at multiple demand levels.
Each simulation run spawns vehicles stochastically so every episode
sees different traffic — prevents policy from memorizing one fixed scenario.

Usage:
    python scripts/generate_stochastic_routes.py
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

NET_DIR   = Path("networks/external/RESCO/grid4x4")
SOURCE    = NET_DIR / "grid4x4_dense.rou.xml"
SIM_END   = 1000  # seconds, must match env max_steps

VARIANTS = [
    ("light",   0.50),   # ~500 vehicles
    ("medium",  0.75),   # ~750 vehicles
    ("dense",   1.00),   # ~1000 vehicles (matches current)
    ("heavy",   1.25),   # ~1250 vehicles
]

def extract_od_probs(rou_path: Path, sim_end: int) -> list[tuple[str, str, float]]:
    tree = ET.parse(rou_path)
    root = tree.getroot()
    od_counts: Counter = Counter()
    for v in root.findall("vehicle"):
        edges = v.find("route").get("edges").split()
        od_counts[(edges[0], edges[-1])] += 1
    total = sum(od_counts.values())
    # probability per second = count / sim_end
    return [(src, dst, count / sim_end) for (src, dst), count in od_counts.items()]

def write_flow_rou(out_path: Path, od_probs: list, scale: float, sim_end: int) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- Stochastic flow-based route file. Scale={scale:.2f} (~{int(sum(p for _,_,p in od_probs)*sim_end*scale)} vehicles/ep) -->',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
    ]
    for i, (src, dst, base_prob) in enumerate(od_probs):
        prob = base_prob * scale
        lines.append(
            f'    <flow id="f{i}" from="{src}" to="{dst}"'
            f' begin="0" end="{sim_end}" probability="{prob:.6f}"'
            f' departLane="free" departSpeed="max"/>'
        )
    lines.append("</routes>")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    exp_vehicles = int(sum(p for _, _, p in od_probs) * sim_end * scale)
    print(f"  Written: {out_path.name}  (expected ~{exp_vehicles} vehicles/episode)")

def main():
    print(f"Reading OD pairs from {SOURCE}...")
    od_probs = extract_od_probs(SOURCE, SIM_END)
    print(f"  Found {len(od_probs)} unique OD pairs")

    for name, scale in VARIANTS:
        out_path = NET_DIR / f"grid4x4_{name}_stochastic.rou.xml"
        write_flow_rou(out_path, od_probs, scale, SIM_END)

    print("\nDone. Use these files in your config for multi-route training.")
    print("Each episode will have different vehicle counts and departures.")

if __name__ == "__main__":
    main()
