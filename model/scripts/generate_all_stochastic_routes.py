"""Generate stochastic flow-based route files for all 7 training networks.

Handles both <trip from/to> and <vehicle><route edges> formats.
Writes 4 demand-level variants per network: light/medium/dense/heavy.

Usage:
    python scripts/generate_all_stochastic_routes.py
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SIM_DURATION = 1000   # episode length: max_steps(200) * delta_time(5) = 1000s

VARIANTS = [
    ("light",  0.50),
    ("medium", 0.75),
    ("dense",  1.00),
    ("heavy",  1.25),
]

# Per-network config:
#   dir        — folder containing the source route file
#   source     — source .rou.xml filename
#   begin_time — SUMO simulation start time (seconds)
#   time_window — total duration of source traffic (seconds), used to compute rate
#   element    — "trip" (has from/to attrs) or "vehicle" (has <route edges>)
NETWORKS = [
    {
        "name":        "ingolstadt1",
        "dir":         "networks/external/RESCO/ingolstadt1",
        "source":      "ingolstadt1.rou.xml",
        "begin_time":  57600,
        "time_window": 3600,
        "element":     "trip",
    },
    {
        "name":        "ingolstadt7",
        "dir":         "networks/external/RESCO/ingolstadt7",
        "source":      "ingolstadt7.rou.xml",
        "begin_time":  57600,
        "time_window": 3600,
        "element":     "trip",
    },
    {
        "name":        "ingolstadt21",
        "dir":         "networks/external/RESCO/ingolstadt21",
        "source":      "ingolstadt21.rou.xml",
        "begin_time":  57600,
        "time_window": 3600,
        "element":     "trip",
    },
    {
        "name":        "cologne1",
        "dir":         "networks/external/RESCO/cologne1",
        "source":      "cologne1.rou.xml",
        "begin_time":  25200,
        "time_window": 3600,
        "element":     "trip",
    },
    {
        "name":        "cologne8",
        "dir":         "networks/external/RESCO/cologne8",
        "source":      "cologne8.rou.xml",
        "begin_time":  25200,
        "time_window": 3600,
        "element":     "trip",
    },
    {
        "name":        "arterial4x4",
        "dir":         "networks/external/RESCO/arterial4x4",
        "source":      "arterial4x4_1.rou.xml",
        "begin_time":  0,
        "time_window": 3000,
        "element":     "vehicle",
    },
    {
        "name":        "pasubio",
        "dir":         "networks/external/bologna_pasubio/pasubio",
        "source":      "pasubio.rou.xml",
        "begin_time":  0,
        "time_window": 3600,
        "element":     "vehicle",
    },
]


def extract_od_probs(rou_path: Path, element: str, time_window: int) -> list:
    """Return list of (from_edge, to_edge, probability_per_second)."""
    root = ET.parse(rou_path).getroot()
    od_counts: Counter = Counter()

    if element == "trip":
        for t in root.findall("trip"):
            src = t.get("from", "")
            dst = t.get("to", "")
            if src and dst:
                od_counts[(src, dst)] += 1
    else:  # vehicle with inline <route edges>
        for v in root.findall("vehicle"):
            r = v.find("route")
            if r is None:
                continue
            edges = r.get("edges", "").split()
            if len(edges) >= 1:
                src = edges[0]
                dst = edges[-1]
                od_counts[(src, dst)] += 1

    return [(src, dst, count / time_window) for (src, dst), count in od_counts.items()]


def write_flow_rou(
    out_path: Path,
    od_probs: list,
    scale: float,
    begin_time: int,
    network_name: str,
    variant: str,
) -> None:
    end_time = begin_time + SIM_DURATION
    exp_vehicles = int(sum(p for _, _, p in od_probs) * SIM_DURATION * scale)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- Stochastic flow route file: {network_name} {variant} (scale={scale:.2f}, ~{exp_vehicles} veh/ep) -->',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
    ]
    for i, (src, dst, base_prob) in enumerate(od_probs):
        prob = base_prob * scale
        lines.append(
            f'    <flow id="f{i}" from="{src}" to="{dst}"'
            f' begin="{begin_time}" end="{end_time}"'
            f' probability="{prob:.6f}"'
            f' departLane="free" departSpeed="max"/>'
        )
    lines.append("</routes>")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {out_path.name}  (~{exp_vehicles} veh/ep)")


def main():
    for net in NETWORKS:
        net_dir  = Path(net["dir"])
        src_path = net_dir / net["source"]
        name     = net["name"]
        begin    = net["begin_time"]
        window   = net["time_window"]
        elem     = net["element"]

        print(f"\n{name}: reading {src_path.name} ...")
        od_probs = extract_od_probs(src_path, elem, window)
        print(f"  {len(od_probs)} unique OD pairs")

        for variant, scale in VARIANTS:
            out_path = net_dir / f"{name}_{variant}_stochastic.rou.xml"
            write_flow_rou(out_path, od_probs, scale, begin, name, variant)

    print("\nDone. All stochastic route files written.")


if __name__ == "__main__":
    main()
