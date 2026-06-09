"""Generate capacity-calibrated stochastic route files for any SUMO network.

This universal generator works with any network provided a .net.xml file.
It optionally accepts a source .rou.xml file for existing OD pair distribution,
or generates synthetic OD pairs if none is provided.

Demand levels (vehicles IN NETWORK at steady state):
  light:  25% of jam capacity
  medium: 45% of jam capacity
  dense:  65% of jam capacity
  heavy:  90% of jam capacity

Capacity is calculated as:
  jam_capacity = total_lane_length / 7.5 (meters)
  where 7.5m = vehicle_length (5m) + min_gap (2.5m)

Usage:
    # With source route file (recommended):
    python scripts/generate_capacity_based_routes.py \\
        --net networks/external/RESCO/ingolstadt1/ingolstadt1.net.xml \\
        --source networks/external/RESCO/ingolstadt1/ingolstadt1.rou.xml \\
        --output networks/external/RESCO/ingolstadt1 \\
        --name ingolstadt1 \\
        --begin-time 57600 \\
        --time-window 3600

    # Without source route file (synthetic OD pairs):
    python scripts/generate_capacity_based_routes.py \\
        --net networks/generated/toronto/toronto.net.xml \\
        --output networks/generated/toronto \\
        --name toronto
"""

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import math


VEHICLE_LENGTH = 5.0      # meters
MIN_GAP = 2.5             # meters
JAM_CAPACITY_DIVISOR = VEHICLE_LENGTH + MIN_GAP  # 7.5 meters per vehicle

SIM_DURATION = 1000  # episode seconds (max_steps(200) * delta_time(5))

DEMAND_VARIANTS = [
    ("light", 0.25),
    ("medium", 0.45),
    ("dense", 0.65),
    ("heavy", 0.90),
]


def calculate_jam_capacity(net_path: Path) -> float:
    """Calculate jam capacity from .net.xml file.

    Sums total lane length and divides by standard vehicle spacing.
    """
    root = ET.parse(net_path).getroot()
    total_length = 0.0

    for edge in root.findall(".//edge"):
        for lane in edge.findall("lane"):
            length = float(lane.get("length", 0))
            total_length += length

    jam_capacity = total_length / JAM_CAPACITY_DIVISOR
    return jam_capacity


def extract_od_from_source(rou_path: Path, element_type: str) -> Counter:
    """Extract origin-destination pairs from source .rou.xml file.

    Args:
        rou_path: Path to source route file
        element_type: "trip" for <trip from/to> or "vehicle" for <vehicle><route edges>

    Returns:
        Counter of (from_edge, to_edge) pairs and their frequencies
    """
    root = ET.parse(rou_path).getroot()
    od_counts: Counter = Counter()

    if element_type == "trip":
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

    return od_counts


def generate_synthetic_od_pairs(net_path: Path, num_pairs: int = 20) -> Counter:
    """Generate synthetic OD pairs from network entry/exit points.

    Uses all edges that are entry or exit points (edges with low in/out degree).
    Falls back to random edges if insufficient boundary edges exist.

    Args:
        net_path: Path to .net.xml file
        num_pairs: Target number of OD pairs to generate

    Returns:
        Counter with synthetic OD pair frequencies (uniform distribution)
    """
    root = ET.parse(net_path).getroot()

    # Find all edges
    all_edges = []
    for edge in root.findall(".//edge"):
        edge_id = edge.get("id", "")
        if edge_id and not edge_id.startswith(":"):  # Skip internal edges
            all_edges.append(edge_id)

    if len(all_edges) < 2:
        raise ValueError(f"Network has only {len(all_edges)} edges, need at least 2 for OD pairs")

    # Simple heuristic: assume edges are somewhat sorted, pick from start and end regions
    # or if few edges, just pair them up
    if len(all_edges) <= num_pairs * 2:
        # Few edges, pair them systematically
        od_pairs = []
        for i in range(len(all_edges)):
            for j in range(i + 1, min(i + 3, len(all_edges))):
                od_pairs.append((all_edges[i], all_edges[j]))
        od_counts = Counter({pair: 1 for pair in od_pairs})
    else:
        # Many edges, pick boundary edges more likely
        boundary_indices = list(range(min(10, len(all_edges)))) + list(range(max(10, len(all_edges) - 10), len(all_edges)))
        boundary_edges = [all_edges[i] for i in boundary_indices if i < len(all_edges)]

        od_counts = Counter()
        for i in range(num_pairs):
            src = boundary_edges[i % len(boundary_edges)]
            dst = boundary_edges[(i + 1) % len(boundary_edges)]
            od_counts[(src, dst)] += 1

    return od_counts


def estimate_avg_trip_time(net_path: Path) -> float:
    """Estimate average trip time based on network size.

    Uses heuristic: sqrt(number_of_edges) * 2 seconds per edge at 10 m/s.
    For typical networks: ~20-50 edges → 40-100 seconds

    Can be overridden manually if actual trip data is available.
    """
    root = ET.parse(net_path).getroot()
    num_edges = len(list(root.findall(".//edge")))

    # Heuristic: avg path length ~ sqrt(num_edges), at 10 m/s avg speed
    estimated_edges_per_trip = math.sqrt(num_edges) * 1.5
    estimated_trip_time = estimated_edges_per_trip * 20  # 20s per edge on average

    return max(20.0, estimated_trip_time)  # minimum 20 seconds


def write_stochastic_routes(
    out_path: Path,
    od_probs: list,
    scale: float,
    begin_time: int,
    network_name: str,
    variant: str,
    utilization_pct: int,
    jam_capacity: float,
) -> int:
    """Write stochastic flow XML file.

    Returns:
        Expected number of vehicles in network at steady state.
    """
    end_time = begin_time + SIM_DURATION
    exp_vehicles = int(sum(p for _, _, p in od_probs) * SIM_DURATION * scale)
    exp_in_network = int(jam_capacity * utilization_pct / 100.0)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- {network_name} {variant}: {utilization_pct}% jam capacity ({exp_in_network} veh in network, ~{exp_vehicles} veh/ep, scale={scale:.2f}x) -->',
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

    return exp_vehicles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net", type=Path, required=True,
        help="Path to .net.xml file"
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Path to source .rou.xml file (optional; if not provided, synthetic OD pairs are generated)"
    )
    parser.add_argument(
        "--source-type", choices=["trip", "vehicle"], default="trip",
        help="Type of elements in source route file: 'trip' for <trip from/to>, 'vehicle' for <vehicle><route edges>"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for generated route files"
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Network name (used in output filenames)"
    )
    parser.add_argument(
        "--begin-time", type=int, default=0,
        help="SUMO simulation start time (seconds)"
    )
    parser.add_argument(
        "--time-window", type=int, default=3600,
        help="Source route file time window (seconds)"
    )
    parser.add_argument(
        "--avg-trip-time", type=float, default=None,
        help="Average trip time (seconds); if not provided, estimated from network size"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.net.exists():
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if args.source and not args.source.exists():
        raise FileNotFoundError(f"Source route file not found: {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1: Calculate jam capacity
    print(f"\n=== {args.name} ===")
    print(f"Reading network: {args.net}")
    jam_capacity = calculate_jam_capacity(args.net)
    print(f"Jam capacity: {jam_capacity:.0f} vehicles")
    print()

    # Step 2: Extract or generate OD pairs
    if args.source:
        print(f"Reading source routes: {args.source}")
        od_counts = extract_od_from_source(args.source, args.source_type)
        od_probs = [(src, dst, count / args.time_window) for (src, dst), count in od_counts.items()]
        source_flow = sum(p for _, _, p in od_probs)  # veh/s
        print(f"  {len(od_probs)} unique OD pairs")
        print(f"  Source flow rate: {source_flow:.4f} veh/s")
    else:
        print("No source route file provided; generating synthetic OD pairs...")
        od_counts = generate_synthetic_od_pairs(args.net)
        od_probs = [(src, dst, count / args.time_window) for (src, dst), count in od_counts.items()]
        source_flow = sum(p for _, _, p in od_probs)
        print(f"  Generated {len(od_probs)} synthetic OD pairs")
        print(f"  Source flow rate: {source_flow:.4f} veh/s")

    # Step 3: Estimate average trip time if not provided
    if args.avg_trip_time:
        avg_trip = args.avg_trip_time
        print(f"Using provided avg trip time: {avg_trip:.0f} seconds")
    else:
        avg_trip = estimate_avg_trip_time(args.net)
        print(f"Estimated avg trip time: {avg_trip:.0f} seconds")

    print()

    # Step 4: Generate 4 variants
    print("Generating variants:")
    for variant_name, utilization in DEMAND_VARIANTS:
        veh_in_net = jam_capacity * utilization
        flow_rate = veh_in_net / avg_trip
        scale = flow_rate / source_flow

        out_path = args.output / f"{args.name}_{variant_name}_stochastic.rou.xml"
        exp_veh = write_stochastic_routes(
            out_path, od_probs, scale, args.begin_time, args.name, variant_name,
            int(utilization * 100), jam_capacity
        )

        print(f"  {variant_name:8s} ({int(utilization*100):2d}% cap): "
              f"{int(veh_in_net):5d} veh in network, "
              f"~{exp_veh:6d} veh/ep, scale={scale:.1f}x  -> {out_path.name}")

    print(f"\nDone. All 4 route files written to {args.output}")


if __name__ == "__main__":
    main()
