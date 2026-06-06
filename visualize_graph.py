"""Visualize the typed-edge graph produced by graph_builder for a SUMO net.xml.

Usage:
    python visualize_graph.py data/networks/cross_smoke.net.xml
    python visualize_graph.py data/networks/linear_two.net.xml
    python visualize_graph.py data/networks/pass_through.net.xml

Requires: matplotlib
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Add project root to path so data.graph_builder resolves
sys.path.insert(0, str(Path(__file__).parent))
from data.graph_builder import build_graph


# ---------------------------------------------------------------------------
# Parse junction positions from net.xml (not done by graph_builder)
# ---------------------------------------------------------------------------

def _parse_positions(net_path: Path) -> dict:
    root = ET.parse(net_path).getroot()
    positions = {}
    for j in root.findall("junction"):
        jid = j.get("id")
        x = j.get("x")
        y = j.get("y")
        if jid and x is not None and y is not None:
            positions[jid] = (float(x), float(y))
    return positions


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _layout(node_ids, positions):
    """Return {node_id: (x, y)} using XML positions where available,
    falling back to a simple circular layout."""
    coords = {}
    for i, nid in enumerate(node_ids):
        if nid in positions:
            coords[nid] = positions[nid]
        else:
            angle = 2 * np.pi * i / max(len(node_ids), 1)
            coords[nid] = (np.cos(angle) * 200, np.sin(angle) * 200)
    return coords


# ---------------------------------------------------------------------------
# Arrow drawing
# ---------------------------------------------------------------------------

def _draw_edge(ax, src_xy, dst_xy, color, linestyle, offset_factor=0.06):
    """Draw a curved arrow from src to dst with a small lateral offset so
    bidirectional edges don't perfectly overlap."""
    sx, sy = src_xy
    dx, dy = dst_xy

    # Perpendicular unit vector for offset
    ex, ey = dx - sx, dy - sy
    length = np.hypot(ex, ey) or 1.0
    px, py = -ey / length, ex / length

    off = offset_factor * length
    sx2 = sx + px * off
    sy2 = sy + py * off
    dx2 = dx + px * off
    dy2 = dy + py * off

    ax.annotate(
        "",
        xy=(dx2, dy2),
        xytext=(sx2, sy2),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.8,
            linestyle=linestyle,
            connectionstyle="arc3,rad=0.15",
        ),
    )


# ---------------------------------------------------------------------------
# Phase features table
# ---------------------------------------------------------------------------

def _draw_phase_table(ax, node_ids, phase_features, node_meta):
    ax.axis("off")
    ax.set_title("Phase Features  (1 = lane goes green)", fontsize=10, pad=8)

    rows = []
    col_labels = []
    row_labels = []

    for node_idx, nid in enumerate(node_ids):
        feats = phase_features[node_idx]
        n_phases = node_meta[node_idx]["num_phases"]
        n_lanes = feats[0].shape[0] if feats else 0

        if not col_labels:
            col_labels = [f"lane {i}" for i in range(n_lanes)]

        for p_idx in range(n_phases):
            row_labels.append(f"{nid} · phase {p_idx}")
            rows.append([f"{v:.0f}" for v in feats[p_idx].tolist()])

    if not rows:
        ax.text(0.5, 0.5, "(no phases)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        return

    # Pad rows that have fewer columns than the widest row
    max_cols = max(len(r) for r in rows)
    col_labels = col_labels[:max_cols]
    padded = [r + ["—"] * (max_cols - len(r)) for r in rows]

    table = ax.table(
        cellText=padded,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Colour cells: green = 1, white = 0, gray = "—"
    for (r, c), cell in table.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#dddddd")
        else:
            val = padded[r - 1][c] if r - 1 < len(padded) else "—"
            if val == "1":
                cell.set_facecolor("#c8f0c8")
            elif val == "0":
                cell.set_facecolor("#ffffff")
            else:
                cell.set_facecolor("#eeeeee")


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualize(net_path: Path, save_path: Path = None):
    net_path = Path(net_path)
    g = build_graph(net_path)
    positions = _parse_positions(net_path)

    node_ids = g["node_ids"]
    edge_index = g["edge_index"]
    edge_type = g["edge_type"]
    phase_features = g["phase_features"]
    node_meta = g["node_meta"]

    coords = _layout(node_ids, positions)

    # Figure layout: graph on top, phase table on bottom
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.35)
    ax_graph = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # --- Graph panel ---
    ax_graph.set_aspect("equal")
    ax_graph.set_title(
        f"{net_path.name}  —  {len(node_ids)} node(s), "
        f"{(edge_type == 0).sum().item()} flow edge(s), "
        f"{(edge_type == 1).sum().item()} coord edge(s)",
        fontsize=11,
    )
    ax_graph.axis("off")

    # Draw edges
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        etype = edge_type[i].item()
        src_xy = coords[node_ids[src_idx]]
        dst_xy = coords[node_ids[dst_idx]]
        if etype == 0:
            _draw_edge(ax_graph, src_xy, dst_xy, color="#2563eb", linestyle="solid")
        else:
            _draw_edge(ax_graph, src_xy, dst_xy, color="#dc2626", linestyle="dashed")

    # Draw nodes
    node_radius = 18
    for nid in node_ids:
        x, y = coords[nid]
        circle = plt.Circle((x, y), node_radius, color="#fbbf24", ec="#92400e", lw=2, zorder=3)
        ax_graph.add_patch(circle)
        n_phases = node_meta[node_ids.index(nid)]["num_phases"]
        ax_graph.text(x, y + 2, nid, ha="center", va="center",
                      fontsize=9, fontweight="bold", zorder=4)
        ax_graph.text(x, y - 7, f"{n_phases}ph", ha="center", va="center",
                      fontsize=7, color="#374151", zorder=4)

    # Auto-scale with margin
    all_x = [v[0] for v in coords.values()]
    all_y = [v[1] for v in coords.values()]
    margin = 60
    if len(all_x) > 1:
        ax_graph.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax_graph.set_ylim(min(all_y) - margin, max(all_y) + margin)
    else:
        ax_graph.set_xlim(all_x[0] - 100, all_x[0] + 100)
        ax_graph.set_ylim(all_y[0] - 100, all_y[0] + 100)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#2563eb", label="Flow edge  (upstream → downstream)"),
        mpatches.Patch(color="#dc2626", label="Coord edge (downstream → upstream)"),
    ]
    ax_graph.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # --- Phase table panel ---
    _draw_phase_table(ax_table, node_ids, phase_features, node_meta)

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_graph.py <path/to/net.xml> [output.png]")
        sys.exit(1)

    net = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    visualize(net, save_path=out)
