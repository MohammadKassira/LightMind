from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET


def convert_osm_to_sumo(osm_path: str, session_id: str) -> dict:
    output_dir = f"data/sessions/{session_id}/sumo"
    os.makedirs(output_dir, exist_ok=True)

    net_file = f"{output_dir}/network.net.xml"

    sumo_home = os.environ.get("SUMO_HOME", "/Users/hasanhaidar/Library/Python/3.9/lib/python/site-packages/sumo")
    # Always resolve to the pip-installed sumo package if the env var points somewhere that doesn't have the typemaps
    typemap_check = os.path.join(sumo_home, "data", "typemap", "osmNetconvert.typ.xml")
    if not os.path.exists(typemap_check):
        sumo_home = "/Users/hasanhaidar/Library/Python/3.9/lib/python/site-packages/sumo"
    env = os.environ.copy()
    env["SUMO_HOME"] = sumo_home

    result = subprocess.run(
        [
            "netconvert",
            "--osm-files", osm_path,
            "--output-file", net_file,
            "--geometry.remove",
            "--roundabouts.guess",
            "--ramps.guess",
            "--junctions.join",
            "--tls.guess-signals",
            "--tls.discard-simple",
            "--tls.join",
            "--osm.sidewalks", "false",
            "--osm.crossings", "false",
            "--keep-edges.by-vclass", "passenger",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    if result.returncode != 0:
        raise Exception(f"netconvert failed: {result.stderr}")

    network_data = extract_network_data(net_file)

    return {"net_file": net_file, "network_data": network_data}


def extract_network_data(net_file: str) -> dict:
    tree = ET.parse(net_file)
    root = tree.getroot()

    traffic_lights: list[dict] = []
    junctions: list[dict] = []
    edges: list[dict] = []

    for junction in root.findall("junction"):
        j_type = junction.get("type", "")
        j_id = junction.get("id")
        x = float(junction.get("x", 0))
        y = float(junction.get("y", 0))

        junctions.append({"id": j_id, "x": x, "y": y, "type": j_type})

        if j_type in ("traffic_light", "traffic_light_unregulated"):
            traffic_lights.append({"id": j_id, "x": x, "y": y})

    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue

        lanes = edge.findall("lane")
        if not lanes:
            continue

        shape = lanes[0].get("shape", "")
        if shape:
            coords = []
            for point in shape.split():
                px, py = point.split(",")
                coords.append({"x": float(px), "y": float(py)})

            edges.append({
                "id": edge.get("id"),
                "coords": coords,
                "speed": float(lanes[0].get("speed", 13.9)),
            })

    return {
        "traffic_lights": traffic_lights,
        "junctions": junctions,
        "edges": edges,
        "tl_count": len(traffic_lights),
        "junction_count": len(junctions),
        "edge_count": len(edges),
    }


def convert_sumo_coords_to_latlon(net_file: str, x: float, y: float) -> dict:
    """Convert SUMO internal XY coordinates to real lat/lon using the net.xml location element."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    location = root.find("location")

    if location is None:
        return {"lat": 0.0, "lon": 0.0}

    net_offset = location.get("netOffset", "0,0").split(",")
    conv_boundary = location.get("convBoundary", "0,0,0,0").split(",")
    orig_boundary = location.get("origBoundary", "0,0,0,0").split(",")

    offset_x = float(net_offset[0])
    offset_y = float(net_offset[1])

    orig_coords = [float(c) for c in orig_boundary]
    conv_coords = [float(c) for c in conv_boundary]

    if conv_coords[2] == conv_coords[0] or conv_coords[3] == conv_coords[1]:
        return {"lat": 0.0, "lon": 0.0}

    lon = orig_coords[0] + (x - offset_x - conv_coords[0]) / (conv_coords[2] - conv_coords[0]) * (orig_coords[2] - orig_coords[0])
    lat = orig_coords[1] + (y - offset_y - conv_coords[1]) / (conv_coords[3] - conv_coords[1]) * (orig_coords[3] - orig_coords[1])

    return {"lat": lat, "lon": lon}


def get_network_summary(network_data: dict, net_file: str) -> dict:
    """Convert all SUMO XY positions to lat/lon.

    Primary: sumolib.net.readNet (handles the real cartographic projection).
    Fallback: manual linear interpolation from the <location> element.
    """
    import logging
    log = logging.getLogger(__name__)

    # ── sumolib path ────────────────────────────────────────────────────────────
    try:
        import sumolib  # noqa: PLC0415
        net = sumolib.net.readNet(net_file)

        tls_latlon = []
        for tl in network_data["traffic_lights"]:
            lon, lat = net.convertXY2LonLat(tl["x"], tl["y"])
            tls_latlon.append({"id": tl["id"], "lat": lat, "lon": lon})

        road_segments = []
        for edge in network_data["edges"][:300]:
            seg = []
            for pt in edge["coords"]:
                lon, lat = net.convertXY2LonLat(pt["x"], pt["y"])
                seg.append({"lat": lat, "lon": lon})
            if len(seg) >= 2:
                road_segments.append({"id": edge["id"], "coords": seg})

        log.info("sumolib projection: %d TLs, %d segments", len(tls_latlon), len(road_segments))
        return {
            "traffic_lights": tls_latlon,
            "road_segments": road_segments,
            "stats": {
                "tl_count": len(tls_latlon),
                "junction_count": network_data["junction_count"],
                "edge_count": network_data["edge_count"],
            },
        }
    except ImportError as exc:
        log.warning("sumolib not available (%s) — falling back to manual projection", exc)
    except Exception as exc:
        log.warning("sumolib.net.readNet failed for %s (%s) — falling back", net_file, exc)

    # ── manual fallback ─────────────────────────────────────────────────────────
    try:
        tls_latlon = []
        for tl in network_data["traffic_lights"]:
            c = convert_sumo_coords_to_latlon(net_file, tl["x"], tl["y"])
            tls_latlon.append({"id": tl["id"], "lat": c["lat"], "lon": c["lon"]})

        road_segments = []
        for edge in network_data["edges"][:300]:
            seg = []
            for pt in edge["coords"]:
                c = convert_sumo_coords_to_latlon(net_file, pt["x"], pt["y"])
                seg.append({"lat": c["lat"], "lon": c["lon"]})
            if len(seg) >= 2:
                road_segments.append({"id": edge["id"], "coords": seg})

        return {
            "traffic_lights": tls_latlon,
            "road_segments": road_segments,
            "stats": {
                "tl_count": len(tls_latlon),
                "junction_count": network_data.get("junction_count", 0),
                "edge_count": network_data.get("edge_count", 0),
            },
            "fallback": True,
        }
    except Exception as exc:
        log.error("Manual projection also failed: %s", exc)
        return {
            "traffic_lights": [],
            "road_segments": [],
            "stats": {
                "tl_count": 0,
                "junction_count": network_data.get("junction_count", 0),
                "edge_count": network_data.get("edge_count", 0),
            },
            "error": str(exc),
        }
