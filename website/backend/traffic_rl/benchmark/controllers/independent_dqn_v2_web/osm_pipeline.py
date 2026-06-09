from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .exceptions import WebIntegrationError
from .utils import ensure_dir, json_dump, now_iso, require_exists, run_command


def validate_osm_input(osm_path: Path) -> dict[str, Any]:
    stage = "osm_validation"
    require_exists(osm_path, stage=stage)
    if osm_path.suffix.lower() not in {".osm", ".xml"}:
        raise WebIntegrationError(stage, f"Expected .osm upload, got: {osm_path.name}")

    try:
        root = ET.parse(osm_path).getroot()
    except Exception as exc:
        raise WebIntegrationError(stage, f"OSM XML parse failed: {exc}") from exc

    if root.tag != "osm":
        raise WebIntegrationError(stage, f"Root tag must be <osm>, got <{root.tag}>")

    node_count = sum(1 for _ in root.iter("node"))
    way_count = sum(1 for _ in root.iter("way"))
    relation_count = sum(1 for _ in root.iter("relation"))
    highway_way_count = 0
    signalized_nodes = 0
    for node in root.iter("node"):
        for tag in node.findall("tag"):
            if tag.attrib.get("k") == "highway" and tag.attrib.get("v") == "traffic_signals":
                signalized_nodes += 1
                break
    for way in root.iter("way"):
        for tag in way.findall("tag"):
            if tag.attrib.get("k") == "highway":
                highway_way_count += 1
                break

    if node_count == 0 or way_count == 0:
        raise WebIntegrationError(stage, "OSM must include at least one node and one way.")
    if highway_way_count == 0:
        raise WebIntegrationError(stage, "OSM must include drivable highways (no way with tag k='highway').")

    return {
        "checked_at_utc": now_iso(),
        "osm_path": str(osm_path),
        "node_count": node_count,
        "way_count": way_count,
        "relation_count": relation_count,
        "highway_way_count": highway_way_count,
        "signalized_nodes_tagged": signalized_nodes,
    }


def convert_osm_to_sumo_net(osm_path: Path, output_dir: Path, network_name: str) -> dict[str, Any]:
    stage = "osm_to_sumo_conversion"
    ensure_dir(output_dir)
    net_file = output_dir / f"{network_name}.net.xml"
    conversion_stdout = output_dir / "netconvert_stdout.txt"
    conversion_stderr = output_dir / "netconvert_stderr.txt"

    cmd = [
        "netconvert",
        "--osm-files",
        str(osm_path),
        "--output-file",
        str(net_file),
        "--geometry.remove",
        "true",
        "--ramps.guess",
        "true",
        "--roundabouts.guess",
        "true",
        "--junctions.join",
        "true",
        "--tls.guess-signals",
        "true",
        "--tls.discard-simple",
        "false",
        "--tls.join",
        "true",
        "--no-warnings",
        "true",
    ]
    run_command(cmd, stage=stage, stdout_path=conversion_stdout, stderr_path=conversion_stderr)

    if not net_file.exists():
        raise WebIntegrationError(stage, f"netconvert did not produce net file: {net_file}")

    return {
        "converted_at_utc": now_iso(),
        "net_file": str(net_file),
        "netconvert_command": cmd,
        "netconvert_stdout": str(conversion_stdout),
        "netconvert_stderr": str(conversion_stderr),
    }


def detect_traffic_lights(net_file: Path) -> dict[str, Any]:
    stage = "traffic_light_detection"
    require_exists(net_file, stage=stage)

    try:
        root = ET.parse(net_file).getroot()
    except Exception as exc:
        raise WebIntegrationError(stage, f"Net XML parse failed: {exc}") from exc

    tl_ids = sorted({elem.attrib.get("id", "") for elem in root.iter("tlLogic") if elem.attrib.get("id")})
    controlled_tls_ids = sorted({elem.attrib.get("tl", "") for elem in root.iter("connection") if elem.attrib.get("tl")})
    junction_signalized_count = sum(
        1
        for j in root.iter("junction")
        if str(j.attrib.get("type", "")).lower() in {"traffic_light", "traffic_light_right_on_red"}
    )

    accepted_tls_ids = sorted(set(tl_ids) | set(controlled_tls_ids))
    if not accepted_tls_ids:
        raise WebIntegrationError(
            stage,
            "No traffic-light controllers found in converted network. Independent DQN needs at least one controllable TLS.",
        )

    return {
        "detected_at_utc": now_iso(),
        "tl_logic_count": len(tl_ids),
        "controlled_connection_tls_count": len(controlled_tls_ids),
        "signalized_junction_count": int(junction_signalized_count),
        "accepted_tls_ids": accepted_tls_ids,
    }


def validate_sample_sumo_boot(net_file: Path, output_dir: Path) -> dict[str, Any]:
    stage = "sample_conversion_validation"
    ensure_dir(output_dir)
    empty_route_file = output_dir / "empty.rou.xml"
    stdout_path = output_dir / "sample_sumo_stdout.txt"
    stderr_path = output_dir / "sample_sumo_stderr.txt"

    empty_route_file.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<routes></routes>\n',
        encoding="utf-8",
    )

    cmd = [
        "sumo",
        "--net-file",
        str(net_file),
        "--route-files",
        str(empty_route_file),
        "--begin",
        "0",
        "--end",
        "20",
        "--no-step-log",
        "true",
        "--no-warnings",
        "true",
    ]
    run_command(cmd, stage=stage, stdout_path=stdout_path, stderr_path=stderr_path)

    return {
        "validated_at_utc": now_iso(),
        "sumo_command": cmd,
        "sample_stdout": str(stdout_path),
        "sample_stderr": str(stderr_path),
        "empty_route_file": str(empty_route_file),
    }


def write_conversion_report(
    *,
    output_path: Path,
    validation: dict[str, Any],
    conversion: dict[str, Any],
    tls_detection: dict[str, Any],
    sample_boot: dict[str, Any],
) -> None:
    payload = {
        "created_at_utc": now_iso(),
        "validation": validation,
        "conversion": conversion,
        "traffic_light_detection": tls_detection,
        "sample_conversion_boot_check": sample_boot,
    }
    json_dump(output_path, payload)
