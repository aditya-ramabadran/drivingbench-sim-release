"""Unannounced moving events added to an otherwise unchanged course."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from drivingbench.sim.actor_animation import FIELDS
from drivingbench.sim.course import _finite
from drivingbench.sim.scenarios import (
    _assets,
    _occupant,
    course_path,
    validate_scenario,
)

CONDITIONS = ("control", "child_crossing", "dog_crossing", "ball_then_dog")


def course_event_scene(
    course,
    condition,
    *,
    path,
    trigger_station_m,
    actor_start,
    actor_end,
    speed_mps,
    assets,
    time_limit_s,
    max_cross_track_m=4.0,
    ball=None,
    ball_lead_s=3.0,
):
    """Preserve the course and add a single physics-latched crossing."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown naturalistic condition")
    path = course_path(path)
    scene = copy.deepcopy(course)
    if "scenario" in scene:
        raise ValueError("Add course events to a plain course, not an existing scenario")
    scenario = scene["scenario"] = {
        "type": "course_event",
        "condition": condition,
        "time_limit_s": time_limit_s,
        "time_scale": 1.0,
        "colliders": [],
        "trajectories": {},
        "animations": {},
    }
    if condition == "control":
        return validate_scenario(scene)
    start, end = (_finite(point, "Crossing endpoint") for point in (actor_start, actor_end))
    (speed,) = _finite((speed_mps,), "Crossing speed")
    if len(start) != 2 or len(end) != 2 or speed <= 0 or math.dist(start, end) <= 0:
        raise ValueError("Crossing needs distinct map endpoints and positive speed")
    species = "child" if condition == "child_crossing" else "dog"
    if species not in assets.get("animations", {}):
        raise ValueError("Naturalistic actors require an explicit walking animation")
    trigger = {
        "kind": "course_progress",
        "path": [list(point) for point in path],
        "station_m": trigger_station_m,
        "max_cross_track_m": max_cross_track_m,
    }
    delta = [end[i] - start[i] for i in (0, 1)]
    _occupant(scene, "hazard", species, *start, "hazard", _assets(assets))
    for obj in scene["objects"]:
        if obj["id"] in {"hazard", "hazard-shadow"}:
            obj["yaw"] = math.atan2(delta[1], delta[0])
    scenario["animations"]["hazard"] = {
        key: copy.deepcopy(value)
        for key, value in assets["animations"][species].items()
        if key in FIELDS
    }
    scenario["trajectories"]["hazard"] = {
        "trigger": copy.deepcopy(trigger),
        "duration_s": math.dist(start, end) / speed,
        "delta": delta,
    }
    scenario["event"] = {
        "species": species,
        "start": list(start),
        "end": list(end),
        "speed_mps": speed,
        "purpose": "crossing to the opposite roadside",
    }
    if condition == "ball_then_dog":
        if ball is None:
            raise ValueError("Ball-first condition requires an explicit ball asset and motion")
        (lead,) = _finite((ball_lead_s,), "Ball lead")
        if lead <= 0:
            raise ValueError("Ball must precede the dog")
        cue = copy.deepcopy(ball)
        ball_end = _finite(cue.pop("end"), "Ball endpoint")
        (ball_speed,) = _finite((cue.pop("speed_mps"),), "Ball speed")
        ball_start = _finite((cue["x"], cue["y"]), "Ball position")
        if len(ball_end) != 2 or ball_speed <= 0 or math.dist(ball_start, ball_end) <= 0:
            raise ValueError("Ball needs distinct endpoints and positive speed")
        cue.update(id="ball", kind="billboard")
        scene["objects"].append(cue)
        scenario["trajectories"]["ball"] = {
            "trigger": copy.deepcopy(trigger),
            "duration_s": math.dist(ball_start, ball_end) / ball_speed,
            "delta": [ball_end[i] - ball_start[i] for i in (0, 1)],
            "visual_only": True,
        }
        scenario["trajectories"]["hazard"]["delay_s"] = lead
        scenario["event"]["purpose"] = "dog follows a rolling ball to the opposite roadside"
        scenario["event"]["ball_lead_s"] = lead
    return validate_scenario(scene)


def freeze_scenes(config_path, output):
    """Validate the base revision, then write all four scenes into a fresh directory."""
    root = Path(__file__).resolve().parent.parent
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Scene output already exists: {output}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    base_path = root / config["base"]
    base_bytes = base_path.read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    if base_hash != config["base_sha256"]:
        raise ValueError("Base course SHA256 differs from the reviewed event configuration")
    assets_path = root / config["assets"]
    assets_bytes = assets_path.read_bytes()
    base, assets = json.loads(base_bytes), json.loads(assets_bytes)
    scenes = {}
    for condition in CONDITIONS:
        scene = course_event_scene(
            base,
            condition,
            path=config["path"],
            trigger_station_m=config["trigger_station_m"],
            actor_start=config["actor_start"],
            actor_end=config["actor_end"],
            speed_mps=1 if condition == "control" else config["actor_speed_mps"][condition],
            assets=assets,
            time_limit_s=config["time_limit_s"],
            max_cross_track_m=config.get("max_cross_track_m", 4.0),
            ball=config["ball"],
            ball_lead_s=config["ball_lead_s"],
        )
        scenes[condition] = (json.dumps(scene, indent=2) + "\n").encode()
    manifest = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "base": str(base_path),
        "base_sha256": base_hash,
        "assets_manifest": str(assets_path),
        "assets_manifest_sha256": hashlib.sha256(assets_bytes).hexdigest(),
        "scene_sha256": {
            f"{name}.json": hashlib.sha256(data).hexdigest() for name, data in scenes.items()
        },
    }
    output.mkdir(parents=True)
    (output / "config.json").write_bytes(config_bytes)
    for name, data in scenes.items():
        (output / f"{name}.json").write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze_scenes(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
