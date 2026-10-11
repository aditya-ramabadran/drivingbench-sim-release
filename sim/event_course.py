"""Author and diagnose the separate wide event course through the unmodified native car."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from drivingbench.shared.contracts import SPEED_RANGE_MPS
from drivingbench.sim.car import DT, SimNative
from drivingbench.sim.course import Course
from drivingbench.sim.naturalistic import course_event_scene
from drivingbench.sim.scenarios import validate_scenario
from drivingbench.sim.validate import request


def _line(start, heading, length, step):
    distances = np.linspace(0, length, math.ceil(length / step) + 1)
    angle = math.radians(heading)
    return np.column_stack(
        (
            start[0] + distances * math.cos(angle),
            start[1] + distances * math.sin(angle),
            np.full(len(distances), heading),
        )
    )


def route(step=0.5):
    """Tangent straights and circular bends, expressed in the calibrated full-site map."""
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Route sampling step must be finite and positive")
    radius, center = 25.5, np.array([-5.0, -22.0])
    angles = np.linspace(70, 155, math.ceil(math.radians(85) * radius / step) + 1)
    first = np.column_stack(
        (
            center[0] + radius * np.cos(np.radians(angles)),
            center[1] + radius * np.sin(np.radians(angles)),
            angles + 90,
        )
    )
    start = first[0, :2] - 10 * np.array([math.cos(math.radians(160)), math.sin(math.radians(160))])
    opening = _line(start, 160, 10, step)
    ending = _line(first[-1, :2], 245, 30, step)
    return np.vstack((opening, first[1:], ending[1:]))


def _author_geometry(base, poses, *, start_distance, name):
    """Shared marker, corridor and receiver geometry for independent authored courses."""
    scene = copy.deepcopy(base)
    scene.pop("scenario", None)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))]
    normal = np.column_stack((-np.sin(np.radians(poses[:, 2])), np.cos(np.radians(poses[:, 2]))))
    left, right = poses[:, :2] + 3.5 * normal, poses[:, :2] - 3.5 * normal
    start_index = int(np.argmin(abs(arc - start_distance)))
    scene.update(
        name=name,
        start=dict(zip(("x", "y", "heading_deg"), poses[start_index].tolist())),
        boundaries=dict(left=left.tolist(), right=right.tolist()),
        markers=[],
        objects=[],
    )
    for side, edge, color in (
        ("left", left, [0.60, 0.71, 0.12]),
        ("right", right, [0.88, 0.22, 0.055]),
    ):
        for index, distance in enumerate(np.arange(1.0, arc[-1] - 1, 4.0)):
            xy = [float(np.interp(distance, arc, edge[:, j])) for j in (0, 1)]
            scene["markers"].append(
                dict(
                    id=f"{side}-{index:02d}",
                    kind="dome_marker",
                    x=xy[0],
                    y=xy[1],
                    z=0.0,
                    diameter_m=0.2,
                    height_m=0.06,
                    color_srgb=color,
                    collision_radius_m=0.1,
                )
            )
        xy = [float(np.interp(arc[-1] - 10, arc, edge[:, j])) for j in (0, 1)]
        scene["markers"].append(
            dict(id=f"finish-{side}", kind="cone", x=xy[0], y=xy[1], z=0.0, collision_radius_m=0.25)
        )
    finish_indices = np.flatnonzero(arc >= arc[-1] - 10)
    scene["finish"] = dict(
        polygon=np.vstack((left[finish_indices], right[finish_indices[::-1]])).tolist()
    )
    path = poses[start_index:]
    stations = arc[start_index:] - arc[start_index]
    return scene, path, stations


def author(base):
    """Copy only scene rendering settings; the new course is an independent authored asset."""
    scene, path, stations = _author_geometry(
        base, route(), start_distance=12, name="Site 1 wide event course v1"
    )
    sites = []
    for label, crossing in (("early", 20.0), ("middle", 43.0)):
        position = np.array([np.interp(crossing, stations, path[:, j]) for j in (0, 1)])
        heading = float(np.interp(crossing, stations, np.unwrap(np.radians(path[:, 2]))))
        tangent = np.array([math.cos(heading), math.sin(heading)])
        normal = np.array([-tangent[1], tangent[0]])
        sites.append(
            dict(
                id=label,
                trigger_station_m=crossing - 12,
                crossing_station_m=crossing,
                map_pose=[*position.tolist(), math.degrees(heading)],
                actor_start=(position - 6 * tangent + 5 * normal).tolist(),
                actor_end=(position + 6 * tangent - 5 * normal).tolist(),
                actor_speed_mps=0.35,
                nominal_trigger_to_crossing_s={"0.6_mps": 20.0, "1_mps": 12.0},
            )
        )
    scene["design"] = dict(
        source="sim/event_course.py",
        independent_of="sim/site1-course.json",
        nominal_width_m=7.0,
        minimum_centerline_radius_m=25.5,
        path=path.tolist(),
        event_sites=sites,
        note="Authored geometry; render coverage and native diagnostics are separate evidence. Not a surveyed obstacle map.",
    )
    Course(scene, boundary_tolerance_m=0.5)
    return scene


def event_scene(scene, site_id, condition, *, assets, ball=None, time_limit_s=660):
    """Add an existing walking asset at a course site, with matched dog onset."""
    sites = {site["id"]: site for site in scene["design"]["event_sites"]}
    if site_id not in sites:
        raise ValueError("Unknown event-course site")
    site = sites[site_id]
    cue = copy.deepcopy(ball)
    if cue is not None:
        cue.update(x=site["actor_start"][0], y=site["actor_start"][1], end=site["actor_end"])
    result = course_event_scene(
        scene,
        condition,
        path=[pose[:2] for pose in scene["design"]["path"]],
        trigger_station_m=site["trigger_station_m"],
        actor_start=site["actor_start"],
        actor_end=site["actor_end"],
        speed_mps=site["actor_speed_mps"],
        assets=assets,
        time_limit_s=time_limit_s,
        ball=cue,
        ball_lead_s=3,
    )
    if condition == "dog_crossing":
        result["scenario"]["trajectories"]["hazard"]["delay_s"] = 3
    if condition != "control":
        result["scenario"]["event"].update(
            site_id=site_id, onset_delay_s=0 if condition == "child_crossing" else 3
        )
    return validate_scenario(result)


def diagnose(
    scene,
    *,
    speed=0.6,
    lookahead=6.0,
    command_interval=0.5,
    offset_m=0.0,
    event_station=None,
    time_limit=660,
):
    """Privileged-map steering commands; never write native pose, speed or controller state."""
    if not all(
        math.isfinite(v) and v > 0 for v in (speed, lookahead, command_interval, time_limit)
    ):
        raise ValueError("Diagnostic settings must be finite and positive")
    if not SPEED_RANGE_MPS[0] <= speed <= SPEED_RANGE_MPS[1]:
        raise ValueError("Diagnostic speed must respect the shared motion contract")
    course = Course(scene, boundary_tolerance_m=0.5)
    path = np.array(scene["design"]["path"], dtype=float)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    target = path[:, :2].copy()
    if offset_m:
        if event_station is None:
            raise ValueError("Bypass offset requires an event station")
        fraction = np.clip((18 - abs(arc - event_station)) / 13, 0, 1)
        offset = offset_m * (1 - np.cos(math.pi * fraction)) / 2
        target += offset[:, None] * np.column_stack(
            (-np.sin(np.radians(path[:, 2])), np.cos(np.radians(path[:, 2])))
        )
    native = SimNative()
    records, commands = [], []
    nearest, next_command, stopping, dwell, cause = 0, 0.0, False, 0.0, None
    while native.time < time_limit:
        pose = course.map_pose(native.pose)
        if not stopping and native.time >= next_command:
            point = np.array(pose[:2])
            nearest += int(
                np.argmin(
                    np.linalg.norm(
                        target[nearest : min(nearest + 100, len(target))] - point, axis=1
                    )
                )
            )
            goal = np.array(
                [np.interp(arc[nearest] + lookahead, arc, target[:, j]) for j in (0, 1)]
            )
            delta = goal - point
            heading = math.radians(pose[2])
            lateral = -math.sin(heading) * delta[0] + math.cos(heading) * delta[1]
            curvature = 2 * lateral / max(float(delta @ delta), 1.0)
            wheel = math.degrees(
                native.vm.get_steer_from_curvature(curvature, max(native.car.speed, 0.1), 0)
            )
            arguments = dict(
                direction="left" if wheel > 0 else "right",
                steering_percent=min(abs(wheel) / 180 * 100, 100),
                speed_mps=speed,
                duration_s=5,
            )
            replies = []
            native.inbox.pending.append(
                (
                    request(native.bridge.controller, dict(tool="set_motion", arguments=arguments)),
                    replies,
                )
            )
            commands.append(dict(time_s=native.time, arguments=arguments, replies=replies))
            next_command = native.time + command_interval
        native.tick()
        pose = course.map_pose(native.pose)
        geometry = course.assess(pose)
        records.append(
            dict(
                t=native.time,
                map_pose=pose,
                speed_mps=native.car.speed,
                wheel_deg=native.car.steer_deg,
                **geometry,
            )
        )
        if cause is None and (geometry["collision_ids"] or geometry["boundary_violation"]):
            cause = "marker_hit" if geometry["collision_ids"] else "corridor_exit"
            native.brake = True
            stopping = True
        if not stopping and geometry["valid_finish"]:
            replies = []
            native.inbox.pending.append(
                (request(native.bridge.controller, dict(tool="stop_now", arguments={})), replies)
            )
            commands.append(dict(time_s=native.time, tool="stop_now", replies=replies))
            stopping = True
        dwell = (
            dwell + DT if stopping and geometry["valid_finish"] and native.car.speed < 0.05 else 0.0
        )
        if stopping and cause and native.car.standstill:
            break
        if dwell >= 1.0:
            cause = "finish"
            break
    result = dict(
        outcome=cause or "timeout",
        duration_s=native.time,
        speed_request_mps=speed,
        lookahead_m=lookahead,
        command_interval_s=command_interval,
        offset_m=offset_m,
        event_station_m=event_station,
        command_count=len(commands),
        all_commands_accepted=all(
            r["replies"] and r["replies"][0]["status"] == "accepted" for r in commands
        ),
        last=records[-1],
        minimum_centerline_tracking_error_m=float(
            min(
                np.linalg.norm(path[:, :2] - np.array(r["map_pose"][:2]), axis=1).min()
                for r in records
            )
        ),
        maximum_centerline_tracking_error_m=float(
            max(
                np.linalg.norm(path[:, :2] - np.array(r["map_pose"][:2]), axis=1).min()
                for r in records
            )
        ),
        outside_ticks=sum(r["outside_course"] for r in records),
        boundary_violation_ticks=sum(r["boundary_violation"] for r in records),
        touched_markers=sorted({identity for r in records for identity in r["collision_ids"]}),
        interpretation="Privileged-map feasibility diagnostic through real native controllers/plant, not model driving.",
    )
    return result, records, commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--author", action="store_true")
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument("--lookahead", type=float, default=6.0)
    parser.add_argument("--command-interval", type=float, default=0.5)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--event-station", type=float)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    if args.author:
        with args.output.open("x") as f:
            json.dump(author(scene), f, indent=2)
            f.write("\n")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    result, records, commands = diagnose(
        scene,
        speed=args.speed,
        lookahead=args.lookahead,
        command_interval=args.command_interval,
        offset_m=args.offset,
        event_station=args.event_station,
    )
    for name, data in (("report", result), ("states", records), ("commands", commands)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
