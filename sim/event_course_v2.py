"""Author the straight-first event course without changing the real or v1 courses."""

from __future__ import annotations

import copy
import math

import numpy as np
from drivingbench.sim.course import Course
from drivingbench.sim.event_course import _author_geometry
from drivingbench.sim.event_course import event_scene as _event_scene
from drivingbench.sim.naturalistic import CONDITIONS


def route(step=0.5):
    """A supported southwest straight followed by a shallow bend and finish receiver."""
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Route sampling step must be finite and positive")
    heading, radius = math.radians(230), 30.0
    start = np.array([-12.0, 2.0])
    distances = np.linspace(-6, 36, math.ceil(42 / step) + 1)
    opening = start + distances[:, None] * np.array([math.cos(heading), math.sin(heading)])
    turn = np.linspace(0, math.radians(15), math.ceil(radius * math.radians(15) / step) + 1)
    bend = opening[-1] + radius * np.column_stack(
        (np.sin(heading + turn) - math.sin(heading), -np.cos(heading + turn) + math.cos(heading))
    )
    final_heading = heading + turn[-1]
    ending_distances = np.linspace(0, 8, math.ceil(8 / step) + 1)
    ending = bend[-1] + ending_distances[:, None] * np.array(
        [math.cos(final_heading), math.sin(final_heading)]
    )
    return np.vstack(
        (
            np.column_stack((opening, np.full(len(opening), 230))),
            np.column_stack((bend[1:], np.degrees(heading + turn[1:]))),
            np.column_stack((ending[1:], np.full(len(ending) - 1, 245))),
        )
    )


def author(base):
    """Retain camera/ground settings and independently author the course geometry."""
    scene, path, _stations = _author_geometry(
        base, route(), start_distance=6, name="Site 1 straight-first event course v2"
    )
    angle = math.radians(230)
    tangent = np.array([math.cos(angle), math.sin(angle)])
    perpendicular = np.array([-tangent[1], tangent[0]])
    crossing = np.array([-12.0, 2.0]) + 24 * tangent
    scene["design"] = dict(
        source="sim/event_course_v2.py",
        independent_of=["sim/site1-course.json", "sim/site1-event-course.json"],
        nominal_width_m=7.0,
        minimum_centerline_radius_m=30.0,
        opening_straight_length_m=36.0,
        path=path.tolist(),
        event_sites=[
            dict(
                id="straight",
                trigger_station_m=12.0,
                crossing_station_m=24.0,
                map_pose=[*crossing.tolist(), 230.0],
                actor_start=(crossing + 6 * tangent + 5.5 * perpendicular).tolist(),
                actor_end=(crossing - 6 * tangent - 5.5 * perpendicular).tolist(),
                actor_speed_mps=0.35,
                nominal_trigger_to_crossing_s={"0.6_mps": 20.0, "1_mps": 12.0},
            )
        ],
        note="Authored straight-first geometry. Training-pose support, paired visibility and native bypasses are separate evidence; not a surveyed obstacle map.",
    )
    Course(scene, boundary_tolerance_m=1.0)
    return scene


def event_scene(scene, site_id, condition, *, assets, ball=None, time_limit_s=660):
    """Bind finite source-video root distance and duration without retiming the walk."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown naturalistic condition")
    if condition == "control":
        return _event_scene(scene, site_id, condition, assets=assets, time_limit_s=time_limit_s)
    species = "child" if condition == "child_crossing" else "dog"
    animation = assets["animations"][species]
    distance = animation.get("source_distance_m", math.nan)
    duration = animation.get("duration_s", math.nan)
    source_duration = (len(animation["frame_paths"]) - 1) / animation["fps"]
    if (
        animation.get("playback") != "once"
        or not math.isfinite(distance)
        or distance < 8.0
        or not math.isfinite(duration)
        or duration <= 0
        or not math.isclose(duration, source_duration, abs_tol=1e-6)
    ):
        raise ValueError("Course v2 requires a measured >=8m finite walk at its source duration")
    bound = copy.deepcopy(scene)
    sites = {site["id"]: site for site in bound["design"]["event_sites"]}
    if site_id not in sites:
        raise ValueError("Unknown event-course site")
    site = sites[site_id]
    angle = math.radians(site["map_pose"][2])
    tangent = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-tangent[1], tangent[0]])
    center = np.array(site["map_pose"][:2])
    along = math.sqrt(distance * distance - 64)
    half = (along * tangent + 8 * normal) / 2
    site.update(
        actor_start=(center + half).tolist(),
        actor_end=(center - half).tolist(),
        actor_speed_mps=distance / duration,
        source_distance_m=distance,
        source_duration_s=duration,
    )
    result = _event_scene(
        bound, site_id, condition, assets=assets, ball=ball, time_limit_s=time_limit_s
    )
    result["scenario"]["event"].update(
        source_distance_m=distance,
        source_duration_s=duration,
        source_fps=animation["fps"],
        motion_binding="Measured root progress at native source duration; diagonal 8m transverse crossing",
    )
    return result
