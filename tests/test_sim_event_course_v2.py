import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from drivingbench.sim.course import Course
from drivingbench.sim.event_course import event_scene
from drivingbench.sim.event_course_v2 import author, route
from drivingbench.sim.scenarios import ActorTimeline, course_progress
from shapely.geometry import Point, Polygon


def test_scene_is_independent_and_reproducible():
    base = json.loads(Path("sim/site1-course.json").read_text())
    original = copy.deepcopy(base)
    scene = author(base)
    assert scene == json.loads(Path("sim/site1-event-course-v2.json").read_text())
    assert base == original
    assert scene["background"] == base["background"]
    assert scene["wide_camera"] == base["wide_camera"]
    assert scene["boundaries"] != base["boundaries"]
    course = Course(scene, boundary_tolerance_m=1, strict_boundary_tolerance_m=0.5)
    evidence = course.assess(course.map_pose((0, 0, 0)))
    assert not evidence["collision_ids"]
    assert not evidence["outside_course"]
    assert not evidence["strict_boundary_violation"]
    left, right = (np.array(scene["boundaries"][side]) for side in ("left", "right"))
    assert np.linalg.norm(left - right, axis=1) == pytest.approx(7)


def test_crossing_precedes_bend_and_has_a_forward_roadside_approach():
    scene = author({"frame": "site1_full_ground"})
    path = np.array(scene["design"]["path"])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    assert 51 < arc[-1] < 53
    assert path[arc <= 36, 2] == pytest.approx(230)
    assert path[-1, 2] == pytest.approx(245)
    site = scene["design"]["event_sites"][0]
    assert site["trigger_station_m"] == 12
    assert site["crossing_station_m"] == 24
    corridor = Polygon(scene["boundaries"]["left"] + scene["boundaries"]["right"][::-1])
    for endpoint in ("actor_start", "actor_end"):
        assert not corridor.contains(Point(site[endpoint]))
    angle = math.radians(scene["start"]["heading_deg"])
    tangent = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-tangent[1], tangent[0]])
    delta = np.array(site["actor_start"]) - np.array([scene["start"][k] for k in ("x", "y")])
    assert delta @ tangent == pytest.approx(30)
    assert delta @ normal == pytest.approx(5.5)
    assert math.degrees(math.atan2(delta @ normal, delta @ tangent)) < 11
    assert math.dist(site["actor_start"], site["actor_end"]) / site["actor_speed_mps"] > 46


def test_v2_reuses_latched_timeline_and_dog_onset_is_matched():
    scene = author({"frame": "site1_full_ground"})
    assets = {
        "assets": {"dog": "dog.png", "child": "child.png"},
        "animations": {
            name: {"frame_paths": [f"{name}.png"], "fps": 2} for name in ("dog", "child")
        },
    }
    plain = event_scene(scene, "straight", "dog_crossing", assets=assets)
    cue = event_scene(
        scene,
        "straight",
        "ball_then_dog",
        assets=assets,
        ball=dict(asset_path="ball.png", width_m=0.24, height_m=0.24, speed_mps=0.8),
    )
    assert plain["scenario"]["trajectories"]["hazard"] == cue["scenario"]["trajectories"]["hazard"]
    timeline = ActorTimeline(plain)
    path = scene["design"]["path"]
    assert timeline.update(path[20], 1) == {}
    stations = [course_progress([p[:2] for p in path], path[i][:2])[0] for i in (20, 36)]
    trigger = scene["design"]["event_sites"][0]["trigger_station_m"]
    onset = 1 + (trigger - stations[0]) / (stations[1] - stations[0]) * 4 + 3
    assert timeline.update(path[36], 5)["hazard"] == pytest.approx(onset)
    assert timeline.update(path[0], 10)["hazard"] == pytest.approx(onset)


@pytest.mark.parametrize("step", [0, -1, float("nan"), float("inf")])
def test_invalid_sampling_rejected(step):
    with pytest.raises(ValueError, match="positive"):
        route(step)


def test_source_walk_keeps_its_measured_distance_and_duration():
    from drivingbench.sim.event_course_v2 import event_scene as video_scene

    base = author({"frame": "site1_full_ground"})
    source = {
        "frame_paths": [f"child{i}.png" for i in range(25)],
        "fps": 24,
        "playback": "once",
        "progress_fraction": np.linspace(0, 1, 25).tolist(),
        "source_distance_m": 10.0,
        "duration_s": 1.0,
    }
    assets = {"assets": {"child": "child.png"}, "animations": {"child": source}}
    scene = video_scene(base, "straight", "child_crossing", assets=assets)
    site = scene["design"]["event_sites"][0]
    assert math.dist(site["actor_start"], site["actor_end"]) == pytest.approx(10)
    assert scene["scenario"]["trajectories"]["hazard"]["duration_s"] == pytest.approx(1)
    assert scene["scenario"]["event"]["source_distance_m"] == 10
    assert base["design"]["event_sites"][0]["actor_speed_mps"] == 0.35
    source["duration_s"] = 2
    with pytest.raises(ValueError, match="source duration"):
        video_scene(base, "straight", "child_crossing", assets=assets)
    source["duration_s"] = 1
    source["source_distance_m"] = 7.9
    with pytest.raises(ValueError, match=">=8m"):
        video_scene(base, "straight", "child_crossing", assets=assets)
