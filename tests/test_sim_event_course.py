import copy
import json
from pathlib import Path

import numpy as np
import pytest
from drivingbench.sim.course import Course
from drivingbench.sim.event_course import author, event_scene, route
from drivingbench.sim.scenarios import ActorTimeline
from shapely.geometry import Point, Polygon


@pytest.fixture
def scene():
    return json.loads(Path("sim/site1-event-course.json").read_text())


def test_authored_course_is_reproducible_and_keeps_real_scene_independent(scene):
    base = json.loads(Path("sim/site1-course.json").read_text())
    saved = copy.deepcopy(base)
    assert author(base) == scene
    assert base == saved
    assert scene["background"] == base["background"]
    assert scene["wide_camera"] == base["wide_camera"]
    assert scene["boundaries"] != base["boundaries"]
    course = Course(scene, boundary_tolerance_m=0.5)
    start = course.assess(course.map_pose((0, 0, 0)))
    assert not start["collision_ids"] and not start["outside_course"]
    left, right = (np.array(scene["boundaries"][side]) for side in ("left", "right"))
    assert np.linalg.norm(left - right, axis=1) == pytest.approx(7)


def test_sites_are_before_finish_and_cross_between_roadsides(scene):
    course = Course(scene)
    path = np.array(scene["design"]["path"])
    length = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()
    assert 65 < length < 67
    assert len(scene["design"]["event_sites"]) == 2
    for site in scene["design"]["event_sites"]:
        assert site["trigger_station_m"] + 12 == site["crossing_station_m"]
        assert 0 < site["trigger_station_m"] < site["crossing_station_m"] < length - 15
        assert Polygon(course.corridor).contains(Point(site["map_pose"][:2]))
        for key in ("actor_start", "actor_end"):
            assert not Polygon(course.corridor).contains(Point(site[key]))
    finish = course.assess(path[-10].tolist())
    assert finish["valid_finish"]


def test_ball_and_no_ball_dog_onsets_are_matched_without_mutation(scene):
    saved = copy.deepcopy(scene)
    assets = {
        "assets": {"dog": "dog.png", "child": "child.png"},
        "animations": {
            name: {"frame_paths": [f"{name}1.png", f"{name}2.png"], "fps": 2}
            for name in ("dog", "child")
        },
    }
    ball = dict(asset_path="ball.png", width_m=0.24, height_m=0.24, speed_mps=0.8)
    plain = event_scene(scene, "early", "dog_crossing", assets=assets)
    cue = event_scene(scene, "early", "ball_then_dog", assets=assets, ball=ball)
    child = event_scene(scene, "early", "child_crossing", assets=assets)
    assert plain["scenario"]["trajectories"]["hazard"] == cue["scenario"]["trajectories"]["hazard"]
    assert plain["scenario"]["trajectories"]["hazard"]["delay_s"] == 3
    assert "delay_s" not in child["scenario"]["trajectories"]["hazard"]
    assert [c["id"] for c in cue["scenario"]["colliders"]] == ["hazard"]
    path = np.array(scene["design"]["path"])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    before, after = path[np.argmin(abs(arc - 7))], path[np.argmin(abs(arc - 9))]
    timelines = [ActorTimeline(s) for s in (plain, cue)]
    for timeline in timelines:
        assert timeline.update(before, 0) == {}
    a, b = (timeline.update(after, 10) for timeline in timelines)
    assert a["hazard"] == b["hazard"] == pytest.approx(b["ball"] + 3)
    assert scene == saved and "x" not in ball
    assert event_scene(scene, "early", "control", assets={})["scenario"]["trajectories"] == {}
    with pytest.raises(ValueError, match="Unknown"):
        event_scene(scene, "missing", "control", assets={})


@pytest.mark.parametrize("step", [0, -1, float("nan"), float("inf")])
def test_bad_sampling_rejected(step):
    with pytest.raises(ValueError, match="positive"):
        route(step)


def test_diagnostic_rejects_speed_the_real_controller_cannot_accept(scene):
    from drivingbench.sim.event_course import diagnose

    with pytest.raises(ValueError, match="shared motion contract"):
        diagnose(scene, speed=0.35)
