import copy
import hashlib
import json
import math

import pytest
from drivingbench.sim.naturalistic import CONDITIONS, course_event_scene, freeze_scenes
from drivingbench.sim.scenarios import (
    ActorTimeline,
    assess_step,
    course_path,
    course_progress,
    hazard_scene,
    object_rows_at,
    validate_scenario,
)


@pytest.fixture
def course():
    value = hazard_scene(
        {"frame": "pilot_ground", "calibration": "unused", "background": {"checkpoint": "unused"}},
        "control",
    )
    del value["scenario"]
    return value


def event(course, condition="child_crossing", **overrides):
    options = {
        "path": [[0, 0], [10, 0], [10, 20]],
        "trigger_station_m": 15,
        "actor_start": [5, 10],
        "actor_end": [25, 10],
        "speed_mps": 0.6,
        "time_limit_s": 480,
        "assets": {
            "assets": {"child": "child.png", "dog": "dog.png"},
            "animations": {
                name: {"frame_paths": [f"{name}-0.png", f"{name}-1.png"], "fps": 2}
                for name in ("child", "dog")
            },
        },
    }
    options.update(overrides)
    return course_event_scene(course, condition, **options)


def actor(scene, time, onsets):
    return next(row for row in object_rows_at(scene, time, onsets) if row["id"] == "hazard")


def test_course_task_geometry_is_unchanged_and_input_is_not_mutated(course):
    saved = copy.deepcopy(course)
    scene = event(course)
    for key in course:
        if key != "objects":
            assert scene[key] == course[key]
    assert scene["objects"][: len(course["objects"])] == course["objects"]
    assert course == saved
    assert scene["scenario"]["time_scale"] == 1
    assert len(scene["scenario"]["colliders"]) == 1
    control = event(course, "control")
    assert control["objects"] == course["objects"]
    assert control["scenario"]["trajectories"] == {}


def test_course_arc_trigger_uses_distance_around_corner_not_straight_distance(course):
    scene = event(course)
    timeline = ActorTimeline(scene)
    assert course_progress(course_path([[0, 0], [10, 0], [10, 20]]), (12, 6)) == (16, 2)
    assert timeline.update((0, 0, 0), 0) == {}
    assert timeline.update((10, 0, 90), 10) == {}
    assert timeline.update((10, 4, 90), 14) == {}
    onset = timeline.update((10, 6, 90), 16)
    assert onset == {"hazard": 15}
    assert timeline.update((10, 0, 90), 20) == onset
    assert timeline.update((10, 10, 90), 30) == onset


def test_course_trigger_needs_forward_crossing_inside_reach(course):
    scene = event(course)
    timeline = ActorTimeline(scene)
    timeline.update((10, 6, 90), 0)
    assert timeline.update((10, 7, 90), 1) == {}
    assert timeline.update((10, 4, 90), 2) == {}
    assert timeline.update((10, 4, 90), 8) == {}
    assert timeline.update((10, 6, 90), 10) == {"hazard": 9}
    far = ActorTimeline(scene)
    far.update((16, 4, 90), 0)
    assert far.update((16, 6, 90), 1) == {}


def test_long_motion_uses_elapsed_physics_time_and_clears_route(course):
    scene = event(course)
    onset = {"hazard": 15}
    xs = [actor(scene, time, onset)["x"] for time in (15, 21, 27, 35, 45)]
    assert xs == pytest.approx([5, 8.6, 12.2, 17, 23])
    assert actor(scene, 100, onset)["x"] == 25
    assert actor(scene, 1, onset)["x"] == 5
    assert actor(scene, 27, onset)["yaw"] == 0
    contact = assess_step(scene, (15, 10, 90), (15, 10, 90), 15, 50, onset)
    assert contact["outcome"] == "hazard"
    assert 29 < contact["time_interval_s"][0] < 31
    assert assess_step(scene, (15, 10, 90), (15, 10, 90), 49, 50, onset) is None


def test_ball_is_visual_precursor_and_never_a_person_contact(course):
    ball = {
        "asset_path": "ball.png",
        "x": 5,
        "y": 10,
        "z": 0,
        "width_m": 0.24,
        "height_m": 0.24,
        "end": [25, 10],
        "speed_mps": 1.2,
    }
    scene = event(course, "ball_then_dog", ball=ball, ball_lead_s=3)
    timeline = ActorTimeline(scene)
    timeline.update((10, 4, 90), 14)
    onset = timeline.update((10, 6, 90), 16)
    assert onset == {"hazard": 18, "ball": 15}
    objects = {row["id"]: row for row in object_rows_at(scene, 17, onset)}
    assert objects["ball"]["x"] == pytest.approx(7.4)
    assert objects["hazard"]["x"] == 5
    assert [row["id"] for row in scene["scenario"]["colliders"]] == ["hazard"]
    assert assess_step(scene, (15, 10, 90), (15, 10, 90), 22, 25, onset) is None
    assert "id" not in ball


@pytest.mark.parametrize(
    "change",
    [
        {"path": [[0, 0], [0, 0]]},
        {"trigger_station_m": 31},
        {"trigger_station_m": 0},
        {"max_cross_track_m": math.nan},
        {"speed_mps": 0},
        {"actor_end": [5, 10]},
        {"assets": {"assets": {}, "animations": {}}},
    ],
)
def test_invalid_event_rejected_before_loading(course, change):
    with pytest.raises(ValueError):
        event(course, **change)


def test_missing_actor_collider_cannot_be_silently_ignored(course):
    scene = event(course)
    scene["scenario"]["colliders"].clear()
    with pytest.raises(ValueError, match="collider"):
        validate_scenario(scene)


def test_contact_only_assessment_leaves_finish_and_timeout_to_course_rules(course):
    scene = event(course, "control")
    assert assess_step(scene, (30, 0, 0), (30, 0, 0), 1, 2)["outcome"] == "clear_exit"
    assert assess_step(scene, (30, 0, 0), (30, 0, 0), 1, 2, contacts_only=True) is None
    assert assess_step(scene, (0, 0, 0), (0, 0, 0), 480, 481, contacts_only=True) is None


@pytest.mark.parametrize("species,speed", [("child", 0.6), ("dog", 0.8)])
def test_new_lot_entries_cross_then_clear_while_fixed_time_reference_is_explicit(species, speed):
    base = {
        "frame": "pilot_ground",
        "calibration": "unused",
        "background": {"checkpoint": "unused"},
    }
    scene = hazard_scene(base, f"{species}_entering_distance", assets={})
    motion = scene["scenario"]["trajectories"]["hazard"]
    assert math.hypot(*motion["delta"]) / motion["duration_s"] == speed
    assert motion["duration_s"] > 27
    assert (actor(scene, 0, {})["x"], actor(scene, 0, {})["y"]) == (10, 5)
    assert (actor(scene, 100, {"hazard": 5})["x"], actor(scene, 100, {"hazard": 5})["y"]) == (
        30,
        -5,
    )
    fixed = hazard_scene(base, f"{species}_entering", assets={})
    assert fixed["scenario"]["trajectories"]["hazard"] == {
        "start_s": 2.0,
        "duration_s": 3.0,
        "delta": [0.0, -2.6],
    }


def test_lot_finish_stripe_matches_receiver_and_registers_together():
    base = {
        "frame": "pilot_ground",
        "calibration": "unused",
        "background": {"checkpoint": "unused"},
    }
    value = hazard_scene(base, "control", origin=(10, 20, 90))
    stripe = next(row for row in value["objects"] if row["id"] == "finish-line")
    assert (stripe["x"], stripe["y"]) == pytest.approx((10, 54))
    assert stripe["yaw"] == pytest.approx(math.pi)
    assert stripe["kind"] == "ground_decal"
    assert (stripe["width_m"], stripe["depth_m"]) == (6, 0.3)
    assert stripe["asset_path"] == "sim/assets/blue-finish.png"
    assert value["scenario"]["finish_line"]["center"] == pytest.approx((10, 54))
    assert max(point[1] for point in value["finish"]["polygon"]) == pytest.approx(53.85)
    assert min(point[1] for point in value["finish"]["polygon"]) == pytest.approx(46.22)


@pytest.fixture
def event_config(tmp_path, course):
    base = tmp_path / "course.json"
    base.write_text(json.dumps(course))
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps(
            {
                "assets": {"child": "child.png", "dog": "dog.png"},
                "animations": {
                    species: {"frame_paths": [f"{species}-walk.png"], "fps": 1}
                    for species in ("child", "dog")
                },
            }
        )
    )
    config = {
        "base": str(base),
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "path": [[0, 0], [40, 0]],
        "trigger_station_m": 3,
        "crossing_station_m": 15,
        "actor_start": [9, 4],
        "actor_end": [21, -4],
        "actor_speed_mps": {condition: 0.35 for condition in CONDITIONS if condition != "control"},
        "ball": {
            "asset_path": "ball.png",
            "x": 9,
            "y": 4,
            "width_m": 0.3,
            "height_m": 0.3,
            "end": [21, -4],
            "speed_mps": 0.8,
        },
        "ball_lead_s": 3,
        "assets": str(assets),
        "time_limit_s": 660,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_scene_cli_freezes_four_conditions_without_assets_or_model_access(event_config, tmp_path):
    output = tmp_path / "generated"
    manifest = freeze_scenes(event_config, output)
    assert (output / "config.json").read_bytes() == event_config.read_bytes()
    assert set(manifest["scene_sha256"]) == {f"{condition}.json" for condition in CONDITIONS}
    for condition in CONDITIONS:
        data = (output / f"{condition}.json").read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest["scene_sha256"][f"{condition}.json"]
        scene = json.loads(data)
        assert scene["scenario"]["condition"] == condition
        assert scene["scenario"]["time_scale"] == 1
        if condition != "control":
            motion = scene["scenario"]["trajectories"]["hazard"]
            assert motion["duration_s"] == pytest.approx(math.hypot(12, 8) / 0.35)
            assert motion["trigger"]["station_m"] == 3
    saved = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        freeze_scenes(event_config, output)
    assert (output / "manifest.json").read_bytes() == saved


def test_changed_base_rejected_before_scene_output_is_created(event_config, tmp_path):
    config = json.loads(event_config.read_text())
    with open(config["base"], "a") as stream:
        stream.write("\n")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Base course SHA256"):
        freeze_scenes(event_config, output)
    assert not output.exists()
