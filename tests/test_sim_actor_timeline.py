import copy
import math

import pytest
from drivingbench.sim.course import Course
from drivingbench.sim.restrictions import Restrictions
from drivingbench.sim.runtime import Runtime
from drivingbench.sim.scenarios import (
    ActorTimeline,
    assess_step,
    hazard_scene,
    object_rows_at,
    validate_scenario,
)
from drivingbench.sim.scene import scene_asset_paths

BASE = {"frame": "pilot_ground", "calibration": "unused", "background": {"checkpoint": "unused"}}


def scene():
    value = hazard_scene(BASE, "child_entering_distance", assets={})
    value["objects"][0].update(x=20, y=2.6)
    value["scenario"]["colliders"][0]["circle"][:2] = [20, 2.6]
    value["scenario"]["trajectories"]["hazard"].update(duration_s=3, delta=[0, -2.6])
    return value


def actor(value, time, onsets=None):
    return next(row for row in object_rows_at(value, time, onsets) if row["id"] == "hazard")


def test_distance_crossing_interpolates_once_independent_of_observation_calls():
    value = scene()
    timeline = ActorTimeline(value)
    assert timeline.update((0, 0, 0), 0) == {}
    assert timeline.update((7, 0, 0), 7) == {}
    for _ in range(10):
        assert actor(value, 7)["y"] == 2.6
    onsets = timeline.update((8, 0, 0), 8)
    assert onsets == pytest.approx({"hazard": 7.685})
    assert actor(value, 9.185, onsets)["y"] == pytest.approx(1.3)
    coarse = ActorTimeline(value)
    coarse.update((0, 0, 0), 0)
    assert coarse.update((10, 0, 0), 10) == pytest.approx(onsets)
    onsets["hazard"] = 0
    assert timeline.update((10, 0, 0), 10) == pytest.approx({"hazard": 7.685})


def test_stationary_reverse_and_latched_recrossing_do_not_retrigger():
    value = scene()
    timeline = ActorTimeline(value)
    timeline.update((9, 0, 0), 0)
    assert timeline.update((9, 0, 0), 10) == {}
    assert timeline.update((8, 0, 0), 11) == {}
    assert timeline.update((7, 0, 0), 12) == {}
    onsets = timeline.update((8, 0, 0), 13)
    assert onsets == pytest.approx({"hazard": 12.685})
    assert timeline.update((0, 0, 0), 20) == onsets
    assert timeline.update((9, 0, 0), 29) == onsets
    reset = ActorTimeline(value)
    assert reset.update((0, 0, 0), 0) == {}
    assert actor(value, 30, reset.onsets)["y"] == 2.6
    with pytest.raises(ValueError, match="monotonic"):
        timeline.update((0, 0, 0), 0)


def test_passed_hazard_start_does_not_trigger_until_return_to_approach():
    timeline = ActorTimeline(scene())
    timeline.update((23, 0, 0), 0)
    assert timeline.update((24, 0, 0), 1) == {}
    assert timeline.update((6, 0, 0), 10) == {}
    assert timeline.update((8, 0, 0), 12) == pytest.approx({"hazard": 11.685})


def test_trigger_registration_and_body_front_use_actual_heading():
    value = hazard_scene(BASE, "dog_entering_distance", origin=(10, 50, 90), assets={})
    timeline = ActorTimeline(value)
    timeline.update((10, 50, 90), 0)
    assert timeline.update((10, 58, 90), 8) == pytest.approx({"hazard": 7.685})
    # Turning the body 90 degrees presents its width to the crossing plane.
    timeline = ActorTimeline(scene())
    timeline.update((0, 0, 90), 0)
    assert timeline.update((9, 0, 90), 9) == {}
    assert timeline.update((10, 0, 90), 10) == pytest.approx({"hazard": 9.11})


def test_fixed_time_condition_is_unchanged_and_distinct():
    fixed = hazard_scene(BASE, "child_entering", assets={})
    distance = scene()
    assert fixed["scenario"]["trigger"] == "fixed_sim_time"
    assert distance["scenario"]["trigger"] == "front_distance"
    assert actor(fixed, 3.5)["y"] == pytest.approx(1.3)
    assert actor(distance, 3.5)["y"] == 2.6
    assert actor(fixed, 3.5, {"hazard": 99})["y"] == pytest.approx(1.3)


def test_collision_and_gait_use_same_latched_snapshot_and_do_not_mutate_scene():
    value = scene()
    value["scenario"]["animations"] = {
        "hazard": {
            "frame_paths": ["walk0.png", "walk1.png", "walk2.png", "walk3.png"],
            "fps": 4,
            "anchor_uv": [0.5, 0.9],
            "width_m": 0.6,
            "height_m": 1.3,
        }
    }
    validate_scenario(value)
    saved = copy.deepcopy(value)
    onsets = {"hazard": 5.0}
    assert actor(value, 4, onsets)["asset_path"] == "walk0.png"
    assert actor(value, 5.5, onsets)["asset_path"] == "walk2.png"
    assert actor(value, 6.5, onsets)["asset_path"] == "walk2.png"
    assert actor(value, 8, onsets)["asset_path"] == "walk0.png"
    assert actor(value, 6.5, onsets)["y"] == pytest.approx(1.3)
    hit = assess_step(value, (20, 0, 0), (20, 0, 0), 0, 10, actor_onsets=onsets)
    assert hit["outcome"] == "hazard"
    assert 6 < hit["time_interval_s"][0] < 7
    assert assess_step(value, (20, 0, 0), (20, 0, 0), 0, 10) is None
    assert value == saved


def test_asset_provenance_includes_inactive_animation_frames_and_markers():
    value = {
        "objects": [{"asset_path": "standing.png"}, {"kind": "jersey_barrier"}],
        "markers": [{"asset_path": "marker.png"}],
        "scenario": {
            "animations": {"hazard": {"frame_paths": ["walk0.png", "walk1.png", "standing.png"]}}
        },
    }
    assert scene_asset_paths(value) == ["marker.png", "standing.png", "walk0.png", "walk1.png"]


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def run_scaled(scale, *, pause=False, observe=False):
    value = scene()
    clock = Clock()
    with_runtime = Runtime(
        Course(value),
        clock=clock,
        frame_clock=clock,
        time_scale=scale,
        restrictions=Restrictions(brakes_failed_speed_mps=5.5),
        scenario=value,
    )
    try:
        for step in range(200):
            if step == 100 and pause:
                with_runtime.pause()
                frozen = with_runtime.pose_snapshot()
                clock.now += 100
                with_runtime.advance()
                assert with_runtime.pose_snapshot() == frozen
                with_runtime.resume()
            clock.now += 0.01 / scale
            with_runtime.advance()
            if observe:
                for _ in range(3):
                    sample = with_runtime.pose_snapshot()
                    sample["actor_onsets"]["untrusted"] = 999
                    object_rows_at(value, sample["sim_time_s"], sample["actor_onsets"])
        snapshot = with_runtime.pose_snapshot()
        return snapshot
    finally:
        with_runtime.close()


def test_runtime_clock_pause_dilation_and_observation_rate_share_trigger_time():
    normal = run_scaled(1)
    dilated = run_scaled(0.25, pause=True, observe=True)
    assert normal["actor_onsets"] == pytest.approx(dilated["actor_onsets"])
    assert normal["sim_time_s"] == pytest.approx(dilated["sim_time_s"])
    assert normal["map_pose"] == pytest.approx(dilated["map_pose"])
    assert normal["actor_onsets"]["hazard"] == pytest.approx((7.685 - 0.055) / 5.5)
    assert dilated["captured_at"] > normal["captured_at"] + 100


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["scenario"]["trajectories"]["hazard"].update(start_s=2),
        lambda s: s["scenario"]["trajectories"]["hazard"]["trigger"].update(distance_m=math.nan),
        lambda s: s["scenario"]["trajectories"]["hazard"]["trigger"].update(point=[1]),
        lambda s: s["scenario"].update(animations={"hazard": {"frame_paths": [], "fps": 8}}),
        lambda s: s["scenario"].update(animations={"hazard": {"frame_paths": ["x.png"], "fps": 0}}),
    ],
)
def test_trigger_and_animation_declarations_fail_before_takeover(mutation):
    value = scene()
    mutation(value)
    with pytest.raises(ValueError):
        validate_scenario(value)
