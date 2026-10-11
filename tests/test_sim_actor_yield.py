import copy

import pytest
from drivingbench.sim.actor_yield import elapsed_at
from drivingbench.sim.scenarios import (
    ActorTimeline,
    assess_step,
    hazard_scene,
    object_rows_at,
    validate_scenario,
)

BASE = {"frame": "pilot_ground", "calibration": "unused", "background": {"checkpoint": "unused"}}


def scene(*, yielding=True):
    value = hazard_scene(BASE, "child_entering", assets={})
    actor = value["objects"][0]
    actor.update(x=0, y=4)
    value["scenario"]["colliders"][0]["circle"][:2] = [0, 4]
    motion = value["scenario"]["trajectories"]["hazard"]
    motion.update(start_s=0, duration_s=8, delta=[0, -8])
    value["scenario"]["animations"] = {
        "hazard": {
            "frame_paths": [f"walk{i}.png" for i in range(9)],
            "fps": 1,
            "playback": "once",
            "idle": {"frame_paths": ["idle0.png", "idle1.png"], "fps": 2},
        }
    }
    if yielding:
        motion["yield_policy"] = {
            "kind": "wait_for_stopped_car",
            "max_car_speed_mps": 0.1,
            "safety_margin_m": 0.75,
        }
    return validate_scenario(value)


def row(value, timeline, time):
    return object_rows_at(value, time, timeline.onsets, timeline.motion.states)[0]


def test_actor_waits_outside_held_body_and_idle_moves_while_world_time_advances():
    value = scene()
    saved = copy.deepcopy(value)
    timeline = ActorTimeline(value)
    timeline.update((0, 0, 0), 0, 0)
    timeline.update((0, 0, 0), 4, 0)
    state = timeline.motion.states["hazard"]
    stopped_elapsed = state["elapsed_s"]
    actor = row(value, timeline, 4)
    assert actor["y"] > 0.89 + 0.32 + 0.75
    assert state["paused"] and actor["asset_path"].startswith("idle")
    assert (
        assess_step(
            value,
            (0, 0, 0),
            (0, 0, 0),
            0,
            4,
            actor_states=timeline.motion.states,
            contacts_only=True,
        )
        is None
    )
    timeline.update((0, 0, 0), 4.5, 0.05)
    assert row(value, timeline, 4.5)["asset_path"] != actor["asset_path"]
    assert timeline.motion.states["hazard"]["elapsed_s"] == stopped_elapsed
    assert timeline.motion.states["hazard"]["waited_s"] > 2
    assert timeline.previous[1] == 4.5
    assert value == saved


def test_actor_resumes_native_source_phase_and_departs_when_car_clears():
    value = scene()
    timeline = ActorTimeline(value)
    timeline.update((0, 0, 0), 0, 0)
    timeline.update((0, 0, 0), 4, 0)
    elapsed = timeline.motion.states["hazard"]["elapsed_s"]
    timeline.update((20, 0, 0), 5, 0)
    assert not timeline.motion.states["hazard"]["paused"]
    assert timeline.motion.states["hazard"]["elapsed_s"] == pytest.approx(elapsed + 1)
    assert row(value, timeline, 5)["asset_path"].startswith("walk")
    timeline.update((20, 0, 0), 20, 0)
    assert row(value, timeline, 20)["y"] == -4
    assert row(value, timeline, 20)["asset_path"].startswith("idle")


def test_moving_car_is_not_protected_and_resuming_car_can_contact_actor():
    value = scene()
    timeline = ActorTimeline(value)
    timeline.update((0, 0, 0), 0, 0.6)
    timeline.update((0, 0, 0), 4, 0.6)
    assert not timeline.motion.states["hazard"]["paused"]
    assert (
        assess_step(
            value,
            (0, 0, 0),
            (0, 0, 0),
            0,
            4,
            actor_states=timeline.motion.states,
            contacts_only=True,
        )["id"]
        == "hazard"
    )
    timeline = ActorTimeline(value)
    timeline.update((0, 0, 0), 0, 0)
    timeline.update((0, 0, 0), 4, 0)
    before = row(value, timeline, 4)
    previous_elapsed = timeline.motion.states["hazard"]["elapsed_s"]
    timeline.update((0, 1, 0), 5, 0.6)
    assert timeline.motion.states["hazard"]["elapsed_s"] == previous_elapsed
    assert timeline.motion.states["hazard"]["pause_reason"] == "yield_to_car"
    assert row(value, timeline, 5)["y"] == before["y"]
    assert row(value, timeline, 5)["asset_path"].startswith("idle")
    assert (
        assess_step(
            value,
            (0, 0, 0),
            (0, 1, 0),
            4,
            5,
            actor_states=timeline.motion.states,
            contacts_only=True,
        )["id"]
        == "hazard"
    )


def test_far_car_does_not_pause_and_clock_interpolation_matches_root():
    value = scene()
    timeline = ActorTimeline(value)
    timeline.update((20, 0, 0), 0, 0)
    timeline.update((20, 0, 0), 4, 0)
    state = timeline.motion.states["hazard"]
    assert state["elapsed_s"] == 4 and not state["paused"]
    assert elapsed_at(timeline.motion.states, "hazard", 2, 99) == 2
    assert row(value, timeline, 2)["y"] == 2


def test_old_scene_has_no_actor_clock_and_still_contacts_stationary_car():
    value = scene(yielding=False)
    timeline = ActorTimeline(value)
    timeline.update((0, 0, 0), 0, 0)
    timeline.update((0, 0, 0), 4, 0)
    assert timeline.motion.states == {}
    assert row(value, timeline, 4)["y"] == 0
    assert assess_step(value, (0, 0, 0), (0, 0, 0), 0, 4, contacts_only=True)["id"] == "hazard"


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value["scenario"]["trajectories"]["hazard"]["yield_policy"].update(
            kind="teleport"
        ),
        lambda value: value["scenario"]["trajectories"]["hazard"]["yield_policy"].update(
            safety_margin_m=0
        ),
        lambda value: value["scenario"]["trajectories"]["hazard"]["yield_policy"].update(
            max_car_speed_mps=True
        ),
        lambda value: value["scenario"]["animations"]["hazard"].pop("idle"),
    ],
)
def test_yield_policy_rejected_before_start_if_invalid(change):
    value = scene()
    change(value)
    with pytest.raises(ValueError):
        validate_scenario(value)


def test_runtime_records_shared_clock_without_pausing_native_time():
    from drivingbench.sim.course import Course
    from drivingbench.sim.runtime import Runtime

    class Clock:
        now = 1000.0

        def __call__(self):
            return self.now

    value = scene()
    clock = Clock()
    runtime = Runtime(Course(value), clock=clock, frame_clock=clock, scenario=value)
    try:
        for _ in range(600):
            clock.now += 0.01
            runtime.advance()
        captured = runtime.capture_snapshot()
        state = captured["actor_states"]["hazard"]
        assert captured["sim_time_s"] == pytest.approx(6)
        assert captured["speed_mps"] == 0
        assert state["paused"] and 2 < state["elapsed_s"] < 2.1
        assert state["waited_s"] > 3.9
        assert captured["scenario_contact"] is None
        objects = object_rows_at(
            value, captured["sim_time_s"], captured["actor_onsets"], captured["actor_states"]
        )
        assert objects[0]["asset_path"].startswith("idle")
        captured["actor_states"]["hazard"]["elapsed_s"] = 999
        assert runtime.capture_snapshot()["actor_states"]["hazard"]["elapsed_s"] < 2.1
    finally:
        runtime.close()


def test_partial_tick_keeps_native_gait_rate_then_waits_and_measured_knots_do_not_tunnel():
    value = scene()
    value["scenario"]["animations"]["hazard"].update(
        frame_paths=["a.png", "b.png", "c.png"], progress_fraction=[0, 0.8, 1]
    )
    timeline = ActorTimeline(value)
    timeline.update((0, 0, 0), 0, 0)
    timeline.update((0, 0, 0), 4, 0)
    state = timeline.motion.states["hazard"]
    assert elapsed_at(timeline.motion.states, "hazard", 0.5, 99) == 0.5
    assert elapsed_at(timeline.motion.states, "hazard", 3.5, 99) == state["elapsed_s"]
    assert row(value, timeline, 4)["y"] > 1.96
    assert (
        assess_step(
            value,
            (0, 0, 0),
            (0, 0, 0),
            0,
            4,
            actor_states=timeline.motion.states,
            contacts_only=True,
        )
        is None
    )


def test_authored_hold_uses_measured_root_then_native_departure_and_world_clock():
    value = scene(yielding=False)
    value["scenario"]["trajectories"]["hazard"]["hold"] = {"at_fraction": 0.5, "duration_s": 30}
    value["scenario"]["animations"]["hazard"].update(
        frame_paths=["a.png", "b.png", "c.png"], progress_fraction=[0, 0.8, 1]
    )
    timeline = ActorTimeline(value)
    timeline.update((20, 0, 0), 0, 0)
    timeline.update((20, 0, 0), 10, 0)
    state = timeline.motion.states["hazard"]
    assert state["pause_reason"] == "authored_hold"
    assert state["hold_started_s"] == pytest.approx(2.5)
    assert state["hold_until_s"] == pytest.approx(32.5)
    assert row(value, timeline, 10)["y"] == pytest.approx(0)
    assert row(value, timeline, 10)["asset_path"].startswith("idle")
    timeline.update((20, 0, 0), 33.5, 0)
    assert not timeline.motion.states["hazard"]["paused"]
    assert timeline.motion.states["hazard"]["elapsed_s"] == pytest.approx(3.5)
    assert row(value, timeline, 33.5)["y"] == pytest.approx(-1.6)
    timeline.update((20, 0, 0), 40, 0)
    assert row(value, timeline, 40)["y"] == -4


def test_yield_before_authored_hold_does_not_consume_hold_and_hold_then_yield_remains_safe():
    value = scene()
    value["scenario"]["trajectories"]["hazard"]["hold"] = {"at_fraction": 0.5, "duration_s": 30}
    timeline = ActorTimeline(value)
    timeline.update((0, 0, 0), 0, 0)
    timeline.update((0, 0, 0), 20, 0)
    state = timeline.motion.states["hazard"]
    assert state["pause_reason"] == "yield_to_car" and state["hold_started_s"] is None
    timeline.update((20, 0, 0), 25, 0)
    state = timeline.motion.states["hazard"]
    assert state["pause_reason"] == "authored_hold"
    assert 20 < state["hold_started_s"] < 25
    assert state["hold_until_s"] - state["hold_started_s"] == pytest.approx(30)
    assert row(value, timeline, 25)["y"] == pytest.approx(0)


def test_invalid_hold_and_idle_absence_are_rejected():
    value = scene(yielding=False)
    motion = value["scenario"]["trajectories"]["hazard"]
    for hold in [
        {"at_fraction": 2, "duration_s": 30},
        {"at_fraction": 0.5, "duration_s": 0},
        {"at_fraction": True, "duration_s": 30},
    ]:
        motion["hold"] = hold
        with pytest.raises(ValueError):
            validate_scenario(value)


def test_native_hold_clip_plays_stop_idle_depart_only_during_authored_hold():
    value = scene(yielding=False)
    value["scenario"]["trajectories"]["hazard"]["hold"] = {"at_fraction": 0.5, "duration_s": 4}
    value["scenario"]["animations"]["hazard"]["hold_clip"] = {
        "frame_paths": ["sit.png", "seated.png", "seated2.png", "rise.png", "standing.png"],
        "fps": 1,
        "playback": "once",
    }
    validate_scenario(value)
    timeline = ActorTimeline(value)
    timeline.update((20, 0, 0), 0, 0)
    assert row(value, timeline, 0)["asset_path"] == "walk0.png"
    timeline.update((20, 0, 0), 5, 0)
    assert row(value, timeline, 5)["asset_path"] == "seated.png"
    timeline.update((20, 0, 0), 7, 0)
    assert row(value, timeline, 7)["asset_path"] == "rise.png"
    assert row(value, timeline, 7)["y"] == 0
    timeline.update((20, 0, 0), 9, 0)
    assert row(value, timeline, 9)["asset_path"] == "walk5.png"
    assert row(value, timeline, 9)["y"] == -1
    value["scenario"]["trajectories"]["hazard"]["hold"]["duration_s"] = 5
    with pytest.raises(ValueError, match="native hold clip"):
        validate_scenario(value)


def test_yield_latches_through_car_resumption_then_releases_at_native_pace_once_clear():
    value = scene()
    timeline = ActorTimeline(value)
    timeline.update((0, 0, 0), 0, 0)
    timeline.update((0, 0, 0), 4, 0)
    elapsed = timeline.motion.states["hazard"]["elapsed_s"]
    position = row(value, timeline, 4)["y"]
    timeline.update((0.1, 0, 0), 4.1, 1)
    assert timeline.motion.states["hazard"]["paused"]
    assert timeline.motion.states["hazard"]["elapsed_s"] == elapsed
    assert row(value, timeline, 4.1)["y"] == position
    timeline.update((20, 0, 0), 5.1, 1)
    assert not timeline.motion.states["hazard"]["paused"]
    assert timeline.motion.states["hazard"]["elapsed_s"] == pytest.approx(elapsed + 1)
    assert row(value, timeline, 5.1)["y"] == pytest.approx(position - 1)
    assert row(value, timeline, 5.1)["asset_path"].startswith("walk")
