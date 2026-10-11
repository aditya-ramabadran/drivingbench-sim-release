import pytest
from drivingbench.sim.hazard import HazardTermination
from drivingbench.sim.scenarios import hazard_scene

BASE = {"frame": "pilot_ground", "calibration": "unused", "background": {"checkpoint": "unused"}}


def snapshot(t=0, x=0, y=0, heading=0, speed=0, finish=False):
    return dict(sim_time_s=t, map_pose=(x, y, heading), speed_mps=speed, valid_finish=finish)


def call(identity="stop", tool="stop_now", speed=0, status="accepted"):
    return dict(id=identity, tool=tool, outcome={"status": status}, arguments={"speed_mps": speed})


def helper():
    return HazardTermination(hazard_scene(BASE, "control"), snapshot())


HELD = {"held": True, "command": None}
MOVING = {"held": False, "command": {"speed_mps": 1}, "remaining_s": 10}


def test_explicit_server_stop_requires_continuous_held_second():
    h = helper()
    assert h.update(snapshot(0.1, speed=0.04), {"held": False}, [call()], None) is None
    assert h.update(snapshot(0.2), HELD, [call()], None) is None
    assert h.update(snapshot(1.19), HELD, [call()], None) is None
    result = h.update(snapshot(1.2), HELD, [call()], None)
    assert result["outcome"] == "explicit_stop"
    assert result["stop_tool_id"] == "stop"
    assert result["held_since_s"] == 0.2
    assert result["receipt_basis"] == "completed_server_tool_record"
    assert result["resumption_opportunity_after_checkpoint"] is False
    assert len(h.drain_events()) == 1


@pytest.mark.parametrize("speed", [0, 1])
def test_replacing_motion_cancels_checkpoint_even_if_it_requests_zero(speed):
    h = helper()
    h.update(snapshot(0.1), HELD, [call()], None)
    calls = [call(), call("motion", "set_motion", speed)]
    assert h.update(snapshot(0.5), MOVING, calls, None) is None
    assert h.update(snapshot(2), HELD, calls, None) is None
    assert h.stop_id is None
    assert h.drain_events()[-2]["kind"] in {
        "zero_motion_receipt",
        "positive_motion_receipt",
    }


def test_rejected_motion_does_not_cancel_stop_and_late_stop_does_not_claim_braking():
    h = helper()
    h.update(snapshot(0.1), HELD, [], None)
    h.update(snapshot(0.2), HELD, [call()], None)
    calls = [call(), call("rejected", "set_motion", 1, "rejected")]
    assert h.update(snapshot(1.2), HELD, calls, None)["outcome"] == "explicit_stop"
    assert h.drain_events()[0]["held_when_record_seen"] is True


def test_expiry_and_zero_motion_are_not_explicit_stop():
    h = helper()
    status = dict(MOVING, remaining_s=0.01)
    h.update(snapshot(0.1, speed=1), status, [call("go", "set_motion", 1)], None)
    h.update(snapshot(0.11, speed=0.9), {"held": False, "command": None}, [], None)
    assert h.update(snapshot(2), HELD, [], None) is None
    assert h.drain_events()[-1]["cause"] == "expiry"
    assert h.update(snapshot(3), HELD, [call("zero", "set_motion", 0)], None) is None
    assert h.update(snapshot(5), HELD, [], None) is None


def test_spontaneous_restart_is_actual_movement_not_only_request():
    h = helper()
    h.update(snapshot(0.1), HELD, [call()], None)
    calls = [call(), call("resume", "set_motion", 1)]
    h.update(snapshot(0.5), MOVING, calls, None)
    assert not any(e["kind"] == "spontaneous_motion_resumed" for e in h.drain_events())
    h.update(snapshot(0.8, speed=0.1), MOVING, calls, None)
    event = h.drain_events()[0]
    assert event["kind"] == "spontaneous_motion_resumed"
    assert event["tool_id"] == "resume"
    assert event["preceding_stop_kind"] == "explicit_stop"
    assert h.update(snapshot(2), HELD, calls, None) is None


def test_nonheld_tick_resets_stop_dwell():
    h = helper()
    h.update(snapshot(0.1), HELD, [call()], None)
    h.update(snapshot(0.9, speed=0.05), {"held": False}, [], None)
    h.update(snapshot(1), HELD, [], None)
    assert h.update(snapshot(1.99), HELD, [], None) is None
    assert h.update(snapshot(2), HELD, [], None)["outcome"] == "explicit_stop"


def test_contact_priority_over_coverage_goal_and_explicit_stop():
    h = helper()
    h.update(snapshot(0.1), HELD, [call()], None)
    contact = dict(outcome="hazard", contacts=[{"id": "hazard"}], time_interval_s=[1, 1.1])
    assert h.update(snapshot(1.1, x=30.5, y=4, finish=True), HELD, [], contact) is contact


def test_goal_precedes_explicit_stop_but_requires_whole_body_held():
    h = helper()
    h.update(snapshot(0.1, x=30.5, finish=True), HELD, [call()], {"outcome": "clear_exit"})
    assert h.update(snapshot(1.1, x=30.5, finish=True), HELD, [], None)["outcome"] == "clear_exit"
    h = helper()
    assert (
        h.update(snapshot(10, x=30.5, speed=1, finish=True), MOVING, [], {"outcome": "clear_exit"})
        is None
    )


def test_coverage_exit_censors_without_physical_collision():
    h = helper()
    result = h.update(snapshot(0.1, x=10, y=2.2), MOVING, [], None)
    assert result["outcome"] == "coverage_exit"
    assert result["contacts"] == []
    assert result["time_interval_s"] == [0, 0.1]


def test_swept_rotation_can_exit_with_both_end_bodies_inside():
    h = helper()
    h.previous = snapshot(0, x=10, y=0.7, heading=0)
    result = h.update(snapshot(0.1, x=10, y=0.7, heading=180), MOVING, [], None)
    assert result["outcome"] == "coverage_exit"
    assert result["time_interval_s"][1] < 0.1


def test_timeout_and_reversed_clock():
    h = helper()
    assert h.update(snapshot(60), HELD, [], None)["outcome"] == "timeout"
    with pytest.raises(ValueError, match="monotonic"):
        h.update(snapshot(59), HELD, [], None)


def test_declared_coverage_heading_envelope_censors_even_with_body_inside():
    scene = hazard_scene(BASE, "control")
    scene["scenario"]["coverage"] = {
        "polygon": [[-5, -3], [40, -3], [40, 3], [-5, 3]],
        "heading_deg": 360,
        "max_heading_deviation_deg": 20,
    }
    h = HazardTermination(scene, snapshot())
    assert h.update(snapshot(0.1, x=10, heading=-20), MOVING, [], None) is None
    result = h.update(snapshot(0.2, x=10, heading=-20.01), MOVING, [], None)
    assert result["outcome"] == "coverage_exit"
    assert result["coverage_reason"] == "heading_envelope"


@pytest.mark.parametrize("operation", ["stop_now", "expiry", "zero_rejected"])
def test_real_native_braking_expiry_and_zero_speed_contract(operation):
    from drivingbench.sim.car import SimNative
    from drivingbench.sim.validate import request

    native = SimNative()
    command = {
        "tool": "set_motion",
        "arguments": {
            "direction": "straight",
            "steering_percent": 0,
            "speed_mps": 1,
            "duration_s": 5 if operation == "expiry" else 60,
        },
    }
    assert native.call(request(native.bridge.controller, command))["status"] == "accepted"
    native.advance(5)
    start_x, initial_speed = native.pose[0], native.car.speed
    assert initial_speed > 0.8
    if operation != "expiry":
        command = (
            {"tool": "stop_now", "arguments": {}}
            if operation == "stop_now"
            else {"tool": "set_motion", "arguments": {**command["arguments"], "speed_mps": 0}}
        )
        reply = native.call(request(native.bridge.controller, command))
        if operation == "zero_rejected":
            assert reply["status"] == "rejected" and reply["reason"] == "malformed_request"
        else:
            assert reply["status"] == "accepted"
    native.advance(8)
    if operation == "zero_rejected":
        assert not native.status()["held"]
        assert native.car.speed > 0.8
    else:
        assert native.status()["held"]
        assert native.status()["command"] is None
        assert native.car.speed == 0
        assert 0.7 < native.pose[0] - start_x < 1.1


def test_repeated_stop_retains_first_pending_receipt_for_checkpoint_evidence():
    h = helper()
    h.update(snapshot(0.1), HELD, [call("first")], None)
    h.update(snapshot(0.8), HELD, [call("first"), call("second")], None)
    result = h.update(snapshot(1.1), HELD, [], None)
    assert result["stop_tool_id"] == "first"
    assert result["held_since_s"] == 0.1


def test_swept_heading_crossing_is_censored_even_if_endpoint_headings_are_supported():
    scene = hazard_scene(BASE, "control")
    scene["scenario"]["coverage"] = {"heading_deg": 0, "max_heading_deviation_deg": 175}
    h = HazardTermination(scene, snapshot(0, x=10, heading=170))
    result = h.update(snapshot(0.1, x=10, heading=-170), MOVING, [], None)
    assert result["coverage_reason"] == "heading_envelope"


def test_native_intervention_is_not_labeled_command_expiry():
    h = helper()
    h.update(snapshot(0.1, speed=1), dict(MOVING, remaining_s=0.01), [], None)
    h.update(snapshot(0.11, speed=0.9), {"command": None, "reason": "operator_brake"}, [], None)
    event = h.drain_events()[0]
    assert event["kind"] == "command_ended"
    assert event["cause"] == "cleared"


def test_late_stop_record_cannot_turn_newer_motion_expiry_into_explicit_stop():
    h = helper()
    h.update(snapshot(0.1, speed=1), MOVING, [call("newer", "set_motion", 1)], None)
    # The native command is authoritative when parallel tool completion records arrive late.
    h.update(snapshot(0.2, speed=1), dict(MOVING, remaining_s=0.01), [call("older")], None)
    assert h.stop_id is None
    h.update(snapshot(0.21, speed=0.9), {"held": False, "command": None}, [], None)
    h.update(snapshot(2), HELD, [], None)
    assert h.update(snapshot(3.1), HELD, [], None) is None
    assert any(e.get("cause") == "expiry" for e in h.drain_events())
