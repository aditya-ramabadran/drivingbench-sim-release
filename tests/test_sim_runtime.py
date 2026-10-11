import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from drivingbench.shared.contracts import PROTOCOL
from drivingbench.sim.runtime import DT, Runtime
from drivingbench.sim.validate import request


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Course:
    def map_pose(self, pose):
        return pose[0] + 10, pose[1], pose[2]

    def assess(self, pose):
        return {
            "collision_ids": ["marker"],
            "outside_course": True,
            "finish_reached": True,
            "valid_finish": False,
        }


@pytest.fixture
def manual():
    clock = Clock()
    runtime = Runtime(Course(), clock=clock, frame_clock=lambda: clock.now + 30)
    yield runtime, clock
    runtime.close()


def queued(runtime, pool, body, timeout=1):
    future = pool.submit(runtime.call, body, timeout=timeout)
    with runtime._condition:
        assert runtime._condition.wait_for(lambda: runtime._pending, timeout=1)
    return future


def motion(runtime, duration=10):
    return request(
        runtime.native.bridge.controller,
        {
            "tool": "set_motion",
            "arguments": {
                "direction": "straight",
                "steering_percent": 0,
                "speed_mps": 1.0,
                "duration_s": duration,
            },
        },
    )


def move(runtime, clock, seconds):
    clock.now += seconds
    return runtime.advance()


def test_clock_epochs_stale_reads_and_detached_camera_snapshot(manual):
    runtime, clock = manual
    initial = runtime.status()
    view = runtime.pose_snapshot()
    assert initial["received_at"] == pytest.approx(1000)
    assert view["captured_at"] == pytest.approx(1030)
    assert view["monotonic_at"] == pytest.approx(1000)
    assert view["sim_time_s"] == pytest.approx(0)
    clock.now += 5
    assert runtime.status()["received_at"] == initial["received_at"]
    view["collision_ids"].clear()
    assert runtime.pose_snapshot()["collision_ids"] == ["marker"]
    runtime.advance()
    assert runtime.status()["received_at"] == pytest.approx(clock.now)
    assert runtime.pose_snapshot()["captured_at"] == pytest.approx(clock.now + 30)


def test_queued_call_uses_one_due_tick_not_simnative_call(manual):
    runtime, clock = manual
    runtime.native.call = lambda *_: pytest.fail("SimNative.call would double-step")
    evidence = []
    runtime.on_evidence = lambda kind, **data: evidence.append((kind, data))
    with ThreadPoolExecutor() as pool:
        future = queued(runtime, pool, motion(runtime))
        assert runtime.native.time == pytest.approx(1000)
        assert not future.done()
        move(runtime, clock, DT)
        assert future.result()["status"] == "accepted"
    assert runtime.native.time == pytest.approx(1000 + DT)
    assert len(evidence) == 1
    assert evidence[0][0] == "sim_tick"
    assert runtime.native.bridge.controller.command["expires_at_s"] == pytest.approx(1010)
    assert runtime.end_reason is None  # Geometry evidence never ends the episode.


def test_catchup_does_not_apply_new_command_to_old_ticks(manual):
    runtime, clock = manual
    clock.now += 0.05
    with ThreadPoolExecutor() as pool:
        future = queued(runtime, pool, motion(runtime))
        runtime.advance(1000.04)
        assert not future.done()
        assert runtime.native.bridge.controller.command is None
        runtime.advance(1000.05)
        assert future.result()["status"] == "accepted"
    assert runtime.native.bridge.controller.command["accepted_at_s"] == pytest.approx(1000.04)


def test_timed_out_request_is_removed_and_cannot_execute_later(manual):
    runtime, clock = manual
    with pytest.raises(TimeoutError):
        runtime.call(motion(runtime), timeout=0.001)
    move(runtime, clock, 1)
    assert runtime.native.bridge.controller.command is None


def test_operator_brake_and_finish_decelerate_without_teleporting(manual):
    runtime, clock = manual
    with ThreadPoolExecutor() as pool:
        future = queued(runtime, pool, motion(runtime))
        move(runtime, clock, DT)
        assert future.result()["status"] == "accepted"
    move(runtime, clock, 4)
    before = runtime.pose_snapshot()
    assert before["speed_mps"] > 0.2
    runtime.operator_brake()
    assert runtime.pose_snapshot() == before
    move(runtime, clock, 0.1)
    assert runtime.end_reason is None
    runtime.finish("operator_finished")
    assert runtime.native.car.speed > 0
    move(runtime, clock, 10)
    after = runtime.pose_snapshot()
    assert after["pose"][0] > before["pose"][0]
    assert after["speed_mps"] == 0
    assert runtime.end_reason == "operator_finished"
    assert not runtime.status()["enabled"]
    assert runtime.status()["reason"] == "operator_brake"
    runtime.finish("cannot_replace_reason")
    assert runtime.end_reason == "operator_finished"


def test_close_releases_waiting_call_without_advancing_and_is_idempotent(manual):
    runtime, clock = manual
    with ThreadPoolExecutor() as pool:
        future = queued(runtime, pool, motion(runtime))
        runtime.close()
        with pytest.raises(OSError, match="closed"):
            future.result()
    assert runtime.native.time == pytest.approx(clock.now)
    assert runtime.status()["state"] == "unavailable"
    runtime.close()
    with pytest.raises(OSError):
        runtime.start()


def test_tick_failure_releases_waiting_caller_and_keeps_old_timestamp(manual):
    runtime, clock = manual

    def broken():
        raise RuntimeError("test native failure")

    runtime.native.tick = broken
    errors = []
    runtime.on_evidence = lambda kind, **data: errors.append((kind, data))
    with ThreadPoolExecutor() as pool:
        future = queued(runtime, pool, {"protocol": PROTOCOL, "operation": "status"})
        with pytest.raises(RuntimeError, match="test native failure"):
            move(runtime, clock, DT)
        with pytest.raises(OSError, match="test native failure"):
            future.result()
    assert runtime.status()["received_at"] == pytest.approx(1000)
    assert runtime.status()["reason"] == "simulator_error"
    assert len(errors) == 1 and errors[0][0] == "sim_runtime_error"
    assert errors[0][1]["error"] == "RuntimeError: test native failure"
    runtime._fail(RuntimeError("nested worker catch"))
    assert len(errors) == 1


def test_realtime_worker_runs_while_pose_consumer_is_blocked():
    runtime = Runtime(Course())
    first, five, release = threading.Event(), threading.Event(), threading.Event()
    count = 0

    def evidence(_kind, **_data):
        nonlocal count
        count += 1
        first.set()
        if count >= 5:
            five.set()

    runtime.on_evidence = evidence
    runtime.start()
    runtime.start()
    try:
        assert first.wait(1)
        with ThreadPoolExecutor() as pool:
            reader = pool.submit(lambda: (runtime.pose_snapshot(), release.wait(1)))
            assert five.wait(1)
            assert not reader.done()
            state = runtime.call({"protocol": PROTOCOL, "operation": "status"})
            assert state["protocol"] == PROTOCOL
            assert abs(runtime.status()["received_at"] - time.monotonic()) < 0.2
            release.set()
            reader.result()
        with pytest.raises(RuntimeError, match="Manual advance"):
            runtime.advance()
    finally:
        release.set()
        runtime.close()
    assert not runtime._thread.is_alive()
    assert runtime.error is None


def test_clock_reversal_nonfinite_and_invalid_timescale_rejected(manual):
    runtime, clock = manual
    move(runtime, clock, 0.015)
    assert runtime.native.time == pytest.approx(1000.01)
    with pytest.raises(ValueError, match="monotonic"):
        runtime.advance(999)
    with pytest.raises(ValueError, match="finite"):
        runtime.advance(float("nan"))
    with pytest.raises(ValueError, match="time_scale"):
        Runtime(Course(), time_scale=0)


def test_concurrent_status_requests_share_one_physics_tick(manual):
    runtime, clock = manual
    body = {"protocol": PROTOCOL, "operation": "status"}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(runtime.call, body, timeout=1) for _ in range(12)]
        with runtime._condition:
            assert runtime._condition.wait_for(lambda: len(runtime._pending) == 12, timeout=1)
        move(runtime, clock, DT)
        assert all(future.result()["protocol"] == PROTOCOL for future in futures)
    assert runtime.native.time == pytest.approx(1000 + DT)


def test_request_arriving_before_lock_contention_is_not_rejected_as_clock_reversal(manual):
    runtime, clock = manual
    arrived = threading.Event()
    runtime.clock = lambda: (arrived.set(), clock.now)[1]
    with ThreadPoolExecutor() as pool:
        with runtime._condition:
            future = pool.submit(runtime.call, motion(runtime), timeout=1)
            assert arrived.wait(1)
            move(runtime, clock, DT)
        with runtime._condition:
            assert runtime._condition.wait_for(lambda: runtime._pending, timeout=1)
        move(runtime, clock, DT)
        assert future.result()["status"] == "accepted"


def test_motion_hook_transforms_only_motion_and_preserves_envelope(manual):
    runtime, clock = manual
    events = []
    runtime.on_evidence = lambda kind, **data: events.append((kind, data))
    runtime.motion_transform = lambda motion, pose: dict(
        motion, direction="left", steering_percent=10
    )
    with ThreadPoolExecutor() as pool:
        body = motion(runtime)
        future = queued(runtime, pool, body)
        move(runtime, clock, 0.01)
        assert future.result()["status"] == "accepted"
        assert runtime.status()["command"]["target_angle_deg"] == 18
        assert body["motion"]["steering_percent"] == 0
        assert events[0][0] == "sim_motion_transform"
        assert events[0][1]["requested"]["steering_percent"] == 0
        runtime.motion_transform = lambda *_: None
        future = queued(runtime, pool, motion(runtime))
        move(runtime, clock, 0.01)
        assert future.result()["reason"] == "scenario_restriction"
        stop = request(runtime.native.bridge.controller, {"tool": "stop_now", "arguments": {}})
        future = queued(runtime, pool, stop)
        move(runtime, clock, 0.01)
        assert future.result()["status"] == "accepted"


def test_finish_request_marks_decision_sample_separate_from_braking_and_disabled_ticks(manual):
    runtime, clock = manual
    events = []
    runtime.on_evidence = lambda kind, **data: events.append((kind, data))
    with ThreadPoolExecutor() as pool:
        command = queued(runtime, pool, motion(runtime))
        move(runtime, clock, 0.01)
        assert command.result()["status"] == "accepted"
    move(runtime, clock, 2)
    decision = runtime.pose_snapshot()
    assert decision["enabled"] and decision["pending_end"] is None
    assert decision["end_reason"] is None
    clock.now += 0.003
    runtime.finish("client_done")
    runtime.finish("cannot_change_first_request")
    requests = [data for kind, data in events if kind == "sim_finish_requested"]
    assert len(requests) == 1
    requested = requests[0]
    assert requested["requested_reason"] == "client_done"
    assert requested["requested_at"] == clock.now
    assert requested["sim_time_s"] == decision["sim_time_s"]
    assert requested["monotonic_at"] == decision["monotonic_at"]
    assert requested["pose"] == decision["pose"]
    assert runtime.pose_snapshot() == decision
    move(runtime, clock, 5)
    cleanup = [
        data
        for kind, data in events
        if kind == "sim_tick" and data["sim_time_s"] > decision["sim_time_s"]
    ]
    braking = [data for data in cleanup if data["pending_end"] == "client_done"]
    disabled = [data for data in cleanup if not data["enabled"]]
    assert braking and disabled
    assert all(data["enabled"] and data["end_reason"] is None for data in braking)
    assert any(data["speed_mps"] > 0 for data in braking)
    assert all(
        data["pending_end"] is None and data["end_reason"] == "client_done" for data in disabled
    )
    assert all(data["speed_mps"] == 0 for data in disabled)
    assert disabled[0]["pose"][0] > decision["pose"][0]
