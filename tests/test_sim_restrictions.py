from concurrent.futures import ThreadPoolExecutor

import pytest
from drivingbench.shared.contracts import PROTOCOL
from drivingbench.sim.restrictions import Restrictions, runtime_for
from drivingbench.sim.runtime import Runtime
from drivingbench.sim.validate import request


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


class Course:
    def map_pose(self, pose):
        return pose

    def assess(self, _pose):
        return {}


def envelope(runtime, tool="set_motion", **overrides):
    args = dict(direction="left", steering_percent=3, speed_mps=1, duration_s=5)
    args.update(overrides)
    return request(
        runtime.native.bridge.controller,
        {"tool": tool, "arguments": args if tool == "set_motion" else {}},
    )


def queue(runtime, pool, body):
    future = pool.submit(runtime.call, body, timeout=1)
    with runtime._condition:
        assert runtime._condition.wait_for(lambda: runtime._pending, timeout=1)
    return future


def advance(runtime, clock, seconds):
    clock.now += seconds
    runtime.advance()


@pytest.fixture
def scaled():
    clock = Clock()
    runtime = Runtime(Course(), clock=clock, frame_clock=lambda: clock.now + 30, time_scale=1 / 6)
    yield runtime, clock
    runtime.close()


def test_scaled_expiry_uses_simulated_seconds_and_frames_use_wall_seconds(scaled):
    runtime, clock = scaled
    with ThreadPoolExecutor() as pool:
        future = queue(runtime, pool, envelope(runtime))
        advance(runtime, clock, 0.06)
        assert future.result()["status"] == "accepted"
    advance(runtime, clock, 3)
    assert runtime.status()["state"] == "executing"
    assert runtime.pose_snapshot()["sim_time_s"] == pytest.approx(0.51)
    assert runtime.pose_snapshot()["captured_at"] == pytest.approx(1033.06)
    advance(runtime, clock, 27.12)
    assert runtime.status()["command"] is None
    assert runtime.status()["received_at"] == pytest.approx(1030.18)
    old_frame = runtime.pose_snapshot()
    clock.now += 10
    assert runtime.pose_snapshot() == old_frame


def test_pause_stop_status_and_resume_preserve_capture_age_and_control_time(scaled):
    runtime, clock = scaled
    with ThreadPoolExecutor() as pool:
        future = queue(runtime, pool, envelope(runtime, duration_s=5))
        advance(runtime, clock, 0.06)
        assert future.result()["status"] == "accepted"
        runtime.pause()
        before = runtime.pose_snapshot()
        command = runtime.status()["command"].copy()
        advance(runtime, clock, 60)
        assert runtime.status()["command"] == command
        assert runtime.pose_snapshot() == before
        assert runtime.status()["received_at"] == pytest.approx(clock.now)
        future = queue(runtime, pool, {"protocol": PROTOCOL, "operation": "status"})
        runtime.advance()
        assert future.result()["protocol"] == PROTOCOL
        future = queue(runtime, pool, envelope(runtime, "stop_now"))
        runtime.advance()
        assert future.result()["status"] == "accepted"
        assert runtime.status()["command"] is None
        assert runtime.pose_snapshot() == before
    runtime.resume()
    advance(runtime, clock, 0.06)
    assert runtime.pose_snapshot()["sim_time_s"] == pytest.approx(0.02)
    assert runtime.pose_snapshot()["captured_at"] == pytest.approx(clock.now + 30)


def test_motion_accepted_while_paused_begins_on_resume(scaled):
    runtime, clock = scaled
    runtime.pause()
    with ThreadPoolExecutor() as pool:
        future = queue(runtime, pool, envelope(runtime))
        advance(runtime, clock, 30)
        assert future.result()["status"] == "accepted"
    assert runtime.status()["command"]["expires_at_s"] == pytest.approx(1005)
    assert runtime.native.time == pytest.approx(1000)
    runtime.resume()
    advance(runtime, clock, 3)
    assert runtime.status()["command"] is not None
    advance(runtime, clock, 27.2)
    assert runtime.status()["command"] is None


def test_failed_brakes_ignore_stop_expiry_and_operator_brake_but_keep_steering():
    clock = Clock()
    runtime = Runtime(
        Course(),
        clock=clock,
        restrictions=Restrictions(brakes_failed_speed_mps=5.5, uniform_steering_percent=12),
    )
    try:
        with ThreadPoolExecutor() as pool:
            future = queue(runtime, pool, envelope(runtime))
            advance(runtime, clock, 0.01)
            assert future.result()["status"] == "accepted"
            assert runtime.status()["command"]["target_angle_deg"] == pytest.approx(21.6)
            advance(runtime, clock, 0.5)
            assert abs(runtime.native.car.steer_deg) > 0
            future = queue(runtime, pool, envelope(runtime, "stop_now"))
            advance(runtime, clock, 0.01)
            assert future.result()["status"] == "accepted"
        advance(runtime, clock, 3)
        assert runtime.native.car.speed == 5.5
        runtime.operator_brake()
        before = runtime.pose_snapshot()["pose"]
        advance(runtime, clock, 1)
        assert runtime.native.car.speed == 5.5
        assert runtime.pose_snapshot()["pose"] != before
        events = []
        runtime.on_evidence = lambda kind, **data: events.append((kind, data))
        runtime.terminate("divider_tip")
        impact_pose = runtime.pose_snapshot()
        advance(runtime, clock, 30)
        assert runtime.pose_snapshot() == impact_pose
        assert runtime.native.car.speed == 0
        assert runtime.end_reason == "divider_tip"
        assert events[0][0] == "sim_terminal_impact"
        assert events[0][1]["speed_mps"] == 5.5
        assert not runtime.status()["enabled"]
        with pytest.raises(RuntimeError, match="cannot resume"):
            runtime.resume()
    finally:
        runtime.close()


def test_latency_does_not_block_physics_and_rejected_commands_leave_controller_unchanged():
    clock = Clock()
    runtime = Runtime(
        Course(),
        clock=clock,
        restrictions=Restrictions(
            brakes_failed_speed_mps=5.5,
            accepted_commands=("observe", "stop_now"),
            command_latency_s=0.5,
        ),
    )
    try:
        start = runtime.native.car.x
        with ThreadPoolExecutor() as pool:
            future = queue(runtime, pool, envelope(runtime))
            advance(runtime, clock, 0.4)
            assert not future.done()
            assert runtime.native.car.x - start == pytest.approx(2.2)
            advance(runtime, clock, 0.1)
            assert future.result()["reason"] == "scenario_restriction"
            assert runtime.status()["command"] is None
    finally:
        runtime.close()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_time_scales(value):
    with pytest.raises(ValueError, match="time_scale"):
        Runtime(Course(), time_scale=value)


@pytest.mark.parametrize(
    "declaration",
    [
        {"brakes_failed_speed_mps": 6.1},
        {"uniform_steering_percent": 101},
        {"command_latency_s": -1},
        {"accepted_commands": ["drive"]},
        {"accepted_commands": ["observe", "observe"]},
    ],
)
def test_invalid_restrictions(declaration):
    with pytest.raises(ValueError):
        Restrictions(**declaration)


def test_declared_factory_and_fixed_steering_preserve_straight_and_direction():
    declaration = {"time_scale": 1 / 6, "restrictions": {"uniform_steering_percent": 17}}
    runtime = runtime_for(declaration)(Course(), time_scale=declaration["time_scale"])
    try:
        assert runtime.time_scale == 1 / 6
        for direction, value, expected in (
            ("left", 2, 17),
            ("right", 90, 17),
            ("straight", 0, 0),
            ("left", 0, 0),
        ):
            transformed = runtime.restrictions.transform(
                dict(direction=direction, steering_percent=value), {}
            )
            assert transformed == dict(direction=direction, steering_percent=expected)
    finally:
        runtime.close()


def test_live_paused_worker_services_stop_and_watchdog_without_refreshing_frame():
    import time

    runtime = Runtime(Course(), paused=True)
    runtime.start()
    try:
        before = runtime.pose_snapshot()
        reply = runtime.call(envelope(runtime))
        assert reply["status"] == "accepted"
        reply = runtime.call(envelope(runtime, "stop_now"))
        assert reply["status"] == "accepted"
        runtime.call({"protocol": PROTOCOL, "operation": "status"})
        assert time.monotonic() - runtime.status()["received_at"] < 0.2
        assert runtime.pose_snapshot() == before
        assert runtime.status()["command"] is None
    finally:
        runtime.close()


def test_failed_brakes_command_expires_but_speed_does_not():
    clock = Clock()
    runtime = Runtime(Course(), clock=clock, restrictions=Restrictions(brakes_failed_speed_mps=5.5))
    try:
        with ThreadPoolExecutor() as pool:
            future = queue(runtime, pool, envelope(runtime))
            advance(runtime, clock, 0.01)
            assert future.result()["status"] == "accepted"
        advance(runtime, clock, 6)
        assert runtime.status()["command"] is None
        assert runtime.native.car.speed == 5.5
        assert runtime.native.car.x > 20
    finally:
        runtime.close()


def test_latency_timeout_removes_motion_and_status_bypasses_latency():
    clock = Clock()
    runtime = Runtime(Course(), clock=clock, restrictions=Restrictions(command_latency_s=10))
    try:
        with pytest.raises(TimeoutError):
            runtime.call(envelope(runtime), timeout=0.001)
        with ThreadPoolExecutor() as pool:
            future = queue(runtime, pool, {"protocol": PROTOCOL, "operation": "status"})
            advance(runtime, clock, 0.01)
            assert future.result()["protocol"] == PROTOCOL
        advance(runtime, clock, 11)
        assert runtime.status()["command"] is None
    finally:
        runtime.close()


def test_pause_from_tick_callback_does_not_deadlock_or_catch_up_on_resume():
    clock = Clock()
    runtime = Runtime(Course(), clock=clock)
    try:
        runtime.on_evidence = lambda kind, **_data: runtime.pause() if kind == "sim_tick" else None
        advance(runtime, clock, 1)
        assert runtime.pose_snapshot()["sim_time_s"] == pytest.approx(0.01)
        runtime.on_evidence = None
        runtime.resume()
        advance(runtime, clock, 0.01)
        assert runtime.pose_snapshot()["sim_time_s"] == pytest.approx(0.02)
    finally:
        runtime.close()


def test_terminal_contact_serializes_with_an_inflight_due_tick():
    import threading

    clock = Clock()
    runtime = Runtime(Course(), clock=clock, restrictions=Restrictions(brakes_failed_speed_mps=5.5))
    entered, release, terminating = threading.Event(), threading.Event(), threading.Event()
    tick = runtime._tick
    events = []
    runtime.on_evidence = lambda kind, **data: events.append(kind)

    def delayed_tick():
        entered.set()
        assert release.wait(1)
        tick()

    def terminate():
        terminating.set()
        runtime.terminate("divider_tip")

    runtime._tick = delayed_tick
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            clock.now += 0.01
            ticking = pool.submit(runtime.advance)
            assert entered.wait(1)
            ending = pool.submit(terminate)
            assert terminating.wait(1)
            assert not ending.done()
            release.set()
            ticking.result()
            ending.result()
        frozen = runtime.pose_snapshot()
        advance(runtime, clock, 1)
        assert runtime.pose_snapshot() == frozen
        assert events == ["sim_tick", "sim_terminal_impact", "sim_terminal_state"]
    finally:
        release.set()
        runtime.close()


def test_very_slow_clock_keeps_service_healthy_but_frame_and_physics_old():
    clock = Clock()
    runtime = Runtime(Course(), clock=clock, frame_clock=lambda: clock.now + 30, time_scale=0.001)
    try:
        initial = runtime.pose_snapshot()
        advance(runtime, clock, 3)
        assert runtime.status()["received_at"] == pytest.approx(clock.now)
        assert runtime.pose_snapshot() == initial
        assert runtime.native.time == pytest.approx(1000)
        with ThreadPoolExecutor() as pool:
            future = queue(runtime, pool, {"protocol": PROTOCOL, "operation": "status"})
            runtime.advance()
            assert future.result()["protocol"] == PROTOCOL
        clock.now += 3
        assert runtime.status()["received_at"] == pytest.approx(1003)
        assert runtime.pose_snapshot() == initial
        advance(runtime, clock, 4.01)
        assert runtime.pose_snapshot()["sim_time_s"] == pytest.approx(0.01)
        assert runtime.pose_snapshot()["captured_at"] == pytest.approx(1040)
    finally:
        runtime.close()


def test_due_tick_block_never_receives_idle_heartbeat():
    import threading

    clock = Clock()
    runtime = Runtime(Course(), clock=clock, time_scale=0.001)
    entered, release = threading.Event(), threading.Event()
    tick = runtime.native.tick

    def blocked_tick():
        entered.set()
        assert release.wait(1)
        tick()

    runtime.native.tick = blocked_tick
    try:
        clock.now += 10
        with ThreadPoolExecutor() as pool:
            future = pool.submit(runtime.advance)
            assert entered.wait(1)
            # Tick owns the condition, so inspect the captured snapshot directly.
            assert runtime._snapshot["received_at"] == pytest.approx(1000)
            release.set()
            future.result()
    finally:
        release.set()
        runtime.close()
