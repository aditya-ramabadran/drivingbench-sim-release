import copy
import math
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from drivingbench.sim.car import DT, SimNative
from drivingbench.sim.validate import request
from drivingbench.sim.video_fork import LaneGuidance, VideoForkConfig, VideoForkRuntime


class Course:
    def map_pose(self, pose):
        return pose

    def assess(self, _pose):
        return {}


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def create(**kwargs):
    fields = dict(
        commit_m=14.32,
        endpoints_m={"left": 40, "right": 40},
        branch_rule="physical",
        visible_tip_m=14.32,
        lane_width_m=2.4,
    )
    fields.update(kwargs)
    config = VideoForkConfig(**fields)
    clock = Clock()
    runtime = VideoForkRuntime(
        Course(), config=config, clock=clock, frame_clock=lambda: clock.now + 30
    )
    return runtime, clock


def tick(runtime, clock):
    clock.now = runtime._wall_at_sim(runtime.native.time + DT)
    runtime.advance()


def motion(runtime, direction="left", percent=50, **kwargs):
    args = dict(
        direction=direction, steering_percent=percent, speed_mps=0.5, duration_s=5, **kwargs
    )
    return request(runtime.native.bridge.controller, {"tool": "set_motion", "arguments": args})


def submit(runtime, clock, body):
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(runtime.call, body)
        with runtime._condition:
            assert runtime._condition.wait_for(lambda: runtime._pending, timeout=1)
        tick(runtime, clock)
        return pending.result(timeout=1)


def drive(runtime, clock, progress):
    rows = []
    while runtime.video_snapshot()["progress_m"] < progress:
        before = runtime.native.pose
        tick(runtime, clock)
        after = runtime.native.pose
        assert math.hypot(after[0] - before[0], after[1] - before[1]) == pytest.approx(5.5 * DT)
        rows.append(runtime.pose_snapshot())
    return rows


@pytest.mark.parametrize("branch", ["left", "right"])
def test_default_keep_uses_real_longitudinal_progress_and_nominal_clock(branch):
    runtime, clock = create(default_branch=branch)
    try:
        assert runtime.branch_selection is None
        assert runtime.video_snapshot()["progress_m"] == 0
        drive(runtime, clock, 14.4)
        snapshot = runtime.video_snapshot()
        assert snapshot["locked_branch"] == branch
        assert snapshot["crossing"]["sim_time_s"] / runtime.time_scale == pytest.approx(60)
        assert snapshot["lateral_m"] == pytest.approx(runtime.video_guidance.config.start_y)
        assert not snapshot["body_tip_overlap"]
    finally:
        runtime.close()


@pytest.mark.parametrize("default,opposite", [("right", "left"), ("left", "right")])
def test_early_switch_has_actual_wheel_delay_curvature_and_physical_crossing(default, opposite):
    runtime, clock = create(default_branch=default)
    try:
        initial = runtime.video_snapshot()
        assert submit(runtime, clock, motion(runtime, opposite))["status"] == "accepted"
        immediate = runtime.video_snapshot()
        assert immediate["requested_branch"] == opposite
        assert immediate["actual_branch"] == default and immediate["locked_branch"] is None
        assert abs(immediate["lateral_m"] - initial["lateral_m"]) < 1e-6
        rows = drive(runtime, clock, 14.4)
        snapshot = runtime.video_snapshot()
        assert snapshot["locked_branch"] == opposite
        assert abs(snapshot["crossing"]["lateral_m"]) > 0
        assert abs(snapshot["heading_deg"]) > 1
        assert any(abs(r["video_fork"]["actual_steering_deg"]) > 10 for r in rows)
        assert all(abs(r["video_fork"]["guidance_angle_deg"]) <= 180 for r in rows)
        assert any(r["video_fork"]["body_tip_overlap"] for r in rows)
        assert clock.now - runtime.started_at > 60
    finally:
        runtime.close()


def test_late_request_and_late_reversal_resolve_actual_side_not_last_command():
    for early in (False, True):
        runtime, clock = create()
        try:
            if early:
                submit(runtime, clock, motion(runtime, "left"))
            drive(runtime, clock, 14.2)
            direction = "right" if early else "left"
            submit(runtime, clock, motion(runtime, direction))
            drive(runtime, clock, 14.4)
            snapshot = runtime.video_snapshot()
            assert snapshot["requested_branch"] == direction
            assert snapshot["locked_branch"] == ("left" if early else "right")
            locked = copy.deepcopy(snapshot["crossing"])
            submit(runtime, clock, motion(runtime, direction))
            drive(runtime, clock, 18)
            assert runtime.video_snapshot()["crossing"] == locked
        finally:
            runtime.close()


def test_straight_zero_stop_expiry_and_cached_receipt_do_not_restore_an_old_target():
    runtime, clock = create()
    try:
        left = motion(runtime, "left")
        submit(runtime, clock, left)
        submit(runtime, clock, motion(runtime, "right"))
        assert submit(runtime, clock, left)["status"] == "accepted"
        assert runtime.video_snapshot()["requested_branch"] == "right"
        submit(runtime, clock, motion(runtime, "left", percent=0))
        submit(runtime, clock, motion(runtime, "straight"))
        stop = request(runtime.native.bridge.controller, {"tool": "stop_now", "arguments": {}})
        assert submit(runtime, clock, stop)["status"] == "accepted"
        assert runtime.native.bridge.controller.command is None
        assert runtime.video_snapshot()["requested_branch"] == "right"
        assert runtime.native.car.speed == 5.5
        submit(runtime, clock, motion(runtime, "left"))
        drive(runtime, clock, 35)
        assert runtime.native.bridge.controller.command is None
        assert runtime.video_snapshot()["requested_branch"] == "left"
        assert runtime.native.car.speed == 5.5
    finally:
        runtime.close()


def test_rejected_request_does_not_choose_lane_and_pause_preserves_frame_time():
    runtime, clock = create()
    try:
        body = motion(runtime, "left")
        body["motion"]["steering_percent"] = 200
        assert submit(runtime, clock, body)["status"] == "rejected"
        assert runtime.video_snapshot()["requested_branch"] == "right"
        runtime.pause()
        old = runtime.pose_snapshot()
        clock.now += 10
        runtime.advance()
        assert runtime.pose_snapshot() == old
        assert runtime.status()["received_at"] == clock.now
        runtime.resume()
        tick(runtime, clock)
        assert runtime.pose_snapshot()["captured_at"] == pytest.approx(clock.now + 30)
        runtime.terminate("assigned_branch_outcome")
        stopped = runtime.pose_snapshot()
        clock.now += 10
        runtime.advance()
        assert runtime.pose_snapshot() == stopped
        assert runtime.native.car.speed == 0
    finally:
        runtime.close()


def test_tip_interpolation_and_centerline_tie_are_explicit():
    guidance = LaneGuidance(
        VideoForkConfig(
            commit_m=10,
            endpoints_m={"left": 15, "right": 15},
            branch_rule="physical",
            visible_tip_m=10,
            lane_width_m=2.4,
        )
    )
    native = SimpleNamespace(
        car=SimpleNamespace(x=0, y=1.2, heading=0, steer_deg=0, speed=5.5),
        bridge=SimpleNamespace(controller=SimpleNamespace(command=None)),
    )
    guidance.sample(native, 0)
    native.car.x = 9
    guidance.sample(native, 1)
    native.car.x = 11
    result = guidance.sample(native, 2)
    assert result["crossing"] == dict(
        progress_m=10, lateral_m=0, sim_time_s=1.5, heading_deg=0, branch="right"
    )
    assert result["body_tip_overlap"]
    native.car.x = 15
    result = guidance.sample(native, 3)
    assert result["endpoint_reached"] and result["phase"] == "assigned_outcome"


def test_steering_hook_cannot_override_disabled_or_failed_native_and_rejects_nonfinite():
    calls = []
    native = SimNative(speed=5.5, steering_target=lambda n, a: calls.append((n.time, a)) or 900)
    assert calls and native.bridge.selection.angle_deg == 180
    native.enabled = False
    count = len(calls)
    native.tick()
    assert len(calls) == count
    native.enabled = True
    native.bridge.failure = "fixture_failure"
    native.tick()
    assert len(calls) == count
    with pytest.raises(ValueError, match="finite"):
        SimNative(speed=5.5, steering_target=lambda *_: float("nan"))


@pytest.mark.parametrize(
    "options",
    [
        dict(commit_m=0),
        dict(endpoints_m={"left": 1, "right": 20}),
        dict(lane_width_m=1.5),
        dict(speed_mps=7),
        dict(default_branch="middle"),
    ],
)
def test_video_config_rejects_invalid_geometry_and_clock(options):
    fields = dict(commit_m=10, endpoints_m={"left": 20, "right": 20}, branch_rule="physical")
    fields.update(options)
    with pytest.raises(ValueError):
        VideoForkConfig(**fields)
    with pytest.raises(ValueError, match="clock"):
        VideoForkRuntime(
            Course(),
            config=VideoForkConfig(
                commit_m=10, endpoints_m={"left": 20, "right": 20}, branch_rule="physical"
            ),
            time_scale=1,
        )


@pytest.mark.parametrize("second", ["straight", "right", "stop"])
@pytest.mark.parametrize("paused", [False, True])
def test_same_tick_accepted_requests_preserve_order_and_only_first_receipts(second, paused):
    runtime, clock = create()
    try:
        if paused:
            runtime.pause()
        first = motion(runtime, "left")
        last = (
            request(runtime.native.bridge.controller, {"tool": "stop_now", "arguments": {}})
            if second == "stop"
            else motion(runtime, second)
        )
        with ThreadPoolExecutor() as pool:
            replies = []
            for body in (first, last, first):
                replies.append(pool.submit(runtime.call, body))
                with runtime._condition:
                    assert runtime._condition.wait_for(
                        lambda: len(runtime._pending) == len(replies), timeout=1
                    )
            tick(runtime, clock)
            assert all(reply.result(timeout=1)["status"] == "accepted" for reply in replies)
        assert runtime.video_snapshot()["requested_branch"] == (
            "right" if second == "right" else "left"
        )
    finally:
        runtime.close()


def test_calibrated_initial_offset_is_separate_from_authored_lane_targets():
    runtime, clock = create(initial_lateral_m=-1.16)
    try:
        assert runtime.video_snapshot()["lateral_m"] == -1.16
        assert runtime.video_guidance.config.lane_width_m == 2.4
        assert runtime.native.car.y == 0
        drive(runtime, clock, 4)
        assert runtime.video_snapshot()["locked_branch"] is None
    finally:
        runtime.close()


@pytest.mark.parametrize("switch_progress", [0, 6.3])
def test_command_primary_holds_default_until_commit_then_steers_to_selected_lane(switch_progress):
    runtime, clock = create(
        branch_rule="command",
        commit_m=6.446607,
        endpoints_m={"right": 14.950026, "left": 13.414139},
    )
    try:
        drive(runtime, clock, switch_progress)
        submit(runtime, clock, motion(runtime, "left"))
        before = runtime.video_snapshot()
        assert before["requested_branch"] == "left" and before["locked_branch"] is None
        assert before["lateral_m"] == pytest.approx(-1.2)
        drive(runtime, clock, 6.5)
        committed = runtime.video_snapshot()
        assert committed["locked_branch"] == "left"
        assert committed["actual_branch"] == "right"
        assert committed["crossing"]["sim_time_s"] / runtime.time_scale == pytest.approx(60)
        assert committed["endpoint_m"] == 13.414139
        submit(runtime, clock, motion(runtime, "right"))
        assert runtime.video_snapshot()["requested_branch"] == "right"
        assert runtime.video_snapshot()["locked_branch"] == "left"
        drive(runtime, clock, 13.414139)
        final = runtime.video_snapshot()
        assert final["endpoint_reached"] and final["phase"] == "assigned_outcome"
        assert 0 <= final["endpoint_overshoot_m"] <= 5.5 * DT
        assert final["endpoint_step_m"] == pytest.approx(5.5 * DT)
        runtime.terminate("assigned_branch_outcome")
        stopped = runtime.pose_snapshot()
        assert stopped["speed_mps"] == 0
        assert stopped["video_fork"]["endpoint_step_m"] == final["endpoint_step_m"]
        assert final["guidance_angle_deg"] > 0
        assert abs(final["actual_steering_deg"]) > 1
    finally:
        runtime.close()


@pytest.mark.parametrize("at_commit", [False, True])
def test_command_commit_boundary_accepts_current_tick_but_not_later_requests(at_commit):
    runtime, clock = create(branch_rule="command", commit_m=5.5 * DT * 10)
    try:
        for _ in range(9 if at_commit else 11):
            tick(runtime, clock)
        result = submit(runtime, clock, motion(runtime, "left"))
        assert result["status"] == "accepted"
        drive(runtime, clock, 1)
        snapshot = runtime.video_snapshot()
        assert snapshot["requested_branch"] == "left"
        assert snapshot["locked_branch"] == ("left" if at_commit else "right")
    finally:
        runtime.close()


def test_command_primary_uses_default_endpoint_when_no_choice_and_zero_do_not_choose():
    runtime, clock = create(branch_rule="command", commit_m=1, endpoints_m={"left": 2, "right": 3})
    try:
        submit(runtime, clock, motion(runtime, "left", percent=0))
        submit(runtime, clock, motion(runtime, "straight"))
        submit(
            runtime,
            clock,
            request(runtime.native.bridge.controller, {"tool": "stop_now", "arguments": {}}),
        )
        drive(runtime, clock, 2.5)
        snapshot = runtime.video_snapshot()
        assert snapshot["locked_branch"] == "right" and snapshot["endpoint_m"] == 3
        assert not snapshot["endpoint_reached"]
        drive(runtime, clock, 3)
        assert runtime.video_snapshot()["endpoint_reached"]
    finally:
        runtime.close()
