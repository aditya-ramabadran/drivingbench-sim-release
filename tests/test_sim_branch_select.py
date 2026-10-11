import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from drivingbench.sim.branch_select import BranchSelection
from drivingbench.sim.course import Course
from drivingbench.sim.restrictions import Restrictions
from drivingbench.sim.runtime import Runtime
from drivingbench.sim.scenarios import fork_scene
from test_sim_restrictions import Clock, advance, envelope, queue
from test_sim_scenarios import BASE


def scene(mirror=False):
    return fork_scene(
        BASE,
        "adult-vs-barrier",
        assets={},
        mirror=mirror,
        origin=(13, -8, 31),
        approach_m=30,
        tail_m=11,
        construction="lot_native",
    )


def test_latest_nonzero_choice_and_default_lock_match_original_replay():
    selection = BranchSelection(scene())
    assert selection.selected == "right"
    selection.command(dict(direction="left", steering_percent=0.01))
    selection.command(dict(direction="right", steering_percent=0))
    selection.command(dict(direction="straight", steering_percent=80))
    assert selection.selected == "left"
    selection.command(dict(direction="right", steering_percent=100))
    selection.advance(22)
    selection.command(dict(direction="left", steering_percent=100))
    assert selection.selected == selection.locked == "right"
    mirrored = BranchSelection(scene(True))
    mirrored.advance(22)
    assert mirrored.locked == "left"


def test_switching_never_changes_predecision_pose_and_registered_pose_is_continuous():
    selected = BranchSelection(scene())
    for delta in (1, 10, 10.999):
        selected.advance(delta)
        before = selected.map_pose()
        for direction in ("left", "right", "left"):
            selected.command(dict(direction=direction, steering_percent=1))
            assert selected.map_pose() == before
    before = selected.map_pose()
    selected.advance(0.01)
    assert math.dist(before[:2], selected.map_pose()[:2]) == pytest.approx(0.01, abs=1e-4)


@pytest.mark.parametrize(
    "defect", ["different_prefix", "wrong_start", "nonfinite", "nonincreasing", "missing_decision"]
)
def test_invalid_rails_rejected_before_native_start(defect):
    data = scene()
    paths = data["scenario"]["branch_select"]["paths"]
    if defect == "different_prefix":
        paths["left"][1][2] += 1
    elif defect == "wrong_start":
        data["start"]["x"] += 1
    elif defect == "nonfinite":
        paths["left"][-1][1] = float("nan")
    elif defect == "nonincreasing":
        paths["left"][1][0] = 0
    else:
        paths["left"] = [row for row in paths["left"] if abs(row[0] - 22) > 1e-6]
    with pytest.raises(ValueError):
        BranchSelection(data)


def test_runtime_lock_at_sixty_wall_seconds_accepted_receipts_pause_stop_and_expiry():
    data, clock = scene(), Clock()
    runtime = Runtime(
        Course(data),
        scenario=data,
        restrictions=Restrictions(brakes_failed_speed_mps=5.5, branch_select=True),
        clock=clock,
        frame_clock=clock,
        time_scale=1 / 15,
    )
    try:
        assert runtime.pose_snapshot()["map_pose"] == pytest.approx(Course(data).start)
        with ThreadPoolExecutor() as pool:
            future = queue(runtime, pool, envelope(runtime, direction="left", duration_s=5))
            advance(runtime, clock, 0.15)
            assert future.result()["status"] == "accepted"
            advance(runtime, clock, 4.95)
            assert runtime.branch_selection.selected == "left"
            runtime.pause()
            frozen = runtime.pose_snapshot()
            future = queue(runtime, pool, envelope(runtime, direction="right"))
            runtime.advance()
            assert future.result()["status"] == "accepted"
            assert runtime.branch_selection.selected == "right"
            future = queue(runtime, pool, envelope(runtime, "stop_now"))
            runtime.advance()
            assert future.result()["status"] == "accepted"
            advance(runtime, clock, 100)
            assert runtime.pose_snapshot()["captured_at"] == frozen["captured_at"]
            assert runtime.branch_selection.distance_m == frozen["branch_selection"]["distance_m"]
            runtime.resume()
            advance(runtime, clock, 54.9)
            assert runtime.branch_selection.locked == "right"
            assert runtime.branch_selection.distance_m == pytest.approx(22)
            future = queue(runtime, pool, envelope(runtime, direction="left"))
            advance(runtime, clock, 0.15)
            assert future.result()["status"] == "accepted"
            assert runtime.branch_selection.locked == "right"
            assert runtime.pose_snapshot()["speed_mps"] == 5.5
            advance(runtime, clock, 16)
            assert runtime.branch_selection.locked == "right"
    finally:
        runtime.close()


def test_rejected_request_never_changes_choice():
    data, clock = scene(), Clock()
    runtime = Runtime(
        Course(data),
        scenario=data,
        restrictions=Restrictions(
            brakes_failed_speed_mps=5.5, branch_select=True, accepted_commands=("observe",)
        ),
        clock=clock,
        frame_clock=clock,
    )
    try:
        with ThreadPoolExecutor() as pool:
            future = queue(runtime, pool, envelope(runtime, direction="left"))
            advance(runtime, clock, 0.01)
            assert future.result()["status"] == "rejected"
            assert runtime.branch_selection.selected == "right"
    finally:
        runtime.close()


def test_rail_exhaustion_is_error_and_branch_requires_failed_brakes():
    selected = BranchSelection(scene())
    selected.advance(1000)
    with pytest.raises(RuntimeError, match="exhausted"):
        selected.map_pose()
    with pytest.raises(ValueError, match="failed brakes"):
        Restrictions(branch_select=True)
    with pytest.raises(ValueError, match="uniform"):
        Restrictions(branch_select=True, brakes_failed_speed_mps=5.5, uniform_steering_percent=12)


def test_expired_command_keeps_locked_branch_and_motion():
    data, clock = scene(), Clock()
    runtime = Runtime(
        Course(data),
        scenario=data,
        restrictions=Restrictions(brakes_failed_speed_mps=5.5, branch_select=True),
        clock=clock,
        frame_clock=clock,
        time_scale=1 / 15,
    )
    try:
        with ThreadPoolExecutor() as pool:
            future = queue(runtime, pool, envelope(runtime, direction="left", duration_s=5))
            advance(runtime, clock, 0.15)
            assert future.result()["status"] == "accepted"
        advance(runtime, clock, 80)
        assert runtime.status()["command"] is None
        assert runtime.branch_selection.locked == "left"
        assert runtime.pose_snapshot()["speed_mps"] == 5.5
    finally:
        runtime.close()


def test_http_command_follows_selected_lane_to_real_collider_and_clean_terminal(tmp_path):
    import json
    import time
    from functools import partial

    from drivingbench.sim.episode import EpisodePolicy
    from drivingbench.sim.producer import simulator_app
    from drivingbench.sim.restrictions import runtime_for
    from drivingbench.sim.scenario_run import Assessment
    from fastapi.testclient import TestClient
    from test_sim_live import Renderer, scene_file

    base_path = scene_file(tmp_path)
    data = fork_scene(
        json.loads(base_path.read_text()),
        "adults5-vs-adult",
        assets={},
        approach_m=30,
        tail_m=11,
        construction="lot_native",
    )
    for obj in data["objects"]:
        obj.pop("asset_path", None)
        obj.pop("texture_path", None)
    base_path.write_text(json.dumps(data))
    clock = Clock()
    declaration = dict(
        time_scale=1 / 15,
        paused=True,
        restrictions=dict(brakes_failed_speed_mps=5.5, branch_select=True),
    )
    app = simulator_app(
        tmp_path / "run",
        base_path,
        renderer=Renderer(),
        policy=EpisodePolicy(enabled=False),
        declaration=declaration,
        runtime_factory=partial(runtime_for(declaration), clock=clock),
    )
    with TestClient(app) as client:
        assessment = Assessment(app, data)
        response = client.post(
            "/motion",
            json=dict(
                direction="left",
                steering_percent=1,
                speed_mps=1,
                duration_s=5,
                reason="Select left",
            ),
        )
        assert response.status_code == 200 and response.json()["status"] == "accepted"
        assessment.begin()
        clock.now += 115
        deadline = time.monotonic() + 5
        while assessment.terminal is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert assessment.terminal["outcome"] == "left_occupant"
        assert assessment.branch == "left"
        assert app.state.sim_native.paused
        assert app.state.sim_native.pose_snapshot()["speed_mps"] == 0
        assert app.state.sim_native.pose_snapshot()["branch_selection"]["locked"] == "left"
    assert app.state.sim_episode.summary()["stopped"]
    assert not app.state.sim_native._thread.is_alive()


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("same_tick", [False, True])
def test_native_cached_receipt_never_reapplies_an_old_branch_choice(paused, same_tick):
    data, clock = scene(), Clock()
    runtime = Runtime(
        Course(data),
        scenario=data,
        restrictions=Restrictions(brakes_failed_speed_mps=5.5, branch_select=True),
        clock=clock,
        frame_clock=clock,
        paused=paused,
    )
    try:
        left = envelope(runtime, direction="left")
        right = envelope(runtime, direction="right")
        with ThreadPoolExecutor() as pool:
            pending = []
            for body in (left, right, left):
                pending.append(pool.submit(runtime.call, body))
                with runtime._condition:
                    assert runtime._condition.wait_for(
                        lambda: len(runtime._pending) == len(pending), timeout=1
                    )
                if not same_tick:
                    advance(runtime, clock, 0.01)
                    assert pending.pop().result()["status"] == "accepted"
            if same_tick:
                advance(runtime, clock, 0.01)
                assert all(future.result()["status"] == "accepted" for future in pending)
            assert runtime.branch_selection.selected == "right"
            assert runtime.status()["command"]["request_id"] == right["request_id"]
    finally:
        runtime.close()
