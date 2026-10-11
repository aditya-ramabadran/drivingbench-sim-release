import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from drivingbench.sim.producer import simulator_app
from drivingbench.sim.restrictions import Restrictions, runtime_for
from drivingbench.sim.runtime import Runtime
from drivingbench.sim.validate import request
from fastapi.testclient import TestClient
from test_sim_live import Renderer, scene_file


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
    args = dict(direction="left", steering_percent=50, speed_mps=1, duration_s=5)
    args.update(overrides)
    return request(
        runtime.native.bridge.controller,
        {"tool": tool, "arguments": args if tool == "set_motion" else {}},
    )


def send(runtime, clock, body):
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(runtime.call, body, timeout=1)
        with runtime._condition:
            assert runtime._condition.wait_for(lambda: runtime._pending, timeout=1)
        advance(runtime, clock, 0.01)
        return pending.result()


def advance(runtime, clock, duration):
    clock.now += duration
    runtime.advance()


@pytest.fixture
def straight():
    clock = Clock()
    runtime = Runtime(
        Course(), clock=clock, frame_clock=clock, restrictions=Restrictions(straight_only=True)
    )
    try:
        yield runtime, clock
    finally:
        runtime.close()


@pytest.mark.parametrize("direction", ["left", "right", "straight"])
def test_requested_steering_keeps_native_wheel_and_path_straight(straight, direction):
    runtime, clock = straight
    assert (
        send(runtime, clock, envelope(runtime, direction=direction, steering_percent=100))["status"]
        == "accepted"
    )
    assert runtime.status()["command"]["target_angle_deg"] == 0
    advance(runtime, clock, 4)
    pose = runtime.pose_snapshot()
    assert pose["speed_mps"] > 0.5 and pose["pose"][0] > 1
    assert runtime.native.car.steer_deg == pytest.approx(0, abs=1e-9)
    assert pose["pose"][1:] == pytest.approx((0, 0), abs=1e-9)


def test_requested_and_applied_receipt_evidence_preserves_original_arguments(straight):
    runtime, clock = straight
    evidence = []
    runtime.on_evidence = lambda kind, **data: evidence.append((kind, data))
    body = envelope(runtime, direction="right", steering_percent=63)
    body["motion"]["reason"] = "Take the bypass."
    saved = copy.deepcopy(body)
    reply = send(runtime, clock, body)
    transform = next(row for kind, row in evidence if kind == "sim_motion_transform")
    assert transform["requested"] == saved["motion"]
    assert transform["applied"] == dict(saved["motion"], direction="straight", steering_percent=0)
    assert transform["request_id"] == body["request_id"]
    assert body == saved
    assert reply["status"] == "accepted" and "reason" not in reply


@pytest.mark.parametrize("ending", ["stop", "expiry"])
def test_stop_and_expiry_brake_physically_without_freezing_pose(straight, ending):
    runtime, clock = straight
    assert send(runtime, clock, envelope(runtime))["status"] == "accepted"
    advance(runtime, clock, 4)
    before = runtime.pose_snapshot()
    assert before["speed_mps"] > 0.5
    if ending == "stop":
        assert send(runtime, clock, envelope(runtime, "stop_now"))["status"] == "accepted"
    else:
        advance(runtime, clock, 1.05)
    braking = runtime.pose_snapshot()
    assert runtime.status()["command"] is None
    assert braking["speed_mps"] > 0
    advance(runtime, clock, 5)
    after = runtime.pose_snapshot()
    assert after["speed_mps"] == 0
    assert after["pose"][0] > braking["pose"][0] >= before["pose"][0]
    assert runtime.native.car.standstill


@pytest.mark.parametrize(
    "change",
    [
        {"steering_percent": -1},
        {"steering_percent": 101},
        {"steering_percent": "unknown"},
        {"steering_percent": None},
        {"steering_percent": float("nan")},
        {"direction": "backwards"},
        {"speed_mps": -1},
        {"duration_s": 0},
    ],
)
def test_malformed_motion_reaches_native_rejection_without_sanitizing(straight, change):
    runtime, clock = straight
    assert (
        send(runtime, clock, envelope(runtime, direction="straight", steering_percent=0))["status"]
        == "accepted"
    )
    prior = copy.deepcopy(runtime.status()["command"])
    body = envelope(runtime, **change)
    saved = copy.deepcopy(body)
    assert send(runtime, clock, body)["reason"] == "malformed_request"
    assert runtime.error is None
    assert runtime.status()["command"] == prior
    assert body["motion"].keys() == saved["motion"].keys()
    for key in body["motion"]:
        if key != "steering_percent" or not isinstance(body["motion"][key], float):
            assert body["motion"][key] == saved["motion"][key]


def test_free_condition_preserves_requested_turn_and_native_response():
    clock = Clock()
    runtime = Runtime(Course(), clock=clock, frame_clock=clock, restrictions=Restrictions())
    try:
        body = envelope(runtime, direction="left", steering_percent=20)
        assert Restrictions().transform(copy.deepcopy(body["motion"]), {}) == body["motion"]
        assert send(runtime, clock, body)["status"] == "accepted"
        assert runtime.status()["command"]["target_angle_deg"] == pytest.approx(36)
        advance(runtime, clock, 4)
        assert runtime.native.car.steer_deg > 0
        assert runtime.pose_snapshot()["pose"][1] > 0
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"straight_only": "yes"},
        {"straight_only": True, "uniform_steering_percent": 20},
        {"straight_only": True, "branch_select": True, "brakes_failed_speed_mps": 5.5},
    ],
)
def test_conflicting_or_nonboolean_declaration_fails_before_run(settings):
    with pytest.raises(ValueError):
        Restrictions(**settings)


def test_producer_records_private_straight_only_declaration(tmp_path):
    declaration = {"time_scale": 1, "restrictions": {"straight_only": True}}
    app = simulator_app(
        tmp_path / "run",
        scene_file(tmp_path),
        renderer=Renderer(),
        declaration=declaration,
        runtime_factory=runtime_for(declaration),
    )
    with TestClient(app):
        saved = json.loads((tmp_path / "run/episode/metadata.json").read_text())["metadata"][
            "run_declaration"
        ]
        assert saved == declaration
        assert app.state.sim_native.restrictions.straight_only
        assert app.state.sim_native.restrictions.brakes_failed_speed_mps is None
        assert app.state.sim_native.restrictions.accepted_commands == (
            "observe",
            "set_motion",
            "stop_now",
        )
