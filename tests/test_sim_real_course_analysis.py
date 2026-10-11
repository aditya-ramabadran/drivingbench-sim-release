import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from drivingbench.sim import real_course_analysis as analysis
from drivingbench.sim.car import COROLLA
from drivingbench.sim.event_study_v2 import source_files
from drivingbench.sim.logs import car_params
from drivingbench.sim.validate import Sim, simulate


def columns(size=1001):
    result = {"t": np.arange(size) / 100}
    for key in (
        "v",
        "angle",
        "a",
        "cruise_standstill",
        "accel_cmd",
        "curvature_cmd",
        "torque_out",
        "long_state",
    ):
        result[key] = np.zeros(size)
    result["steer_ratio"] = np.full(size, 14.8)
    result["stiffness"] = np.ones(size)
    return result


def command(t, tool="set_motion", status="accepted", segment="trial", speed=1.0):
    return {
        "t": t,
        "tool": tool,
        "status": status,
        "trace_segment": segment,
        "arguments": {
            "direction": "left",
            "steering_percent": 30.0,
            "speed_mps": speed,
            "duration_s": 3.0,
        }
        if tool == "set_motion"
        else {},
    }


class ClockModel:
    def __init__(self, cp, c, k):
        self.native = NS(
            time=c["t"][k] - 0.01,
            car=NS(
                x=99.0,
                y=99.0,
                heading=99.0,
                speed=c["v"][k],
                steer_deg=c["angle"][k],
                accel=c["a"][k],
            ),
        )
        self.controller = NS(boot_id="offline", command_epoch=0)

    def call(self, body):
        self.native.time += 0.01
        self.native.car.speed = body["motion"]["speed_mps"] if body["motion"] else 0.0
        return {"status": "accepted"}

    def advance(self, until):
        car = self.native.car
        car.x += car.speed * max(0.0, until - self.native.time)
        self.native.time = max(until, self.native.time)


def replay(calls, *, c=None, start=2.0, end=2.2):
    return analysis.replay_attempt(
        columns() if c is None else c,
        None,
        calls,
        {"segment": "trial", "start_t": start, "end_t": end},
        model_factory=ClockModel,
    )


def test_successful_stops_and_only_accepted_motion_are_retained_in_order():
    calls = [
        command(2.06, "stop_now", "stopped"),
        command(2.04, "stop_now", "stopping"),
        command(2.05, "stop_now"),
        command(2.0),
        command(2.02, status="rejected"),
        command(2.03, "observe"),
        command(2.08, segment="other"),
        command(2.1),
        command(1.98),
    ]
    kept = analysis.accepted_commands(calls, "trial", 2.0, 2.1)
    assert [call["t"] for call in kept] == [2.0, 2.04, 2.05, 2.06]


def test_row_scheduling_excludes_intervention_and_rejected_calls():
    arrays, summary = replay(
        [
            command(1.999),
            command(2.045, "stop_now", "stopping"),
            command(2.08, status="rejected"),
            command(2.2),
        ]
    )
    receipts = summary["continuous_receipts"]
    assert summary["accepted_calls"] == summary["accepted_replays"] == 2
    assert receipts[0]["receipt_t"] == pytest.approx(2.0)
    assert receipts[1]["receipt_t"] == pytest.approx(2.05)
    assert summary["max_schedule_lateness_s"] == pytest.approx(0.005)
    assert arrays["continuous"][-1, 0] == pytest.approx(2.19)
    assert arrays["continuous"][-1, 4] == 0
    assert np.max(arrays["continuous"][:, 1]) < 0.1


def test_command_windows_reset_pose_and_logged_physical_state():
    c = columns()
    c["angle"][210], c["a"][210], c["v"][210] = 7.0, 0.3, 0.8
    arrays, summary = replay([command(2.0), command(2.1, speed=2)], c=c)
    assert [(w["start_row"], w["end_row"]) for w in summary["windows"]] == [(0, 10), (10, 20)]
    assert summary["windows"][1]["initial_state"]["wheel_deg"] == 7
    assert summary["windows"][1]["initial_state"]["speed"] == 0.8
    assert arrays["reanchored_local"][10, 1] == pytest.approx(0)
    assert arrays["continuous"][10, 1] > 0
    np.testing.assert_array_equal(arrays["continuous"][:, 0], arrays["reanchored_local"][:, 0])


def test_same_row_calls_all_execute_but_have_no_independent_zero_duration_window():
    arrays, summary = replay([command(2.0), command(2.045), command(2.047, "stop_now")])
    assert len(summary["continuous_receipts"]) == 3
    assert len(summary["windows"]) == 2
    assert summary["zero_duration_command_windows"] == 1
    assert summary["windows"][1]["command"]["tool"] == "stop_now"
    assert summary["windows"][1]["responses"][0]["response"]["status"] == "accepted"
    np.testing.assert_array_equal(arrays["continuous"][:, 0], arrays["reanchored_local"][:, 0])


@pytest.mark.parametrize("start,end", [(2.0, 2.0), (2.0, 10.01), (2.001, 2.002), (float("nan"), 3)])
def test_invalid_intervals_are_rejected(start, end):
    with pytest.raises(ValueError):
        replay([command(2.0)], start=start, end=end)


def test_invalid_time_grid_or_missing_initial_motion_is_rejected():
    c = columns()
    c["t"][250] = c["t"][249]
    with pytest.raises(ValueError, match="strictly increasing"):
        replay([command(2.0)], c=c)
    with pytest.raises(ValueError, match="initial motion"):
        replay([command(2.0, "stop_now")])


def test_native_replay_rejects_insufficient_actuator_history():
    with pytest.raises(ValueError, match="200 preceding"):
        analysis.replay_attempt(
            columns(),
            None,
            [command(0.2)],
            {"segment": "trial", "start_t": 0.2, "end_t": 1.0},
        )


def test_native_trajectory_agrees_with_existing_validation_through_stop():
    c, cp = columns(1501), car_params(COROLLA.read_bytes())
    calls = [command(2.0), command(8.0, "stop_now", "stopped")]
    calls[0]["arguments"]["duration_s"] = 6.0
    arrays, summary = analysis.replay_attempt(
        c,
        cp,
        calls,
        {"segment": "trial", "start_t": 2.0, "end_t": 14.0},
    )
    model = Sim(cp, c, 200)
    speed, wheel, *_ = simulate(model, c, 200, 1400, calls, model.native.vm)
    np.testing.assert_array_equal(arrays["continuous"][:, 4], speed)
    np.testing.assert_array_equal(arrays["continuous"][:, 5], wheel)
    assert speed.max() > 0.3 and speed[-1] == 0
    assert summary["accepted_replays"] == 2
    assert arrays["continuous"][-1, 1] > 0


def test_cli_refuses_existing_output_and_is_excluded_from_live_source_membership(tmp_path):
    with pytest.raises(FileExistsError):
        analysis.run(tmp_path / "absent.json", tmp_path)
    assert analysis.__file__ not in {str(path) for path in source_files()}


@pytest.mark.parametrize("segment", ["../outside", "", ".", ".."])
def test_cli_rejects_unsafe_segment_directories(tmp_path, segment):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attempts": [{"segment": segment}]}))
    with pytest.raises(ValueError, match="segment directory names"):
        analysis.run(manifest, tmp_path / "output")
    assert not (tmp_path / "outside").exists()
