import json

import pytest
from drivingbench.replay.dynamics import Dynamics, collect


def telemetry(t, speed, command=None, **extra):
    return dict(
        kind="telemetry",
        received_at=t,
        speed_mps=speed,
        steering_deg=0,
        command=command,
        enabled=True,
        sensors_valid=True,
        reason=None,
        lateral_active=True,
        **extra,
    )


def write_events(tmp_path, rows):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_fit_collection_keeps_normal_expiry_but_excludes_operator_stop(tmp_path):
    command = dict(request_id="a", accepted_at_s=0, speed_mps=1, target_angle_deg=0)
    rows = [
        telemetry(0, 0, command),
        telemetry(0.1, 0.2, command),
        telemetry(0.2, 0.8, command),
        telemetry(0.3, 0.6),
        telemetry(0.4, 0.3),
        telemetry(0.5, 0),
    ]
    path = write_events(tmp_path, rows)
    launches, stops, _, _ = collect([path])
    assert len(launches) == len(stops) == 1
    assert stops[0]["initial"] == 0.8
    rows[3]["brake_pressed"] = True
    path = write_events(tmp_path, rows)
    assert collect([path])[1] == []


def test_incomplete_stop_not_fit(tmp_path):
    command = dict(request_id="a", accepted_at_s=0, speed_mps=1, target_angle_deg=0)
    path = write_events(
        tmp_path,
        [telemetry(0, 1, command), telemetry(0.1, 0.8), telemetry(0.2, 0.7), telemetry(0.3, 0.5)],
    )
    assert collect([path])[1] == []


def test_report_load_and_invalid_parameters(tmp_path):
    path = tmp_path / "dynamics.json"
    path.write_text(json.dumps({"parameters": {"brake_mps2": 0.7}}))
    assert Dynamics.load(path).brake_mps2 == 0.7
    for kwargs in ({"brake_mps2": 0}, {"launch_delay_s": -1}, {"speed_noise_mps": float("nan")}):
        with pytest.raises(ValueError):
            Dynamics(**kwargs)
