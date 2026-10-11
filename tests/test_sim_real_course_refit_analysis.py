import math
from dataclasses import replace

import numpy as np
import pytest
from drivingbench.sim.car import SimNative
from drivingbench.sim.event_study_v2 import source_files
from drivingbench.sim.plant import Corolla, Params
from drivingbench.sim.real_course_refit_analysis import (
    CandidateCorolla,
    CandidateSim,
    RefitConfig,
    make_plant,
    measured_launch_age,
)
from drivingbench.sim.validate import Sim, request


@pytest.mark.parametrize(
    "kwargs",
    [
        {"yaw_coefficients": (1, 0)},
        {"yaw_coefficients": (math.nan, 0, 0)},
        {"launch_boost_mps2": -1},
        {"launch_boost_tau_s": 0},
    ],
)
def test_invalid_candidate_coefficients_rejected(kwargs):
    with pytest.raises(ValueError):
        RefitConfig(**kwargs)


def test_yaw_form_has_bounded_angle_dependence_and_explicit_direction_term():
    config = RefitConfig(yaw_coefficients=(1, -0.1, -0.05))
    assert config.yaw_scale(0) == 1
    assert config.yaw_scale(180) == pytest.approx(0.85)
    assert config.yaw_scale(-180) == pytest.approx(0.9)
    assert config.yaw_scale(900) == 0.65
    assert RefitConfig(yaw_coefficients=(1.1, 0.1, 0)).yaw_scale(900) == 1.15


def test_measured_launch_age_does_not_invent_an_initial_departure():
    ages = measured_launch_age([1.0, 1.0, 0.0, 0.01, 0.1, 0.2, 0.0, 0.1])
    assert np.isinf(ages[:4]).all()
    np.testing.assert_allclose(ages[4:6], [0, 0.01])
    assert math.isinf(ages[6]) and ages[7] == 0


def test_zero_residual_candidate_matches_baseline_native_through_expiry_and_stop():
    config = RefitConfig()
    baseline = SimNative(params=config.params)
    candidate = SimNative(params=config.params, plant_factory=make_plant(config))
    call = {
        "tool": "set_motion",
        "arguments": {
            "direction": "left",
            "steering_percent": 60.0,
            "speed_mps": 1.0,
            "duration_s": 6.0,
        },
    }
    for model in [baseline, candidate]:
        assert model.call(request(model.bridge.controller, call))["status"] == "accepted"
    for at in (2.0, 4.0, 6.0, 10.0):
        for model in [baseline, candidate]:
            model.advance(at)
        for key in ("speed", "accel", "steer_deg", "heading", "x", "y"):
            assert getattr(candidate.car, key) == getattr(baseline.car, key)
    assert candidate.car.speed == 0
    assert candidate.car.x > 1


def test_launch_residual_decays_and_does_not_change_steering_response():
    p = replace(Params(), launch_s=0.05, accel_delay_s=0.03, accel_tau_s=0.1, accel_gain=1)
    config = RefitConfig(params=p, launch_boost_mps2=1, launch_boost_tau_s=0.7)
    baseline = Corolla(p, lambda angle, speed: 0.0)
    candidate = CandidateCorolla(p, lambda angle, speed: 0.0, config=config)
    for _ in range(200):
        baseline.step(0.1, 0.0, False, True)
        candidate.step(0.1, 0.0, False, True)
    assert candidate.speed > baseline.speed
    assert candidate.steer_deg == baseline.steer_deg == 0
    assert config.launch_boost(0) == 1
    assert 0 < config.launch_boost(2) < 0.06
    assert config.launch_boost(3) == config.launch_boost(math.inf) == 0


@pytest.mark.parametrize("delay", [0.0, 0.04])
def test_launch_residual_is_added_to_the_delayed_input_when_control_is_inactive(delay):
    p = replace(Params(), accel_delay_s=delay, accel_gain=1, accel_tau_s=0.1)
    config = RefitConfig(params=p, launch_boost_mps2=1)
    candidate = CandidateCorolla(p, lambda angle, speed: 0, speed=1, config=config)
    candidate.launch_age_s = 0
    candidate.step(10, 0, False, False)
    assert candidate.accel == pytest.approx(0.1)
    assert candidate.speed == pytest.approx(1.001)
    assert len(candidate.accel_queue) == round(delay / 0.01)


def test_logged_initialization_accepts_custom_factory_and_retains_actuator_history():
    c = {"t": np.arange(401) / 100}
    for name in (
        "v",
        "angle",
        "a",
        "cruise_standstill",
        "accel_cmd",
        "curvature_cmd",
        "torque_out",
        "long_state",
    ):
        c[name] = np.zeros(401)
    c["steer_ratio"], c["stiffness"] = np.full(401, 14.8), np.ones(401)
    c["accel_cmd"][190:200] = 0.3
    config = RefitConfig()
    factory = make_plant(config)
    native = CandidateSim(None, c, 200, config=config, plant_factory=factory)
    baseline = Sim(None, c, 200)
    assert isinstance(native.native.car, CandidateCorolla)
    assert list(native.native.car.accel_queue) == list(baseline.native.car.accel_queue)
    assert native.native.time == baseline.native.time
    import drivingbench.sim.real_course_refit_analysis as module

    assert module.__file__ not in {str(path) for path in source_files()}
