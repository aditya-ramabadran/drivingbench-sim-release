import math

import numpy as np
import pytest
from drivingbench.sim.plant import Corolla, Params
from drivingbench.sim.real_course_yaw_uncertainty_analysis import (
    ResidualYaw,
    UncertainCorolla,
    perturb_start,
    yaw_stratum,
)


class ZeroNoise:
    def rate(self, speed, wheel_deg):
        return 0.0


def test_zero_process_noise_matches_live_pose_and_actuators():
    class CenteredDraw:
        params = Params()

        def launch(self, held):
            return self.params

    draw = CenteredDraw()

    def curvature(angle, speed):
        return angle / 40

    expected = Corolla(draw.params, curvature)
    actual = UncertainCorolla(draw.params, curvature, draw=draw, noise=ZeroNoise())
    for index in range(800):
        args = (0.8 if index < 500 else -0.8, 0.8, True, True)
        expected.step(*args)
        actual.step(*args)
        for field in ("x", "y", "heading", "speed", "accel", "steer_deg"):
            assert getattr(expected, field) == getattr(actual, field)


def test_start_error_rotates_with_car():
    assert perturb_start((10, 20, math.pi / 2), (2, 1, 0.1)) == pytest.approx(
        (9, 22, math.pi / 2 + 0.1)
    )


def test_bins_are_sign_symmetric_with_declared_edges():
    assert yaw_stratum(0.5, 59.99) == 0
    assert yaw_stratum(1, -60) == 4
    assert yaw_stratum(2, -120) == 8


def test_process_block_retains_order_and_stops_at_rest():
    group = {"id": "a", "bias_rad_s": [0.02] * 9}
    block = {"group": "a", "stratum": 0, "centered_rad_s": np.arange(100) / 1000}
    noise = ResidualYaw(np.random.default_rng(0), [group], [block], "a")
    assert noise.rate(0.5, 0) == pytest.approx(0.02)
    assert noise.rate(0.5, 0) == pytest.approx(0.021)
    assert noise.rate(0, 0) == 0
    assert noise.rate(0.5, 0) == pytest.approx(0.02)


def test_pose_noise_keeps_actuators_identical():
    class CenteredDraw:
        params = Params()

        def launch(self, held):
            return self.params

    class FixedNoise:
        def rate(self, speed, wheel_deg):
            return 0.1 if speed else 0.0

    def curvature(angle, speed):
        return angle / 40

    draw = CenteredDraw()
    baseline = Corolla(draw.params, curvature, speed=1.0)
    actual = UncertainCorolla(draw.params, curvature, draw=draw, noise=FixedNoise(), speed=1.0)
    for _ in range(100):
        baseline.step(0.2, 0.8, True, True)
        actual.step(0.2, 0.8, True, True)
        for field in ("speed", "accel", "steer_deg"):
            assert getattr(baseline, field) == getattr(actual, field)
    assert actual.heading - baseline.heading == pytest.approx(0.1)
    assert actual.y != baseline.y
