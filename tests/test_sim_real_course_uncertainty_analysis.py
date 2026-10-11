"""Prediction-region edge cases without local footage."""

import numpy as np
import pytest
from drivingbench.sim.plant import Corolla, Params
from drivingbench.sim.real_course_uncertainty_analysis import LiveResponseCorolla, regions


def test_centered_response_draw_executes_live_plant_exactly():
    class Centered:
        params = Params()
        launches = 0

        def launch(self, held):
            self.launches += 1
            return self.params

    draw = Centered()

    def curvature(angle, speed):
        return angle / 15

    real = Corolla(draw.params, curvature)
    sampled = LiveResponseCorolla(draw.params, curvature, draw=draw)
    for index in range(800):
        args = (0.7 if index < 500 else -1.0, 0.8, True, True)
        real.step(*args)
        sampled.step(*args)
        assert [sampled.x, sampled.y, sampled.heading, sampled.speed, sampled.steer_deg] == [
            real.x, real.y, real.heading, real.speed, real.steer_deg,
        ]  # fmt: skip
    assert draw.launches == 1


def test_singular_predictions_do_not_admit_displaced_truth():
    draws = np.zeros((200, 3))
    inside = regions(draws, [0, 0, 0])
    outside = regions(draws, [0.001, 0, 0])
    assert all(x["position_covered"] for x in inside["bands"])
    assert not any(x["position_covered"] for x in outside["bands"])


def test_wrapped_heading_interval_crosses_pi_without_losing_membership():
    draws = np.zeros((200, 3))
    draws[:, 2] = np.pi + np.linspace(-0.2, 0.2, 200)
    result = regions(draws, [0, 0, -np.pi + 0.01])
    assert all(x["heading_covered"] for x in result["bands"])
    assert result["mean_heading_error_deg"] < 1


def test_broad_heading_distribution_and_joint_mass_are_explicit():
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(2000, 3))
    draws[:, 2] = rng.uniform(-np.pi, np.pi, len(draws))
    result = regions(draws, [0, 0, 0])
    assert np.diff(result["bands"][1]["heading_offsets_deg"])[0] > 180
    assert result["bands"][0]["in_draw_intersection_mass"] < 0.35
    assert result["bands"][1]["in_draw_intersection_mass"] < 0.85


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.ones((5, 2)), [[0, 0, np.nan]] * 5])
def test_invalid_draws_rejected(bad):
    with pytest.raises(ValueError, match="finite pose"):
        regions(bad, [0, 0, 0])
