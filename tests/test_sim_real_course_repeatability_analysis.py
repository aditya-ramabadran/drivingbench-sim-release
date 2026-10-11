"""Empirical draw semantics without real logs or gitignored populations."""

import pytest
from drivingbench.sim.plant import Params
from drivingbench.sim.real_course_refit_analysis import RefitConfig
from drivingbench.sim.real_course_repeatability_analysis import ResponseCorolla, ResponseDraw


def population():
    groups = [
        {"id": "a", "route": "r", "session_ids": ["s"], "steer_gain_multiplier": 2,
         "steer_hold_multiplier": 3},
    ]  # fmt: skip
    launches = [
        {"id": "free", "group": "a", "route": "r", "session_ids": ["s"],
         "held": False, "delay_s": 1.5, "lurch_mps2": 2.0},
        {"id": "held", "group": "a", "route": "r", "session_ids": ["s"],
         "held": True, "delay_s": 3.5, "lurch_mps2": 4.0},
    ]  # fmt: skip
    return dict(
        groups=groups,
        launches=launches,
        steering_center={"gain_multiplier_median": 2, "hold_multiplier_median": 3},
        launch_center={
            "False": {"delay_median_s": 1.5, "lurch_median_mps2": 2},
            "True": {"delay_median_s": 3.5, "lurch_median_mps2": 4},
        },
    )


def test_centered_joint_draw_preserves_baseline():
    config = RefitConfig()
    draw = ResponseDraw(config, 12, **population())
    assert draw.params == config.params
    assert draw.launch(False) == config.params
    assert draw.launch(True) == config.params
    assert [x["sample_id"] for x in draw.launch_log] == ["free", "held"]


def test_delay_and_lurch_pair_stays_together():
    args = population()
    args["launches"][0].update(delay_s=2.5, lurch_mps2=3)
    config = RefitConfig(params=Params(launch_s=1.2, launch_accel_mps2=2))
    draw = ResponseDraw(config, 12, **args)
    sampled = draw.launch(False)
    assert sampled.launch_s == pytest.approx(2.2)
    assert sampled.launch_accel_mps2 == pytest.approx(3)
    assert sampled.hold_launch_s == config.params.hold_launch_s


@pytest.mark.parametrize("scope", ["same_session", "same_route", "all_training_routes"])
def test_fallback_preserves_hold_stratum_and_records_scope(scope):
    args = population()
    sample = args["launches"][1]
    sample["group"] = "another"
    if scope != "same_session":
        sample["session_ids"] = ["different"]
    if scope == "all_training_routes":
        sample["route"] = "different"
    draw = ResponseDraw(RefitConfig(), 12, **args)
    draw.launch(True)
    assert draw.launch_log[-1]["sample_id"] == "held"
    assert draw.launch_log[-1]["pool_scope"] == scope


def test_missing_hold_stratum_is_not_replaced_with_free_launch():
    args = population()
    args["launches"] = args["launches"][:1]
    draw = ResponseDraw(RefitConfig(), 12, **args)
    with pytest.raises(ValueError, match="conditional launch stratum"):
        draw.launch(True)


def test_joint_steering_scaling_and_floor_are_explicit():
    args = population()
    args["groups"][0].update(steer_gain_multiplier=1, steer_hold_multiplier=6)
    args["launches"][0]["delay_s"] = 0.01
    config = RefitConfig()
    draw = ResponseDraw(config, 12, **args)
    assert draw.params.steer_gain == config.params.steer_gain / 2
    assert draw.params.return_gain == config.params.return_gain / 2
    assert draw.params.steer_gain_per_mps == config.params.steer_gain_per_mps / 2
    assert draw.params.steer_hold_deg == config.params.steer_hold_deg * 2
    assert draw.launch(False).launch_s == 0.1
    assert draw.launch_log[-1]["delay_floor_applied"] is True


def test_one_sample_per_positive_standstill_request():
    draw = ResponseDraw(RefitConfig(), 12, **population())
    car = ResponseCorolla(draw.config.params, lambda wheel, speed: 0, draw=draw)
    for _ in range(20):
        car.step(1, 0, False, True)
    assert len(draw.launch_log) == 1
    car.step(0, 0, False, True)
    car.step(1, 0, False, True)
    assert len(draw.launch_log) == 2


def test_repeat_seed_has_same_correlated_records():
    left = ResponseDraw(RefitConfig(), [4, 5, 6], **population())
    right = ResponseDraw(RefitConfig(), [4, 5, 6], **population())
    for held in [False, True, False]:
        assert left.launch(held) == right.launch(held)
    assert left.launch_log == right.launch_log
