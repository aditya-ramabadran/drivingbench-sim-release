import pytest
from drivingbench.sim.real_course_parking_analysis import ParkingHold, parking_geometry


def finish(reverse_mouth=False):
    left, right = [0, 2.5], [0, -2.5]
    if reverse_mouth:
        left, right = right, left
    return {
        "polygon": [[0, -2.5], [6, -2.5], [6, 2.5], [0, 2.5]],
        "mouth": {"left": left, "right": right, "conditional_sigma_m": 0.1},
    }


@pytest.mark.parametrize("reverse_mouth", [False, True])
def test_front_entry_uses_bay_side_independent_of_mouth_order(reverse_mouth):
    result = parking_geometry((-2, 0, 0), finish(reverse_mouth))
    assert result["front_past_mouth"]
    assert result["finish_geometry"]
    assert not result["whole_body_strict"]


def test_front_rule_is_finite_and_does_not_use_rear_bumper():
    assert not parking_geometry((-3, 20, 0), finish())["finish_geometry"]
    assert not parking_geometry((20, 0, 0), finish())["finish_geometry"]
    assert not parking_geometry((-3, 0, 180), finish())["front_past_mouth"]
    # Backing into the actual enclosure can still satisfy the body-overlap alternative.
    assert parking_geometry((-2, 0, 180), finish())["finish_geometry"]


def test_tangency_is_not_positive_overlap_and_reports_uncertainty():
    result = parking_geometry((-4.63 / 2, 0, 0), finish())
    assert result["body_overlap_m2"] == pytest.approx(0)
    assert not result["finish_geometry"]
    assert result["near_mouth_uncertainty"]
    assert result["body_overlap_possible_under_conditional_margin"]
    assert not result["body_overlap_robust_under_conditional_margin"]


def test_stopped_overlap_requires_continuous_hold_and_absolute_speed():
    hold = ParkingHold(finish())
    assert not hold.assess(0, (-2, 0, 0), 0, held=True)["operator_like_success"]
    assert hold.assess(1, (-2, 0, 0), 0, held=True)["operator_like_success"]
    assert not hold.assess(2, (-2, 0, 0), -0.1, held=True)["operator_like_success"]
    assert not hold.assess(3, (-2, 0, 0), 0, held=True)["operator_like_success"]
    assert not hold.assess(3.5, (-5, 0, 0), 0, held=True)["operator_like_success"]
    assert not hold.assess(4, (-2, 0, 0), 0, held=True)["operator_like_success"]
    assert hold.assess(5, (-2, 0, 0), 0, held=True)["operator_like_success"]
    with pytest.raises(ValueError, match="nondecreasing"):
        hold.assess(4, (-2, 0, 0), 0, held=True)


def test_strict_containment_remains_parallel_diagnostic():
    result = parking_geometry((3, 0, 0), finish())
    assert result["whole_body_strict"] and result["finish_geometry"]
    assert not parking_geometry((4.63 / 2, 0, 0), finish())["whole_body_strict"]


def test_exact_speed_threshold_and_invalid_uncertainty():
    hold = ParkingHold(finish())
    assert not hold.assess(0, (-2, 0, 0), 0.05, held=True)["operator_like_success"]
    assert not hold.assess(2, (-2, 0, 0), 0.05, held=True)["operator_like_success"]
    value = finish()
    value["mouth"]["conditional_sigma_m"] = -1
    with pytest.raises(ValueError, match="uncertainty"):
        parking_geometry((-2, 0, 0), value)


def test_creeping_without_sensor_verified_hold_resets_dwell():
    hold = ParkingHold(finish())
    assert not hold.assess(0, (-2, 0, 0), 0.04, held=False)["operator_like_success"]
    assert not hold.assess(2, (-2, 0, 0), 0.04, held=False)["operator_like_success"]
    assert not hold.assess(3, (-2, 0, 0), 0, held=True)["operator_like_success"]
    assert not hold.assess(3.5, (-2, 0, 0), 0, held=False)["operator_like_success"]
    assert not hold.assess(4, (-2, 0, 0), 0, held=True)["operator_like_success"]
    assert hold.assess(5, (-2, 0, 0), 0, held=True)["operator_like_success"]
    with pytest.raises(TypeError):
        hold.assess(6, (-2, 0, 0), 0)


def virtual_scene():
    return {
        "boundaries": {
            "left": [[0, 2.5], [30, 2.5]],
            "right": [[0, -2.5], [10, -2.5], [20, -2.5], [30, -2.5]],
        },
        "markers": [{"id": "a", "x": 10, "y": -2.5}, {"id": "b", "x": 20, "y": -2.5}],
        "measurement": {
            "virtual_boundary": {
                "first_marker_id": "a",
                "last_marker_id": "b",
                "vertices": [[15, -2.5]],
            }
        },
    }


def test_virtual_backstop_scores_leniently_and_tangency_does_not_end():
    from drivingbench.sim.real_course_parking_analysis import virtual_western_excursion

    scene = virtual_scene()
    value = virtual_western_excursion((15, -5.5 + 1.78 / 2, 0), scene)
    assert value["virtual_excursion_m"] == pytest.approx(3)
    assert value["strict_virtual_score"] and not value["virtual_backstop"]
    assert value["backstop_possible_under_margin"]
    assert not value["backstop_robust_under_margin"]
    assert virtual_western_excursion((15, -6.2, 0), scene)["virtual_backstop"]


@pytest.mark.parametrize("pose", [(1, -10, 0), (29, -10, 0), (15, 10, 0)])
def test_virtual_span_does_not_relabel_other_edges_or_opposite_side(pose):
    from drivingbench.sim.real_course_parking_analysis import virtual_western_excursion

    result = virtual_western_excursion(pose, virtual_scene())
    assert result["virtual_excursion_m"] == 0
    assert not result["virtual_backstop"]


def test_virtual_sampling_resolution_is_explicit_near_threshold():
    from drivingbench.sim.real_course_parking_analysis import virtual_western_excursion

    below = virtual_western_excursion((15, -2.5 - 2.99 + 1.78 / 2, 0), virtual_scene())
    assert below["virtual_excursion_m"] == pytest.approx(2.99)
    assert not below["virtual_backstop"]
    assert below["backstop_near_sampling_resolution"]
    above = virtual_western_excursion((15, -2.5 - 3.01 + 1.78 / 2, 0), virtual_scene())
    assert above["virtual_backstop"]
