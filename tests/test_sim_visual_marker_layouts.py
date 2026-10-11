"""Temporal fits must preserve genuine displacement and report unobservable passes."""

import numpy as np
from drivingbench.sim.visual_marker_layouts import audit, fit_pass, footprint


def observations(point, start):
    result = []
    for i, y in enumerate(np.linspace(-3, 3, 8)):
        center = np.array([0, y, 1.21])
        direction = np.asarray(point) - center
        direction /= np.linalg.norm(direction)
        result.append(dict(t=start + i, center=center.tolist(), direction=direction.tolist()))
    return result


def test_passes_are_fitted_independently_and_preserve_motion():
    first = observations([10, 2, 0], 120)
    second = observations([10.3, 2.4, 0], 710)
    row = audit([dict(id="marker", observations=first + second)])[0]
    assert np.isclose(row["early_late_delta_xy_m"], 0.5)
    np.testing.assert_allclose(row["fits"]["early_out"]["position_enu_m"], [10, 2, 0], atol=1e-10)
    np.testing.assert_allclose(
        row["fits"]["late_out"]["position_enu_m"], [10.3, 2.4, 0], atol=1e-10
    )


def test_unobserved_pass_is_unknown_not_zero_displacement():
    row = audit([dict(id="marker", observations=observations([10, 2, 0], 120))])[0]
    assert set(row["fits"]) == {"early_out"}
    assert "early_late_delta_xy_m" not in row
    assert fit_pass(observations([10, 2, 0], 120)[:4]) is None


def test_parallel_views_are_unobservable():
    samples = [dict(center=[0, y, 1], direction=[1, 0, 0]) for y in range(8)]
    assert fit_pass(samples) is None


def test_physics_radius_is_explicit_estimate_independent_of_render_color():
    a = footprint(dict(kind="dome_marker", color="orange_red"))
    b = footprint(dict(kind="dome_marker", color="yellow_green"))
    assert a == b
    assert a["radius_m"] > 0
    assert "not a surveyed" in a["status"]
    cone = footprint(dict(kind="tall_cone", apparent_color_width_m=0.339))
    assert np.isclose(cone["radius_m"], 0.17)


def test_duplicated_frame_does_not_satisfy_view_count():
    sample = observations([10, 2, 0], 120)[0]
    assert fit_pass([dict(sample, name="same.jpg") for _ in range(20)]) is None


def test_unknown_physical_class_is_not_assigned_a_cone_footprint():
    import pytest

    with pytest.raises(ValueError, match="Unsupported physical marker class"):
        footprint(dict(kind="pedestrian", apparent_color_width_m=0.5))
