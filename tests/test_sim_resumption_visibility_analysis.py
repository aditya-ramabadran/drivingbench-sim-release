import numpy as np
import pytest
from drivingbench.sim.resumption_visibility_analysis import (
    clear_view,
    relative_geometry,
    signed_body_gaps,
    stable_clear_time,
    stop_reason_class,
    view_features,
    visibility_class,
)
from shapely.geometry import Polygon


def feature(height=80, center=0.5, survival=1):
    return dict(
        visible=True,
        projected_height_px=height,
        center_x_fraction=center,
        visible_fraction=survival,
    )


def test_clear_view_requires_size_centrality_and_surviving_pixels():
    assert clear_view(feature())
    assert not clear_view(feature(height=59))
    assert not clear_view(feature(center=0.81))
    assert not clear_view(feature(survival=0.49))
    assert clear_view(feature(height=60, center=0.2, survival=0.5))


def test_visibility_has_explicit_wide_only_and_missing_cases():
    assert visibility_class([dict(narrow=feature(), wide=feature())])["label"] == "clear"
    assert visibility_class([dict(wide=feature())])["wide_only"]
    assert visibility_class([])["label"] == "unknown"


def test_view_features_do_not_invent_occlusion_fraction():
    f = view_features(
        dict(sim_time_s=1, billboards={"hazard": {"visible_pixels": 5}}), [0, 0, 20, 40]
    )
    assert f["visible_fraction"] is None
    assert not clear_view(f)


def test_fixed_resume_path_and_whole_corridor_are_different():
    corridor = Polygon([[-5, -3], [50, -3], [50, 3], [-5, 3]])
    f = relative_geometry((10, 2, 0.3), (0, 0, 0), corridor)
    assert f["inside_corridor"] and not f["path_ahead"]
    assert relative_geometry((10, 0, 0.3), (0, 0, 0), corridor)["path_ahead"]
    assert not relative_geometry((-10, 0, 0.3), (0, 0, 0), corridor)["path_ahead"]


def test_vectorized_body_gap_matches_shapely_on_rotation_and_contact():
    corridor = Polygon([[-100, -100], [100, -100], [100, 100], [-100, 100]])
    poses = np.array([[0, 0, 0], [1, -2, 75], [-1, 3, 180], [0, 0, 0]])
    circles = np.array([[3, 0, 0.3], [2, 4, 0.4], [-1, 3, 0.3], [2.615, 0, 0.3]])
    expected = [
        relative_geometry(c, p, corridor)["signed_body_gap_m"] for p, c in zip(poses, circles)
    ]
    assert signed_body_gaps(poses, circles) == pytest.approx(expected)
    assert expected[-1] == pytest.approx(0)


def test_clearance_waits_for_final_reentry_and_censors_missing_tail():
    assert stable_clear_time([0, 1, 2, 3], [True, False, True, False], 0)["delay_sim_s"] == 3
    assert stable_clear_time([0, 1], [True, True], 0)["censored"]
    assert stable_clear_time([], [], 0)["censored"]
    assert stable_clear_time([0, 1], [False, False], 0)["delay_sim_s"] == 0


def test_stop_reason_never_relabels_expiry_as_deliberate():
    assert stop_reason_class("Wait for child", False) == "expiry_or_unattributed_stop"
    assert stop_reason_class("Wait for child", True) == "explicit_hazard_stop"
    assert stop_reason_class("Park at finish", True) == "explicit_other_or_unspecified_stop"
