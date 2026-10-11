import numpy as np
import pytest
from drivingbench.sim.actor_video_assets import (
    height_scale_factors,
    matte_anchor,
    metric_track,
    monotone_progress,
    nearby_components,
    normalized_extent,
)


def test_height_trend_does_not_follow_gait_or_brief_matte_loss():
    heights = np.full(101, 180.0)
    heights[::2] += 2
    heights[50] = 130
    trend, factors = height_scale_factors(heights, fps=24, reference_height_px=180)
    assert np.all((trend >= 180) & (trend <= 182))
    assert np.ptp(heights[20:40] * factors[20:40]) > 1.9
    assert np.max(factors) < 1.001


def test_height_normalization_removes_slow_growth_isotropically():
    heights = np.linspace(170, 190, 361)
    trend, factors = height_scale_factors(heights, fps=24, reference_height_px=180)
    assert np.all(np.diff(trend) >= 0)
    assert np.max(np.abs(heights * factors - 180)) < 1
    assert factors[0] > 1 and factors[-1] < 1


def test_height_trend_rejects_invalid_measurements():
    with pytest.raises(ValueError):
        height_scale_factors([180, np.nan], fps=24, reference_height_px=180)


def test_matte_islands_keep_head_and_foot_without_far_asphalt():
    boxes = [(30, 30, 65, 100), (38, 6, 57, 26), (38, 104, 62, 111), (110, 20, 160, 120)]
    assert nearby_components(boxes, primary=0, max_gap=5) == (0, 1, 2)


def test_matte_islands_include_adjacent_hand_through_arm():
    assert nearby_components(
        [(10, 20, 30, 80), (32, 30, 40, 40), (42, 31, 47, 38)], primary=0, max_gap=3
    ) == (0, 1, 2)


def test_invalid_matte_component_bounds_are_rejected():
    with pytest.raises(ValueError):
        nearby_components([(0, 0, 0, 3)], primary=0)


def test_monotone_progress_preserves_source_pauses_and_endpoints():
    anchors = [(1, 4), (2, 4), (2, 4), (5, 4), (9, 4)]
    progress, correction = monotone_progress(anchors, direction=(1, 0))
    assert progress.tolist() == [0, 0.125, 0.125, 0.5, 1]
    assert np.all(correction == 0)


def test_isotonic_fit_repairs_small_gait_anchor_reversal_without_uniform_speed():
    progress, correction = monotone_progress([(0, 0), (4, 0), (2, 0), (10, 0)], direction=(1, 0))
    assert progress.tolist() == [0, 0.3, 0.3, 1]
    assert correction.tolist() == [0, -1, 1, 0]


def test_reverse_walk_is_tracked_in_its_actual_direction():
    track = metric_track(
        [(40, 9), (20, 9), (0, 9)],
        nominal_height_m=1.2,
        silhouette_height_px=120,
        direction=(-1, 0),
    )
    assert track["source_distance_m"] == 0.4
    assert track["displacement_m"][-1] == [-0.4, 0]
    assert track["progress_fraction"] == [0, 0.5, 1]


def test_foot_band_ignores_semitransparent_shadow():
    alpha = np.zeros((12, 14), dtype=np.uint8)
    alpha[1:8, 4:10] = 255
    alpha[8:10, 5:9] = 255
    alpha[10:, :] = 90
    anchor, box = matte_anchor(alpha)
    assert anchor == (6.5, 9)
    assert box == (4, 1, 10, 10)


def test_canvas_preserves_raised_foot_and_extending_arm():
    size, anchor = normalized_extent([(2, 1, 8, 10), (0, 2, 11, 8)], [(5, 9), (6, 7)], padding=2)
    assert size == (15, 13)
    assert anchor == (8, 10)


@pytest.mark.parametrize("anchors", [[(0, 0), (0, 0)], [(0, 0), (np.nan, 2)], [(1, 2)]])
def test_bad_or_motionless_tracks_are_rejected(anchors):
    with pytest.raises(ValueError):
        monotone_progress(anchors, direction=(1, 0))


@pytest.mark.parametrize("height", [0, float("nan"), float("inf")])
def test_metric_scale_rejects_nonfinite_and_nonpositive_heights(height):
    with pytest.raises(ValueError):
        metric_track(
            [(0, 0), (10, 0)], nominal_height_m=height, silhouette_height_px=100, direction=(1, 0)
        )
