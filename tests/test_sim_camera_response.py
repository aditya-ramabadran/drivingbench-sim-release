import numpy as np
import pytest
from drivingbench.sim.camera_response import CameraResponse, apply_response, response_profile


def fixture():
    background = np.full((48, 64, 3), 100, np.uint8)
    composite = background.copy()
    composite[12:36, 24:40] = (190, 140, 110)
    surface = np.zeros((48, 64))
    surface[12:36, 24:40] = 5
    coverage = (surface > 0).astype(float)
    return background, composite, np.full((48, 64), 10.0), np.ones((48, 64)), surface, coverage


def test_response_softens_visible_edges_without_changing_distant_background():
    args = fixture()
    profile = CameraResponse(0.55, 0, (1, 1, 1))
    actual = apply_response(*args, profile)
    assert 100 < actual[24, 23, 0] < actual[24, 24, 0] < 190
    assert actual[24, 32, 0] == 190
    assert np.array_equal(actual[:9], args[0][:9])
    assert np.array_equal(actual[:, :21], args[0][:, :21])
    assert not np.shares_memory(actual, args[0])


@pytest.mark.parametrize("name", ["site1-overcast-v1", "site1-splat-softness-v2"])
def test_visible_delta_never_spreads_over_nearer_real_surface_or_unknown_depth(name):
    args = fixture()
    background, composite, depth, opacity, surface, coverage = args
    depth[:, 21:24] = 3
    opacity[:, 40:43] = 0.1
    depth[9:12] = np.nan
    actual = apply_response(*args, response_profile(name))
    np.testing.assert_array_equal(actual[:, 21:24], background[:, 21:24])
    np.testing.assert_array_equal(actual[:, 40:43], background[:, 40:43])
    np.testing.assert_array_equal(actual[9:12], background[9:12])
    assert not np.array_equal(actual[12:36, 24:40], composite[12:36, 24:40])


def test_hidden_or_empty_surfaces_do_not_gain_noise_or_color():
    background, _, depth, opacity, surface, coverage = fixture()
    surface[:] = 0
    coverage[:] = 0
    actual = apply_response(
        background,
        background,
        depth,
        opacity,
        surface,
        coverage,
        response_profile("site1-overcast-v1"),
    )
    np.testing.assert_array_equal(actual, background)


def test_noise_is_reproducible_for_retries_and_varies_for_a_new_capture():
    args = fixture()
    profile = response_profile("site1-overcast-v1")
    first = apply_response(*args, profile, seed=23)
    np.testing.assert_array_equal(first, apply_response(*args, profile, seed=23))
    assert not np.array_equal(first, apply_response(*args, profile, seed=24))
    np.testing.assert_array_equal(first[:8], args[0][:8])


def test_alpha_weighted_color_preserves_transparent_background_contribution():
    background, composite, depth, opacity, surface, coverage = fixture()
    composite[12:36, 24:40] = 125
    coverage[12:36, 24:40] = 0.5
    profile = CameraResponse(0.55, 0, (1.1, 1, 1))
    actual = apply_response(background, composite, depth, opacity, surface, coverage, profile)
    assert tuple(actual[24, 32]) == (132, 125, 125)


def test_rejects_unknown_profiles_and_misaligned_inputs():
    assert response_profile(None) is None
    with pytest.raises(ValueError, match="Unknown"):
        response_profile("site1-made-up")
    args = list(fixture())
    args[2] = args[2][:-1]
    with pytest.raises(ValueError, match="matching"):
        apply_response(*args, response_profile("site1-overcast-v1"))
    args = list(fixture())
    args[-1][0, 0] = np.nan
    with pytest.raises(ValueError, match="coverage"):
        apply_response(*args, response_profile("site1-overcast-v1"))


def test_splat_softness_is_opt_in_and_expands_only_visible_object_support():
    args = fixture()
    original = response_profile("site1-overcast-v1")
    selected = response_profile("site1-splat-softness-v2")
    assert original.sigma_px == 0.55
    assert selected.sigma_px == 1.65
    assert selected.noise_sigma == original.noise_sigma
    assert selected.rgb_gain == original.rgb_gain
    before = apply_response(*args, original, seed=4)
    after = apply_response(*args, selected, seed=4)
    assert after[24, 23, 0] > before[24, 23, 0]
    assert after[24, 24, 0] < before[24, 24, 0]
    changed = np.any(after != args[0], axis=-1)
    assert not changed[: 12 - selected.radius].any()
    assert not changed[:, : 24 - selected.radius].any()
    np.testing.assert_array_equal(after, apply_response(*args, selected, seed=4))


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_softening_an_object_does_not_copy_hidden_background_edges_through_it(alpha):
    background, composite, depth, opacity, surface, coverage = fixture()
    background[:, ::2] = 20
    background[:, 1::2] = 220
    composite[:] = background
    composite[12:36, 24:40] = np.rint(
        120 * alpha + background[12:36, 24:40].astype(float) * (1 - alpha)
    ).astype(np.uint8)
    coverage *= alpha
    profile = CameraResponse(1.65, 0, (1, 1, 1))
    actual = apply_response(background, composite, depth, opacity, surface, coverage, profile)
    np.testing.assert_array_equal(actual[18:30, 29:35], composite[18:30, 29:35])
    np.testing.assert_array_equal(
        actual[:, : 24 - profile.radius], background[:, : 24 - profile.radius]
    )
