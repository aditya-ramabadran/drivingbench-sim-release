"""Ray geometry and native crop bounds without recorded images."""

import numpy as np
import pytest
from drivingbench.sim.visual_cones import crop_bounds, project, rays, triangulate
from drivingbench.sim.visual_pose import camera_to_world


def test_known_two_view_triangulation():
    point = np.array([10, 2, 0.05])
    frames = [dict(transform_matrix=camera_to_world(0, y, 0).tolist()) for y in (-1, 1, 3)]
    observations = [project(frame, point)[0] for frame in frames]
    samples = [rays(frame, pixel) for frame, pixel in zip(frames, observations)]
    centers = np.array([c[0] for c, _ in samples])
    directions = np.array([d[0] for _, d in samples])
    estimated, residual, condition = triangulate(centers, directions)
    np.testing.assert_allclose(estimated, point, atol=1e-10)
    assert max(residual) < 1e-10
    assert condition < 100


def test_parallel_rays_are_not_a_landmark():
    with pytest.raises(np.linalg.LinAlgError):
        triangulate(np.array([[0.0, 0, 1], [0, 1, 1]]), np.array([[1.0, 0, 0], [1, 0, 0]]))


def test_tight_crop_preserves_native_resolution_and_ssim_extent():
    assert crop_bounds((100, 200, 120, 210)) == (96, 196, 124, 214)
    x0, y0, x1, y1 = crop_bounds((100, 200, 106, 202))
    assert x1 - x0 >= 11
    assert y1 - y0 >= 11
    assert crop_bounds((0, 0, 30, 30)) == (0, 0, 34, 34)


def test_multiview_fit_rejects_false_color_association():
    from drivingbench.sim.visual_cones import fit_landmark

    point = np.array([10, 2, 0.05])
    observations = []
    for y in np.linspace(-2, 3, 8):
        center = np.array([0, y, 1.21])
        direction = point - center
        direction /= np.linalg.norm(direction)
        observations.append(dict(center=center.tolist(), direction=direction.tolist()))
    observations.append(dict(center=[0, -5, 1.21], direction=[1, 0, 0]))
    fit = fit_landmark(observations)
    assert fit["count"] == 8
    np.testing.assert_allclose(fit["position"], point, atol=1e-10)
    assert fit_landmark(observations[:4]) is None


def test_landmark_reconstruction_rejects_held_out_before_loading_pixels(tmp_path):
    from drivingbench.sim.visual_cones import landmark_candidates

    with pytest.raises(ValueError, match="training frames only"):
        landmark_candidates([{"split": "test", "file_path": "absent.jpg"}], tmp_path)
