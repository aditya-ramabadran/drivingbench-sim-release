import numpy as np
import pytest
from drivingbench.sim.visual_ground import GroundCalibration, fit_camera, plane_fit
from drivingbench.sim.visual_pose import camera_to_world


def calibration():
    normal = np.array([0.02, -0.01, 1.0])
    normal /= np.linalg.norm(normal)
    return GroundCalibration(normal, 0.2, 1.21, camera_to_world(0, 0, 0)[:3, :3])


def test_plane_fit_recovers_ground_despite_non_ground_outliers():
    rng = np.random.default_rng(8)
    cal = calibration()
    xy = rng.uniform(-20, 20, (100, 2))
    points = np.column_stack((xy, cal.ground_height(*xy.T)))
    points += rng.normal(0, 0.003, points.shape)
    points[:20, 2] += 0.5
    normal, offset, inside = plane_fit(points)
    assert np.allclose(normal, cal.normal, atol=0.001)
    assert offset == pytest.approx(cal.offset, abs=0.003)
    assert inside.sum() == 80
    with pytest.raises(ValueError):
        plane_fit(np.zeros((8, 3)))


def test_ground_basis_and_camera_use_plane_normal_height():
    cal = calibration()
    basis = cal.map_from_ground
    assert np.allclose(basis[:3, :3].T @ basis[:3, :3], np.eye(3))
    assert np.linalg.det(basis[:3, :3]) == pytest.approx(1)
    point = basis @ [3, -4, 0, 1]
    assert point[:3] @ cal.normal + cal.offset == pytest.approx(0)
    camera = cal.camera_from_ground(3, -4, 0.7)
    assert camera[:3, 3] @ cal.normal + cal.offset == pytest.approx(1.21)
    assert np.allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3))
    enu_camera = cal.camera_pose(3, -4, 0.7)
    assert np.allclose(
        enu_camera[:3, 3] - cal.height * cal.normal, [3, -4, cal.ground_height(3, -4)]
    )


def test_ray_intersections_are_camera_depth_and_reject_sky():
    cal = calibration()
    camera = cal.camera_from_ground(0, 0, 0)
    points, depth = cal.ray_intersections(camera, dict(w=3, h=3, fl_x=1, fl_y=1, cx=1.5, cy=1.5))
    assert np.isnan(depth[:2]).all()
    assert np.allclose(depth[2], 1.21)
    assert np.allclose(points[2] @ cal.normal + cal.offset, 0)
    camera_points = (points[2] - camera[:3, 3]) @ camera[:3, :3]
    assert np.allclose(camera_points[:, 2], depth[2])


def test_mount_fit_is_training_only_and_recovers_fixed_rig():
    cal = calibration()
    rows = [
        dict(split="train", heading=h, transform_matrix=cal.camera_pose(i, -i, h))
        for i, h in enumerate(np.linspace(-2, 2, 9))
    ]
    fitted = fit_camera(cal.normal, cal.offset, rows)
    assert fitted.height == pytest.approx(cal.height)
    assert np.allclose(fitted.mount_rotation, cal.mount_rotation)
    with pytest.raises(ValueError, match="training"):
        fit_camera(cal.normal, cal.offset, [dict(rows[0], split="test")])


def test_calibration_rejects_invalid_plane_and_rotation():
    with pytest.raises(ValueError, match="normal"):
        GroundCalibration([0, 0, 2], 0, 1.21, np.eye(3))
    with pytest.raises(ValueError, match="proper rotation"):
        GroundCalibration([0, 0, 1], 0, 1.21, np.diag([1, 1, -1]))
    with pytest.raises(ValueError, match="finite"):
        calibration().camera_pose(np.nan, 0, 0)
