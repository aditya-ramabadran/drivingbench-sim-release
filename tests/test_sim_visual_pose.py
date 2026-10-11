"""Pose seeds and camera conventions, without route artifacts."""

import numpy as np
import pytest
from drivingbench.sim.visual_pose import (
    build_pose_seed,
    camera_to_world,
    integrate,
    register_pose,
    rigid_fit,
    sample_pose,
)

ORIGIN = (10.0, 20.0)


def series():
    t = np.arange(0, 100, 0.01)
    speed = np.where((t > 5) & (t < 95), 1.0, 0.0)
    gear = np.where((t > 45) & (t < 65), 4, 2)
    rate = np.where((t > 20) & (t < 40), 0.03, 0.0)
    heading = 0.6 + np.r_[0, np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(t))]
    xy = integrate(t, speed * np.where(gear == 4, -1, 1), heading) + (3, -2)
    lat = ORIGIN[0] + np.degrees(xy[:, 1] / 6378137)
    lon = ORIGIN[1] + np.degrees(xy[:, 0] / (6378137 * np.cos(np.radians(ORIGIN[0]))))
    return (
        dict(
            t=t,
            v=speed,
            gear=gear,
            gyro_x=rate - 0.025,
            lat=lat,
            lon=lon,
            gps_accuracy=np.ones(len(t)),
        ),
        xy,
        heading,
    )


def test_pose_uses_reverse_and_gyro_bias():
    c, truth, heading = series()
    seed, report = build_pose_seed(c, ORIGIN)
    np.testing.assert_allclose(np.column_stack((seed["x"], seed["y"])), truth, atol=0.02)
    np.testing.assert_allclose(seed["heading"], heading, atol=0.002)
    assert report["blocks"][0]["gyro_rest_bias_rad_s"] == pytest.approx(-0.025)
    assert report["blocks"][0]["reverse_s"] == pytest.approx(19.99)


def test_robust_alignment_rejects_isolated_gps_outlier():
    source = np.column_stack((np.arange(100), np.sin(np.arange(100) / 10)))
    target = register_pose(np.column_stack((source, np.zeros(100))), (3, 4, 0.7))[:, :2]
    target[25] += (1000, -500)
    angle, translation, residual = rigid_fit(source, target, np.ones(100))
    assert angle == pytest.approx(0.7, abs=0.002)
    np.testing.assert_allclose(translation, (3, 4), atol=0.05)
    assert residual[25] > 1000


def test_gaps_and_unknown_moving_gear_are_not_interpolated():
    c, _, _ = series()
    c["gear"][3000:3002] = 0
    keep = (c["t"] < 40) | (c["t"] >= 50)
    c = {k: v[keep] for k, v in c.items()}
    seed, _ = build_pose_seed(c, ORIGIN)
    sampled = sample_pose(seed, [-1, 30.005, 45, 105])
    assert np.isnan(sampled).all()
    assert np.isfinite(sample_pose(seed, [c["t"][0], c["t"][-1]])).all()
    assert np.isnan(seed["x"][3000:3002]).all()


def test_stationary_only_heading_is_unobservable():
    c, _, _ = series()
    c["v"][:] = 0
    seed, report = build_pose_seed(c, ORIGIN)
    assert np.isnan(seed["heading"]).all()
    assert report["blocks"][0]["status"] == "insufficient moving GPS baseline"


def test_rejects_duplicate_clock():
    c, _, _ = series()
    c["t"][4] = c["t"][3]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_pose_seed(c, ORIGIN)


def test_camera_axes_and_registration():
    camera = camera_to_world(3, 4, 0)
    np.testing.assert_allclose(camera[:3, 3], (3, 4, 1.21))
    np.testing.assert_allclose(camera[:3, 2], (1, 0, 0))  # forward is east
    np.testing.assert_allclose(camera[:3, 0], (0, -1, 0))  # right is south
    np.testing.assert_allclose(camera[:3, 1], (0, 0, -1))  # down is down
    np.testing.assert_allclose(
        register_pose((2, 1, 0.2), (10, 20, np.pi / 2)), (9, 22, np.pi / 2 + 0.2)
    )
    tilted = camera_to_world(0, 0, 0.3, (0.01, 0.05, -0.1))
    np.testing.assert_allclose(tilted[:3, :3].T @ tilted[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(tilted[:3, :3]) == pytest.approx(1)


def test_mount_yaw_is_inverse_device_from_calib():
    # Positive rpyCalib yaw points the optical axis left in the road's FLU frame.
    camera = camera_to_world(0, 0, 0, rpy=(0, 0, 0.1))
    np.testing.assert_allclose(camera[:3, 2], (np.cos(0.1), np.sin(0.1), 0))


def test_odometry_checks_reverse_sign_and_vendor_delay():
    c, _, _ = series()
    rows = []
    for i in range(0, len(c["t"]), 20):
        rows.append(
            dict(
                t=c["t"][i] + 0.1,
                trans=[c["v"][i] * (-1 if c["gear"][i] == 4 else 1), 0, 0],
                rot=[0, 0, -(c["gyro_x"][i] + 0.025)],
            )
        )
    _, report = build_pose_seed(c, ORIGIN, odometry=rows)
    diagnostics = report["camera_odometry"]
    assert diagnostics["fused"] is False
    for direction in ("forward", "reverse", "rest"):
        assert diagnostics[direction]["samples"] > 10
        assert diagnostics[direction]["speed_abs_median"] == pytest.approx(0)
        assert diagnostics[direction]["yaw_rate_abs_median"] == pytest.approx(0)
