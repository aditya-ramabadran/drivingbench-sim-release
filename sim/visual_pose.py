"""GPS-anchored pose seeds; ENU metres, heading counterclockwise from east."""

from __future__ import annotations

import numpy as np

DRIVE, REVERSE = 2, 4


def enu(latitude, longitude, origin):
    """Local tangent-plane east/north coordinates for this small, flat site."""
    lat0, lon0 = np.radians(origin)
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.column_stack(((lon - lon0) * np.cos(lat0), lat - lat0)) * 6378137.0


def rigid_fit(source, target, accuracy):
    """Robust SE(2) fit, without a scale change; returns angle, translation, residuals."""
    weight = 1 / np.maximum(accuracy, 0.5) ** 2
    for _ in range(8):
        a = np.average(source, axis=0, weights=weight)
        b = np.average(target, axis=0, weights=weight)
        p, q = source - a, target - b
        dot = np.sum(weight * np.sum(p * q, axis=1))
        cross = np.sum(weight * (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]))
        angle = np.arctan2(cross, dot)
        cs, sn = np.cos(angle), np.sin(angle)
        rotation = np.array([[cs, -sn], [sn, cs]])
        translation = b - rotation @ a
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        weight = 1 / np.maximum(accuracy, 0.5) ** 2 * np.minimum(1, 2 / np.maximum(residual, 1e-9))
    return angle, translation, residual


def integrate(t, speed, heading):
    velocity = speed[:, None] * np.column_stack((np.cos(heading), np.sin(heading)))
    steps = (velocity[1:] + velocity[:-1]) / 2 * np.diff(t)[:, None]
    return np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))


def build_pose_seed(columns, origin, max_gap_s=0.1, odometry=None):
    """Smooth weak GPS anchors onto gear-aware dead reckoning, separately across gaps.

    Unobservable blocks have NaN poses. GPS residuals measure seed consistency, not truth.
    The input series carries forward GPS fixes; fits therefore sample at most 10 Hz.
    """
    t, speed, gear, gyro = (np.asarray(columns[k]) for k in ("t", "v", "gear", "gyro_x"))
    if len(t) == 0 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Pose input needs finite, strictly increasing timestamps")
    gps = enu(columns["lat"], columns["lon"], origin)
    accuracy = np.asarray(columns["gps_accuracy"])
    valid = np.isfinite(speed) & np.isfinite(gyro) & np.isfinite(gear)
    valid &= (speed < 0.05) | np.isin(gear, (DRIVE, REVERSE))
    good_gps = (
        np.isfinite(gps).all(axis=1) & np.isfinite(accuracy) & (accuracy > 0) & (accuracy < 5)
    )
    signed = speed * np.where(gear == REVERSE, -1, 1)
    cut = np.r_[True, (np.diff(t) > max_gap_s) | ~valid[:-1] | ~valid[1:]]
    block = np.cumsum(cut) - 1
    result = {
        "t": t.copy(),
        "x": np.full(len(t), np.nan),
        "y": np.full(len(t), np.nan),
        "heading": np.full(len(t), np.nan),
        "block": block,
    }
    report = {"origin_lat_lon": list(origin), "max_gap_s": max_gap_s, "blocks": []}
    starts = np.r_[0, np.flatnonzero(np.diff(block)) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(t)]):
        if end - start < 2 or not valid[start]:
            continue
        sl = slice(start, end)
        bt, bv, bg = t[sl], signed[sl], gyro[sl]
        rest = speed[sl] < 0.01
        bias = float(np.median(bg[rest])) if np.any(rest) else None
        row = {
            "block": int(block[start]),
            "start": float(bt[0]),
            "end": float(bt[-1]),
            "gyro_rest_bias_rad_s": bias,
            "reverse_s": float(np.sum(np.diff(bt) * (gear[sl][:-1] == REVERSE))),
        }
        report["blocks"].append(row)
        if bias is None:
            row["status"] = "no stationary gyro bias"
            continue
        yaw_rate = np.where(rest, 0, bg - bias)
        yaw = np.r_[0, np.cumsum((yaw_rate[1:] + yaw_rate[:-1]) / 2 * np.diff(bt))]
        path = integrate(bt, bv, yaw)
        anchors, offsets, fit_errors = [], [], []
        # Forty-second overlapping fits keep metre-scale GPS noise below path displacement.
        centers = np.unique(np.r_[bt[0], np.arange(bt[0] + 20, bt[-1], 20), bt[-1]])
        for center in centers:
            lo, hi = np.searchsorted(bt, [center - 20, center + 20])
            idx = np.arange(lo, hi, 10)
            idx = idx[good_gps[sl][idx]]
            if len(idx) < 20 or np.linalg.norm(np.std(path[idx], axis=0)) < 2:
                continue
            angle, _, residual = rigid_fit(path[idx], gps[sl][idx], accuracy[sl][idx])
            anchors.append(center)
            offsets.append(angle)
            fit_errors.extend(residual.tolist())
        row["heading_anchors"] = len(anchors)
        if not anchors:
            row["status"] = "insufficient moving GPS baseline"
            continue
        heading = yaw + np.interp(bt, anchors, np.unwrap(offsets))
        path = integrate(bt, bv, heading)
        fixes, corrections = [], []
        for center in centers:
            lo, hi = np.searchsorted(bt, [center - 10, center + 10])
            idx = np.arange(lo, hi, 10)
            idx = idx[good_gps[sl][idx]]
            if len(idx):
                fixes.append(center)
                corrections.append(np.median(gps[sl][idx] - path[idx], axis=0))
        correction = np.column_stack(
            [np.interp(bt, fixes, np.array(corrections)[:, j]) for j in range(2)]
        )
        path += correction
        result["x"][sl], result["y"][sl], result["heading"][sl] = path[:, 0], path[:, 1], heading
        residual = np.linalg.norm(path[good_gps[sl]] - gps[sl][good_gps[sl]], axis=1)
        row.update(
            status="seeded",
            gps_residual_median_m=float(np.median(residual)),
            gps_residual_p95_m=float(np.percentile(residual, 95)),
            window_fit_residual_median_m=float(np.median(fit_errors)),
        )
    report["valid_rows"] = int(np.isfinite(result["x"]).sum())
    report["total_rows"] = len(t)
    if odometry is not None:
        report["camera_odometry"] = odometry_consistency(columns, result, report, odometry)
    return result, report


def odometry_consistency(columns, seed, report, records):
    """Check learned camera odometry against the seed's wheel/gyro inputs, without fusion."""
    out = {"time_correction_s": -0.1, "fused": False}
    if not records:
        return out
    t = columns["t"]
    times = np.array([r["t"] - 0.1 for r in records])
    index = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    bias = np.full(len(t), np.nan)
    for row in report["blocks"]:
        if row["status"] == "seeded":
            bias[seed["block"] == row["block"]] = row["gyro_rest_bias_rad_s"]
    gear, speed = columns["gear"][index], columns["v"][index]
    signed = speed * np.where(gear == REVERSE, -1, 1)
    yaw_rate = columns["gyro_x"][index] - bias[index]
    odom_speed = np.array([r["trans"][0] for r in records])
    odom_yaw = -np.array([r["rot"][2] for r in records])
    valid = np.isfinite(yaw_rate) & (np.abs(t[index] - times) <= 0.05)
    groups = {
        "forward": (gear == DRIVE) & (speed > 0.2),
        "reverse": (gear == REVERSE) & (speed > 0.2),
        "rest": speed < 0.01,
    }
    for name, group in groups.items():
        use = valid & group & np.isfinite(odom_speed) & np.isfinite(odom_yaw)
        row = {"samples": int(use.sum())}
        for label, truth, observed in (
            ("speed", signed, odom_speed),
            ("yaw_rate", yaw_rate, odom_yaw),
        ):
            if np.any(use):
                error = np.abs(truth[use] - observed[use])
                row[f"{label}_abs_median"] = float(np.median(error))
                row[f"{label}_abs_p95"] = float(np.percentile(error, 95))
                if np.std(truth[use]) > 1e-8 and np.std(observed[use]) > 1e-8:
                    row[f"{label}_correlation"] = float(
                        np.corrcoef(truth[use], observed[use])[0, 1]
                    )
        out[name] = row
    return out


def sample_pose(seed, times):
    """Interpolate x/y/unwrapped heading; leave gaps and unobservable rows as NaN."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    t = seed["t"]
    right = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    left = np.maximum(right - 1, 0)
    left = np.where(t[right] == times, right, left)
    valid = (times >= t[0]) & (times <= t[-1]) & (seed["block"][left] == seed["block"][right])
    pose = np.column_stack([np.interp(times, t, seed[k]) for k in ("x", "y", "heading")])
    pose[~valid] = np.nan
    return pose


def register_pose(local, origin_pose):
    """Map SimNative's x-forward/y-left/CCW pose through a chosen site's start pose."""
    x, y, heading = np.asarray(local).T
    ox, oy, angle = origin_pose
    cs, sn = np.cos(angle), np.sin(angle)
    return np.array([ox + cs * x - sn * y, oy + sn * x + cs * y, angle + heading]).T


def camera_to_world(x, y, heading, rpy=(0, 0, 0), height=1.21):
    """OpenCV camera-to-ENU; rpyCalib is device-from-calib, as in vendored camera.py.

    No measured camera/GPS fore-aft offset is available; x/y represent the camera ground point.
    """
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    device_from_calib = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    cs, sn = np.cos(heading), np.sin(heading)
    world_from_road = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
    device_from_view = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    result = np.eye(4)
    result[:3, :3] = world_from_road @ np.diag([1, -1, -1]) @ device_from_calib.T @ device_from_view
    result[:3, 3] = (x, y, height)
    return result
