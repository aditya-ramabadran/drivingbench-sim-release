"""Selection and fixed registration for real-camera versus open-loop simulator videos."""

from __future__ import annotations

import numpy as np
from drivingbench.sim.visual_pose import register_pose


def median_window(rows, session):
    """Choose the session's representative complete window, never its minimum error."""
    eligible = [row for row in rows if row["session"] == session and "30s" in row["sim"]]
    if not eligible:
        raise ValueError("Session has no complete 30-second validation window")
    errors = [row["sim"]["30s"]["position_m"] for row in eligible]
    if not np.isfinite(errors).all():
        raise ValueError("Validation errors must be finite")
    median = float(np.median(errors))
    distances = np.abs(np.asarray(errors) - median)
    tied = [row for row, gap in zip(eligible, distances) if np.isclose(gap, distances.min())]
    return min(tied, key=lambda row: row["t"])


def register_trajectory(local, start):
    """One start alignment; subsequent poses retain all native drift, in radians."""
    local, start = np.asarray(local, dtype=float), np.asarray(start, dtype=float)
    if local.ndim != 2 or local.shape[1] != 3 or len(local) == 0 or start.shape != (3,):
        raise ValueError("Expected nonempty Nx3 trajectory and one three-component start")
    if not np.isfinite(local).all() or not np.isfinite(start).all():
        raise ValueError("Trajectory and registration must be finite")
    shifted = local.copy()
    delta = shifted[:, :2] - shifted[0, :2]
    angle = -shifted[0, 2]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shifted[:, :2] = delta @ rotation.T
    shifted[:, 2] -= shifted[0, 2]
    return register_pose(shifted, start)
