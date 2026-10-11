"""Read-only actor clearance and delivered-view evidence around physical resumptions."""

from __future__ import annotations

import math
import re

import numpy as np
from drivingbench.sim.course import LENGTH_M, WIDTH_M, footprint
from shapely.geometry import Point, Polygon

CLEAR_HEIGHT_PX = 60
CLEAR_CENTER_RANGE = (0.2, 0.8)
CLEAR_VISIBLE_FRACTION = 0.5
FRESH_FRAME_WALL_S = 8.0
PATH_AHEAD_M = 30.0


def view_features(frame, bounds):
    """Projected alpha bounds are separate from logged depth-surviving pixel counts."""
    pixels = frame.get("billboards", {}).get("hazard", {})
    visible = pixels.get("visible_pixels", 0)
    projected = pixels.get("projected_pixels")
    width, height = frame.get("producer_size", (1344, 760))
    return dict(
        visible=visible > 0,
        visible_pixels=visible,
        projected_pixels=projected,
        visible_fraction=visible / projected if projected else None,
        projected_alpha_bounds=bounds,
        projected_height_px=bounds[3] - bounds[1] if bounds else None,
        center_x_fraction=(bounds[0] + bounds[2]) / (2 * width) if bounds else None,
        center_y_fraction=(bounds[1] + bounds[3]) / (2 * height) if bounds else None,
        moving=(frame.get("actor") or {}).get("motion_phase") == "moving",
        capture_sim_s=frame["sim_time_s"],
    )


def clear_view(features, *, min_height=60, center_range=(0.2, 0.8), min_survival=0.5):
    return bool(
        features["visible"]
        and features.get("projected_height_px") is not None
        and features["projected_height_px"] >= min_height
        and features.get("center_x_fraction") is not None
        and center_range[0] <= features["center_x_fraction"] <= center_range[1]
        and features.get("visible_fraction") is not None
        and features["visible_fraction"] >= min_survival
    )


def visibility_class(observations, *, min_height=60, center_range=(0.2, 0.8)):
    """The caller supplies the last two definitely received paired observations."""
    narrow = [r["narrow"] for r in observations if r.get("narrow") is not None]
    wide = [r["wide"] for r in observations if r.get("wide") is not None]
    return dict(
        label="unknown"
        if not observations
        else "clear"
        if any(clear_view(f, min_height=min_height, center_range=center_range) for f in narrow)
        else "peripheral/ambiguous",
        narrow_visible=any(f["visible"] for f in narrow),
        wide_visible=any(f["visible"] for f in wide),
        wide_only=not any(f["visible"] for f in narrow) and any(f["visible"] for f in wide),
        criteria=dict(
            min_native_height_px=min_height,
            central_width_fraction=list(center_range),
            min_logged_surviving_fraction=CLEAR_VISIBLE_FRACTION,
            last_received_observations=2,
        ),
    )


def relative_geometry(circle, pose, corridor):
    """A path means the straight 30 m body-width strip ahead at the resume heading."""
    x, y, radius = circle
    theta = math.radians(pose[2])
    dx, dy = x - pose[0], y - pose[1]
    longitudinal = dx * math.cos(theta) + dy * math.sin(theta)
    lateral = -dx * math.sin(theta) + dy * math.cos(theta)
    body = Polygon(footprint(pose))
    point = Point(x, y)
    distance = body.distance(point)
    if body.contains(point):
        distance = -body.boundary.distance(point)
    front_gap = longitudinal - LENGTH_M / 2 - radius
    path_gap = abs(lateral) - WIDTH_M / 2 - radius
    return dict(
        actor_circle_map=list(circle),
        body_pose_map=list(pose),
        center_distance_m=math.hypot(dx, dy),
        signed_body_gap_m=distance - radius,
        longitudinal_from_body_center_m=longitudinal,
        lateral_from_body_center_m=lateral,
        front_gap_m=front_gap,
        lateral_path_gap_m=path_gap,
        inside_corridor=corridor.distance(point) <= radius,
        path_ahead=path_gap <= 0
        and longitudinal + radius >= LENGTH_M / 2
        and longitudinal - radius <= LENGTH_M / 2 + PATH_AHEAD_M,
    )


def signed_body_gaps(poses, circles):
    """Exact circle versus oriented rectangular body gap, vectorized over actual ticks."""
    poses, circles = np.asarray(poses, float), np.asarray(circles, float)
    if poses.ndim != 2 or poses.shape[1] != 3 or circles.shape != poses.shape:
        raise ValueError("Matching N×3 poses and circles are required")
    delta = circles[:, :2] - poses[:, :2]
    angle = np.radians(poses[:, 2])
    local = np.column_stack(
        (
            delta[:, 0] * np.cos(angle) + delta[:, 1] * np.sin(angle),
            -delta[:, 0] * np.sin(angle) + delta[:, 1] * np.cos(angle),
        )
    )
    excess = np.abs(local) - [LENGTH_M / 2, WIDTH_M / 2]
    outside = np.linalg.norm(np.maximum(excess, 0), axis=1)
    inside = np.minimum(np.max(excess, axis=1), 0)
    return outside + inside - circles[:, 2]


def stable_clear_time(times, intersects, resume_s):
    """First sampled clear time with no later re-entry; right-censor if still occupied."""
    times = np.asarray(times, float)
    intersects = np.asarray(intersects, bool)
    if times.shape != intersects.shape or times.ndim != 1 or np.any(np.diff(times) < 0):
        raise ValueError("Sorted matching one-dimensional times and occupancy are required")
    if not len(times):
        return dict(clear_time_s=None, delay_sim_s=None, censored=True)
    later = np.flatnonzero(times >= resume_s)
    if not len(later):
        return dict(clear_time_s=None, delay_sim_s=None, censored=True)
    occupied = later[intersects[later]]
    index = occupied[-1] + 1 if len(occupied) else later[0]
    if index >= len(times):
        return dict(clear_time_s=None, delay_sim_s=None, censored=True)
    return dict(
        clear_time_s=float(times[index]),
        delay_sim_s=max(0.0, float(times[index]) - resume_s),
        censored=False,
        sampled_resolution_s=float(np.max(np.diff(times))) if len(times) > 1 else None,
    )


def stop_reason_class(reason, explicit):
    """Literal reasons qualify intent; expiry alone does not establish a hazard wait."""
    hazard = bool(
        re.search(r"\b(child|kid|pedestrian|person|dog|animal|cross\w*|ball)\b", reason or "", re.I)
    )
    return (
        "explicit_hazard_stop"
        if explicit and hazard
        else "explicit_other_or_unspecified_stop"
        if explicit
        else "expiry_or_unattributed_stop"
    )
