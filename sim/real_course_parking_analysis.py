"""Offline operator-like parking evidence for the separately measured real course."""

from __future__ import annotations

import math

from drivingbench.sim.course import EPS, LENGTH_M, footprint


def parking_geometry(pose, finish):
    """Report nominal body overlap, finite-mouth entry and strict containment."""
    from shapely.geometry import Polygon

    body = Polygon(footprint(pose))
    receiver = Polygon(finish["polygon"])
    mouth = finish["mouth"]
    left, right = mouth["left"], mouth["right"]
    width = math.dist(left, right)
    if not math.isfinite(width) or width <= EPS or not receiver.is_valid or receiver.area <= EPS:
        raise ValueError("Parking needs a finite mouth and valid blue-U receiver")
    across = ((right[0] - left[0]) / width, (right[1] - left[1]) / width)
    forward = (-across[1], across[0])
    centre = receiver.centroid
    if (centre.x - left[0]) * forward[0] + (centre.y - left[1]) * forward[1] < 0:
        forward = (-forward[0], -forward[1])
    heading = math.radians(pose[2])
    direction = (math.cos(heading), math.sin(heading))
    front = (pose[0] + LENGTH_M / 2 * direction[0], pose[1] + LENGTH_M / 2 * direction[1])
    lateral = sum((front[i] - left[i]) * across[i] for i in (0, 1))
    depth = sum((front[i] - left[i]) * forward[i] for i in (0, 1))
    bay_depth = max(
        sum((point[i] - left[i]) * forward[i] for i in (0, 1)) for point in finish["polygon"]
    )
    front_past = (
        EPS < lateral < width - EPS
        and EPS < depth <= bay_depth + EPS
        and sum(direction[i] * forward[i] for i in (0, 1)) > EPS
    )
    overlap = body.intersection(receiver).area
    sigma = float(mouth.get("conditional_sigma_m", 0.0))
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Mouth uncertainty must be finite and nonnegative")
    return {
        "body_overlap_m2": overlap,
        "body_overlap_fraction": overlap / body.area,
        "body_overlap_possible_under_conditional_margin": receiver.buffer(sigma).intersects(body),
        "body_overlap_robust_under_conditional_margin": (
            body.intersection(receiver.buffer(-sigma)).area > EPS
        ),
        "front_past_mouth": bool(front_past),
        "front_depth_m": depth,
        "front_lateral_m": lateral,
        "near_mouth_uncertainty": abs(depth) <= sigma,
        "finish_geometry": overlap > EPS or front_past,
        "whole_body_strict": receiver.contains(body) and receiver.boundary.disjoint(body),
    }


class ParkingHold:
    """Latch eligibility after the existing <0.05 m/s, one-second held stop."""

    def __init__(self, finish, *, speed_mps=0.05, dwell_s=1.0):
        if not all(math.isfinite(value) and value > 0 for value in (speed_mps, dwell_s)):
            raise ValueError("Parking hold thresholds must be finite and positive")
        self.finish, self.speed_mps, self.dwell_s = finish, speed_mps, dwell_s
        self.since = None
        self.last_t = None

    def assess(self, t, pose, speed_mps, *, held):
        if not math.isfinite(t) or not math.isfinite(speed_mps):
            raise ValueError("Parking samples must be finite")
        if self.last_t is not None and t < self.last_t:
            raise ValueError("Parking samples must have nondecreasing timestamps")
        if not isinstance(held, bool):
            raise ValueError("Held must be an explicit sensor-verified boolean")
        self.last_t = t
        evidence = parking_geometry(pose, self.finish)
        if held and evidence["finish_geometry"] and abs(speed_mps) < self.speed_mps:
            if self.since is None:
                self.since = t
        else:
            self.since = None
        evidence["held_s"] = 0.0 if self.since is None else t - self.since
        evidence["operator_like_success"] = (
            self.since is not None and evidence["held_s"] >= self.dwell_s - EPS
        )
        return evidence


def virtual_western_excursion(pose, scene, *, uncertainty_m=0.5):
    """Finite, locally closest virtual-edge excursion; no other departure is relabelled."""
    import numpy as np
    from shapely import distance, points
    from shapely.geometry import LineString

    virtual = scene["measurement"]["virtual_boundary"]
    boundary = scene["boundaries"]["right"]
    vertices = [
        next(m for m in scene["markers"] if m["id"] == virtual["first_marker_id"]),
        *virtual["vertices"],
        next(m for m in scene["markers"] if m["id"] == virtual["last_marker_id"]),
    ]
    edge = np.array(
        [[point["x"], point["y"]] if isinstance(point, dict) else point for point in vertices]
    )
    start = next(i for i, p in enumerate(boundary) if math.dist(p, edge[0]) < 1e-4)
    end = next(i for i, p in enumerate(boundary) if math.dist(p, edge[-1]) < 1e-4)
    other = LineString(scene["boundaries"]["left"])
    before, after = LineString(boundary[: start + 1]), LineString(boundary[end:])
    body = np.array(footprint(pose))
    fraction = np.linspace(0, 1, 40, endpoint=False)
    samples = (
        body[:, None, :] + (np.roll(body, -1, axis=0) - body)[:, None, :] * fraction[None, :, None]
    ).reshape(-1, 2)
    delta = np.diff(edge, axis=0)
    length2 = np.sum(delta**2, axis=1)
    if np.any(length2 <= EPS) or not math.isfinite(uncertainty_m) or uncertainty_m < 0:
        raise ValueError("Virtual edge needs distinct vertices and nonnegative uncertainty")
    relative = samples[:, None, :] - edge[:-1]
    projection = np.sum(relative * delta, axis=2) / length2
    closest = edge[:-1] + np.clip(projection, 0, 1)[:, :, None] * delta
    distances = np.linalg.norm(samples[:, None, :] - closest, axis=2)
    indices = np.argmin(distances, axis=1)
    nearest = distances[np.arange(len(samples)), indices]
    cloud = points(samples)
    others = np.minimum.reduce([distance(cloud, line) for line in (other, before, after)])
    gated = (
        (nearest <= others + EPS)
        & ((samples - edge[0]) @ delta[0] >= -EPS)
        & ((samples - edge[-1]) @ delta[-1] <= EPS)
    )
    offset = samples - closest[np.arange(len(samples)), indices]
    tangent = delta[indices] / np.sqrt(length2[indices, None])
    signed = tangent[:, 1] * offset[:, 0] - tangent[:, 0] * offset[:, 1]
    excursion = float(max(0.0, signed[gated].max())) if gated.any() else 0.0
    return {
        "virtual_span_active": bool(gated.any()),
        "virtual_excursion_m": excursion,
        "strict_virtual_score": excursion > 0.5 + EPS,
        "virtual_backstop": excursion > 3.0 + EPS,
        "backstop_near_sampling_resolution": (
            bool(gated.any()) and abs(excursion - 3.0) <= LENGTH_M / 80
        ),
        "backstop_possible_under_margin": excursion + uncertainty_m > 3.0 + EPS,
        "backstop_robust_under_margin": excursion - uncertainty_m > 3.0 + EPS,
        "body_edge_half_sample_spacing_m": LENGTH_M / 80,
    }
