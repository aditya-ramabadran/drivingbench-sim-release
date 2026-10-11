"""Offline live-plant response draws and empirical marginal prediction regions."""

from __future__ import annotations

import numpy as np
from drivingbench.sim.plant import Corolla


class LiveResponseCorolla(Corolla):
    """Sample measured response variation without altering the live plant equations."""

    def __init__(self, p, curvature, dt=0.01, *, draw, **state):
        super().__init__(draw.params, curvature, dt, **state)
        self.draw, self.draw_requested = draw, False

    def step(self, accel_cmd, torque, lat_active, long_active, brake=False):
        requesting = self.standstill and long_active and accel_cmd > 0
        if requesting and not self.draw_requested:
            self.p = self.draw.launch(bool(self.hold))
            self.draw_requested = True
        if not requesting:
            self.draw_requested = False
        return super().step(accel_cmd, torque, lat_active, long_active, brake)


def wrapped(angle):
    """Signed angular displacement in [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def regions(draws, truth, levels=(0.5, 0.9)):
    """Empirical ellipses and circular-centered intervals, not joint nominal regions."""
    draws, truth = np.asarray(draws, dtype=float), np.asarray(truth, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if (
        draws.ndim != 2 or draws.shape[1] != 3 or len(draws) < 3 or truth.shape != (3,)
        or not np.isfinite(draws).all() or not np.isfinite(truth).all()
        or levels.ndim != 1 or not np.isfinite(levels).all()
        or np.any((levels <= 0) | (levels >= 1))
    ):  # fmt: skip
        raise ValueError("Need finite pose draws, a pose truth and probabilities inside (0, 1)")
    center = draws[:, :2].mean(axis=0)
    covariance = np.cov(draws[:, :2], rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    # A 0.1 mm numerical floor permits deterministic/collinear predictions without padding them.
    values = np.maximum(values, 1e-8)
    precision = (vectors / values) @ vectors.T
    delta = draws[:, :2] - center
    radius2 = np.einsum("ni,ij,nj->n", delta, precision, delta)
    thresholds = np.quantile(radius2, levels)
    actual = truth[:2] - center
    actual_radius2 = float(actual @ precision @ actual)
    resultant = np.mean(np.exp(1j * draws[:, 2]))
    heading_center = float(np.angle(resultant)) if abs(resultant) > 1e-12 else 0.0
    angular = wrapped(draws[:, 2] - heading_center)
    truth_angular = float(wrapped(truth[2] - heading_center))
    bounds = np.quantile(angular, np.column_stack(((1 - levels) / 2, (1 + levels) / 2)))
    rows = []
    for level, threshold, (low, high) in zip(levels, thresholds, bounds):
        position_in = radius2 <= threshold + 1e-12
        heading_in = (angular >= low - 1e-12) & (angular <= high + 1e-12)
        pc = actual_radius2 <= threshold + 1e-12
        hc = low - 1e-12 <= truth_angular <= high + 1e-12
        rows.append(
            {
                "level": float(level),
                "position_covered": bool(pc),
                "heading_covered": bool(hc),
                "intersection_covered": bool(pc and hc),
                "in_draw_position_mass": float(position_in.mean()),
                "in_draw_heading_mass": float(heading_in.mean()),
                "in_draw_intersection_mass": float((position_in & heading_in).mean()),
                "ellipse_axes_m": np.sqrt(values * threshold).tolist(),
                "heading_offsets_deg": np.degrees([low, high]).tolist(),
            }
        )
    return {
        "position_mean": center.tolist(),
        "position_covariance": covariance.tolist(),
        "ellipse_eigenvectors": vectors.tolist(),
        "heading_center_rad": heading_center,
        "heading_resultant_length": float(abs(resultant)),
        "truth_mahalanobis2": actual_radius2,
        "mean_position_error_m": float(np.linalg.norm(actual)),
        "mean_heading_error_deg": float(abs(np.degrees(truth_angular))),
        "bands": rows,
    }
