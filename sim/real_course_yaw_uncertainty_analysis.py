"""Training-only yaw residual blocks for offline live-plant uncertainty checks."""

from __future__ import annotations

import math

import numpy as np
from drivingbench.sim.real_course_uncertainty_analysis import LiveResponseCorolla


def yaw_stratum(speed, wheel_deg):
    """Speed and wheel-angle bins declared before held-out evaluation."""
    return 3 * int(np.searchsorted((1.0, 2.0), speed, side="right")) + int(
        np.searchsorted((60.0, 120.0), abs(wheel_deg), side="right")
    )


class ResidualYaw:
    """One correlated run effect and consecutive one-second residual blocks."""

    def __init__(self, rng, groups, blocks, group_id):
        self.rng, self.blocks = rng, blocks
        self.group = next((x for x in groups if x["id"] == group_id), None)
        if self.group is None:
            self.group = groups[int(rng.integers(len(groups)))]
        self.candidates = {}
        for stratum in range(9):
            candidates = [x for x in blocks if x["stratum"] == stratum]
            same = [x for x in candidates if x["group"] == self.group["id"]]
            self.candidates[stratum] = same or candidates
        self.index, self.block, self.stratum = 100, None, None

    def rate(self, speed, wheel_deg):
        if speed <= 0.05:
            self.index, self.block, self.stratum = 100, None, None
            return 0.0
        stratum = yaw_stratum(speed, wheel_deg)
        if self.index >= 100 or stratum != self.stratum:
            candidates = self.candidates[stratum]
            if not candidates:
                return 0.0
            self.block = candidates[int(self.rng.integers(len(candidates)))]
            self.index, self.stratum = 0, stratum
        value = self.block["centered_rad_s"][self.index]
        self.index += 1
        return float(value + self.group["bias_rad_s"][stratum])


class UncertainCorolla(LiveResponseCorolla):
    """Unchanged actuators, with offline sampled pose-process yaw residuals."""

    def __init__(self, *args, noise, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise = noise

    def step(self, *args, **kwargs):
        previous = self.x, self.y, self.heading
        super().step(*args, **kwargs)
        self.heading += self.noise.rate(self.speed, self.steer_deg) * self.dt
        self.x = previous[0] + self.speed * math.cos(self.heading) * self.dt
        self.y = previous[1] + self.speed * math.sin(self.heading) * self.dt


def perturb_start(start, residual):
    """Apply a body-frame start error without changing its registered mean."""
    start, residual = np.asarray(start), np.asarray(residual)
    c, s = np.cos(start[2]), np.sin(start[2])
    delta = np.array(((c, -s), (s, c))) @ residual[:2]
    return [start[0] + delta[0], start[1] + delta[1], start[2] + residual[2]]
