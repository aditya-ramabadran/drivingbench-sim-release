"""A declared binary lane choice over validated, continuous map-space paths."""

from __future__ import annotations

import bisect
import math


class BranchSelection:
    def __init__(self, scene):
        scenario = scene["scenario"]
        config = scenario["branch_select"]
        self.decision_m = float(config["decision_m"])
        self.paths = {
            side: tuple(tuple(float(value) for value in row) for row in config["paths"][side])
            for side in ("left", "right")
        }
        if not math.isfinite(self.decision_m) or self.decision_m <= 0:
            raise ValueError("Branch decision distance must be finite and positive")
        for rows in self.paths.values():
            if (
                len(rows) < 3
                or any(len(row) != 4 or not all(map(math.isfinite, row)) for row in rows)
                or rows[0][0] != 0
                or any(b[0] <= a[0] for a, b in zip(rows, rows[1:]))
                or not any(abs(row[0] - self.decision_m) < 1e-8 for row in rows)
                or rows[-1][0] <= self.decision_m
            ):
                raise ValueError(
                    "Branch paths need increasing distances and an explicit decision point"
                )
        prefixes = [
            tuple(row for row in rows if row[0] <= self.decision_m + 1e-8)
            for rows in self.paths.values()
        ]
        if prefixes[0] != prefixes[1]:
            raise ValueError("Branch paths must share the entire approach through commitment")
        start = tuple(float(scene["start"][key]) for key in ("x", "y", "heading_deg"))
        if any(abs(a - b) > 1e-6 for a, b in zip(start, prefixes[0][0][1:])):
            raise ValueError("Branch path start must match the scene start")
        self.selected = scenario["default_branch"]
        if self.selected not in self.paths:
            raise ValueError("Unknown default branch")
        self.locked = None
        self.distance_m = 0.0
        self.requested_steering_percent = 0.0

    def command(self, motion):
        direction, percent = motion["direction"], motion["steering_percent"]
        self.requested_steering_percent = (
            percent * {"left": 1, "right": -1, "straight": 0}[direction]
        )
        if self.locked is None and direction in self.paths and percent > 0:
            self.selected = direction

    def advance(self, distance_m):
        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("Branch progress must be finite and nonnegative")
        self.distance_m += distance_m
        if self.locked is None and self.distance_m >= self.decision_m - 1e-8:
            self.locked = self.selected

    def snapshot(self):
        return dict(
            selected=self.selected,
            locked=self.locked,
            distance_m=self.distance_m,
            decision_m=self.decision_m,
            requested_steering_percent=self.requested_steering_percent,
        )

    def map_pose(self):
        rows = self.paths[self.locked or self.selected]
        if self.distance_m > rows[-1][0] + 1e-8:
            raise RuntimeError("Branch path exhausted before a terminal contact")
        index = max(
            0,
            min(len(rows) - 2, bisect.bisect_right([row[0] for row in rows], self.distance_m) - 1),
        )
        a, b = rows[index : index + 2]
        fraction = (self.distance_m - a[0]) / (b[0] - a[0])
        angle = (b[3] - a[3] + 180) % 360 - 180
        return (
            a[1] + fraction * (b[1] - a[1]),
            a[2] + fraction * (b[2] - a[2]),
            a[3] + fraction * angle,
        )

    def apply(self, native, start):
        x, y, heading = self.map_pose()
        theta = math.radians(start[2])
        dx, dy = x - start[0], y - start[1]
        native.car.x = dx * math.cos(theta) + dy * math.sin(theta)
        native.car.y = -dx * math.sin(theta) + dy * math.cos(theta)
        native.car.heading = math.radians(heading - start[2])
