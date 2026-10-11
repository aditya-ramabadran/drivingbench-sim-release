"""Reviewed paired-camera coverage, with explicit pose and asset boundaries."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
from drivingbench.sim.producer import file_hash
from shapely.geometry import Point, Polygon


def heading_matches(heading, cell):
    return abs((heading - cell["heading_deg"] + 180) % 360 - 180) <= cell["heading_tolerance_deg"]


def can_connect(first, second, max_gap_m):
    (a, pa), (b, pb) = first, second
    delta = abs((a["heading_deg"] - b["heading_deg"] + 180) % 360 - 180)
    return (
        a["narrow"] == a["wide"] == b["narrow"] == b["wide"] == "pass"
        and delta <= a["heading_tolerance_deg"] + b["heading_tolerance_deg"]
        and pa.distance(pb) <= max_gap_m
    )


def connect_gaps(source, max_gap_m=0.5):
    """Declare bounded connections between paired-pass cells with overlapping yaw guards."""
    coverage = CoverageCertificate(source)
    data = deepcopy(coverage.data)
    pairs = []
    for first, second in combinations(coverage.cells, 2):
        if can_connect(first, second, max_gap_m):
            pairs.append([first[0]["id"], second[0]["id"]])
    data["gap_connections"] = {"max_gap_m": max_gap_m, "pairs": pairs}
    CoverageCertificate(data)
    return data


def local_pose(map_pose, start):
    """Map body pose to renderer-local x/y/heading degrees."""
    x, y, heading = map(float, map_pose)
    sx, sy, sh = map(float, start)
    c, s = math.cos(math.radians(sh)), math.sin(math.radians(sh))
    dx, dy = x - sx, y - sy
    return c * dx + s * dy, -s * dx + c * dy, heading - sh


def stations(x, y, heading, spacing_m=2.0):
    """Sample travelled-distance stations, preserving the final pose and yaw wraps."""
    values = np.asarray([x, y, heading], dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != 3
        or values.shape[1] < 2
        or not np.isfinite(values).all()
        or not math.isfinite(spacing_m)
        or spacing_m <= 0
    ):
        raise ValueError("Finite equal-length path arrays and positive spacing are required")
    distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(values[0]), np.diff(values[1])))]
    keep = np.r_[np.diff(distance) > 1e-8, True]
    distance, values = distance[keep], values[:, keep]
    if distance[-1] == 0:
        return np.array([[values[0, -1], values[1, -1], values[2, -1], 0.0]])
    query = np.unique(np.r_[np.arange(0, distance[-1], spacing_m), distance[-1]])
    return np.column_stack(
        [np.interp(query, distance, row) for row in (values[0], values[1], np.unwrap(values[2]))]
        + [query]
    )


class CoverageCertificate:
    """Fail closed unless a paired, reviewed spatial cell and heading guard both pass."""

    def __init__(self, source):
        self.data = (
            json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else source
        )
        if self.data.get("schema") != 1 or self.data.get("map_frame") != "site1_full_ground":
            raise ValueError("Unknown rendered coverage certificate")
        if not isinstance(self.data.get("approved"), bool):
            raise ValueError("Coverage approval must be explicit")
        self.cells = []
        identities = set()
        for row in self.data.get("cells", []):
            identity = row["id"]
            polygon = Polygon(row["polygon_xy"])
            center = float(row["heading_deg"])
            tolerance = float(row["heading_tolerance_deg"])
            if (
                identity in identities
                or not polygon.is_valid
                or polygon.area <= 0
                or not np.isfinite(polygon.exterior.coords).all()
                or not math.isfinite(center)
                or not 0 < tolerance <= 180
            ):
                raise ValueError("Invalid coverage cell")
            identities.add(identity)
            for camera in ("narrow", "wide"):
                if row.get(camera) not in ("pass", "fail", "unknown"):
                    raise ValueError("Every cell needs both camera reviews")
            self.cells.append((row, polygon))
        self.paired = [
            (row, polygon) for row, polygon in self.cells if row["narrow"] == row["wide"] == "pass"
        ]
        self.connections = []
        declaration = self.data.get("gap_connections")
        if declaration is not None:
            self.max_gap_m = float(declaration["max_gap_m"])
            if not math.isfinite(self.max_gap_m) or not 0 < self.max_gap_m <= 0.5:
                raise ValueError("Gap connections must be bounded at no more than 0.5 m")
            indexed = {row["id"]: (row, polygon) for row, polygon in self.cells}
            seen = set()
            for pair in declaration["pairs"]:
                if len(pair) != 2 or pair[0] == pair[1] or any(key not in indexed for key in pair):
                    raise ValueError("Gap connections need two distinct existing cells")
                identity = tuple(sorted(pair))
                if identity in seen:
                    raise ValueError("Duplicate gap connection")
                seen.add(identity)
                first, second = (indexed[key] for key in pair)
                a, b = first[1], second[1]
                if not can_connect(first, second, self.max_gap_m):
                    raise ValueError("Only nearby paired-pass cells can connect")
                self.connections.append((first, second, a.union(b).convex_hull))

    def validate_assets(self, root):
        hashes = self.data.get("source_hashes", {})
        review = self.data.get("review_manifest")
        if not hashes or not review or not self.data.get("review_manifest_sha256"):
            raise ValueError("Rendered coverage needs immutable asset and review provenance")
        for path, expected in hashes.items():
            if file_hash(Path(root) / path) != expected:
                raise ValueError(f"Coverage asset changed: {path}")
        if file_hash(Path(root) / review) != self.data["review_manifest_sha256"]:
            raise ValueError("Coverage review changed")
        return self

    def measure(self, map_pose):
        """Report strict review membership and independent body-reference spatial/yaw excess."""
        if not self.data["approved"] or not self.paired:
            raise ValueError("Coverage measurement requires approved nonempty paired reviews")
        support = self(map_pose)
        x, y, heading = map(float, map_pose)
        point = Point(x, y)
        candidates = [
            (
                polygon.distance(point),
                max(
                    0.0,
                    abs((heading - row["heading_deg"] + 180) % 360 - 180)
                    - row["heading_tolerance_deg"],
                ),
                row["id"],
            )
            for row, polygon in self.paired
        ]
        distance, yaw, identity = min(candidates)
        yaw_compatible = [distance for distance, yaw, _ in candidates if yaw == 0]
        return dict(
            support,
            distance_to_any_paired_pass_parent_m=distance,
            nearest_spatial_parent=identity,
            yaw_excess_at_nearest_parent_deg=yaw,
            nearest_yaw_compatible_parent_distance_m=(
                min(yaw_compatible) if yaw_compatible else None
            ),
        )

    def __call__(self, map_pose):
        pose = np.asarray(map_pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("Expected finite map x/y/heading degrees")
        if not self.data["approved"]:
            return {"supported": False, "reason": "coverage_not_approved", "cells": []}
        point = Point(*pose[:2])
        eligible = {row["id"] for row, _ in self.cells if heading_matches(pose[2], row)}
        matching = []
        for row, polygon in self.cells:
            if (
                row["narrow"] == row["wide"] == "pass"
                and row["id"] in eligible
                and polygon.covers(point)
            ):
                matching.append(row["id"])
        if not matching and self.connections:
            excluded = any(
                row["id"] in eligible
                and (row["narrow"] != "pass" or row["wide"] != "pass")
                and polygon.covers(point)
                for row, polygon in self.cells
            )
            if not excluded:
                for (first, a), (second, b), hull in self.connections:
                    if (
                        first["id"] in eligible
                        and second["id"] in eligible
                        and hull.covers(point)
                        and a.distance(point) + b.distance(point) <= self.max_gap_m
                    ):
                        return {
                            "supported": True,
                            "reason": "reviewed_connection",
                            "cells": [first["id"], second["id"]],
                        }
        return {
            "supported": bool(matching),
            "reason": "reviewed_pair" if matching else "outside_reviewed_pose_envelope",
            "cells": matching,
        }
