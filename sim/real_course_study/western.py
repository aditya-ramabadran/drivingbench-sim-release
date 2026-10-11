"""Finite outward body-area guard for the separately approved virtual western edge."""

from __future__ import annotations

import math

from drivingbench.sim.course import EPS, footprint


class WesternBackstop:
    """Score a body polygon against a 3 m finite outward region, not edge samples."""

    def __init__(self, vertices, *, scene_extent_m=200.0, threshold_m=3.0, quad_segs=64):
        from shapely.geometry import LineString
        from shapely.prepared import prep

        if (
            not math.isfinite(scene_extent_m)
            or not math.isfinite(threshold_m)
            or not 0 < threshold_m < scene_extent_m
            or not isinstance(quad_segs, int)
            or quad_segs < 8
        ):
            raise ValueError("Backstop needs finite positive extents and adequate arc resolution")
        line = LineString(vertices)
        if not line.is_simple or line.length <= EPS:
            raise ValueError("Virtual span must be a finite simple polyline")
        self.threshold_m = threshold_m
        self.arc_allowance_m = threshold_m * (1 / math.cos(math.pi / (4 * quad_segs)) - 1)
        # Inflate the round buffer until each inscribed chord covers the exact-radius disk.
        inner = line.buffer(threshold_m + self.arc_allowance_m, quad_segs=quad_segs)
        outer = line.buffer(
            -scene_extent_m, single_sided=True, join_style="round", quad_segs=quad_segs
        )
        self.region = outer.difference(inner)
        self.prepared = prep(self.region)

    @classmethod
    def from_scene(cls, scene):
        virtual = scene["measurement"]["virtual_boundary"]
        markers = {row["id"]: row for row in scene["markers"]}
        first, last = (markers[virtual[key]] for key in ("first_marker_id", "last_marker_id"))
        vertices = [[first["x"], first["y"]], *virtual["vertices"], [last["x"], last["y"]]]
        coordinates = scene["boundaries"]["left"] + scene["boundaries"]["right"]
        extent = max(math.dist(a, b) for a in coordinates for b in coordinates) + 50
        return cls(vertices, scene_extent_m=extent)

    def __call__(self, pose):
        from shapely.geometry import Polygon

        body = Polygon(footprint(pose))
        area = body.intersection(self.region).area if self.prepared.intersects(body) else 0.0
        return {
            "virtual_backstop": area > EPS,
            "backstop_outward_overlap_m2": area,
            "backstop_threshold_m": self.threshold_m,
            "backstop_arc_allowance_m": self.arc_allowance_m,
            "backstop_geometry": "finite outward single-sided band; flat span caps; round-distance guard",
            "backstop_locality_difference": "No offline nearest-other-boundary ownership gate; rendered coverage gates reachable poses separately",
        }
