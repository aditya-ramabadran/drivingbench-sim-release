"""Opted-in approved course interpretation; original measured-course defaults stay separate."""

from __future__ import annotations

import math
from dataclasses import dataclass

from drivingbench.sim.course import EPS, LENGTH_M, WIDTH_M, Course, _pose
from drivingbench.sim.real_course_parking_analysis import (
    parking_geometry,
    virtual_western_excursion,
)
from drivingbench.sim.real_course_study.runtime import RealCoursePolicy
from drivingbench.sim.real_course_study.runtime import runtime_factory as measured_runtime_factory
from drivingbench.sim.real_course_study.western import WesternBackstop

EXCURSION_M = 2.5
FRONT_GAP_M = 1.0


@dataclass(frozen=True)
class RefinedPolicy(RealCoursePolicy):
    boundary_tolerance_m: float = EXCURSION_M


class RefinedCourse(Course):
    """Smooth corridor, raw marker circles and finite-width whole-bumper parking."""

    def __init__(self, scene, *, western_score=virtual_western_excursion):
        from shapely.geometry import LineString, Polygon

        super().__init__(scene, boundary_tolerance_m=EXCURSION_M, strict_boundary_tolerance_m=0.5)
        self.scene = scene
        raw = scene.get("refinement", {}).get("raw_boundaries")
        if raw is None:
            raise ValueError("Refined course requires its unchanged raw boundary reference")
        self.western_score = western_score
        self.western_guard = (
            WesternBackstop.from_scene(scene)
            if western_score is virtual_western_excursion
            else None
        )
        self.receiver = Polygon(scene["finish"]["polygon"])
        mouth = scene["finish"]["mouth"]
        self.mouth_left, right = tuple(mouth["left"]), tuple(mouth["right"])
        self.mouth_width = math.dist(self.mouth_left, right)
        if not math.isfinite(self.mouth_width) or self.mouth_width <= EPS:
            raise ValueError("Parking needs a finite nonzero mouth")
        self.across = tuple((right[i] - self.mouth_left[i]) / self.mouth_width for i in (0, 1))
        inward = (-self.across[1], self.across[0])
        centroid = self.receiver.centroid
        sign = (
            1
            if sum((centroid.coords[0][i] - self.mouth_left[i]) * inward[i] for i in (0, 1)) > 0
            else -1
        )
        self.inward = tuple(sign * value for value in inward)
        self.finite_mouth = LineString([self.mouth_left, right])
        self.boundary_rule.update(
            live_rule="whole body more than 2.5 m beyond approved smooth corridor",
            strict_reference="raw dome contact or more than 0.50 m beyond smooth corridor",
            raw_reference="unchanged raw boundaries retained in scene provenance for offline scoring",
            legacy_guards="western >3 m and reviewed-parent >5 m are private diagnostic scores",
        )

    def parking(self, pose):
        from shapely.geometry import Point

        evidence = parking_geometry(pose, self.scene["finish"])
        heading = math.radians(pose[2])
        direction = (math.cos(heading), math.sin(heading))
        leftward = (-direction[1], direction[0])
        front = tuple(pose[i] + LENGTH_M / 2 * direction[i] for i in (0, 1))
        bumper = [
            tuple(front[i] + side * WIDTH_M / 2 * leftward[i] for i in (0, 1)) for side in (-1, 1)
        ]
        laterals = sorted(
            sum((point[i] - self.mouth_left[i]) * self.across[i] for i in (0, 1))
            for point in bumper
        )
        depth = sum((front[i] - self.mouth_left[i]) * self.inward[i] for i in (0, 1))
        facing = sum(direction[i] * self.inward[i] for i in (0, 1)) > EPS
        width_clear = EPS < laterals[0] and laterals[-1] < self.mouth_width - EPS
        front_point = Point(front)
        inside = self.receiver.contains(front_point)
        before = width_clear and -FRONT_GAP_M - EPS <= depth <= EPS
        return {
            **evidence,
            "legacy_area_or_centre_geometry": evidence["finish_geometry"],
            "front_entire_bumper_within_mouth_width": width_clear,
            "front_bumper_lateral_range_m": laterals,
            "front_inside_U": inside,
            "front_before_finite_mouth": before,
            "front_distance_to_finite_mouth_m": front_point.distance(self.finite_mouth),
            "forward_facing": facing,
            "finish_geometry": bool(facing and width_clear and (inside or before)),
            "geometry_rule": "whole bumper within finite mouth width; front centre inside U or at most 1 m before it, facing inward",
        }

    def assess(self, pose):
        pose = _pose(pose)
        evidence = super().assess(pose)
        parking = self.parking(pose)
        western = (
            self.western_guard(pose)
            if self.western_guard is not None
            else self.western_score(pose, self.scene)
        )
        evidence.update(
            parking=parking,
            western={**western, "score_only": True},
            strict_whole_body_finish=parking["whole_body_strict"],
            finish_reached=parking["finish_geometry"],
            valid_finish=parking["finish_geometry"] and not evidence["obstacle_collision_ids"],
        )
        return evidence


def runtime_factory(
    scene, coverage, *, registered_start=None, native_factory=None, western_score=None
):
    """Opt into refined geometry while sharing the original held/capture/confirmation lifecycle."""
    base = measured_runtime_factory(
        scene,
        coverage,
        registered_start=registered_start,
        native_factory=native_factory,
        western_score=western_score,
        course_factory=RefinedCourse,
    )

    class RefinedRuntime(base):
        def _capture(self):
            status, evidence = super()._capture()
            evidence["legacy_lot_safety_backstop"] = evidence["lot_safety_backstop"]
            evidence["lot_safety_backstop"] = None
            return status, evidence

    return RefinedRuntime
