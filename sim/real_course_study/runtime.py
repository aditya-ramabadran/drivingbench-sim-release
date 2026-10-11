"""Measured-course geometry and held parking around the unchanged live runtime."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from drivingbench.sim.course import Course, _pose
from drivingbench.sim.episode import EpisodePolicy
from drivingbench.sim.real_course_parking_analysis import (
    parking_geometry,
    virtual_western_excursion,
)
from drivingbench.sim.real_course_study.western import WesternBackstop
from drivingbench.sim.runtime import Runtime

PARKING_CONFIRMATION_S = 5.0
LOT_SAFETY_DISTANCE_M = 5.0


@dataclass(frozen=True)
class RealCoursePolicy(EpisodePolicy):
    stop_on_marker_contact: bool = False
    stop_on_exit: bool = True
    strict_boundary_tolerance_m: float | None = 0.5
    time_limit_s: float = 660.0

    def finish_eligible(self, evidence):
        return bool(
            evidence["finish_reached"]
            and evidence.get("held") is True
            and not evidence.get("positive_motion_command")
            and not evidence.get("obstacle_collision_ids")
            and not evidence.get("coverage_exit")
            and not evidence.get("lot_safety_backstop")
            and not evidence["boundary_violation"]
            and abs(evidence["speed_mps"]) < self.finish_speed_mps
        )


class RealCourse(Course):
    """Dome/body and interpolated marker spans are scores, not physical walls."""

    def __init__(self, scene, *, western_score=virtual_western_excursion):
        super().__init__(scene, boundary_tolerance_m=0.5, strict_boundary_tolerance_m=0.5)
        self.scene, self.western_score = scene, western_score
        self.western_guard = (
            WesternBackstop.from_scene(scene)
            if western_score is virtual_western_excursion
            else None
        )
        self.boundary_rule = {
            **self.boundary_rule,
            "live_rule": "virtual western excursion >3 m; reviewed coverage is reported; lot backstop >5 m",
            "strict_reference": "any dome/body contact or >0.50 m whole-corridor excursion",
            "measured_span_rule": "score only; interpolated marker rows are not physical walls",
        }

    def assess(self, pose):
        result = super().assess(pose)
        parking = parking_geometry(pose, self.scene["finish"])
        western = self.western_score(pose, self.scene)
        if self.western_guard is not None:
            western = {
                **western,
                "offline_sampled_backstop": western["virtual_backstop"],
                **self.western_guard(pose),
            }
        result.update(
            strict_whole_body_finish=parking["whole_body_strict"],
            parking=parking,
            western=western,
            boundary_violation=western["virtual_backstop"],
            finish_reached=parking["finish_geometry"],
            valid_finish=parking["finish_geometry"] and not result["obstacle_collision_ids"],
        )
        return result


def runtime_factory(
    scene,
    coverage,
    *,
    registered_start=None,
    native_factory=None,
    western_score=None,
    course_factory=RealCourse,
):
    """Bind reviewed body-pose measurements; missing spatial evidence has no default."""
    if not callable(getattr(coverage, "measure", None)):
        raise ValueError("A rendered-coverage measurement is required")

    registered_start = None if registered_start is None else _pose(registered_start)

    class RealCourseRuntime(Runtime):
        def __init__(self, course, **kwargs):
            self._parking_deadline = None
            options = {} if native_factory is None else {"native_factory": native_factory}
            if western_score is not None:
                options["western_score"] = western_score
            native_options = {k: v for k, v in options.items() if k == "native_factory"}
            course_options = {k: v for k, v in options.items() if k == "western_score"}
            measured = course_factory(scene, **course_options)
            measured.start = course.start if registered_start is None else registered_start
            super().__init__(measured, **native_options, **kwargs)

        def _capture(self):
            status, evidence = super()._capture()
            evidence["native_hold"] = {
                key: status.get(key)
                for key in ("held", "standstill", "sensors_valid", "car_state_valid")
            }
            evidence["held"] = all(value is True for value in evidence["native_hold"].values())
            command = status.get("command")
            evidence["positive_motion_command"] = bool(command and command["speed_mps"] > 0)
            evidence["parking_confirmation_deadline_s"] = (
                None if self._parking_deadline is None else self._parking_deadline - self.started_at
            )
            support = coverage.measure(evidence["map_pose"])
            if not isinstance(support, dict) or not isinstance(support.get("supported"), bool):
                raise ValueError("Coverage measurement must return an explicit supported boolean")
            distance = support.get("distance_to_any_paired_pass_parent_m")
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (float, int))
                or not math.isfinite(distance)
                or distance < 0
            ):
                raise ValueError("Coverage measurement needs a finite nonnegative spatial distance")
            evidence["rendered_coverage"] = support
            evidence["coverage_exit"] = None
            evidence["lot_safety_backstop"] = support if distance > LOT_SAFETY_DISTANCE_M else None
            return status, evidence

        def finish(self, reason):
            if reason != "client_done":
                return super().finish(reason)
            with self._condition:
                if self._closed:
                    raise OSError("simulator_closed")
                if self.pending_end or self.end_reason or self._parking_deadline is not None:
                    return
                self._parking_deadline = self.native.time + PARKING_CONFIRMATION_S
                self.native.brake = True
                evidence = copy.deepcopy(self._pose)
                deadline = self._parking_deadline - self.started_at
                self._condition.notify_all()
            if self.on_evidence:
                self.on_evidence(
                    "sim_parking_confirmation", deadline_sim_time_s=deadline, **evidence
                )

        def _prepare(self, request, transformed):
            if self._parking_deadline is not None and request.body.get("operation") == "motion":
                request.replies.append({"status": "rejected", "reason": "operator_brake"})
                return None
            return super()._prepare(request, transformed)

        def _tick(self):
            super()._tick()
            with self._condition:
                expired = (
                    self._parking_deadline is not None
                    and self.native.time >= self._parking_deadline - 1e-9
                )
            if expired:
                super().finish("client_done")

    return RealCourseRuntime
