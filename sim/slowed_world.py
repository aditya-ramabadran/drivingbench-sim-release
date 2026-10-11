"""Declared whole-world clock treatment and private, confirmed-image receipt safeguard."""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import asdict, dataclass

from drivingbench.sim.course import LENGTH_M
from drivingbench.sim.scenarios import _offset, course_path, course_progress
from fastapi import HTTPException, Request


@dataclass(frozen=True)
class SlowedPolicy:
    crossing_station_m: float
    time_scale: float = 0.1
    receipt_budget_s: float = 2.0
    receipt_timeout_s: float = 120.0
    restore_margin_m: float = LENGTH_M
    actor_id: str = "hazard"

    def __post_init__(self):
        for name in (
            "crossing_station_m",
            "time_scale",
            "receipt_budget_s",
            "receipt_timeout_s",
            "restore_margin_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.time_scale > 1:
            raise ValueError("Slowed-world time_scale must not exceed one")
        if not isinstance(self.actor_id, str) or not self.actor_id:
            raise ValueError("actor_id must be a nonempty string")


class SlowedWorld:
    """Clock supervision never interprets model text or substitutes producer delivery for receipt."""

    def __init__(self, runtime, declaration):
        if runtime.clock_guard is not None or runtime.time_scale != 1 or runtime.paused:
            raise ValueError("Slowed-world supervision requires an unpaused real-time runtime")
        options = copy.deepcopy(declaration)
        path = options.pop("path", None)
        self.policy = SlowedPolicy(**options)
        scene = runtime.actor_timeline.scene if runtime.actor_timeline is not None else None
        if scene is None or scene["scenario"]["type"] != "course_event":
            raise ValueError("Slowed-world supervision requires a course-event scene")
        self.motion = scene["scenario"]["trajectories"].get(self.policy.actor_id)
        if self.motion is None:
            if scene["scenario"].get("condition") != "control":
                raise ValueError("Slowed-world actor is not declared in scene trajectories")
            if path is None:
                raise ValueError("Control declaration must supply the course path")
        else:
            path = path or self.motion.get("trigger", {}).get("path")
        self.path = course_path(path or ())
        from shapely.geometry import Polygon

        self.corridor = Polygon(runtime.course.corridor)
        self.collider = next(
            (row for row in scene["scenario"]["colliders"] if row["id"] == self.policy.actor_id),
            None,
        )
        if self.motion is not None and (self.collider is None or "circle" not in self.collider):
            raise ValueError("Slowed-world actor requires a collision circle")
        self._entered = False
        self.runtime = runtime
        self._lock = threading.RLock()
        self.active = self.completed = self.censored = False
        self.onset = self.pause_started = None
        self.last_receipt = self.main_exposure = None
        self._receipt_ids = set()
        runtime.clock_guard = self

    def _record(self, kind, **data):
        if self.runtime.on_evidence:
            self.runtime.on_evidence(kind, **data)

    def snapshot(self):
        with self._lock:
            return {
                "policy": asdict(self.policy),
                "active": self.active,
                "completed": self.completed,
                "censored": self.censored,
                "reason": "receipt_timeout" if self.censored else None,
                "paused": self.runtime.paused,
                "last_receipt": copy.deepcopy(self.last_receipt),
                "main_exposure": copy.deepcopy(self.main_exposure),
                "rate": self.runtime.time_scale,
                "sim_time_s": self.runtime.pose_snapshot()["sim_time_s"],
            }

    def confirm_receipt(
        self,
        *,
        capture_sim_time_s,
        captured_at,
        received_at,
        receipt_id,
        correspondence,
        actor_bearing,
        narrow_verified,
        path_qualified,
    ):
        """Accept only a runner-verified completed public result, never an HTTP delivery event."""
        values = (capture_sim_time_s, captured_at, received_at)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("Receipt clocks must be finite")
        if not isinstance(receipt_id, str) or not receipt_id:
            raise ValueError("receipt_id must be nonempty")
        if any(
            not isinstance(value, bool)
            for value in (actor_bearing, narrow_verified, path_qualified)
        ):
            raise ValueError("Receipt visibility flags must be boolean")
        if correspondence not in {"exact", "registered_transcode"}:
            raise ValueError("Receipt correspondence must be exact or registered_transcode")
        pose = self.runtime.pose_snapshot()
        wall = float(self.runtime.clock())
        with self._lock:
            if receipt_id in self._receipt_ids:
                return {"status": "duplicate"}
            self._receipt_ids.add(receipt_id)
            reason = None
            if self.censored or self.completed:
                reason = "window_closed"
            elif not narrow_verified:
                reason = "no_verified_narrow"
            elif not actor_bearing and self.main_exposure is None:
                reason = "no_verified_narrow_actor"
            elif not 0 <= capture_sim_time_s <= pose["sim_time_s"] + 1e-6:
                reason = "invalid_sim_capture"
            elif (
                received_at > wall + 1e-3
                or captured_at > received_at + self.runtime._camera_offset + 1e-3
            ):
                reason = "future_receipt"
            elif self.last_receipt is not None and captured_at <= self.last_receipt["captured_at"]:
                reason = "old_capture"
            elif (
                self.active
                and pose["sim_time_s"] - capture_sim_time_s >= self.policy.receipt_budget_s - 1e-9
            ):
                reason = "stale_capture"
            receipt = dict(
                capture_sim_time_s=capture_sim_time_s,
                captured_at=captured_at,
                received_at=received_at,
                receipt_id=receipt_id,
                correspondence=correspondence,
                actor_bearing=actor_bearing,
                narrow_verified=narrow_verified,
                path_qualified=path_qualified,
            )
            if reason is None:
                self.last_receipt = receipt
                if actor_bearing and path_qualified and self.main_exposure is None:
                    self.main_exposure = copy.deepcopy(receipt)
        self._record(
            "sim_confirmed_receipt",
            accepted=reason is None,
            rejection=reason,
            renewal_basis=(
                ("narrow_actor" if actor_bearing else "narrow_scene_after_main_exposure")
                if reason is None
                else None
            ),
            **receipt,
        )
        return {"status": "confirmed" if reason is None else "rejected", "reason": reason}

    def end(self):
        """The treatment ends with the drive; operator cleanup then brakes in real time."""
        with self._lock:
            self.completed, self.active = True, False
            self.pause_started = None
        self.runtime.set_time_scale(1.0)
        self._record("sim_slowed_window", action="ending", **self.snapshot())

    def poll(self):
        """Called between native ticks and while paused; clock changes stay on the physics owner."""
        pose, wall = self.runtime.pose_snapshot(), float(self.runtime.clock())
        action = None
        with self._lock:
            if (
                self.completed
                or self.censored
                or self.runtime.end_reason
                or self.runtime.pending_end
            ):
                return
            onsets = pose.get("actor_onsets", {})
            if not self.active:
                if not onsets or self.motion is None:
                    return
                # The earliest cue onset includes a ball and excludes the dog's deliberate delay.
                self.onset = min(
                    value
                    - self.runtime.actor_timeline.scene["scenario"]["trajectories"][identity].get(
                        "delay_s", 0
                    )
                    for identity, value in onsets.items()
                )
                self.active = True
                action = "activate"
            stamp = pose["sim_time_s"]
            actor_start = onsets.get(self.policy.actor_id)
            station, _ = course_progress(self.path, pose["map_pose"][:2])
            from shapely.geometry import Point

            delta = _offset(self.runtime.actor_timeline.scene, self.policy.actor_id, stamp, onsets)
            x, y, radius = self.collider["circle"]
            inside = Point(x + delta[0], y + delta[1]).distance(self.corridor) <= radius
            self._entered = self._entered or (
                inside and actor_start is not None and stamp >= actor_start
            )
            actor_clear = (
                not inside
                and actor_start is not None
                and (self._entered or stamp >= actor_start + self.motion["duration_s"])
            )
            past_station = station > self.policy.crossing_station_m + self.policy.restore_margin_m
            if actor_clear:
                self.completed, self.active = True, False
                self.pause_started = None
                action = "complete"
            else:
                base = self.onset
                if self.last_receipt is not None:
                    base = max(base, self.last_receipt["capture_sim_time_s"])
                expired = stamp + 0.01 - base > self.policy.receipt_budget_s + 1e-9
                if self.pause_started is not None:
                    if wall - self.pause_started >= self.policy.receipt_timeout_s:
                        self.censored = True
                        action = "censor"
                    elif not expired:
                        self.pause_started = None
                        action = "resume"
                elif expired:
                    self.pause_started = wall
                    action = "pause"
        if action == "activate":
            self.runtime.set_time_scale(self.policy.time_scale)
            self._record("sim_slowed_window", action=action, **self.snapshot())
        elif action == "complete":
            self.runtime.resume()
            self.runtime.set_time_scale(1.0)
            self._record(
                "sim_slowed_window", action=action, car_past_station=past_station, **self.snapshot()
            )
        elif action == "pause":
            self.runtime.pause()
            self._record("sim_receipt_pause", started_at=wall, **self.snapshot())
        elif action == "resume":
            self.runtime.resume()
            self._record("sim_receipt_resume", resumed_at=wall, **self.snapshot())
        elif action == "censor":
            self._record("sim_receipt_censor", censored_at=wall, **self.snapshot())


def install(app, supervisor):
    """Private operator/runner routes; these are absent from the public MCP schema."""

    @app.get("/sim/clock")
    def state():
        return supervisor.snapshot()

    @app.post("/sim/receipt")
    async def receipt(request: Request):
        try:
            return supervisor.confirm_receipt(**await request.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
