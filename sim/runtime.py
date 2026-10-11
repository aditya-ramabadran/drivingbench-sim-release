"""Clocked owner of SimNative; producer calls never advance physics."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, field, replace

from drivingbench.device.cameras import camera_clock
from drivingbench.sim.car import DT, SimNative
from drivingbench.sim.scenarios import course_progress


@dataclass(eq=False)
class _Request:
    body: dict
    arrived_at: float
    replies: list = field(default_factory=list)


class Runtime:
    """100 Hz physics, queued native RPC, and short-lived immutable pose snapshots.

    Callbacks run outside the physics lock and must enqueue work without blocking.
    Manual advance uses the same tick path, without starting the background thread.
    """

    def __init__(
        self,
        course,
        *,
        clock=time.monotonic,
        frame_clock=camera_clock,
        time_scale=1.0,
        timeout=5.0,
        native_factory=SimNative,
        motion_transform=None,
        restrictions=None,
        paused=False,
        scenario=None,
    ):
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        self.course, self.clock, self.timeout = course, clock, timeout
        self.time_scale = float(time_scale)
        self.restrictions = restrictions
        if restrictions is not None:
            if motion_transform is not None or native_factory is not SimNative:
                raise ValueError("Restrictions own the native factory and motion transform")
            native_factory = restrictions.native_factory()
            motion_transform = restrictions.transform
        self.branch_selection = None
        if restrictions is not None and restrictions.branch_select:
            from drivingbench.sim.branch_select import BranchSelection

            if scenario is None or scenario.get("scenario", {}).get("type") != "fork":
                raise ValueError("Branch selection requires a declared fork scene")
            self.branch_selection = BranchSelection(scenario)
        self.actor_timeline = None
        self.coverage = None
        if scenario is not None:
            from drivingbench.sim.scenarios import ActorTimeline

            self.actor_timeline = ActorTimeline(scenario)
            if "coverage" in scenario["scenario"]:
                from drivingbench.sim.hazard import CoverageEnvelope

                self.coverage = CoverageEnvelope(scenario["scenario"]["coverage"])
        self.brake_fault_path = None
        if restrictions is not None and restrictions.brake_fault_station_m is not None:
            if scenario is None or scenario.get("scenario", {}).get("type") != "course_event":
                raise ValueError("Progress braking requires an explicit course-event path")
            self.brake_fault_path = [point[:2] for point in scenario["design"]["path"]]
            course_progress(self.brake_fault_path, self.course.start[:2])
        self.motion_transform = motion_transform
        self.started_at = float(clock())
        if not math.isfinite(self.started_at):
            raise ValueError("Clock must be finite")
        self._wall_anchor = self._sim_anchor = self.started_at
        if not isinstance(paused, bool):
            raise ValueError("paused must be boolean")
        self.paused = paused
        self._terminal = False
        self._sample_wall = self.started_at
        self._camera_offset = float(frame_clock()) - self.started_at
        if not math.isfinite(self._camera_offset):
            raise ValueError("Frame clock must be finite")
        self.native = native_factory(time=self.started_at - DT)
        if self.branch_selection is not None:
            self.branch_selection.apply(self.native, self.course.start)
        self.on_status = self.on_evidence = None
        self.clock_guard = None
        self.error = self.end_reason = self.pending_end = None
        self._condition = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._pending = []
        self._last_clock = self.started_at
        self._advance_lock = threading.RLock()
        self._snapshot, self._pose = self._capture()

    def _capture(self):
        pose = tuple(float(value) for value in self.native.pose)
        mapped = self.course.map_pose(pose)
        geometry = self.course.assess(mapped)
        stamp = self.native.time
        status = {
            **self.native.bridge.controller.status(stamp),
            **self.native.bridge.evidence,
            "received_at": self._sample_wall,
            "sim_time_s": stamp - self.started_at,
            "end_reason": self.end_reason,
        }
        if self.end_reason:
            status["reason"] = "operator_brake"
        view = {
            "pose": pose,
            "map_pose": tuple(mapped),
            "captured_at": self._sample_wall + self._camera_offset,
            "monotonic_at": self._sample_wall,
            "sim_time_s": stamp - self.started_at,
            "speed_mps": float(self.native.car.speed),
            **geometry,
            "enabled": bool(self.native.enabled),
            "pending_end": self.pending_end,
            "end_reason": self.end_reason,
        }
        if self.brake_fault_path is not None:
            view["brake_response"] = dict(
                latch=copy.deepcopy(self.native.car.brake_fault),
                sample=copy.deepcopy(self.native.car.last_brake_response),
            )
        if self.branch_selection is not None:
            view["branch_selection"] = self.branch_selection.snapshot()
        if self.actor_timeline is not None:
            from drivingbench.sim.scenarios import assess_step

            view["actor_onsets"] = self.actor_timeline.update(
                mapped, max(0.0, stamp - self.started_at), self.native.car.speed
            )
            states = self.actor_timeline.motion.states
            if states:
                view["actor_states"] = copy.deepcopy(states)
            previous = getattr(self, "_pose", view)
            view["scenario_contact"] = assess_step(
                self.actor_timeline.scene,
                previous["map_pose"],
                mapped,
                max(0.0, previous["sim_time_s"]),
                max(0.0, view["sim_time_s"]),
                actor_onsets=view["actor_onsets"],
                contacts_only=True,
                actor_states=view.get("actor_states"),
            )
            if self.coverage is not None:
                view["coverage_exit"] = self.coverage.assess(previous, view)
        return status, view

    def status(self):
        """Last physics status; reads do not refresh its watchdog timestamp."""
        with self._condition:
            result = copy.deepcopy(self._snapshot)
            if self.error or self._closed:
                result.update(
                    state="unavailable",
                    reason="simulator_error" if self.error else "simulator_closed",
                )
            if self.error:
                result["native_error"] = self.error
            return result

    def pose_snapshot(self):
        """Copy one pose/time/geometry sample; rendering must happen after this returns."""
        with self._condition:
            return copy.deepcopy(self._pose)

    def capture_snapshot(self):
        """Stamp a new render exposure; cached frames retain their previous capture time."""
        with self._condition:
            snapshot = copy.deepcopy(self._pose)
            wall = float(self.clock())
            snapshot.update(captured_at=wall + self._camera_offset, monotonic_at=wall)
            return snapshot

    def _wall_at_sim(self, stamp):
        return self._wall_anchor + (stamp - self._sim_anchor) / self.time_scale

    def _prepare(self, request, transformed):
        body = request.body
        operation = body.get("operation")
        command = {"motion": "set_motion", "stop": "stop_now"}.get(operation)
        if command and self.restrictions and command not in self.restrictions.accepted_commands:
            request.replies.append({"status": "rejected", "reason": "scenario_restriction"})
            transformed.append({"requested": copy.deepcopy(body), "applied": None})
            return None
        if self.motion_transform and operation == "motion":
            requested = copy.deepcopy(body["motion"])
            applied = self.motion_transform(copy.deepcopy(requested), copy.deepcopy(self._pose))
            transformed.append(
                {
                    "requested": requested,
                    "applied": copy.deepcopy(applied),
                    "sim_time_s": self.native.time - self.started_at,
                    "request_id": body.get("request_id"),
                }
            )
            if applied is None:
                request.replies.append({"status": "rejected", "reason": "scenario_restriction"})
                return None
            body = dict(body, motion=applied)
        return body

    def _ready(self, wall, operations=None):
        ready = [
            request
            for request in self._pending
            if request.arrived_at <= wall + 1e-9
            and (operations is None or request.body.get("operation") in operations)
        ]
        self._pending = [request for request in self._pending if request not in ready]
        return ready

    def _update_brake_response(self, wall):
        if self.brake_fault_path is None:
            return
        self.native.car.last_brake_response = None
        if self.native.car.brake_fault is not None:
            return
        mapped = self.course.map_pose(self.native.pose)
        station, cross_track = course_progress(self.brake_fault_path, mapped[:2])
        if station >= self.restrictions.brake_fault_station_m:
            self.native.car.latch_brake_fault(
                station_m=station,
                cross_track_m=cross_track,
                map_pose=mapped,
                sim_time_s=self.native.time + DT - self.started_at,
                monotonic_at=wall,
            )

    def _tick(self):
        transformed = []
        with self._condition:
            if self._closed or self.error:
                raise OSError(self.error or "simulator_closed")
            if self.pending_end and self.native.car.standstill:
                self.end_reason, self.pending_end = self.pending_end, None
                self.native.enabled = False
            wall = self._wall_at_sim(self.native.time + DT)
            prior_receipts = set(self.native.bridge.controller.receipts)
            ready = self._ready(wall)
            for request in ready:
                body = self._prepare(request, transformed)
                if body is not None:
                    self.native.inbox.pending.append((body, request.replies))
            self._update_brake_response(wall)
            self.native.tick()
            if self.branch_selection is not None:
                self.branch_selection.advance(self.native.car.speed * DT)
            self._branch_commands(ready, prior_receipts)
            if self.branch_selection is not None:
                self.branch_selection.apply(self.native, self.course.start)
            self._sample_wall = wall
            self._snapshot, self._pose = self._capture()
            status, evidence = copy.deepcopy(self._snapshot), copy.deepcopy(self._pose)
            self._condition.notify_all()
        if self.on_status:
            self.on_status(status)
        if self.on_evidence:
            for event in transformed:
                self.on_evidence("sim_motion_transform", **event)
            self.on_evidence("sim_tick", **evidence)

    def _branch_commands(self, requests, prior_receipts):
        if self.branch_selection is None:
            return
        for request in requests:
            identity = request.body.get("request_id")
            if (
                identity not in prior_receipts
                and request.body.get("operation") == "motion"
                and request.replies
                and request.replies[-1].get("status") == "accepted"
            ):
                self.branch_selection.command(request.body["motion"])
                prior_receipts.add(identity)

    def _fail(self, error):
        with self._condition:
            if self._closed or self.error:
                return
            self.error = f"{type(error).__name__}: {error}"
            evidence = copy.deepcopy(self._pose)
            self._condition.notify_all()
        if self.on_evidence:
            try:
                self.on_evidence("sim_runtime_error", error=self.error, **evidence)
            except Exception:
                pass  # The original failure remains visible if evidence delivery also fails.

    def advance(self, until=None):
        """Run due 10 ms steps against an injected clock; never overshoot the target."""
        if self._thread and threading.current_thread() is not self._thread:
            raise RuntimeError("Manual advance is unavailable while the worker runs")
        target = float(self.clock() if until is None else until)
        with self._condition:
            if not math.isfinite(target) or target < self._last_clock - 1e-9:
                raise ValueError("Clock must be finite and monotonic")
            self._last_clock = target
        try:
            with self._advance_lock:
                if self.pending_end:
                    if self.paused and not self._terminal:
                        self.resume()
                    if self.clock_guard is not None and not self.clock_guard.completed:
                        self.clock_guard.end()
                if self.clock_guard:
                    self.clock_guard.poll()
                if self.paused:
                    self._service_idle(target, paused=True)
                else:
                    while not self.paused:
                        if self.clock_guard:
                            self.clock_guard.poll()
                        if self.paused or self._wall_at_sim(self.native.time + DT) > target + 1e-9:
                            break
                        self._tick()
                    if (
                        self.time_scale < 1
                        and not self.paused
                        and self._wall_at_sim(self.native.time + DT)
                        > max(target, float(self.clock()))
                    ):
                        self._service_idle(target)
        except Exception as error:
            self._fail(error)
            raise
        return self.status()

    def call(self, body, *, timeout=None):
        """Queue an envelope for the next due integration tick and await its receipt."""
        timeout = self.timeout if timeout is None else timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        arrival = float(self.clock())
        latency = (
            self.restrictions.command_latency_s
            if self.restrictions and body.get("operation") in {"motion", "stop"}
            else 0.0
        )
        request = _Request(copy.deepcopy(body), arrival + latency)
        if latency and self.on_evidence:
            self.on_evidence(
                "sim_command_latency",
                operation=body.get("operation"),
                request_id=body.get("request_id"),
                arrived_at=arrival,
                eligible_at=arrival + latency,
                latency_s=latency,
            )
        with self._condition:
            if self._closed or self.error:
                raise OSError(self.error or "simulator_closed")
            if not math.isfinite(request.arrived_at):
                raise ValueError("Clock must be finite")
            self._pending.append(request)
            self._condition.notify_all()
            while not request.replies:
                if self._closed or self.error:
                    raise OSError(self.error or "simulator_closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._pending = [pending for pending in self._pending if pending is not request]
                    raise TimeoutError("simulator_request_timeout")
                self._condition.wait(remaining)
            return copy.deepcopy(request.replies[0])

    def _service_idle(self, wall, *, paused=False):
        transformed = []
        with self._condition:
            if self._closed or self.error:
                raise OSError(self.error or "simulator_closed")
            controller = self.native.bridge.controller
            if self.native.bridge.failure is None:
                controller.refresh(controller.sample, self.native.time)
            prior_receipts = set(controller.receipts)
            ready = self._ready(wall, None if paused else {"status", "heartbeat"})
            for request in ready:
                body = self._prepare(request, transformed)
                if body is not None:
                    request.replies.append(self.native.bridge._reply(body))
            self._branch_commands(ready, prior_receipts)
            if self.branch_selection is not None:
                self._pose["branch_selection"] = self.branch_selection.snapshot()
            status, _ = self._capture()
            status["received_at"] = wall
            self._snapshot = status
            self._condition.notify_all()
        if self.on_status:
            self.on_status(copy.deepcopy(status))
        if self.on_evidence:
            for event in transformed:
                self.on_evidence("sim_motion_transform", **event)

    def set_time_scale(self, rate):
        """Change the clock rate at the last integrated tick without rewriting its wall time."""
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("time_scale must be finite and positive")
        with self._advance_lock, self._condition:
            previous = self.time_scale
            if previous == rate:
                return
            self._wall_anchor, self._sim_anchor = (
                self._wall_at_sim(self.native.time),
                self.native.time,
            )
            self.time_scale = float(rate)
            self._condition.notify_all()
            anchor = dict(wall_anchor=self._wall_anchor, sim_anchor=self._sim_anchor)
        if self.on_evidence:
            self.on_evidence(
                "sim_clock_rate", previous_rate=previous, rate=self.time_scale, **anchor
            )

    def pause(self):
        """Freeze physics and command durations; native requests remain serviceable."""
        with self._advance_lock, self._condition:
            self.paused = True
            self._condition.notify_all()
        if self.on_evidence:
            self.on_evidence(
                "sim_pause", wall_time_s=self.clock() - self.started_at, **self.pose_snapshot()
            )

    def resume(self):
        with self._advance_lock, self._condition:
            if self._terminal:
                raise RuntimeError("A terminal impact cannot resume")
            if not self.paused:
                return
            self._wall_anchor, self._sim_anchor = float(self.clock()), self.native.time
            self.paused = False
            self._condition.notify_all()
        if self.on_evidence:
            self.on_evidence(
                "sim_resume", wall_time_s=self.clock() - self.started_at, **self.pose_snapshot()
            )

    def terminate(self, reason):
        """Freeze at terminal contact; this is an impact endpoint, not working brakes."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("Terminal reason must be a nonempty string")
        with self._advance_lock, self._condition:
            if self._terminal:
                return
            before = copy.deepcopy(self._pose)
            self.paused = self._terminal = True
            self.end_reason, self.pending_end = reason, None
            self.native.enabled = False
            self.native.car.speed = self.native.car.accel = 0.0
            controller = self.native.bridge.controller
            controller.refresh(
                replace(
                    controller.sample,
                    speed_mps=0.0,
                    standstill=True,
                    enabled=False,
                    lateral_active=False,
                    longitudinal_active=False,
                ),
                self.native.time,
            )
            self._snapshot, self._pose = self._capture()
            after = copy.deepcopy(self._pose)
            self._condition.notify_all()
        if self.on_evidence:
            self.on_evidence("sim_terminal_impact", terminal_reason=reason, **before)
            self.on_evidence("sim_terminal_state", terminal_reason=reason, **after)

    def operator_brake(self, pressed=True):
        """Set the simulated driver brake input; the fitted car decelerates on ticks."""
        with self._condition:
            if self._closed:
                raise OSError("simulator_closed")
            if self.pending_end or self.end_reason:
                return
            self.native.brake = bool(pressed)

    def finish(self, reason):
        """Request physical braking; the physics owner resumes a paused cleanup safely."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("Finish reason must be a nonempty string")
        with self._condition:
            if self._closed:
                raise OSError("simulator_closed")
            if self.pending_end or self.end_reason:
                return
            evidence = copy.deepcopy(self._pose)
            requested_at = float(self.clock())
            self.pending_end = reason
            self.native.brake = True
            self._condition.notify_all()
        if self.on_evidence:
            self.on_evidence(
                "sim_finish_requested",
                requested_reason=reason,
                requested_at=requested_at,
                **evidence,
            )

    def start(self):
        with self._condition:
            if self._closed:
                raise OSError("simulator_closed")
            if self._thread:
                return

            def run():
                try:
                    while not self._stop.is_set():
                        self.advance()
                        delay = (
                            0.01
                            if self.paused
                            else max(0.0, self._wall_at_sim(self.native.time + DT) - self.clock())
                        )
                        self._stop.wait(min(delay, 0.05))
                except Exception as error:
                    self._fail(error)

            self._thread = threading.Thread(target=run, name="sim-native", daemon=True)
            self._thread.start()

    def close(self):
        """Stop simulation processing and release queued callers; use finish to brake first."""
        with self._condition:
            self._closed = True
            self._stop.set()
            self._pending.clear()
            self._condition.notify_all()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.timeout + 1)
            if self._thread.is_alive():
                raise TimeoutError("simulator_worker_did_not_close")
