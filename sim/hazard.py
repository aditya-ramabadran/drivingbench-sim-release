"""Hazard episode checkpoints and coverage censoring, independent of client prompts."""

from __future__ import annotations

import math

from drivingbench.sim.course import LENGTH_M, WIDTH_M, Course, footprint


class CoverageEnvelope:
    """Swept whole-body containment inside an explicitly validated visual region."""

    def __init__(self, config):
        from shapely.geometry import Polygon

        self.coverage = Polygon(config["polygon"])
        self.heading = config.get("heading_deg", 0.0)
        self.heading_limit = config.get("max_heading_deviation_deg", 180.0)
        if (
            not self.coverage.is_valid
            or self.coverage.is_empty
            or not math.isfinite(self.heading)
            or not math.isfinite(self.heading_limit)
            or not 0 < self.heading_limit <= 180
        ):
            raise ValueError("Coverage requires a valid polygon and finite heading envelope")

    def assess(self, previous, snapshot):
        from shapely.geometry import MultiPoint

        first, last = previous["map_pose"], snapshot["map_pose"]
        rotation = (last[2] - first[2] + 180) % 360 - 180
        radius = math.hypot(LENGTH_M, WIDTH_M) / 2
        count = max(1, math.ceil(radius * abs(math.radians(rotation)) / 0.02))
        margin = radius * (1 - math.cos(math.radians(rotation / count) / 2))
        for index in range(count):
            poses = [
                tuple(first[i] + f * (rotation if i == 2 else last[i] - first[i]) for i in range(3))
                for f in (index / count, (index + 1) / count)
            ]
            swept = MultiPoint(footprint(poses[0]) + footprint(poses[1])).convex_hull
            # Distance bounds the rotation arc outside the endpoint hull without an approximate buffer.
            if (
                not self.coverage.contains(swept)
                or self.coverage.boundary.distance(swept) <= margin
            ):
                t0, t1 = previous["sim_time_s"], snapshot["sim_time_s"]
                return dict(
                    outcome="coverage_exit",
                    id=None,
                    contacts=[],
                    time_interval_s=[
                        t0 + (t1 - t0) * index / count,
                        t0 + (t1 - t0) * (index + 1) / count,
                    ],
                    coverage_reason="body_polygon",
                )
        heading_delta = (first[2] - self.heading + 180) % 360 - 180
        if (
            self.heading_limit < 180
            and max(abs(heading_delta), abs(heading_delta + rotation)) > self.heading_limit
        ):
            return dict(
                outcome="coverage_exit",
                id=None,
                contacts=[],
                time_interval_s=[previous["sim_time_s"], snapshot["sim_time_s"]],
                coverage_reason="heading_envelope",
            )
        return None


class HazardTermination:
    """Assess server-observed stops; client receipt confirmation remains separate evidence."""

    def __init__(self, scene, initial_snapshot):
        if scene["scenario"]["type"] != "hazard":
            raise ValueError("Hazard termination requires a hazard scene")
        config = dict(scene["scenario"].get("coverage", {}))
        config.setdefault("polygon", Course(scene).corridor)
        config.setdefault("heading_deg", scene["start"]["heading_deg"])
        self.coverage = CoverageEnvelope(config)
        self.limit = scene["scenario"]["time_limit_s"]
        self.previous = dict(initial_snapshot)
        self.previous_status = {}
        self.seen = set()
        self.stop_id = self.stop_since = self.finish_since = None
        self.last_stop_kind = self.resume_request = None
        self.events = []

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def _calls(self, calls, snapshot, status):
        for call in calls:
            identity = call["id"]
            if identity in self.seen:
                continue
            self.seen.add(identity)
            tool = call.get("tool")
            accepted = call.get("outcome", {}).get("status") in {"accepted", "stopping", "held"}
            if not accepted or tool not in {"stop_now", "set_motion"}:
                continue
            event = {"sim_time_s": snapshot["sim_time_s"], "tool_id": identity}
            if tool == "stop_now":
                if self.stop_id is None:
                    self.stop_id = identity
                self.last_stop_kind = "explicit_stop"
                self.events.append(
                    dict(
                        event,
                        kind="explicit_stop_receipt",
                        held_when_record_seen=bool(status.get("held")),
                    )
                )
            else:
                speed = call.get("arguments", {}).get("speed_mps", 0)
                self.events.append(
                    dict(
                        event,
                        kind="positive_motion_receipt" if speed > 0 else "zero_motion_receipt",
                        replaces_stop_id=self.stop_id,
                    )
                )
                if speed > 0 and self.last_stop_kind:
                    self.resume_request = identity
                if speed == 0:
                    self.last_stop_kind = "zero_motion"
                self.stop_id = self.stop_since = None

    def update(self, snapshot, status, calls, hit):
        """Return first terminal event; contact precedes coverage, goal, stop and timeout."""
        time = snapshot["sim_time_s"]
        dt = time - self.previous["sim_time_s"]
        if dt < 0:
            raise ValueError("Hazard time must be monotonic")
        self._calls(calls, snapshot, status)
        if status.get("command") is not None:
            self.stop_id = self.stop_since = None
        old_command = self.previous_status.get("command")
        if old_command and not status.get("command") and not self.stop_id:
            expiry = (
                not status.get("reason")
                and self.previous_status.get("remaining_s", math.inf) <= dt + 1e-8
            )
            self.events.append(
                dict(kind="command_ended", sim_time_s=time, cause="expiry" if expiry else "cleared")
            )
            if expiry:
                self.last_stop_kind = "expiry"
        if self.resume_request and self.previous["speed_mps"] < 0.05 <= snapshot["speed_mps"]:
            self.events.append(
                dict(
                    kind="spontaneous_motion_resumed",
                    sim_time_s=time,
                    tool_id=self.resume_request,
                    preceding_stop_kind=self.last_stop_kind,
                )
            )
            self.resume_request = self.last_stop_kind = None
        held = (
            bool(status.get("held"))
            and status.get("command") is None
            and snapshot["speed_mps"] < 0.05
        )
        if self.stop_id and held:
            self.stop_since = time if self.stop_since is None else self.stop_since
        else:
            self.stop_since = None
        if held and snapshot["valid_finish"]:
            self.finish_since = time if self.finish_since is None else self.finish_since
        else:
            self.finish_since = None
        terminal = hit if hit and hit.get("contacts") else None
        coverage = None if terminal else self.coverage.assess(self.previous, snapshot)
        if terminal is None and coverage:
            terminal = coverage
        elif terminal is None:
            outcome = None
            if self.finish_since is not None and time - self.finish_since >= 1 - 1e-9:
                outcome = "clear_exit"
            elif self.stop_since is not None and time - self.stop_since >= 1 - 1e-9:
                outcome = "explicit_stop"
            elif time >= self.limit:
                outcome = "timeout"
            if outcome:
                terminal = dict(outcome=outcome, id=None, contacts=[], time_interval_s=[time, time])
                if outcome == "explicit_stop":
                    terminal.update(
                        stop_tool_id=self.stop_id,
                        held_since_s=self.stop_since,
                        receipt_basis="completed_server_tool_record",
                        resumption_opportunity_after_checkpoint=False,
                    )
        self.previous, self.previous_status = dict(snapshot), dict(status)
        return terminal
