"""One attempt's approved stop policy and asynchronous simulation evidence."""

from __future__ import annotations

import copy
import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import SimpleQueue


@dataclass(frozen=True)
class EpisodePolicy:
    enabled: bool = True
    stop_on_contact: bool = True
    stop_on_marker_contact: bool = True
    stop_on_exit: bool = True
    boundary_tolerance_m: float = 0.5
    strict_boundary_tolerance_m: float | None = None
    finish_speed_mps: float = 0.05
    finish_dwell_s: float = 1.0
    time_limit_s: float = 360.0

    def __post_init__(self):
        for value in (
            self.enabled,
            self.stop_on_contact,
            self.stop_on_marker_contact,
            self.stop_on_exit,
        ):
            if not isinstance(value, bool):
                raise ValueError("Policy switches must be booleans")
        if not math.isfinite(self.boundary_tolerance_m) or self.boundary_tolerance_m < 0:
            raise ValueError("Boundary tolerance must be finite and nonnegative")
        if self.strict_boundary_tolerance_m is not None and (
            not math.isfinite(self.strict_boundary_tolerance_m)
            or self.strict_boundary_tolerance_m < 0
        ):
            raise ValueError("Strict boundary tolerance must be finite and nonnegative")
        for value in (self.finish_speed_mps, self.finish_dwell_s, self.time_limit_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Policy thresholds must be finite and positive")

    def finish_eligible(self, evidence):
        """A finish sample eligible for dwell under this run's declared contact policy."""
        hits = (
            evidence["collision_ids"]
            if self.stop_on_marker_contact
            else evidence.get("obstacle_collision_ids", [])
        )
        inside = (
            evidence["valid_finish"] if self.stop_on_marker_contact else evidence["finish_reached"]
        )
        return bool(
            inside
            and not hits
            and not evidence.get("scenario_contact")
            and not evidence.get("coverage_exit")
            and not evidence["outside_course"]
            and evidence["speed_mps"] < self.finish_speed_mps
        )


class Episode:
    """Attach before starting Runtime; close after physics stops to flush the final evidence."""

    def __init__(self, runtime, output, policy=None, *, metadata):
        if runtime.on_evidence is not None:
            raise ValueError("Runtime already has an evidence callback")
        self.runtime, self.policy = runtime, policy or EpisodePolicy()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self._queue = SimpleQueue()
        self._lock = threading.RLock()
        self._closed = False
        self._writer_error = None
        self._sequence = self._ticks = self._dropped = self._exits = 0
        self._touched = set()
        self._actors_touched = set()
        self._first_contact = self._first_exit = self._first_finish = self._cause = None
        self._first_actor_contact = self._first_coverage_exit = None
        self._infrastructure = None
        self._strict_cause = self._first_strict_exit = None
        self._finish_since = None
        self._was_outside = False
        self._last = runtime.pose_snapshot()
        self._start_time = float(self._last["sim_time_s"])
        if not math.isfinite(self._start_time):
            raise ValueError("Episode start time must be finite")
        header = dict(metadata=copy.deepcopy(metadata), policy=asdict(self.policy))
        encoded = json.dumps(header, indent=2, allow_nan=False) + "\n"
        with (self.output / "metadata.json").open("x") as stream:
            stream.write(encoded)
        stream = (self.output / "events.jsonl").open("x")
        self._worker = threading.Thread(
            target=self._write, args=(stream,), name="sim-episode-evidence", daemon=True
        )
        self._worker.start()
        self._callback = self.record
        runtime.on_evidence = self._callback

    def _write(self, stream):
        try:
            with stream:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
                    if item["kind"] in {"sim_frame", "sim_observe_delivery", "sim_clock"}:
                        stream.flush()
                stream.flush()
        except Exception as error:
            self._writer_error = f"{type(error).__name__}: {error}"

    def _enqueue(self, kind, data):
        if self._writer_error:
            self._dropped += 1
            return
        self._sequence += 1
        self._queue.put(dict(sequence=self._sequence, kind=kind, **copy.deepcopy(data)))

    def _fail(self, error, source):
        if self._infrastructure is not None:
            return
        self._infrastructure = dict(
            source=source, error=str(error), evidence=copy.deepcopy(self._last)
        )
        self._enqueue("infrastructure_failure", self._infrastructure)
        try:
            self.runtime.finish("infrastructure_failure")
        except Exception as finish_error:
            self._infrastructure["braking_error"] = str(finish_error)

    def fail(self, error, *, source="runtime"):
        """Record a renderer/runtime/recorder failure separately from course outcomes."""
        with self._lock:
            if not self._closed:
                self._fail(error, source)

    def record(self, kind, **data):
        """Physics callback: update small in-memory aggregates and enqueue, never write files."""
        with self._lock:
            if self._closed:
                return
            if self._writer_error:
                self._fail(self._writer_error, "recorder")
            if kind not in {"sim_tick", "sim_terminal_state"}:
                self._enqueue(kind, data)
                if kind in (
                    "render_error",
                    "sim_render_error",
                    "runtime_error",
                    "sim_runtime_error",
                    "infrastructure_failure",
                ):
                    self._fail(data.get("error", kind), kind)
                return
            t, speed = float(data["sim_time_s"]), float(data["speed_mps"])
            if not math.isfinite(t) or not math.isfinite(speed) or t < self._last["sim_time_s"]:
                self._fail("Nonfinite or nonmonotonic simulation evidence", "runtime")
                return
            self._last = copy.deepcopy(data)
            self._enqueue(kind, data)
            if kind == "sim_terminal_state":
                return
            self._ticks += 1
            hits = data["collision_ids"]
            obstacles = data.get("obstacle_collision_ids", [])
            terminal_hits = hits if self.policy.stop_on_marker_contact else obstacles
            if self.policy.strict_boundary_tolerance_m is not None:
                if "strict_boundary_violation" not in data:
                    self._fail("Missing strict boundary assessment", "runtime")
                    return
                strict_exit = data["strict_boundary_violation"]
                if strict_exit and self._first_strict_exit is None:
                    self._first_strict_exit = copy.deepcopy(data)
                if self._strict_cause is None and (hits or strict_exit):
                    self._strict_cause = dict(
                        reason="marker_hit" if hits else "corridor_exit",
                        evidence=copy.deepcopy(data),
                    )
                    self._enqueue("strict_score_trigger", self._strict_cause)
            actor_contact = data.get("scenario_contact")
            coverage_exit = data.get("coverage_exit")
            outside = bool(data["outside_course"])
            if actor_contact:
                self._actors_touched.update(row["id"] for row in actor_contact["contacts"])
                if self._first_actor_contact is None:
                    self._first_actor_contact = copy.deepcopy(data)
            if coverage_exit and self._first_coverage_exit is None:
                self._first_coverage_exit = copy.deepcopy(data)
            if hits:
                self._touched.update(hits)
                if self._first_contact is None:
                    self._first_contact = copy.deepcopy(data)
            if outside and not self._was_outside:
                self._exits += 1
                if self._first_exit is None:
                    self._first_exit = copy.deepcopy(data)
            self._was_outside = outside
            if data["finish_reached"] and self._first_finish is None:
                self._first_finish = copy.deepcopy(data)
            if self.policy.finish_eligible(data):
                if self._finish_since is None:
                    self._finish_since = t
            else:
                self._finish_since = None
            if self._cause or self._infrastructure or not self.policy.enabled:
                return
            reason = None
            if actor_contact and self.policy.stop_on_contact:
                reason = "hazard_contact"
            elif terminal_hits and self.policy.stop_on_contact:
                reason = "marker_hit" if self.policy.stop_on_marker_contact else "obstacle_contact"
            elif coverage_exit:
                reason = "coverage_exit"
            elif data.get("lot_safety_backstop"):
                reason = "lot_safety_backstop"
            elif data["boundary_violation"] and self.policy.stop_on_exit:
                reason = "corridor_exit"
            elif t - self._start_time >= self.policy.time_limit_s - 1e-9:
                reason = "time_limit"
            elif (
                self._finish_since is not None
                and t - self._finish_since >= self.policy.finish_dwell_s - 1e-9
            ):
                reason = "finish"
            if reason:
                self._cause = dict(reason=reason, evidence=copy.deepcopy(data))
                self._enqueue("episode_trigger", self._cause)
                try:
                    self.runtime.finish(reason)
                except Exception as error:
                    self._fail(error, "runtime")

    def summary(self):
        with self._lock:
            reason = self._cause["reason"] if self._cause else None
            outcome = {
                "finish": "success",
                "marker_hit": "failure",
                "hazard_contact": "failure",
                "obstacle_contact": "failure",
                "coverage_exit": "censored",
                "lot_safety_backstop": "failure",
                "corridor_exit": "failure",
                "time_limit": "timeout",
                None: "unfinished" if self.policy.enabled else "observed",
            }[reason]
            return copy.deepcopy(
                dict(
                    outcome="infrastructure_failure" if self._infrastructure else outcome,
                    first_cause=self._cause,
                    infrastructure_failure=self._infrastructure,
                    ticks=self._ticks,
                    touched_ids=sorted(self._touched),
                    strict_score=(
                        dict(
                            failed=self._strict_cause is not None,
                            first_cause=self._strict_cause,
                            touched_ids=sorted(self._touched),
                            first_boundary_violation=self._first_strict_exit,
                        )
                        if self.policy.strict_boundary_tolerance_m is not None
                        else None
                    ),
                    touched_actor_ids=sorted(self._actors_touched),
                    exit_count=self._exits,
                    first_contact=self._first_contact,
                    first_actor_contact=self._first_actor_contact,
                    first_coverage_exit=self._first_coverage_exit,
                    first_exit=self._first_exit,
                    first_finish=self._first_finish,
                    last_evidence=self._last,
                    elapsed_s=self._last["sim_time_s"] - self._start_time,
                    stopped=self._last["speed_mps"] < self.policy.finish_speed_mps,
                    native_end_reason=self.runtime.end_reason,
                    unqueued_after_writer_error=self._dropped,
                )
            )

    def close(self):
        """Detach, drain the writer, and persist a summary; never call from the physics callback."""
        with self._lock:
            if self._closed:
                return self.summary()
            if self.runtime.error:
                self._fail(self.runtime.error, "runtime")
            self._closed = True
            if self.runtime.on_evidence is self._callback:
                self.runtime.on_evidence = None
            self._queue.put(None)
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            raise TimeoutError("Episode evidence writer did not flush")
        if self._writer_error:
            with self._lock:
                self._fail(self._writer_error, "recorder")
        result = self.summary()
        with (self.output / "summary.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if self._writer_error:
            raise OSError(f"Episode evidence failed: {self._writer_error}")
        return result
