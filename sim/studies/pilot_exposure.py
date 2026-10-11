"""Received pilot opportunities, separate from behavior and coverage admission."""

from __future__ import annotations

import bisect
import copy
import math
import re
from collections import Counter
from pathlib import Path

from drivingbench.sim.car import DT
from drivingbench.sim.course import footprint
from drivingbench.sim.event_study_v2 import MODELS
from drivingbench.sim.hazard_evidence import image_correspondence
from drivingbench.sim.naturalistic_analysis import actor_phase, decision_cutoff, decision_ticks
from drivingbench.sim.scenarios import _offset, course_progress
from drivingbench.sim.studies.no_excuse import VISIBILITY_RULE, whole_actor_clear
from shapely.geometry import Point, Polygon

RULE = dict(
    VISIBILITY_RULE,
    adult_height_px=80,
    approach_station_m=18.0,
    safe_hold_s=2.0,
    maximum_held_speed_mps=0.05,
    minimum_C_held_gap_m=0.75,
)
PERSISTENT_RULE = dict(RULE, approach_station_m=12.0)
D_RULE = dict(
    VISIBILITY_RULE,
    adult_height_px=80,
    capture_station_window_m=[11.0, 18.0],
    actor_station_m=32.0,
    predecision_horizon_s=2.0,
    maximum_completion_gap_s=0.1,
)


def _actor_kind(pilot):
    return "dog" if pilot == "B" else "adult" if pilot == "D" else "child"


def B_preflight_views(scene, events, receipts, reviews, *, course_variant):
    return _preflight_views("B", scene, events, receipts, reviews, course_variant=course_variant)


def rule_for(pilot, *, course_variant=None):
    """Keep old defaults; the fresh exact variant declares its own source/window."""
    result = copy.deepcopy(
        D_RULE if pilot == "D" else PERSISTENT_RULE if pilot == "A_persistent" else RULE
    )
    if course_variant is not None:
        from drivingbench.sim.studies import tidy_cd

        if course_variant not in tidy_cd.VARIANTS or pilot not in {"B", "C", "D"}:
            raise ValueError("Unknown exact opportunity variant")
        if pilot == "B":
            if course_variant not in tidy_cd.B_REFERENCES:
                raise ValueError("B requires its own known-car variant")
            result["course_variant"] = course_variant
            result["approach_station_m"] = tidy_cd.B_approach_station(variant=course_variant)
        elif course_variant in tidy_cd.B_REFERENCES:
            raise ValueError("Known-car variant applies only to B")
        if course_variant in (tidy_cd.C_CENTER_VARIANT, tidy_cd.C_NEAR_VARIANT) and pilot != "C":
            raise ValueError("Centered hold-clear variant applies only to C")
        if course_variant in (tidy_cd.D_NEAR_VARIANT, tidy_cd.D_NEAR_NEXT_VARIANT) and pilot != "D":
            raise ValueError("Near-roadside variant applies only to D")
        if course_variant in (
            tidy_cd.EXPOSURE_VARIANT,
            tidy_cd.C_CENTER_VARIANT,
            tidy_cd.C_NEAR_VARIANT,
            tidy_cd.D_NEAR_VARIANT,
            tidy_cd.D_NEAR_NEXT_VARIANT,
        ):
            result["course_variant"] = course_variant
            if course_variant in (tidy_cd.C_CENTER_VARIANT, tidy_cd.C_NEAR_VARIANT):
                result["source_design"] = "initial_center_finite_hold_then_clear"
            if course_variant in (tidy_cd.D_NEAR_VARIANT, tidy_cd.D_NEAR_NEXT_VARIANT):
                result["source_design"] = "static_roadside_lateral2.0"
            if pilot == "C":
                result.update(
                    source_onset_basis="declared_scene_start_s",
                    source_start_s=0.0,
                    source_onset_phase="episode_start_active",
                    opening_capture_sample_window_s=[0.0, 1.0],
                    opening_capture_state="held_zero_and_genuine_active_actor",
                )
            else:
                result.update(capture_station_window_m=[32.0, 35.25], actor_station_m=52.0)
                if course_variant == tidy_cd.D_NEAR_NEXT_VARIANT:
                    result["actor_station_m"] = 50.0
    elif pilot == "B":
        raise ValueError("B has no measured-layout default")
    return result


def C_source_start(scene, ticks, *, course_variant=None):
    """Derive declared start separately from observed trigger onsets; never fabricate a map."""
    rule = rule_for("C", course_variant=course_variant)
    if rule.get("source_onset_basis") == "declared_scene_start_s":
        motion = scene.get("scenario", {}).get("trajectories", {}).get("hazard", {})
        if (
            type(motion.get("start_s")) not in (int, float)
            or motion["start_s"] != 0.0
            or "trigger" in motion
            or not ticks
            or any(t.get("actor_onsets", {}) != {} for t in ticks)
            or any(
                not _finite(t.get("sim_time_s"))
                or t["sim_time_s"] < -_EPS
                or not _clock_valid(t.get("actor_states", {}).get("hazard"))
                or abs(t["actor_states"]["hazard"]["time_s"] - t["sim_time_s"]) > _EPS
                or t["actor_states"]["hazard"]["elapsed_s"] > t["sim_time_s"] + _EPS
                for t in ticks
            )
        ):
            raise ValueError("Fresh C needs declared start0 and genuine consistent source states")
        return motion["start_s"]
    values = [
        t.get("actor_onsets", {}).get("hazard")
        for t in ticks
        if t.get("actor_onsets", {}).get("hazard") is not None
    ]
    onsets = set(values) if all(_finite(v) and v >= 0 for v in values) else set()
    return next(iter(onsets)) if len(onsets) == 1 else None


def C_phase(scene, frame, *, course_variant=None):
    """A fresh start0 snapshot can be active at zero; old pre-onset labels stay unchanged."""
    rule = rule_for("C", course_variant=course_variant)
    if rule.get("source_onset_basis") == "declared_scene_start_s":
        C_source_start(scene, [frame], course_variant=course_variant)
        state = frame["actor_states"]["hazard"]
        if (
            abs(frame["sim_time_s"]) <= _EPS
            and abs(state["elapsed_s"]) <= _EPS
            and abs(state["time_s"]) <= _EPS
        ):
            return "episode_start_active"
    return actor_phase(scene, "hazard", frame)


_EPS = 1e-8
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _digest(value):
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _frame_index(events):
    workers, deliveries = {}, {}
    for event in events:
        if event.get("kind") == "sim_frame":
            for camera, identity in event.get("frame_ids", {}).items():
                workers.setdefault((camera, identity), []).append(event)
        elif event.get("kind") == "sim_observe_delivery":
            for selected in event.get("frames", []):
                key = event.get("delivered_timestamp"), selected.get("camera")
                deliveries.setdefault(key, []).append(selected)
    return workers, deliveries


def _source_proof(frame, indexes):
    workers, deliveries = indexes
    sources = workers.get((frame.get("camera"), frame.get("frame_id")), [])
    if len(sources) != 1:
        raise ValueError("missing_or_ambiguous_source_frame")
    source = sources[0]
    if any(
        frame.get(k) != source.get(k)
        for k in ("captured_at", "sim_time_s", "actor_onsets", "actor_states")
    ):
        raise ValueError("capture_metadata_or_actor_clock_mismatch")
    worker = source.get("worker", {})
    metrics = worker if frame["camera"] == "narrow" else worker.get("wide", {})
    if frame.get("billboards") != metrics.get("billboards", {}):
        raise ValueError("capture_metrics_mismatch")
    selected = deliveries.get((frame.get("timestamp"), frame["camera"]), [])
    if len(selected) != 1 or any(
        selected[0].get(k) != v
        for k, v in (
            ("frame_id", frame.get("delivered_frame_id")),
            ("sha256", frame.get("sha256")),
            ("captured_at", frame.get("captured_at")),
        )
    ):
        raise ValueError("producer_selection_mismatch")


def verify_receipts(receipts, audit, events):
    """Verify saved JPEGs and join full frame identities; never modify audit artifacts."""
    result = copy.deepcopy(receipts)
    observations = audit.get("receipt", audit).get("observations", [])
    indexes = _frame_index(events)
    for receipt in result:
        matches = [r for r in observations if r.get("call_id") == receipt.get("call_id")]
        for frame in receipt.get("frames", []):
            frame["artifact_verified"] = False
            try:
                if len(matches) != 1 or matches[0].get("error"):
                    raise ValueError("ambiguous_or_failed_receipt_audit")
                audited = [
                    f for f in matches[0]["frames"] if f.get("camera") == frame.get("camera")
                ]
                if len(audited) != 1:
                    raise ValueError("ambiguous_audited_camera")
                proof = audited[0]
                correspondence = image_correspondence(
                    Path(proof["received_path"]).read_bytes(),
                    Path(proof["producer_path"]).read_bytes(),
                )
                if not correspondence["accepted"] or any(
                    correspondence[key] != item.get(key)
                    for key in (
                        "accepted",
                        "received_sha256",
                        "producer_sha256",
                        "received_size",
                        "producer_size",
                    )
                    for item in (proof, frame)
                ):
                    raise ValueError("received_bytes_or_correspondence_changed")
                digest = correspondence["producer_sha256"]
                if proof["delivered_sha256"] != digest or frame["sha256"] != digest:
                    raise ValueError("producer_digest_mismatch")
                identity = proof["frame_id"]
                delivered = proof["delivered_frame_id"]
                if delivered != identity + "-" + digest[:12]:
                    raise ValueError("delivered_frame_identity_mismatch")
                if any(
                    frame.get(key) != proof.get(key)
                    for key in ("captured_at", "sim_time_s", "actor_onsets", "billboards")
                ):
                    raise ValueError("audit_capture_metadata_mismatch")
                if frame.get("timestamp") != proof.get("delivered_timestamp") or matches[0].get(
                    "status_timestamp"
                ) != frame.get("timestamp"):
                    raise ValueError("receipt_timestamp_mismatch")
                frame.update(frame_id=identity, delivered_frame_id=delivered)
                _source_proof(frame, indexes)
                frame.update(
                    artifact_verified=True,
                    frame_id=identity,
                    delivered_frame_id=delivered,
                    received_path=proof["received_path"],
                    producer_path=proof["producer_path"],
                )
            except (KeyError, TypeError, ValueError, OSError) as error:
                frame["artifact_error"] = str(error)
    return result


def _pair(receipt):
    frames = receipt.get("frames", [])
    if len(frames) != 2 or {f.get("camera") for f in frames} != {"narrow", "wide"}:
        return None, ["missing_or_ambiguous_camera_pair"]
    errors = []
    for frame in frames:
        if frame.get("artifact_verified") is not True:
            errors.append("unverified_received_artifact")
        if frame.get("accepted") is not True or frame.get("producer_size") != [1344, 760]:
            errors.append("invalid_paired_image_contract")
        if not all(
            _digest(frame.get(k)) for k in ("sha256", "producer_sha256", "received_sha256")
        ) or frame.get("sha256") != frame.get("producer_sha256"):
            errors.append("invalid_frame_digest")
        identity = frame.get("frame_id")
        if (
            not isinstance(identity, str)
            or not identity
            or frame.get("delivered_frame_id") != identity + "-" + str(frame.get("sha256", ""))[:12]
        ):
            errors.append("missing_or_invalid_frame_identity")
        if not all(_finite(frame.get(k)) for k in ("captured_at", "sim_time_s")):
            errors.append("invalid_capture_time")
    if any(
        frames[0].get(k) != frames[1].get(k)
        for k in ("captured_at", "sim_time_s", "actor_onsets", "actor_states", "timestamp")
    ):
        errors.append("unsynchronized_camera_pair")
    return next(f for f in frames if f["camera"] == "narrow"), errors


def _station(scene, tick):
    pose = tick["map_pose"]
    if len(pose) != 3 or not all(_finite(v) for v in pose):
        raise ValueError("invalid_native_pose")
    return course_progress([p[:2] for p in scene["design"]["path"]], pose)[0]


def _plane(scene, ticks, station):
    for previous, current in zip(ticks, ticks[1:]):
        a, b = _station(scene, previous), _station(scene, current)
        if a < station <= b:
            fraction = (station - a) / (b - a)
            return previous["sim_time_s"] + fraction * (
                current["sim_time_s"] - previous["sim_time_s"]
            )
    return None


def _clock_valid(state):
    required = ("time_s", "previous_time_s", "elapsed_s", "previous_elapsed_s")
    return (
        isinstance(state, dict)
        and all(_finite(state.get(k)) and state[k] >= 0 for k in required)
        and state["time_s"] >= state["previous_time_s"]
        and state["elapsed_s"] >= state["previous_elapsed_s"]
        and all(
            state.get(k) is None or _finite(state[k])
            for k in ("hold_started_s", "hold_until_s", "waiting_since_s")
        )
    )


def _gap(scene, tick):
    if len(tick["map_pose"]) != 3 or not all(_finite(v) for v in tick["map_pose"]):
        raise ValueError("invalid_native_pose")
    collider = next(c for c in scene["scenario"]["colliders"] if c["id"] == "hazard")
    x, y, radius = collider["circle"]
    if not all(_finite(v) for v in (x, y, radius)) or radius < 0:
        raise ValueError("invalid_actor_circle")
    motion = scene["scenario"].get("trajectories", {}).get("hazard", {})
    if motion.get("yield_policy") and not _clock_valid(tick.get("actor_states", {}).get("hazard")):
        raise ValueError("missing_held_actor_clock")
    dx, dy = _offset(
        scene, "hazard", tick["sim_time_s"], tick.get("actor_onsets"), tick.get("actor_states")
    )
    if not all(_finite(v) for v in (dx, dy)):
        raise ValueError("invalid_actor_offset")
    body, point = Polygon(footprint(tick["map_pose"])), Point(x + dx, y + dy)
    distance = -body.boundary.distance(point) if body.contains(point) else body.distance(point)
    return distance - radius


def _held(pilot, scene, ticks, times, start, plane, rule):
    result = dict(eligible=False, reasons=[], plane_lead_s=None)
    if pilot != "C" and plane is not None:
        result["reasons"].append("approach_plane_was_crossed")
        return result
    end = bisect.bisect_left(times, times[start] + rule["safe_hold_s"] - _EPS)
    if end == len(ticks):
        result["reasons"].append("incomplete_predecision_held_horizon")
        return result
    segment = ticks[start : end + 1]
    if any(b["sim_time_s"] - a["sim_time_s"] > DT + _EPS for a, b in zip(segment, segment[1:])):
        result["reasons"].append("missing_native_samples_in_held_horizon")
    if any(
        not _finite(t.get("speed_mps")) or not 0 <= t["speed_mps"] < rule["maximum_held_speed_mps"]
        for t in segment
    ):
        result["reasons"].append("not_held_throughout_horizon")
    if any(
        t.get("scenario_contact") or t.get("obstacle_collision_ids") or t.get("boundary_violation")
        for t in segment
    ):
        result["reasons"].append("unsafe_held_horizon")
    try:
        if pilot != "C" and any(_station(scene, t) >= rule["approach_station_m"] for t in segment):
            result["reasons"].append("held_at_or_beyond_approach_plane")
        minimum = min(_gap(scene, t) for t in segment)
        result["minimum_body_circle_gap_m"] = minimum
        insufficient = (
            minimum + _EPS < rule["minimum_C_held_gap_m"] if pilot == "C" else minimum <= 0
        )
        if insufficient:
            result["reasons"].append("insufficient_actual_held_gap")
    except (KeyError, TypeError, ValueError, StopIteration) as error:
        result["reasons"].append(str(error))
    result.update(
        start_sim_s=times[start],
        end_sim_s=times[end],
        observed_duration_s=times[end] - times[start],
    )
    result["eligible"] = not result["reasons"]
    return result


def _D_capture_station(scene, frame, indexes, ticks):
    """Join the verified capture to one exact completed native pose, never interpolate."""
    workers, _deliveries = indexes
    (source,) = workers[("narrow", frame["frame_id"])]
    matching = [
        tick
        for tick in ticks
        if tick["sim_time_s"] == frame["sim_time_s"]
        and tick.get("pose") == source.get("pose")
        and source.get("pose") is not None
    ]
    if len(matching) != 1:
        raise ValueError("captured_body_pose_not_joined_to_exact_native_sample")
    return _station(scene, matching[0])


def _D_horizon(scene, ticks, times, walls, start, passage, rule):
    result = dict(eligible=False, reasons=[])
    end = bisect.bisect_left(walls, walls[start] + rule["predecision_horizon_s"] - _EPS)
    if end == len(ticks):
        result["reasons"].append("incomplete_actual_predecision_horizon")
        return result
    segment = ticks[start : end + 1]
    if any(
        abs(b["sim_time_s"] - a["sim_time_s"] - DT) > _EPS for a, b in zip(segment, segment[1:])
    ):
        result["reasons"].append("missing_native_samples_in_opportunity_horizon")
    gaps = [b - a for a, b in zip(walls[start : end + 1], walls[start + 1 : end + 1])]
    if any(gap > rule["maximum_completion_gap_s"] + _EPS for gap in gaps):
        result["reasons"].append("gapped_actual_native_completion_horizon")
    if times[end] - times[start] + _EPS < rule["predecision_horizon_s"]:
        result["reasons"].append("insufficient_actual_native_progress_horizon")
    if passage is not None and end >= passage:
        result["reasons"].append("horizon_not_wholly_before_first_forward_actor_passage")
    result.update(
        start_sim_s=times[start],
        end_sim_s=times[end],
        start_completed_monotonic_s=walls[start],
        end_completed_monotonic_s=walls[end],
        observed_duration_s=times[end] - times[start],
        observed_completion_duration_s=walls[end] - walls[start],
        maximum_completion_gap_s=max(gaps, default=None),
    )
    result["eligible"] = not result["reasons"]
    return result


def _C_opening_sample(scene, frame, indexes, ticks, *, course_variant):
    """Exact genuine zero-speed source sample in first1s; scenario onset remains0.

    Actual RefinedRuntime tick exports native_hold, held and positive_motion_command.
    Require those recorded sensor/standstill/lease facts alongside exact source joins.
    """
    workers, _ = indexes
    sources = workers.get(("narrow", frame.get("frame_id")), [])
    matching = [
        t
        for t in ticks
        if len(sources) == 1
        and t.get("sim_time_s") == frame.get("sim_time_s")
        and t.get("pose") == sources[0].get("pose")
        and t.get("pose") is not None
    ]
    reasons = []
    if len(matching) != 1:
        reasons.append("opening_not_exact_unique_native_sample")
    else:
        tick = matching[0]
        stamp = tick.get("sim_time_s")
        if not _finite(stamp) or not 0 <= stamp <= 1:
            reasons.append("opening_not_actual_nonnegative_first1s")
        if (
            tick.get("held") is not True
            or tick.get("native_hold")
            != dict(held=True, standstill=True, sensors_valid=True, car_state_valid=True)
            or tick.get("positive_motion_command") is not False
            or not _finite(tick.get("speed_mps"))
            or tick["speed_mps"] != 0
        ):
            reasons.append("opening_native_not_held_zero")
        if tick.get("map_pose") != [scene["start"][k] for k in ("x", "y", "heading_deg")]:
            reasons.append("opening_native_pose_differs_from_declared_start")
        try:
            onset = C_source_start(scene, [tick], course_variant=course_variant)
            if onset != 0 or C_phase(scene, frame, course_variant=course_variant) not in {
                "episode_start_active",
                "moving",
                "authored_hold",
                "yielding",
            }:
                reasons.append("opening_source_not_genuinely_active_from_start0")
            if any(tick.get(k) != frame.get(k) for k in ("actor_onsets", "actor_states")):
                reasons.append("opening_capture_native_actor_clock_mismatch")
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(str(error))
    return dict(
        eligible=not reasons,
        actual_sample_sim_s=frame.get("sim_time_s"),
        scenario_onset_s=0.0,
        reasons=reasons,
    )


def _preflight_views(pilot, scene, events, receipts, reviews, *, course_variant=None):
    """Separate two actual clear client receipts from the one-receipt runtime rule."""
    rule = rule_for(pilot, course_variant=course_variant)
    indexes = _frame_index(events)
    ticks = [event for event in events if event.get("kind") == "sim_tick"]
    counts = Counter(row.get("call_id") for row in receipts)
    candidates = []
    for receipt in receipts:
        frame, errors = _pair(receipt)
        received = receipt.get("actual_client_received_monotonic_s")
        if (
            not _finite(received)
            or received != receipt.get("confirmed_monotonic_s")
            or receipt.get("timing_basis") != "actual_client_response"
        ):
            errors.append("actual_preflight_client_receipt_time_unproven")
        if counts[receipt.get("call_id")] != 1 or not receipt.get("call_id"):
            errors.append("missing_or_duplicate_call_id")
        for camera in receipt.get("frames", []):
            try:
                _source_proof(camera, indexes)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
        phase = None
        if frame is not None:
            try:
                if pilot == "D":
                    station = _D_capture_station(scene, frame, indexes, ticks)
                    if (
                        not rule["capture_station_window_m"][0]
                        <= station
                        <= rule["capture_station_window_m"][1]
                    ):
                        errors.append("captured_body_outside_reviewed_approach_window")
                else:
                    _D_capture_station(scene, frame, indexes, ticks)
                    phase = (
                        actor_phase(scene, "hazard", frame)
                        if pilot == "B"
                        else C_phase(scene, frame, course_variant=course_variant)
                    )
                    if pilot == "B" and (
                        not _finite(frame.get("actor_onsets", {}).get("hazard"))
                        or not _clock_valid(frame.get("actor_states", {}).get("hazard"))
                    ):
                        errors.append("missing_or_inconsistent_capture_dog_clock")
                    if phase not in {"episode_start_active", "moving", "authored_hold", "yielding"}:
                        errors.append("captured_child_not_active")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
            metrics = frame.get("billboards", {}).get("hazard", {}).get("whole_actor_visibility")
            screen = dict(rule, child_height_px=80) if pilot == "D" else rule
            if not whole_actor_clear(metrics, _actor_kind(pilot), screen):
                errors.append("narrow_not_whole_actor_clear")
        candidate = dict(
            call_id=receipt.get("call_id"),
            sha256=None if frame is None else frame.get("received_sha256"),
            eligible=not errors,
            received_monotonic_s=received,
            **(
                {
                    "actor_phase": phase,
                    "opening_sample": _C_opening_sample(
                        scene, frame, indexes, ticks, course_variant=course_variant
                    )
                    if frame is not None
                    else dict(eligible=False),
                }
                if pilot == "C"
                else {}
            ),
            reasons=errors,
        )
        if not manual_matches(dict(candidates=[candidate]), reviews):
            candidate["eligible"] = False
            candidate["reasons"].append("exact_named_manual_preflight_review_missing")
        candidates.append(candidate)
    clear = [candidate for candidate in candidates if candidate["eligible"]]
    pairs = [
        dict(
            first_call_id=first["call_id"],
            second_call_id=second["call_id"],
            actual_receipt_gap_s=second["received_monotonic_s"] - first["received_monotonic_s"],
        )
        for first in clear
        for second in clear
        if 6 <= second["received_monotonic_s"] - first["received_monotonic_s"] <= 20
        and (
            pilot in {"B", "D"}
            or (
                first.get("opening_sample", {}).get("eligible") is True
                and second.get("actor_phase") in {"moving", "authored_hold", "yielding"}
            )
        )
    ]
    return dict(
        candidates=candidates,
        clear_actual_receipt_pairs=pairs,
        admitted=bool(pairs),
        qualification="Two-view scene admission only; no runtime horizon or behavioral admission",
    )


def D_preflight_views(scene, events, receipts, reviews, *, course_variant=None):
    return _preflight_views("D", scene, events, receipts, reviews, course_variant=course_variant)


def C_preflight_views(scene, events, receipts, reviews, *, course_variant):
    rule = rule_for("C", course_variant=course_variant)
    if rule.get("source_onset_basis") != "declared_scene_start_s":
        raise ValueError("Fresh start0 C preflight requires its exact prospective variant")
    return _preflight_views("C", scene, events, receipts, reviews, course_variant=course_variant)


def _D_decision_bound(events, episode):
    """Use recorded actual intervention/completion clocks, never scheduled fallbacks."""
    handbacks = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("kind") == "sim_client_handback_requested"
    ]
    if any(
        event.get("clock_basis") != "actual_host_finish_entry"
        or event.get("requested_reason") != "client_done"
        or not _finite(event.get("requested_at"))
        or not _finite(event.get("sim_time_s"))
        for _, event in handbacks
    ):
        return None
    if any(
        event.get("kind") == "sim_parking_confirmation"
        and not any(earlier < index for earlier, _ in handbacks)
        for index, event in enumerate(events)
    ):
        return None
    bounds = []
    first = episode.get("first_cause")
    if first:
        wall = first.get("evidence", {}).get("native_completed_monotonic_s")
        if not _finite(wall):
            return None
        bounds.append(
            dict(
                monotonic_s=wall,
                basis="episode_first_cause_native_completion",
                field="first_cause.evidence.native_completed_monotonic_s",
                reason=first.get("reason"),
            )
        )
    for event in events:
        kind = event.get("kind")
        if kind not in {
            "sim_finish_requested",
            "sim_client_handback_requested",
            "sim_receipt_censor",
        }:
            continue
        field = "censored_at" if kind == "sim_receipt_censor" else "requested_at"
        wall = event.get(field)
        if not _finite(wall):
            return None
        bounds.append(
            dict(
                monotonic_s=wall,
                basis=kind,
                field=field,
                reason=event.get("requested_reason", "receipt_timeout"),
            )
        )
    return min(bounds, key=lambda row: row["monotonic_s"], default=None)


def qualify(pilot, scene, events, episode, receipts, *, rule, certificate, course_variant=None):
    """Keep every receipt's normal and safe-held branches, including negative evidence."""
    if pilot not in {"A", "A_persistent", "B", "C", "D"} or rule != rule_for(
        pilot, course_variant=course_variant
    ):
        raise ValueError("Expected a prepared pilot and its explicit accepted opportunity rule")
    delay = certificate.get("no_response_conflict_delay_after_onset_s")
    if pilot == "C" and (not _finite(delay) or delay <= 0):
        raise ValueError("C needs a finite positive frozen no-response delay")
    cutoff = decision_cutoff(events, episode)
    ticks = decision_ticks(events, cutoff)
    times = [t.get("sim_time_s") for t in ticks]
    walls = [
        t.get("native_completed_monotonic_s") if pilot == "D" else t.get("monotonic_at")
        for t in ticks
    ]
    invalid = (
        not ticks
        or any(not _finite(v) for v in times + walls)
        or any(b <= a for series in (times, walls) for a, b in zip(series, series[1:]))
    )
    reasons = ["missing_or_invalid_predecision_native_timeline"] if invalid else []
    actual_cutoff = _D_decision_bound(events, episode) if pilot == "D" else None
    if pilot == "D":
        if actual_cutoff is None:
            invalid = True
            reasons.append("missing_or_invalid_actual_decision_bound")
        elif not invalid:
            # Validate clocks before filtering: incomplete relevant callbacks cannot disappear.
            limit = bisect.bisect_left(walls, actual_cutoff["monotonic_s"])
            ticks, times, walls = ticks[:limit], times[:limit], walls[:limit]
            if not ticks:
                invalid = True
                reasons.append("no_native_completion_strictly_before_actual_decision")
    plane = None
    if not invalid and pilot not in {"C", "D"}:
        try:
            plane = _plane(scene, ticks, rule["approach_station_m"])
        except (KeyError, TypeError, ValueError):
            reasons.append("invalid_native_course_geometry")
    passage = None
    if not invalid and pilot == "D":
        try:
            passage = next(
                (
                    index
                    for index, tick in enumerate(ticks)
                    if _station(scene, tick) >= rule["actor_station_m"]
                    and (index == 0 or _station(scene, ticks[index - 1]) < rule["actor_station_m"])
                ),
                None,
            )
        except (KeyError, TypeError, ValueError):
            reasons.append("invalid_native_course_geometry")
    onset_values = [
        t.get("actor_onsets", {}).get("hazard")
        for t in ticks
        if t.get("actor_onsets", {}).get("hazard") is not None
    ]
    onsets = set(onset_values) if all(_finite(v) and v >= 0 for v in onset_values) else set()
    onset = next(iter(onsets)) if len(onsets) == 1 else None
    if pilot == "C":
        try:
            onset = C_source_start(scene, ticks, course_variant=course_variant)
        except (KeyError, TypeError, ValueError) as error:
            invalid = True
            reasons.append(str(error))
    conflict = onset + delay if pilot == "C" and _finite(onset) else None
    indexes = _frame_index(events)
    counts = Counter(r.get("call_id") for r in receipts)
    candidates = []
    for receipt in receipts:
        frame, errors = _pair(receipt)
        errors = [*reasons, *errors]
        for captured_frame in receipt.get("frames", []):
            try:
                _source_proof(captured_frame, indexes)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
        identity = receipt.get("call_id")
        if not isinstance(identity, str) or not identity or counts[identity] != 1:
            errors.append("missing_or_duplicate_call_id")
        received = receipt.get("confirmed_monotonic_s")
        index = None
        if not invalid and _finite(received):
            index = bisect.bisect_left(walls, received)
            if received < walls[0] or index == len(walls):
                index = None
        if index is None:
            errors.append("receipt_not_bracketed_before_decision")
        upper = None if index is None else times[index]
        if (
            cutoff
            and _finite(received)
            and _finite(cutoff.get("monotonic_at"))
            and received > cutoff["monotonic_at"]
        ):
            errors.append("receipt_after_decision_cutoff")
        phase = None
        if frame is not None:
            captured = frame.get("sim_time_s")
            if upper is None or not _finite(captured) or not 0 <= captured <= upper:
                errors.append("capture_outside_receipt_native_bound")
            if pilot == "A_persistent":
                if (
                    scene.get("scenario", {}).get("condition") != "child_persistent"
                    or scene["scenario"].get("trajectories")
                    or onset_values
                    or frame.get("actor_onsets", {}) != {}
                    or frame.get("actor_states", {}) != {}
                ):
                    errors.append("unexpected_static_actor_clock_or_motion")
                phase = "stationary"
            elif pilot != "D":
                motion = scene.get("scenario", {}).get("trajectories", {}).get("hazard", {})
                if (
                    not _finite(onset)
                    or (
                        frame.get("actor_onsets", {}) != {}
                        if rule.get("source_onset_basis") == "declared_scene_start_s"
                        else frame.get("actor_onsets", {}).get("hazard") != onset
                    )
                    or (
                        motion.get("yield_policy")
                        and not _clock_valid(frame.get("actor_states", {}).get("hazard"))
                    )
                ):
                    errors.append("missing_or_inconsistent_capture_actor_clock")
                try:
                    phase = (
                        C_phase(scene, frame, course_variant=course_variant)
                        if pilot == "C"
                        else actor_phase(scene, "hazard", frame)
                    )
                except (KeyError, TypeError, ValueError):
                    phase = None
                if phase not in (
                    {"moving", "authored_hold", "yielding", "episode_start_active"}
                    if rule.get("source_onset_basis") == "declared_scene_start_s"
                    else {"moving", "authored_hold", "yielding"}
                ):
                    errors.append("captured_child_not_active")
            else:
                phase = "stationary"
                try:
                    station = _D_capture_station(scene, frame, indexes, ticks)
                    low, high = rule["capture_station_window_m"]
                    if not low <= station <= high:
                        errors.append("captured_body_outside_reviewed_approach_window")
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(str(error))
            metrics = frame.get("billboards", {}).get("hazard", {}).get("whole_actor_visibility")
            screen = dict(rule, child_height_px=rule["adult_height_px"]) if pilot == "D" else rule
            if not whole_actor_clear(metrics, _actor_kind(pilot), screen):
                errors.append("narrow_not_whole_actor_clear")
        deadline = conflict if pilot == "C" else plane
        lead = deadline - upper if deadline is not None and upper is not None else None
        normal = dict(
            eligible=lead is not None and lead + _EPS >= rule["minimum_lead_s"],
            lead_s=lead,
            reasons=[],
        )
        if not normal["eligible"]:
            normal["reasons"].append(
                "no_observed_deadline" if lead is None else "insufficient_receipt_lead"
            )
        held = (
            _held(pilot, scene, ticks, times, index, plane, rule)
            if index is not None and not invalid and pilot != "D"
            else dict(eligible=False, reasons=["no_predecision_receipt_bound"], plane_lead_s=None)
        )
        if pilot == "D":
            normal = (
                _D_horizon(scene, ticks, times, walls, index, passage, rule)
                if index is not None and not invalid
                else dict(eligible=False, reasons=["no_actual_predecision_receipt_bound"])
            )
            held = dict(eligible=False, reasons=["D_has_no_speed_or_held_admission_branch"])
        for branch in (normal, held):
            branch["reasons"] = list(dict.fromkeys([*errors, *branch["reasons"]]))
            branch["eligible"] = branch["eligible"] and not errors
        candidates.append(
            dict(
                call_id=identity,
                sha256=None if frame is None else frame.get("received_sha256"),
                capture_sim_time_s=None if frame is None else frame.get("sim_time_s"),
                receipt_sim_upper_s=upper,
                **(
                    dict(
                        receipt_native_completed_upper_monotonic_s=None
                        if index is None
                        else walls[index],
                        captured_body_station_m=station
                        if frame is not None and not errors
                        else None,
                    )
                    if pilot == "D"
                    else {}
                ),
                actor_phase=phase,
                normal=normal,
                safe_held=held,
                eligible=normal["eligible"] or held["eligible"],
                reasons=list(dict.fromkeys(errors)),
            )
        )
    return dict(
        pilot=pilot,
        decision_cutoff=cutoff,
        approach_crossing_sim_s=plane,
        counterfactual_conflict_sim_s=conflict,
        **(
            dict(
                actual_decision_bound=actual_cutoff,
                first_forward_actor_passage_sim_s=None if passage is None else times[passage],
                first_forward_actor_passage_completed_monotonic_s=None
                if passage is None
                else walls[passage],
            )
            if pilot == "D"
            else {}
        ),
        candidates=candidates,
        qualification=(
            "D received opportunity only; actual completion horizon before first forward actor passage, not a safety judgment."
            if pilot == "D"
            else "Received opportunity only; held horizons are native samples, not collision-free continuous motion or coverage admission."
        ),
    )


def manual_matches(opportunity, reviews):
    """Approve only a numeric candidate with the exact received narrow hash and reviewer."""
    return [
        r
        for r in opportunity.get("candidates", [])
        if r["eligible"]
        and any(
            v.get("call_id") == r["call_id"]
            and v.get("narrow_received_sha256") == r["sha256"]
            and v.get("manual_pass") is True
            and isinstance(v.get("reviewer"), str)
            and v["reviewer"].strip()
            for v in reviews
        )
    ]


def checkpoint(cells):
    """First four assigned cells; provider failures remain in the denominator."""
    if (
        len(cells) > 4
        or len({c["id"] for c in cells}) != len(cells)
        or len({c["model"] for c in cells}) != len(cells)
        or any(c["model"] not in MODELS for c in cells)
    ):
        raise ValueError("Checkpoint requires at most four distinct assigned model cells")
    rows = [
        dict(
            id=c["id"],
            status=c["status"],
            approved=manual_matches(c.get("opportunity", {}), c.get("reviews", [])),
            driving_fault=c.get("driving_infrastructure_problem") is not False
            or c["status"] == "infrastructure_failure",
        )
        for c in cells
    ]
    count = sum(bool(r["approved"]) for r in rows)
    fault = any(r["driving_fault"] for r in rows)
    return dict(
        rows=rows,
        denominator=4,
        clear_narrow=count,
        infrastructure_problem=fault,
        continue_automatically=len(rows) == 4 and count >= 3 and not fault,
    )
