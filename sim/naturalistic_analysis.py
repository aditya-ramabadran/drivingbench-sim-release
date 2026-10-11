"""Read-only naturalistic exposure and geometry summaries; never infer intent."""

from __future__ import annotations

import copy
import math

from drivingbench.sim.actor_yield import elapsed_at
from drivingbench.sim.course import footprint
from drivingbench.sim.hazard_evidence import response_timing
from drivingbench.sim.scenarios import _motion_start, _offset
from shapely.geometry import Point, Polygon


def actor_phase(scene, identity, snapshot):
    """Use captured simulated time and latched onset, including delayed actors."""
    motion = scene.get("scenario", {}).get("trajectories", {}).get(identity)
    if motion is None:
        return "stationary"
    start = _motion_start(motion, identity, snapshot.get("actor_onsets"))
    time = snapshot["sim_time_s"]
    if start is None or time <= start:
        return "before_motion"
    state = snapshot.get("actor_states", {}).get(identity, {})
    holding = (
        state.get("hold_started_s") is not None
        and state["hold_started_s"] <= time < state["hold_until_s"]
    )
    if holding:
        return "authored_hold"
    if state.get("paused") and time >= state["waiting_since_s"]:
        return "yielding"
    if (
        elapsed_at(snapshot.get("actor_states"), identity, time, time - start)
        < motion["duration_s"]
    ):
        return "moving"
    return "after_motion"


def exposure_summary(scene, receipt, calls=(), client_actions=()):
    """Keep receipt, visible motion and command ordering distinct for every actor."""
    identities = set(scene.get("scenario", {}).get("trajectories", {}))
    identities.update(c["id"] for c in scene.get("scenario", {}).get("colliders", []))
    result = {}
    for identity in sorted(identities):
        views = []
        for observed in receipt["observations"]:
            for frame in observed["frames"]:
                if (
                    frame.get("accepted")
                    and frame.get("billboards", {}).get(identity, {}).get("visible_pixels", 0) > 0
                ):
                    views.append(
                        dict(
                            call_id=observed["call_id"],
                            event_index=observed["event_index"],
                            camera=frame["camera"],
                            frame_id=frame.get("frame_id"),
                            sim_time_s=frame["sim_time_s"],
                            phase=actor_phase(scene, identity, frame),
                            received_path=frame.get("received_path"),
                        )
                    )
        timings = {}
        for selection in ("first_visible", "first_moving_visible"):
            selected, originals = [], {}
            for observed in receipt["observations"]:
                frames = []
                for frame in observed["frames"]:
                    visible = frame.get("billboards", {}).get(identity, {}).get("visible_pixels", 0)
                    if not frame.get("accepted") or visible <= 0:
                        continue
                    if (
                        selection == "first_moving_visible"
                        and actor_phase(scene, identity, frame) != "moving"
                    ):
                        continue
                    adapted = copy.deepcopy(frame)
                    # The shared timing helper selects the named hazard; retain original evidence below.
                    adapted["billboards"]["hazard"] = {"visible_pixels": visible}
                    frames.append(adapted)
                    originals[(observed["call_id"], frame["camera"])] = frame
                selected.append(dict(observed, frames=frames))
            timing = response_timing(dict(receipt, observations=selected), calls, client_actions)
            for camera, item in timing.items():
                if item is not None:
                    item["frame"] = originals[(item["call_id"], camera)]
                    item["actor_id"] = identity
                    item["interpretation"] = (
                        "Receipt/action ordering, not a causal reaction or attention measure"
                    )
            timings[selection] = timing
        result[identity] = dict(visible_frames=views, timing=timings)
    return result


def _reference(tick):
    return {k: tick[k] for k in ("sequence", "sim_time_s", "map_pose", "speed_mps") if k in tick}


def physical_summary(scene, events, *, terminal_time_s=math.inf):
    """Sample actual moving geometry, retaining signed gaps and authoritative contacts."""
    ticks = [
        e
        for e in events
        if e.get("kind") == "sim_tick"
        and e["sim_time_s"] <= terminal_time_s
        and not e.get("end_reason")
        and e.get("enabled") is not False
    ]
    colliders = scene.get("scenario", {}).get("colliders", [])
    minimum = {c["id"]: None for c in colliders}
    onsets, contacts, stops, resumes, decelerations = {}, [], [], [], []
    moving_seen, stopped, previous_decelerating = False, None, False
    path_length = 0.0
    for index, tick in enumerate(ticks):
        time, speed = tick["sim_time_s"], tick["speed_mps"]
        for identity, onset in tick.get("actor_onsets", {}).items():
            if identity not in onsets:
                onsets[identity] = dict(onset_s=onset, first_logged=_reference(tick))
        for contact in (tick.get("scenario_contact") or {}).get("contacts", []):
            if not any(c["contact"] == contact for c in contacts):
                contacts.append(
                    dict(
                        contact=contact,
                        first_logged=_reference(tick),
                        assessment=tick["scenario_contact"],
                    )
                )
        if index:
            previous = ticks[index - 1]
            path_length += math.dist(previous["map_pose"][:2], tick["map_pose"][:2])
            dt = time - previous["sim_time_s"]
            later = next(
                (t for t in ticks[index : index + 20] if t["sim_time_s"] >= time + 0.1 - 1e-8), None
            )
            decelerating = (
                dt > 0
                and previous["speed_mps"] > 0.1
                and (speed - previous["speed_mps"]) / dt < -0.05
                and later is not None
                and speed - later["speed_mps"] >= 0.02
            )
            if decelerating and not previous_decelerating:
                decelerations.append(_reference(tick))
            previous_decelerating = decelerating
        if speed > 0.05:
            moving_seen = True
            if stopped is not None:
                stopped["end_s"] = time
                stopped["duration_s"] = time - stopped["start_s"]
                resumes.append(_reference(tick))
                stopped = None
        elif moving_seen and stopped is None:
            stopped = dict(start_s=time, end_s=None, duration_s=None, start=_reference(tick))
            stops.append(stopped)
        body = Polygon(footprint(tick["map_pose"]))
        for collider in colliders:
            identity = collider["id"]
            dx, dy = _offset(
                scene, identity, time, tick.get("actor_onsets"), tick.get("actor_states")
            )
            if "circle" in collider:
                x, y, radius = collider["circle"]
                point = Point(x + dx, y + dy)
                distance = body.distance(point)
                if body.contains(point):
                    distance = -body.boundary.distance(point)
                gap = distance - radius
            else:
                shape = Polygon([(x + dx, y + dy) for x, y in collider["polygon"]])
                gap = body.distance(shape)
            if minimum[identity] is None or gap < minimum[identity]["clearance_m"]:
                minimum[identity] = dict(clearance_m=gap, tick=_reference(tick), offset=[dx, dy])
    if stopped is not None and ticks:
        stopped["observed_until_s"] = ticks[-1]["sim_time_s"]
        stopped["observed_duration_s"] = ticks[-1]["sim_time_s"] - stopped["start_s"]
    return dict(
        samples=len(ticks),
        path_length_m=path_length,
        actor_onsets=onsets,
        minimum_sampled_clearance=minimum,
        contacts=contacts,
        moving_to_stopped=stops,
        resumptions=resumes,
        deceleration_onsets=decelerations,
        first_tick=_reference(ticks[0]) if ticks else None,
        last_tick=_reference(ticks[-1]) if ticks else None,
        note="Circle gap is signed and not clipped; polygon overlap has zero distance. Samples do not replace swept runtime contact. Stops use speed<=0.05 after movement, not policy dwell. Terminal freeze must be excluded by supplied cutoff/end_reason; no causal command attribution.",
    )


def decision_cutoff(events, episode):
    """Find the first intervention request, keeping physical braking as later evidence."""
    candidates = []
    first = episode.get("first_cause")
    if first:
        data = first["evidence"]
        candidates.append(
            dict(
                sim_time_s=data["sim_time_s"],
                monotonic_at=data.get("monotonic_at"),
                reason=first["reason"],
                basis="episode_first_cause",
                inclusive=True,
            )
        )
    for row in events:
        kind = row.get("kind")
        if kind in {"sim_finish_requested", "sim_client_handback_requested"}:
            candidates.append(
                dict(
                    sim_time_s=row["sim_time_s"],
                    monotonic_at=row.get("requested_at", row.get("monotonic_at")),
                    reason=row["requested_reason"],
                    basis=kind,
                    inclusive=kind != "sim_client_handback_requested",
                )
            )
        elif kind == "sim_receipt_censor":
            candidates.append(
                dict(
                    sim_time_s=row["sim_time_s"],
                    monotonic_at=row.get("censored_at"),
                    reason="receipt_timeout",
                    basis=kind,
                    inclusive=True,
                )
            )
        elif kind == "sim_tick" and (
            row.get("pending_end") or row.get("end_reason") or row.get("enabled") is False
        ):
            candidates.append(
                dict(
                    sim_time_s=row["sim_time_s"],
                    monotonic_at=row.get("monotonic_at"),
                    reason=row.get("pending_end") or row.get("end_reason") or "native_disabled",
                    basis="first_termination_tick_fallback",
                    inclusive=False,
                )
            )
    return min(candidates, key=lambda row: (row["sim_time_s"], not row["inclusive"]), default=None)


def decision_ticks(events, cutoff):
    """Exclude cleanup and disabled ticks even when their sampled time equals the request."""
    if cutoff is None:
        return []
    return [
        row
        for row in events
        if row.get("kind") == "sim_tick"
        and not row.get("pending_end")
        and not row.get("end_reason")
        and row.get("enabled") is not False
        and (
            row["sim_time_s"] <= cutoff["sim_time_s"]
            if cutoff["inclusive"]
            else row["sim_time_s"] < cutoff["sim_time_s"]
        )
    ]
