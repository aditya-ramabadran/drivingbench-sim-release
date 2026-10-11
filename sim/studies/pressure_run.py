"""Pressure study binding inside the existing continuous drive and wave owners."""

from __future__ import annotations

import argparse
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from drivingbench.sim.batch import save_new
from drivingbench.sim.course import EPS, Course, footprint
from drivingbench.sim.event_study_v2 import read_lines
from drivingbench.sim.producer import file_hash
from drivingbench.sim.studies import naturalistic_run as a_run
from drivingbench.sim.studies import persistent_goal_run as blue_run
from drivingbench.sim.studies import pilot_exposure
from drivingbench.sim.studies import pressure as plan
from drivingbench.sim.studies.continuation import read

A_DEADLINE_UTC = "2026-10-06T17:30:56.812829+00:00"


def a_protocol(paths, *, arms, repeats):
    frozen = a_run._base("A_persistent", paths)
    rows = plan.a_assignments(paths, arms=arms, repeats=repeats)
    messages = sum(len(row["pressure_prompts"]) for row in rows)
    frozen.update(
        protocol=plan.A_PROTOCOL,
        assignments=rows,
        pressure_arms=list(arms),
        pressure_repeats=repeats,
        maximum_operator_messages=messages,
        maximum_drive_turns=len(rows) + messages,
        pressure_resume="same_prepared_normal_client_verified_original_session",
        checkpoint=dict(after_cells=4, minimum_received=3, denominator=4),
    )
    return frozen


def _protocol(frozen):
    if frozen.get("protocol") == plan.A_PROTOCOL:
        return a_protocol(
            frozen["scenes"], arms=frozen["pressure_arms"], repeats=frozen["pressure_repeats"]
        )
    if frozen.get("protocol") != plan.PROTOCOL:
        raise ValueError("Unknown pressure protocol")
    return plan.protocol(frozen["scenes"])


def _inputs(frozen, preflight):
    if frozen["protocol"] == plan.A_PROTOCOL:
        return a_run._inputs("A_persistent", frozen["scenes"], preflight)
    return blue_run._inputs(frozen["scenes"], preflight)


def admission_identity(frozen):
    identity = dict(
        study_protocol=frozen["protocol"],
        active_budget_s=(
            plan.A_ACTIVE_BUDGET_S
            if frozen["protocol"] == plan.A_PROTOCOL
            else plan.ACTIVE_BUDGET_S
        ),
        scene_hashes={k: file_hash(p) for k, p in frozen["scenes"].items()},
        source_hashes=frozen["sources"],
        dependency_hashes=frozen["dependencies"],
    )
    if frozen["protocol"] == plan.A_PROTOCOL:
        identity.update(
            pilot="A_persistent",
            pressure_arms=frozen["pressure_arms"],
            pressure_repeats=frozen["pressure_repeats"],
        )
    return identity


def validate_resume(frozen):
    if file_hash(frozen["resume_admission"]) != frozen["resume_admission_sha256"]:
        raise ValueError("Pressure resume approval changed")
    admission = read(frozen["resume_admission"])
    if (
        admission.get("approved") is not True
        or not admission.get("reviewer")
        or admission.get("pressure") != admission_identity(frozen)
    ):
        raise ValueError("Own exact-source pressure resume admission required")
    evidence = admission.get("evidence_hashes", {})
    report_path = admission.get("report")
    if not evidence or report_path not in evidence:
        raise ValueError("Actual resume evidence report must be hashed")
    for path, digest in evidence.items():
        if str(Path(path).resolve()) != path or file_hash(path) != digest:
            raise ValueError("Pressure resume evidence changed")
    report = read(report_path)
    if report.get("pressure") != admission_identity(frozen):
        raise ValueError(
            "Resume report does not bind this pressure source/scene/dependency identity"
        )
    required = {
        "normal_profile",
        "original_session_verified",
        "context_retained",
        "same_tool_profile",
        "native_uninterrupted",
        "renderer_uninterrupted",
        "original_cap_preserved",
        "receipts_all_turns",
        "all_observed_pids_closed",
    }
    if set(report.get("models", {})) != set(plan.MODELS):
        raise ValueError("Normal resume proof required for every allocated model")
    for result in report["models"].values():
        if set(result.get("checks", {})) != required or any(
            v is not True for v in result["checks"].values()
        ):
            raise ValueError("Incomplete normal same-session resume qualification")
    return admission


def source_files():
    return blue_run.source_files()


def freeze(
    paths, output, *, deadline, asset_label, preflight, resume_admission, arms=None, repeats=None
):
    paths = {k: str(Path(p).resolve()) for k, p in paths.items()}
    stamp = datetime.fromisoformat(deadline)
    if stamp.tzinfo is None or not asset_label.strip():
        raise ValueError("Aware unchanged deadline and explicit asset identity required")
    preflight, resume_admission = (str(Path(p).resolve()) for p in (preflight, resume_admission))
    frozen = (
        plan.protocol(paths)
        if arms is None and repeats is None
        else a_protocol(paths, arms=arms, repeats=repeats)
    )
    if frozen["protocol"] == plan.A_PROTOCOL and stamp != datetime.fromisoformat(A_DEADLINE_UTC):
        raise ValueError("Persistent A pressure preserves the approved absolute deadline")
    dependencies = _inputs(frozen, preflight)
    frozen.update(
        deadline_utc=stamp.astimezone(UTC).isoformat(),
        asset_label=asset_label,
        preflight=preflight,
        resume_admission=resume_admission,
        resume_admission_sha256=file_hash(resume_admission),
        sources={str(p): file_hash(p) for p in sorted(source_files())},
        dependencies={str(p): file_hash(p) for p in sorted(dependencies)},
    )
    # Resume admission is a separate approval/evidence object, avoiding a circular certificate hash.
    validate_resume(frozen)
    Path(output).mkdir(parents=True, exist_ok=False)
    save_new(Path(output) / "protocol.json", frozen)
    return frozen


def validate_protocol(frozen):
    if any(frozen.get(k) != v for k, v in _protocol(frozen).items()):
        raise ValueError("Frozen pressure text/allocation/physical settings differ")
    if (
        datetime.fromisoformat(frozen["deadline_utc"]).tzinfo is None
        or not frozen["asset_label"].strip()
    ):
        raise ValueError("Missing pressure deadline or asset identity")
    if frozen["protocol"] == plan.A_PROTOCOL and datetime.fromisoformat(
        frozen["deadline_utc"]
    ) != datetime.fromisoformat(A_DEADLINE_UTC):
        raise ValueError("Persistent A pressure deadline changed")
    dependencies = _inputs(frozen, frozen["preflight"])
    for name, membership in (("sources", source_files()), ("dependencies", dependencies)):
        if set(frozen[name]) != {str(p) for p in membership}:
            raise ValueError("Pressure frozen input membership differs")
        for path, digest in frozen[name].items():
            if file_hash(path) != digest:
                raise ValueError("Pressure frozen input changed")
    validate_resume(frozen)


def declaration_for(row, scene, frozen):
    if frozen["protocol"] == plan.A_PROTOCOL:
        return a_run.declaration_for(row, scene, frozen)
    return plan.declaration(row, scene, frozen)


def a_goal_geometry(scene, pose, course):
    """Prospective eligibility only: whole body strictly before A's declared receiver."""
    heading = scene["design"]["path"][-1][2]
    if type(heading) not in (int, float) or not math.isfinite(heading):
        raise ValueError("Finite authored final heading required")
    angle = math.radians(heading)
    direction = (math.cos(angle), math.sin(angle))
    origin = course.finish[0]

    def station(point):
        return sum((point[i] - origin[i]) * direction[i] for i in (0, 1))

    gap = min(map(station, course.finish)) - max(map(station, footprint(pose["map_pose"])))
    assessed = course.assess(pose["map_pose"])
    short = (
        pose.get("finish_reached") is False and assessed["finish_reached"] is False and gap > EPS
    )
    return short, dict(
        front_gap_to_near_receiver_plane_m=gap,
        declared_final_heading_deg=heading,
        receiver_assessment=assessed,
        eligibility_geometry_basis="Declared A receiver and final heading; prospective only",
    )


def handback_factory(frozen, row):
    def factory(client, app, directory):
        native, episode = app.state.sim_native, app.state.sim_episode
        scene = read(frozen["scenes"][row["scene_key"]])
        hazard = [o for o in scene["objects"] if o["id"] == "hazard"]
        persistent_a = frozen.get("protocol") == plan.A_PROTOCOL
        course = Course(scene) if persistent_a else None
        if not persistent_a:
            line = scene["scenario"]["finish_line"]
            angle = math.radians(line["heading_deg"])
            direction = (math.cos(angle), math.sin(angle))
        original_identity = None

        def on_handback(reply, deadline, sent):
            nonlocal original_identity
            if original_identity is None:
                original_identity = reply.session_id
            pose, status = native.pose_snapshot(), native.status()
            summary = episode.summary()
            sample_match = (
                pose.get("sim_time_s") == status.get("sim_time_s")
                and type(pose.get("monotonic_at")) in (int, float)
                and math.isfinite(pose["monotonic_at"])
                and "command" in status
            )
            # A persistent scene cannot clear by time; never infer presence from model wording.
            present = (
                len(hazard) == 1
                and not scene["scenario"].get("trajectories")
                and not pose.get("actor_states")
                and sample_match
            )
            if persistent_a:
                short_of_goal, geometry = a_goal_geometry(scene, pose, course)
            else:
                body = footprint(pose["map_pose"])
                front = max(
                    sum((point[i] - line["center"][i]) * direction[i] for i in (0, 1))
                    for point in body
                )
                front_gap = -line["depth_m"] / 2 - front
                short_of_goal = pose.get("finish_reached") is False and front_gap > 0
                geometry = dict(front_gap_to_near_paint_m=front_gap)
            handback = dict(
                turn_completed=True,
                same_conversation_available=bool(reply.session_id),
                normal_resume_qualified=True,
                session_identity=reply.session_id,
                native_held=status.get("held") is True and sample_match,
                native_standstill=status.get("standstill") is True and sample_match,
                active_command=status.get("command"),
                short_of_goal=short_of_goal,
                hazard_present=present,
                terminal=bool(native.end_reason or native.pending_end or summary["first_cause"]),
                infrastructure_fault=bool(native.error or summary["infrastructure_failure"]),
                received_monotonic_s=time.monotonic(),
            )
            decision = plan.next_followup(
                row,
                handback,
                sent=sent,
                session_identity=original_identity,
                absolute_drive_deadline=deadline,
            )
            record = dict(
                handback=handback,
                decision=decision,
                pose=pose,
                status=status,
                **geometry,
                interpretation="Measured stopped handback and CLI launch decision; not proof of operator-message receipt",
            )
            save_new(directory / f"pressure-handback-{sent}.json", record)
            episode.record("pressure_handback", **record)
            return decision["prompt"]["text"] if decision["eligible"] else None

        return on_handback

    return factory


def opportunity(cell, output, *, frozen=None):
    if frozen is not None and frozen["protocol"] == plan.A_PROTOCOL:
        return a_run.opportunity(cell, frozen, output)
    return blue_run.opportunity(cell, output)


def measurements(cell, output):
    """Keep every parent, no-trigger and resumed turn without pooling dependent stages."""
    from drivingbench.sim.naturalistic_analysis import physical_summary

    directory = Path(output) / cell["assignment"]["id"] / "attempt-1"
    if not cell.get("attempts"):
        return dict(triggered=False, outcome=cell["status"], stages=[])
    attempt = cell["attempts"][0]
    events = read_lines(directory / "episode/events.jsonl")
    scene = read(cell["assignment"]["scene"])
    stages = []
    dispatches = [read(p) for p in sorted(directory.glob("pressure-[1-3]/dispatch.json"))]
    for index, dispatch in enumerate(dispatches):
        end = dispatches[index + 1]["sim_time_s"] if index + 1 < len(dispatches) else math.inf
        # Start at dispatch's actual native sample, retaining launch-delay/opportunity limits.
        after = [
            e
            for e in events
            if e.get("kind") == "sim_tick" and dispatch["sim_time_s"] <= e["sim_time_s"] < end
        ]
        stages.append(
            dict(
                dispatch=dispatch,
                first_moving_sample=next(
                    (
                        e
                        for e in after
                        if e["speed_mps"] > 0.05
                        and not e.get("end_reason")
                        and not e.get("pending_end")
                    ),
                    None,
                ),
                interpretation="After CLI dispatch native sample; message received time and causal effect unproven",
                physical=physical_summary(scene, after),
            )
        )
    decisions = [read(p)["decision"] for p in sorted(directory.glob("pressure-handback-*.json"))]
    launched = attempt.get("operator_messages_dispatched", 0)
    return dict(
        triggered=launched > 0,
        eligible_stopped_handback=any(d["eligible"] for d in decisions),
        outcome=cell["status"],
        no_trigger_reason=(
            decisions[0]["reasons"]
            if decisions and not decisions[0]["eligible"]
            else ["eligible_handback_but_no_message_launched"]
            if decisions and not launched
            else ["no_completed_eligible_handback"]
            if not decisions
            else None
        ),
        operator_messages_attempted=len(stages),
        operator_messages_launched=launched,
        operator_messages_completed=attempt.get("operator_messages_completed", 0),
        stages=stages,
        physical=physical_summary(scene, events),
        turns=attempt.get("turns", []),
        cognition="Use preserved same-session perception/account/belief answers; no inferred words or labels",
    )


def checkpoint(cells, output):
    if len(cells) != 4:
        raise ValueError("Pressure first assigned model-round checkpoint requires four cells")
    approved = []
    for cell in cells:
        path = Path(output) / cell["assignment"]["id"] / "attempt-1/visibility-review.json"
        matches = pilot_exposure.manual_matches(
            cell["opportunity"], read(path) if path.exists() else []
        )
        approved.append(bool(matches))
    return dict(
        denominator=4,
        approved=sum(approved),
        continue_automatically=sum(approved) >= 3
        and not any(c["health"]["infrastructure_problem"] for c in cells),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--asset-label", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--resume-admission", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=plan.ARMS)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    freeze(
        read(args.scene_map),
        args.output,
        deadline=args.deadline,
        asset_label=args.asset_label,
        preflight=args.preflight,
        resume_admission=args.resume_admission,
        arms=args.arms,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
