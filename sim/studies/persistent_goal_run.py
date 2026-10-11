"""Exclusive serial collection of the persistent blue-line study."""

from __future__ import annotations

import argparse
import bisect
import copy
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from drivingbench.sim import event_study_v2 as study
from drivingbench.sim.batch import save_new
from drivingbench.sim.course import finish_before_line
from drivingbench.sim.naturalistic_analysis import decision_cutoff, decision_ticks
from drivingbench.sim.producer import file_hash
from drivingbench.sim.scenarios import validate_scenario
from drivingbench.sim.scene import scene_asset_paths
from drivingbench.sim.studies import (
    no_excuse,
    pilot_exposure,
    pilot_health,
)
from drivingbench.sim.studies import (
    persistent_goal as plan,
)
from drivingbench.sim.studies import (
    persistent_goal_preflight as proof,
)
from drivingbench.sim.studies.continuation import postdrive_policy, read
from drivingbench.sim.studies.naturalistic_run import _annotations, await_reviews, runtime_files
from drivingbench.sim.studies.persistent_goal_mcp import configuration

NATIVE_CHECKS = (
    "stop_ceiling_request",
    "expiry",
    "stop_resume",
    "persistent_wait",
    "free_left_bypass",
    "free_right_bypass",
    "contact",
    "finish_before_line",
)


def _scenes(paths):
    plan.assignments(paths)
    shared = []
    for key, path in paths.items():
        scene = validate_scenario(read(path))
        scenario = scene["scenario"]
        if (
            scenario["type"] != "course_event"
            or scenario["time_scale"] != 1
            or scenario["time_limit_s"] != plan.ACTIVE_BUDGET_S
            or scenario["condition"] != key
            or "wide_camera" not in scene
            or scenario.get("trajectories")
            or scenario.get("branch_select")
            or not scenario.get("coverage")
            or scenario["coverage"].get("max_heading_deviation_deg") != 180
        ):
            raise ValueError("Expected paired real-time persistent goal scene without actor motion")
        line = scenario["finish_line"]
        angle = math.radians(line["heading_deg"])
        forward, left = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
        edge = [line["center"][i] - forward[i] * line["depth_m"] / 2 for i in (0, 1)]
        ends = [
            [edge[i] + sign * left[i] * line["width_m"] / 2 for i in (0, 1)] for sign in (1, -1)
        ]
        expected = finish_before_line(*ends)["polygon"]
        actual = scene["finish"]["polygon"]
        if len(actual) != len(expected) or any(
            math.dist(a, b) > 1e-8 for a, b in zip(actual, expected)
        ):
            raise ValueError("Finish must be the shared body-based receiver before the blue line")
        actors = [o for o in scene["objects"] if o["id"] == "hazard"]
        colliders = scenario["colliders"]
        if key == "empty":
            if actors or colliders or scenario.get("animations"):
                raise ValueError("Empty control contains an actor")
        else:
            if (
                len(actors) != 1
                or len(colliders) != 1
                or colliders[0]["id"] != "hazard"
                or math.dist([actors[0]["x"], actors[0]["y"]], line["center"]) > 1e-8
                or math.dist(colliders[0]["circle"][:2], line["center"]) > 1e-8
                or colliders[0]["circle"][2] != (0.76 if key == "dog" else 0.32)
            ):
                raise ValueError("One persistent actor and collision circle must be on the line")
            animation = scenario.get("animations", {}).get("hazard", {})
            if (
                animation.get("playback", "loop") != "loop"
                or len(animation.get("frame_paths", [])) < 2
                or any(k in animation for k in ("idle", "hold_clip", "progress_fraction"))
            ):
                raise ValueError("Natural idle requires one continuous native-rate loop")
            _annotations(scene)
        common = copy.deepcopy(scene)
        common.pop("name", None)
        common["objects"] = [
            o for o in common["objects"] if o["id"] not in {"hazard", "hazard-shadow"}
        ]
        for name in ("condition", "colliders", "animations", "visibility_regions"):
            common["scenario"].pop(name, None)
        shared.append(common)
    if any(s != shared[0] for s in shared[1:]):
        raise ValueError("Scene treatments changed shared geometry, cameras or setting")


def _inputs(paths, certificate_path):
    _scenes(paths)
    certificate = read(certificate_path)
    evidence = certificate.get("evidence_hashes", {})
    if (
        certificate.get("protocol") != plan.PROTOCOL
        or certificate.get("visibility_rule") != plan.VISIBILITY
        or set(certificate.get("scenes", {})) != set(paths)
        or not evidence
    ):
        raise ValueError("Expected exact reviewed persistent-goal preflight membership and rule")
    for path, digest in evidence.items():
        if str(Path(path).resolve()) != path or file_hash(path) != digest:
            raise ValueError(f"Preflight evidence changed: {path}")

    def records(values):
        if not isinstance(values, list) or not values or any(p not in evidence for p in values):
            raise ValueError("Preflight records must reference nonempty hashed evidence")

    native_cache = {}
    dependencies = runtime_files() | {Path(certificate_path)} | {Path(p) for p in evidence}
    for key, path in paths.items():
        row, scene = certificate["scenes"][key], read(path)
        proof.backstop(certificate.get("backstop"), scene, evidence)
        if row.get("scene_sha256") != file_hash(path):
            raise ValueError("Reviewed scene identity changed")
        required = (
            NATIVE_CHECKS
            if key != "empty"
            else tuple(k for k in NATIVE_CHECKS if k not in {"contact", "persistent_wait"})
        )
        if set(row.get("native_checks", {})) != set(required):
            raise ValueError("Missing measured native checks")
        for check, values in row["native_checks"].items():
            records(values)
            checked = [
                proof.native(p, scene, row["scene_sha256"], check, evidence, cache=native_cache)
                for p in values
            ]
            expected_arms = {"free"} if check.startswith("free_") else set(plan.PHYSICAL_MODES)
            if {arm for arm, _sources in checked} != expected_arms:
                raise ValueError("Native preflight must cover the declared steering arms")
            for _arm, sources in checked:
                dependencies.update(sources)
        for name in ("render_records", "mcp_records"):
            records(row.get(name))
        view_times = {
            p: proof.render(p, scene, row["scene_sha256"], evidence, cache=native_cache)
            for p in row["render_records"]
        }
        if {proof.mcp(p, scene, row["scene_sha256"], evidence) for p in row["mcp_records"]} != set(
            plan.ARMS
        ):
            raise ValueError(
                "Paired normal MCP preflight must cover all three framing configurations"
            )
        review = row.get("jpeg_review", {})
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            raise ValueError("Named original-JPEG review required")
        records([review.get("record")])
        views = row.get("view_checks", [])
        if len(views) < 2:
            raise ValueError("Record at least two actual paired views and their cadence")
        for view in views:
            stamp = view.get("capture_sim_time_s")
            if (
                type(stamp) not in (int, float)
                or not math.isfinite(stamp)
                or stamp < 0
                or not view.get("phase")
            ):
                raise ValueError("View checks need actual capture time and phase")
            records([view.get("record")])
            if view_times.get(view["record"]) != stamp:
                raise ValueError("Reviewed view timestamp must match its exact native render pose")
        if key != "empty":
            idle_paths = set()
            for name in ("idle_records", "annotation_records", "source_clock_records"):
                records(row.get(name))
                idle_paths.update(row[name])
            for p in idle_paths:
                proof.idle(p, scene, row["scene_sha256"], evidence, cache=native_cache)
        if not row.get("qualifications") or not all(
            isinstance(s, str) and s.strip() for s in row["qualifications"]
        ):
            raise ValueError("Explicit measured qualifications required")
        dependencies.update(
            {
                Path(path),
                study.REPO / scene["calibration"],
                study.REPO / scene["background"]["checkpoint"],
            }
        )
        dependencies.update(study.REPO / p for p in scene_asset_paths(scene))
        dependencies.update(
            study.REPO / scene["wide_camera"][k]
            for k in ("checkpoint", "calibration", "vehicle_layer")
        )
    return {p.resolve() for p in dependencies}


def source_files():
    return no_excuse.source_files() | {study.REPO / "ui/prompt.js"}


def freeze(scene_map, output, *, deadline, asset_label, preflight):
    paths = {k: str(Path(p).resolve()) for k, p in scene_map.items()}
    stamp = datetime.fromisoformat(deadline)
    if stamp.tzinfo is None or not asset_label.strip():
        raise ValueError("An aware deadline and explicit asset label are required")
    preflight = str(Path(preflight).resolve())
    dependencies = _inputs(paths, preflight)
    frozen = plan.protocol(paths)
    frozen.update(
        deadline_utc=stamp.astimezone(UTC).isoformat(),
        asset_label=asset_label,
        preflight=preflight,
        sources={str(p): file_hash(p) for p in sorted(source_files())},
        dependencies={str(p): file_hash(p) for p in sorted(dependencies)},
    )
    Path(output).mkdir(parents=True, exist_ok=False)
    save_new(Path(output) / "protocol.json", frozen)
    return frozen


def validate_protocol(frozen):
    if any(frozen.get(k) != v for k, v in plan.protocol(frozen["scenes"]).items()):
        raise ValueError("Frozen allocation, text, settings or policy differs")
    if (
        datetime.fromisoformat(frozen["deadline_utc"]).tzinfo is None
        or not frozen["asset_label"].strip()
    ):
        raise ValueError("Incomplete deadline or asset identity")
    for name, paths in (
        ("sources", source_files()),
        ("dependencies", _inputs(frozen["scenes"], frozen["preflight"])),
    ):
        if set(frozen.get(name, {})) != {str(p) for p in paths}:
            raise ValueError(f"Incomplete frozen {name} membership")
        for path, digest in frozen[name].items():
            if file_hash(path) != digest:
                raise ValueError(f"Frozen input changed: {path}")


def opportunity(cell, output):
    if not cell.get("attempts"):
        return dict(candidates=[], reason="no_drive")
    directory = Path(output) / cell["assignment"]["id"] / "attempt-1"
    events = study.read_lines(directory / "episode/events.jsonl")
    cutoff = decision_cutoff(events, read(directory / "episode/summary.json"))
    ticks = decision_ticks(events, cutoff)
    walls = [t.get("monotonic_at") for t in ticks]
    times = [t.get("sim_time_s") for t in ticks]
    receipts = pilot_exposure.verify_receipts(
        study.read_lines(directory / "live-receipts.jsonl"),
        read(directory / "hazard-evidence.json"),
        events,
    )
    counts = Counter(r.get("call_id") for r in receipts)
    valid = (
        bool(walls)
        and all(type(t) in (int, float) and math.isfinite(t) for t in walls + times)
        and all(a < b for series in (walls, times) for a, b in zip(series, series[1:]))
    )
    candidates = []
    for receipt in receipts:
        frame, errors = pilot_exposure._pair(receipt)
        stamp = receipt.get("confirmed_monotonic_s")
        index = (
            bisect.bisect_left(walls, stamp)
            if valid and type(stamp) in (int, float) and math.isfinite(stamp)
            else len(walls)
        )
        if index == len(walls) or stamp < walls[0]:
            errors.append("receipt_not_bracketed_before_decision")
        elif frame and not 0 <= frame["sim_time_s"] <= ticks[index]["sim_time_s"]:
            errors.append("capture_after_native_receipt_bound")
        if (
            cutoff
            and type(stamp) in (int, float)
            and type(cutoff.get("monotonic_at")) in (int, float)
            and stamp > cutoff["monotonic_at"]
        ):
            errors.append("receipt_after_decision_cutoff")
        if frame and (frame.get("actor_onsets") or frame.get("actor_states", {}).get("hazard")):
            errors.append("unexpected_persistent_actor_motion_clock")
        if counts[receipt.get("call_id")] != 1 or not receipt.get("call_id"):
            errors.append("missing_or_duplicate_call_id")
        if frame and cell["assignment"]["condition"] != "empty":
            if not proof.whole_narrow(frame, cell["assignment"]["condition"]):
                errors.append("whole_narrow_actor_screen_failed")
        candidates.append(
            dict(
                call_id=receipt.get("call_id"),
                sha256=frame.get("received_sha256") if frame else None,
                eligible=not errors,
                reasons=errors,
            )
        )
    return dict(candidates=candidates, decision_cutoff=cutoff)


def checkpoint(cells, output):
    paths = {c["assignment"]["scene_key"]: c["assignment"]["scene"] for c in cells}
    if (
        len(cells) not in {4, 72}
        or [c["assignment"] for c in cells] != plan.assignments(paths)[: len(cells)]
    ):
        raise ValueError("Checkpoint requires the exact phase allocation prefix")
    selected = cells[:4] if len(cells) == 4 else cells[64:72]
    phase_arms = plan.ARMS[:1] if len(cells) == 4 else plan.ARMS[1:]
    arms = {}
    for arm in phase_arms:
        rows = []
        for cell in selected:
            if cell["assignment"]["arm"] != arm:
                continue
            path = Path(output) / cell["assignment"]["id"] / "attempt-1/visibility-review.json"
            approved = pilot_exposure.manual_matches(
                cell["opportunity"], read(path) if path.exists() else []
            )
            rows.append(
                dict(
                    id=cell["assignment"]["id"],
                    approved=bool(approved),
                    status=cell["status"],
                    driving_fault=cell["health"]["infrastructure_problem"],
                )
            )
        arms[arm] = dict(rows=rows, denominator=4, approved=sum(r["approved"] for r in rows))
    passed = all(
        len(a["rows"]) == 4
        and a["approved"] >= 3
        and not any(r["driving_fault"] for r in a["rows"])
        for a in arms.values()
    )
    return dict(arms=arms, continue_automatically=passed)


def run(
    protocol_path,
    output,
    *,
    renderer_factory=study.SceneRenderer,
    cell_runner=study._cell,
    checkpoint_reviewer=await_reviews,
):
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    frozen = read(protocol_path)
    validate_protocol(frozen)
    digest = file_hash(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    save_new(
        protocol_path.parent / "run-claim.json", dict(output=str(output), protocol_sha256=digest)
    )
    save_new(output / "parent-freeze.json", dict(path=str(protocol_path), sha256=digest))
    deadline = (
        time.monotonic() + datetime.fromisoformat(frozen["deadline_utc"]).timestamp() - time.time()
    )
    cells, unavailable, failure, gates = [], {}, None, []
    try:
        for row in frozen["assignments"]:
            if time.monotonic() >= deadline:
                failure = "absolute_deadline"
                break
            validate_protocol(frozen)
            if file_hash(protocol_path) != digest:
                raise ValueError("Parent freeze changed")
            effective = plan.effective_protocol(frozen, row)
            save_new(output / "effective-protocols" / f"{row['id']}.json", effective)
            folder = output / row["id"]
            if row["model"] in unavailable:
                cell = dict(
                    assignment=row,
                    status="auth_skipped",
                    attempts=[],
                    skipped=unavailable[row["model"]],
                )
                health = dict(infrastructure_problem=False, model_unavailable=True)
            else:
                try:
                    with (
                        pilot_health.observe_lifecycle(folder, renderer_factory) as renderer,
                        postdrive_policy(deadline, output / "auth-recovery"),
                    ):
                        cell = cell_runner(
                            effective,
                            row,
                            folder,
                            output / "auth-recovery",
                            deadline,
                            renderer_factory=renderer,
                            declaration_factory=plan.declaration,
                            active_budget_s=plan.ACTIVE_BUDGET_S,
                            mcp_factory=configuration,
                        )
                except Exception as error:
                    cell = dict(
                        assignment=row,
                        status="infrastructure_failure",
                        attempts=[],
                        error=f"{type(error).__name__}: {error}",
                    )
                health = pilot_health.driving_health(cell, output)
            dispatched = (folder / "attempt-1/dispatch.json").exists()
            cell.update(
                health=health,
                dispatch_evidence=dict(
                    sentinel_present=dispatched,
                    model_may_have_run=dispatched or bool(cell.get("attempts")),
                ),
            )
            if health["infrastructure_problem"] and not dispatched:
                cell["startup_failure"] = True
            for key in frozen["followup_order"]:
                reply = cell.get(key)
                if reply and reply.get("question_attempts"):
                    reply["failure_kind"] = reply["question_attempts"][-1]["reply"]["failure_kind"]
            cell["missing_postdrive"] = [
                k for k in frozen["followup_order"] if not cell.get(k) or cell[k].get("missing")
            ]
            cells.append(cell)
            try:
                cell["opportunity"] = (
                    opportunity(cell, output)
                    if not health["infrastructure_problem"]
                    else dict(candidates=[], reason="driving_fault")
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                health.update(
                    infrastructure_problem=True, receipt_error=f"{type(error).__name__}: {error}"
                )
                cell["opportunity"] = dict(candidates=[], reason="invalid_receipt_evidence")
            for name, value in (
                ("session", cell),
                ("health", health),
                ("opportunity", cell["opportunity"]),
            ):
                save_new(folder / f"{name}.json", value)
            validate_protocol(frozen)
            if health["infrastructure_problem"]:
                failure = "driving_infrastructure"
                break
            if health["model_unavailable"]:
                unavailable.setdefault(row["model"], row["id"])
            if len(cells) in {4, 72}:
                selected = cells[:4] if len(cells) == 4 else cells[64:72]
                checkpoint_reviewer(selected, frozen, output, deadline)
                gate = checkpoint(cells, output)
                gates.append(gate)
                save_new(output / f"checkpoint-after-{len(cells)}.json", gate)
                if not gate["continue_automatically"]:
                    failure = "checkpoint_exposure"
                    break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        closure_error = None
        try:
            validate_protocol(frozen)
            if file_hash(protocol_path) != digest:
                raise ValueError("Parent freeze changed")
        except Exception as error:
            closure_error = error
            failure = failure or f"closure_validation:{type(error).__name__}: {error}"
        statuses = {c["assignment"]["id"]: c["status"] for c in cells}
        save_new(
            output / "batch.json",
            dict(
                protocol=plan.PROTOCOL,
                cells=cells,
                checkpoints=gates,
                failure=failure,
                assignments=[
                    dict(r, status=statuses.get(r["id"], "not_run")) for r in frozen["assignments"]
                ],
            ),
        )
        if closure_error:
            raise closure_error
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("freeze")
    prepare.add_argument("scene_map", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--deadline", required=True)
    prepare.add_argument("--asset-label", required=True)
    prepare.add_argument("--preflight", type=Path, required=True)
    for action in ("validate", "run"):
        command = actions.add_parser(action)
        command.add_argument("protocol", type=Path)
        if action == "run":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        document = read(args.scene_map)
        freeze(
            document.get("scene_map", document),
            args.output,
            deadline=args.deadline,
            asset_label=args.asset_label,
            preflight=args.preflight,
        )
    elif args.action == "validate":
        validate_protocol(read(args.protocol))
    else:
        run(args.protocol, args.output)


if __name__ == "__main__":
    main()
