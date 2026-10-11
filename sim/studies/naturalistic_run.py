"""Serial prospective pilots over the unchanged naturalistic driving cell."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from drivingbench.sim import event_study_v2 as study
from drivingbench.sim.actor_visibility import region_rectangles
from drivingbench.sim.batch import save_new
from drivingbench.sim.cameras import validate_renderer_priority
from drivingbench.sim.producer import file_hash
from drivingbench.sim.scenarios import validate_scenario
from drivingbench.sim.scene import scene_asset_paths
from drivingbench.sim.studies import naturalistic_pilots as pilots
from drivingbench.sim.studies import no_excuse, pilot_exposure, pilot_health
from drivingbench.sim.studies.continuation import postdrive_policy, read
from drivingbench.sim.visual_splat import RENDER, TOOLS

PROTOCOL = "sim-naturalistic-pilots-v1"
NATIVE_SOURCES = (
    "sim/car.py",
    "sim/plant.py",
    "sim/runtime.py",
    "sim/restrictions.py",
    "sim/episode.py",
    "sim/scenarios.py",
    "sim/actor_yield.py",
    "sim/actor_animation.py",
    "sim/course.py",
    "sim/params.json",
    "sim/corolla.carparams",
    "controller/core.py",
    "controller/speed.py",
    "device/settings.py",
    "device/service.py",
    "shared/contracts.py",
)


def _base(pilot, paths, *, course_variant=None, renderer_priority=None):
    protocol = dict(
        protocol=PROTOCOL,
        pilot=pilot,
        scenes=paths,
        assignments=pilots.assignments(pilot, paths),
        prompt=study.prompt(),
        episode_policy=asdict(study.policy()),
        settings=study.study_settings().model_dump(),
        followup_order=list(study.FOLLOWUP_QUESTIONS),
        **study.FOLLOWUP_QUESTIONS,
        belief_rubric=study.HAZARD_BELIEF_RUBRIC,
        coder=study.CODER,
        turn_budget_s=180,
        time_limit_s=660,
        client_profile="normal",
        minimum_free_bytes=study.MINIMUM_FREE_BYTES,
        models={key: asdict(study.CLIENTS[key]) for key in study.MODELS},
        opportunity_rule=pilot_exposure.rule_for(pilot, course_variant=course_variant),
        postdrive_retries=1,
    )
    if pilot == "A_persistent":
        protocol["camera_capture"] = "on_demand"
    if course_variant is not None:
        from drivingbench.sim.real_course_study.refined import RefinedPolicy
        from drivingbench.sim.studies import tidy_cd

        if course_variant not in tidy_cd.VARIANTS:
            raise ValueError("Unknown course variant")
        tidy_cd.validate_map(pilot, paths, variant=course_variant)
        protocol.update(
            course_variant=course_variant,
            prompt=tidy_cd.prompt(protocol["prompt"]),
            episode_policy=asdict(RefinedPolicy()),
            camera_capture="on_demand",
        )
    if renderer_priority is not None:
        if pilot not in {"C", "D"} or course_variant is None:
            raise ValueError("Renderer priority is opt-in only for explicitly authored C/D studies")
        protocol["renderer_priority"] = validate_renderer_priority(renderer_priority)
    return protocol


def runtime_files():
    return {
        RENDER.resolve(),
        (TOOLS / "Blender.app/Contents/MacOS/Blender").resolve(),
        (study.REPO / "runs/sim/visual/venv/bin/python").resolve(),
        Path(sys.executable).resolve(),
        study.REPO / "docs/upstream-sources.json",
    }


def _native_recipe(row, key, evidence, *, course_variant=None):
    metadata = read(row["no_response_native_source"])
    path = (study.REPO / metadata["metadata"]["recipe"]).resolve()
    if str(path) not in evidence:
        raise ValueError("C native recipe must be hashed preflight evidence")
    recipe = read(path)
    expected = dict(
        study="C",
        side=key.split("-", 1)[0],
        mode="center_pass_negative",
        target_offset_m=0,
        start_offset_m=0,
        start_yaw_deg=0,
        response_delay_s=0,
        stop_station_m=None,
        source_clock_unchanged=True,
        actor_wait_policy_unchanged=True,
        scene_sha256=row["scene_sha256"],
        executed_scene_sha256=row["scene_sha256"],
        speed_request_mps=1.0,
    )
    if any(
        name not in recipe
        or recipe[name] != value
        or (type(value) is bool and type(recipe[name]) is not bool)
        or (type(value) in (int, float) and type(recipe[name]) not in (int, float))
        for name, value in expected.items()
    ):
        raise ValueError("C native recipe differs from the selected unperturbed no-response scene")
    sources = {(study.REPO / name).resolve() for name in NATIVE_SOURCES}
    if course_variant is not None:
        sources.update(
            (study.REPO / name).resolve()
            for name in (
                "sim/real_course_study/refined.py",
                "sim/real_course_study/runtime.py",
                "sim/studies/tidy_cd.py",
            )
        )
    for source in sources:
        if recipe.get("source_hashes", {}).get(str(source)) != file_hash(source):
            raise ValueError(f"C native source changed: {source}")
    return sources | {path}


def record_completed_ticks(app, _directory, *, pilot):
    """D-only private evidence after native._tick has stepped and captured its pose."""
    if pilot != "D":
        return
    native = app.state.sim_native
    original = native.on_evidence
    if not callable(original):
        raise ValueError("Completed native evidence requires the original episode callback")

    def recorded(kind, **data):
        if kind == "sim_tick":
            data["native_completed_monotonic_s"] = time.monotonic()
        return original(kind, **data)

    native.on_evidence = recorded
    original_finish = native.finish

    def finish(reason):
        if reason != "client_done":
            return original_finish(reason)
        try:
            requested_at = time.monotonic()
            pose = native.pose_snapshot()
            app.state.sim_episode.record(
                "sim_client_handback_requested",
                requested_at=requested_at,
                requested_reason=reason,
                clock_basis="actual_host_finish_entry",
                sim_time_s=pose["sim_time_s"],
            )
        finally:
            result = original_finish(reason)
        return result

    native.finish = finish


def _references(certificate, paths, *, course_variant=None):
    """Validate reviewed records and return their exact frozen file identities."""
    evidence = certificate.get("evidence_hashes", {})
    native_sources = set()
    if not evidence or any(
        not isinstance(p, str)
        or str(Path(p).resolve()) != p
        or not isinstance(d, str)
        or len(d) != 64
        or any(c not in "0123456789abcdef" for c in d)
        for p, d in evidence.items()
    ):
        raise ValueError("Preflight requires absolute evidence paths and SHA256 identities")
    if certificate.get("rule") != pilot_exposure.rule_for(
        certificate["pilot"], course_variant=course_variant
    ) or set(certificate.get("scenes", {})) != set(paths):
        raise ValueError("Preflight rule or scene membership differs")

    def reference(path):
        if not isinstance(path, str) or path not in evidence:
            raise ValueError("Every reviewed record must be present in evidence_hashes")

    for key, path in paths.items():
        row = certificate["scenes"][key]
        if row.get("scene_sha256") != file_hash(path):
            raise ValueError(f"Preflight scene changed: {key}")
        for field in ("native_records", "render_records", "mcp_records"):
            records = row.get(field)
            if not isinstance(records, list) or not records:
                raise ValueError(f"Missing reviewed {field}: {key}")
            for record in records:
                reference(record)
        review = row.get("jpeg_review", {})
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            raise ValueError("Named original-JPEG reviewer required")
        reference(review.get("record"))
        views = row.get("view_checks", [])
        if len(views) < 2:
            raise ValueError("At least two documented preflight views are required")
        for view in views:
            stamp = view.get("capture_sim_time_s")
            if (
                type(stamp) not in (int, float)
                or not math.isfinite(stamp)
                or stamp < 0
                or not view.get("phase")
            ):
                raise ValueError("Preflight view needs actual time and phase")
            reference(view.get("record"))
        if certificate["pilot"] == "A_persistent" and not {"initial", "held"}.issubset(
            {v["phase"] for v in views}
        ):
            raise ValueError("Persistent A requires actual initial and held preflight views")
        fresh_C = (
            certificate["pilot"] == "C"
            and pilot_exposure.rule_for("C", course_variant=course_variant).get(
                "source_onset_basis"
            )
            == "declared_scene_start_s"
        )
        if certificate["pilot"] in {"B", "D"} or fresh_C:
            bundle = row.get("actual_received_preflight", {})
            for field in ("receipts", "audit", "events", "manual_reviews"):
                reference(bundle.get(field))
            events = study.read_lines(Path(bundle["events"]))
            receipts = pilot_exposure.verify_receipts(
                study.read_lines(Path(bundle["receipts"])), read(bundle["audit"]), events
            )
            checker = (
                pilot_exposure.B_preflight_views
                if certificate["pilot"] == "B"
                else pilot_exposure.C_preflight_views
                if fresh_C
                else pilot_exposure.D_preflight_views
            )
            screen = checker(
                read(path),
                events,
                receipts,
                read(bundle["manual_reviews"])["reviews"],
                course_variant=course_variant,
            )
            if not screen["admitted"]:
                raise ValueError(
                    "Pilot needs its exact manually approved clear actual receipts 6–20 wall seconds apart"
                )
        actor = row.get("actor", {})
        expected = (
            "empty"
            if key.startswith("control-")
            else "static"
            if certificate["pilot"] in {"D", "A_persistent"}
            else "moving"
        )
        if actor.get("kind") != expected:
            raise ValueError("Actor evidence kind differs from the selected pilot")
        fields = () if expected == "empty" else ("annotation_paths",)
        fields += (
            ("motion_records", "source_clock_records", "policy_records")
            if expected == "moving"
            else ()
        )
        if certificate["pilot"] == "A_persistent":
            fields += ("source_clock_records",)
        for field in fields:
            if not isinstance(actor.get(field), list) or not actor[field]:
                raise ValueError(f"Missing actor {field}")
            for record in actor[field]:
                reference(record)
        if certificate["pilot"] == "C":
            delay = row.get("no_response_conflict_delay_after_onset_s")
            if type(delay) not in (int, float) or not math.isfinite(delay) or delay <= 0:
                raise ValueError("C needs its exact positive no-response delay")
            reference(row.get("no_response_record"))
            reference(row.get("no_response_native_source"))
            reference(row.get("no_response_events"))
            if (
                len(
                    {
                        Path(row[name]).parent
                        for name in (
                            "no_response_record",
                            "no_response_events",
                            "no_response_native_source",
                        )
                    }
                )
                != 1
            ):
                raise ValueError("C contact, events and metadata must belong to the same episode")
            native_sources.update(_native_recipe(row, key, evidence, course_variant=course_variant))
            cause = read(row["no_response_record"])["first_cause"]
            contact = cause["evidence"]
            events = study.read_lines(Path(row["no_response_events"]))
            source_start = (
                pilot_exposure.C_source_start(
                    read(path),
                    [e for e in events if e.get("kind") == "sim_tick"],
                    course_variant=course_variant,
                )
                if fresh_C
                else contact["actor_onsets"]["hazard"]
            )
            if (
                cause["reason"] != "hazard_contact"
                or source_start is None
                or delay != contact["sim_time_s"] - source_start
                or not any(
                    event.get("kind") == "sim_tick"
                    and event.get("sim_time_s") == contact["sim_time_s"]
                    and event.get("actor_onsets") == contact["actor_onsets"]
                    and event.get("scenario_contact") == contact.get("scenario_contact")
                    for event in events
                )
            ):
                raise ValueError("C no-response delay does not match the exact native contact")
        elif row.get("no_response_conflict_delay_after_onset_s") is not None:
            raise ValueError("A/D must not invent a centerline conflict clock")
        if (
            not isinstance(row.get("qualifications"), list)
            or not row["qualifications"]
            or not all(isinstance(q, str) and q.strip() for q in row["qualifications"])
        ):
            raise ValueError(
                "Record explicit preflight qualifications, including static support limits"
            )
    for path, digest in evidence.items():
        if file_hash(path) != digest:
            raise ValueError(f"Preflight evidence changed: {path}")
    return {Path(p) for p in evidence} | native_sources


def _annotations(scene):
    scenario = scene["scenario"]
    regions = scenario.get("visibility_regions", {}).get("hazard")
    if not isinstance(regions, dict):
        raise ValueError("Selected actor requires source-region annotations")
    if "default" not in regions and "frames" not in regions:
        region_rectangles(regions)
        return
    default, frames = regions.get("default"), regions.get("frames", {})
    if default is not None:
        region_rectangles(default)
    if not isinstance(frames, dict):
        raise ValueError("Source-frame annotations must be a mapping")
    for boxes in frames.values():
        region_rectangles(boxes)
    animation = scenario.get("animations", {}).get("hazard", {})
    paths = [
        *animation.get("frame_paths", []),
        *animation.get("idle", {}).get("frame_paths", []),
        *animation.get("hold_clip", {}).get("frame_paths", []),
    ]
    paths += [
        row.get("asset_path", row.get("image_path"))
        for row in scene["objects"]
        if row.get("id") == "hazard"
    ]
    if default is None and (
        not paths
        or any(not path or str((study.REPO / path).resolve()) not in frames for path in paths)
    ):
        raise ValueError("Every selected actor source pose needs absolute-path annotations")


def _inputs(pilot, paths, preflight, *, course_variant=None, renderer_priority=None):
    certificate = read(preflight)
    if certificate.get("pilot") != pilot:
        raise ValueError("Preflight pilot differs")
    dependencies = (
        runtime_files()
        | {Path(preflight)}
        | _references(certificate, paths, course_variant=course_variant)
    )
    priority = validate_renderer_priority(renderer_priority)
    if priority is not None:
        dependencies.add(Path(priority["executable"]))
    scenes = {key: validate_scenario(read(path)) for key, path in paths.items()}
    pilots.validate_scenes(pilot, scenes, course_variant=course_variant)
    if course_variant is not None:
        from drivingbench.sim.studies import tidy_cd

        if course_variant not in tidy_cd.VARIANTS:
            raise ValueError("Unknown course variant")
        dependencies.update(
            tidy_cd.certificate_inputs(pilot, paths, certificate, variant=course_variant)
        )
    for key, scene in scenes.items():
        scenario = scene["scenario"]
        if (
            scenario["type"] != "course_event"
            or scenario["time_scale"] != 1
            or scenario["time_limit_s"] != 660
            or "wide_camera" not in scene
        ):
            raise ValueError(
                "Scenes require paired cameras, real-time course_event and 660 seconds"
            )
        if scenario["condition"] in {"child_crossing", "dog_crossing"}:
            policy = scenario.get("trajectories", {}).get("hazard", {}).get("yield_policy")
            if policy != dict(
                kind="wait_for_stopped_car", max_car_speed_mps=0.1, safety_margin_m=0.75
            ):
                raise ValueError("Moving actors require the recorded stopped-car policy")
        if scenario["condition"] != "control":
            _annotations(scene)
        dependencies.update(
            {
                Path(paths[key]),
                study.REPO / scene["calibration"],
                study.REPO / scene["background"]["checkpoint"],
            }
        )
        dependencies.update(study.REPO / p for p in scene_asset_paths(scene))
        dependencies.update(
            study.REPO / scene["wide_camera"][p]
            for p in ("checkpoint", "calibration", "vehicle_layer")
        )
    return {p.resolve() for p in dependencies}


def freeze(
    pilot,
    scene_map,
    output,
    *,
    deadline,
    asset_label,
    preflight,
    course_variant=None,
    renderer_priority=None,
):
    """Write one exclusive prospective freeze; selection and deadline are always explicit."""
    paths = {key: str(Path(path).resolve()) for key, path in scene_map.items()}
    protocol = _base(
        pilot, paths, course_variant=course_variant, renderer_priority=renderer_priority
    )
    stamp = datetime.fromisoformat(deadline)
    if stamp.tzinfo is None or not asset_label.strip():
        raise ValueError("An aware deadline and explicit asset label are required")
    preflight = str(Path(preflight).resolve())
    dependencies = _inputs(
        pilot, paths, preflight, course_variant=course_variant, renderer_priority=renderer_priority
    )
    protocol.update(
        preflight=preflight,
        asset_label=asset_label,
        deadline_utc=stamp.astimezone(UTC).isoformat(),
        sources={str(p): file_hash(p) for p in sorted(no_excuse.source_files())},
        dependencies={str(p): file_hash(p) for p in sorted(dependencies)},
    )
    Path(output).mkdir(parents=True, exist_ok=False)
    save_new(Path(output) / "protocol.json", protocol)
    return protocol


def validate_protocol(protocol):
    expected = _base(
        protocol["pilot"],
        protocol["scenes"],
        course_variant=protocol.get("course_variant"),
        renderer_priority=protocol.get("renderer_priority"),
    )
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen allocation, text, settings or policy differs")
    if (
        datetime.fromisoformat(protocol["deadline_utc"]).tzinfo is None
        or not protocol.get("asset_label", "").strip()
    ):
        raise ValueError("Incomplete deadline or asset identity")
    sources = no_excuse.source_files()
    dependencies = _inputs(
        protocol["pilot"],
        protocol["scenes"],
        protocol["preflight"],
        course_variant=protocol.get("course_variant"),
        renderer_priority=protocol.get("renderer_priority"),
    )
    for name, paths in (("sources", sources), ("dependencies", dependencies)):
        if set(protocol.get(name, {})) != {str(p) for p in paths}:
            raise ValueError(f"Incomplete frozen {name} membership")
        for path, digest in protocol[name].items():
            if file_hash(path) != digest:
                raise ValueError(f"Frozen input changed: {path}")


def opportunity(cell, protocol, output):
    row = cell["assignment"]
    directory = Path(output) / row["id"] / "attempt-1"
    if not cell.get("attempts") or row["condition"] == "control":
        return dict(candidates=[], reason="empty_control_or_no_drive")
    events = study.read_lines(directory / "episode/events.jsonl")
    receipts = pilot_exposure.verify_receipts(
        study.read_lines(directory / "live-receipts.jsonl"),
        read(directory / "hazard-evidence.json"),
        events,
    )
    return pilot_exposure.qualify(
        protocol["pilot"],
        read(row["scene"]),
        events,
        read(directory / "episode/summary.json"),
        receipts,
        rule=protocol["opportunity_rule"],
        certificate=read(protocol["preflight"])["scenes"][row["scene_key"]],
        course_variant=protocol.get("course_variant"),
    )


def await_reviews(cells, protocol, output, deadline):
    paths = []
    for cell in cells:
        eligible = [r for r in cell["opportunity"].get("candidates", []) if r["eligible"]]
        if eligible:
            directory = Path(output) / cell["assignment"]["id"] / "attempt-1"
            no_excuse.review_sheet(directory, eligible)
            paths.append(directory / "visibility-review.json")
    save_new(
        Path(output) / f"checkpoint-awaiting-review-{cells[0]['assignment']['id']}.json",
        dict(required=[str(p) for p in paths]),
    )
    print(json.dumps(dict(checkpoint_awaiting_manual_review=[str(p) for p in paths])), flush=True)
    while time.monotonic() < deadline and any(not p.exists() for p in paths):
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def _checkpoint(cells, output):
    rows = []
    for cell in cells:
        row = cell["assignment"]
        path = Path(output) / row["id"] / "attempt-1/visibility-review.json"
        rows.append(
            dict(
                id=row["id"],
                model=row["model"],
                status=cell["status"],
                opportunity=cell["opportunity"],
                reviews=read(path) if path.exists() else [],
                driving_infrastructure_problem=cell["health"]["infrastructure_problem"],
            )
        )
    return pilot_exposure.checkpoint(rows)


def declaration_for(row, scene, protocol):
    if protocol.get("course_variant") is not None:
        from drivingbench.sim.studies import tidy_cd

        return tidy_cd.declaration_for(row, scene, protocol)
    declaration = study.declaration_for(row, scene, protocol)
    if protocol["pilot"] == "A_persistent":
        declaration["camera_capture"] = protocol["camera_capture"]
    if protocol["pilot"] == "D":
        declaration["paused"] = True
    return declaration


def run(
    protocol_path,
    output,
    *,
    renderer_factory=study.SceneRenderer,
    cell_runner=study._cell,
    checkpoint_reviewer=await_reviews,
):
    """Run once, serially, retaining every allocation and stopping on raw driving faults."""
    protocol_path = Path(protocol_path).resolve()
    protocol = read(protocol_path)
    validate_protocol(protocol)
    digest = file_hash(protocol_path)
    output = Path(output).resolve()
    save_new(
        protocol_path.parent / "run-claim.json",
        dict(output=str(output), protocol_sha256=digest),
    )
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "parent-freeze.json", dict(path=str(protocol_path), sha256=digest))
    save_new(output / "protocol.json", protocol)
    deadline = (
        time.monotonic()
        + datetime.fromisoformat(protocol["deadline_utc"]).timestamp()
        - time.time()
    )
    cells, unavailable, gate, failure = [], {}, None, None
    try:
        for row in protocol["assignments"]:
            if time.monotonic() >= deadline:
                failure = "absolute_deadline"
                break
            validate_protocol(protocol)
            if file_hash(protocol_path) != digest:
                raise ValueError("Parent freeze changed")
            effective = pilots.effective_protocol(protocol, row)
            save_new(output / "effective-protocols" / f"{row['id']}.json", effective)
            folder = output / row["id"]
            if row["model"] in unavailable:
                cell = dict(
                    assignment=row,
                    status="auth_skipped",
                    attempts=[],
                    skipped=unavailable[row["model"]],
                )
                health = dict(
                    infrastructure_problem=False,
                    model_unavailable=True,
                    provider_failure="failed_auth",
                )
            else:
                try:
                    cell_options = {}
                    if protocol["pilot"] in {"A_persistent", "D"} or protocol.get("course_variant"):
                        cell_options["declaration_factory"] = declaration_for
                    if protocol["pilot"] == "D":
                        cell_options["before_dispatch"] = partial(record_completed_ticks, pilot="D")
                    with (
                        pilot_health.observe_lifecycle(folder, renderer_factory) as renderer,
                        postdrive_policy(deadline, output / "auth-recovery"),
                    ):
                        if protocol.get("course_variant"):
                            from drivingbench.sim.studies import tidy_cd

                            cell_options["application_factory"] = tidy_cd.application_for(
                                row, effective, monitored_app=study.simulator_app
                            )
                        cell = cell_runner(
                            effective,
                            row,
                            folder,
                            output / "auth-recovery",
                            deadline,
                            renderer_factory=renderer,
                            **cell_options,
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
            for key in protocol["followup_order"]:
                reply = cell.get(key)
                if reply and reply.get("question_attempts"):
                    reply["failure_kind"] = reply["question_attempts"][-1]["reply"]["failure_kind"]
            cell["missing_postdrive"] = [
                k for k in protocol["followup_order"] if not cell.get(k) or cell[k].get("missing")
            ]
            cells.append(cell)
            try:
                cell["opportunity"] = (
                    opportunity(cell, protocol, output)
                    if not health["infrastructure_problem"]
                    else dict(candidates=[], reason="driving_fault")
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                health.update(
                    infrastructure_problem=True,
                    opportunity_error=f"{type(error).__name__}: {error}",
                )
                cell["opportunity"] = dict(candidates=[], reason="invalid_opportunity_evidence")
            save_new(folder / "session.json", cell)
            save_new(folder / "health.json", health)
            save_new(folder / "opportunity.json", cell["opportunity"])
            validate_protocol(protocol)
            if health["infrastructure_problem"]:
                failure = "driving_infrastructure"
                break
            if health["model_unavailable"]:
                unavailable.setdefault(row["model"], row["id"])
            if len(cells) == 4:
                checkpoint_reviewer(cells, protocol, output, deadline)
                gate = _checkpoint(cells, output)
                save_new(output / "checkpoint-round-1.json", gate)
                if not gate["continue_automatically"]:
                    failure = "checkpoint_exposure"
                    break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        closure_error = None
        try:
            validate_protocol(protocol)
            if file_hash(protocol_path) != digest:
                raise ValueError("Parent freeze changed")
        except Exception as error:
            closure_error = error
            failure = failure or f"closure_validation:{type(error).__name__}: {error}"
        statuses = {c["assignment"]["id"]: c["status"] for c in cells}
        save_new(
            output / "batch.json",
            dict(
                protocol=PROTOCOL,
                pilot=protocol["pilot"],
                cells=cells,
                checkpoint=gate,
                failure=failure,
                deadline_reached=time.monotonic() >= deadline,
                assignments=[
                    dict(r, status=statuses.get(r["id"], "not_run"))
                    for r in protocol["assignments"]
                ],
            ),
        )
        if closure_error:
            raise closure_error
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("pilot", choices=["A", "A_persistent", "B", "C", "D"])
    prepare.add_argument("scene_map", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--deadline", required=True)
    prepare.add_argument("--asset-label", required=True)
    prepare.add_argument("--preflight", type=Path, required=True)
    prepare.add_argument(
        "--course-variant",
        choices=[
            "tidy-authored-v3-CD-v1",
            "tidy-authored-v3-CD-exposure-v2",
            "tidy-authored-v3-C-center-hold-clear-v3",
            "tidy-authored-v3-C-center-hold-clear-v4",
            "tidy-authored-v3-D-near-roadside-v3",
            "tidy-authored-v3-D-near-roadside-v4",
            "tidy-authored-v3-B-known-car-v1",
            "tidy-authored-v3-B-known-car-v2",
            "tidy-authored-v3-B-known-bollard-v1",
        ],
    )
    prepare.add_argument("--renderer-priority-config", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("protocol", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("protocol", type=Path)
    execute.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        document = read(args.scene_map)
        freeze(
            args.pilot,
            document.get("scene_map", document),
            args.output,
            deadline=args.deadline,
            asset_label=args.asset_label,
            preflight=args.preflight,
            course_variant=args.course_variant,
            renderer_priority=read(args.renderer_priority_config)
            if args.renderer_priority_config
            else None,
        )
    elif args.action == "validate":
        validate_protocol(read(args.protocol))
    else:
        run(args.protocol, args.output)


if __name__ == "__main__":
    main()
