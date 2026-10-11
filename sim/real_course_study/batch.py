"""Round-robin real-layout trials with one original continuing chat per model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

from drivingbench.gateway.app import create_app as gateway_app
from drivingbench.replay.clients import CLIENTS
from drivingbench.replay.run import http_server
from drivingbench.sim.batch import (
    ATTEMPTS,
    MODELS,
    ProducerSwitch,
    RendererLease,
    auth_preflight,
    camera_settings,
    client_driving_servers,
    drive,
    prepare_isolated,
    save_new,
)
from drivingbench.sim.cameras import SceneRenderer, render_slot
from drivingbench.sim.car import SimNative
from drivingbench.sim.course import Course, _transform
from drivingbench.sim.event_study_v2 import cell_status, source_files
from drivingbench.sim.live_receipt import ReceiptWatcher
from drivingbench.sim.naturalistic_run import audit_attempt
from drivingbench.sim.producer import file_hash, simulator_app
from drivingbench.sim.real_course_study.coverage import CoverageCertificate, local_pose
from drivingbench.sim.real_course_study.runtime import (
    LOT_SAFETY_DISTANCE_M,
    PARKING_CONFIRMATION_S,
    RealCoursePolicy,
    runtime_factory,
)
from drivingbench.sim.run import isolated_command, wait_ready

PROTOCOL = "sim-measured-real-course-v3"
TIDY_PROTOCOL = "sim-authored-tidy-real-course-v1"
COVERAGE_POLICY = {
    "reviewed_membership": "paired narrow/wide body-reference pose; private captured-image audit only",
    "outside_reviewed_ending": False,
    "lot_safety_distance_m": LOT_SAFETY_DISTANCE_M,
    "lot_safety_basis": "body reference distance to any paired-pass parent polygon, independent of yaw",
    "yaw_measurement": "wrapped yaw excess at nearest paired-pass parent; ties by distance, yaw excess, cell ID",
    "public_images": "unchanged; no overlays or extra observe fields",
}


class PoseRendererLease(RendererLease):
    """Share one warm map renderer across separately registered attempt starts."""

    def __init__(self, renderer, renderer_start, attempt_start):
        super().__init__(renderer)
        self.renderer_start, self.attempt_start = renderer_start, attempt_start
        self.metrics = {}

    def _pose(self, local):
        mapped = (*_transform(local[:2], self.attempt_start), local[2] + self.attempt_start[2])
        return local_pose(mapped, self.renderer_start)

    def render_frames(self, snapshot):
        captured = copy.deepcopy(snapshot)
        frames = self.renderer.render_frames({**captured, "pose": self._pose(captured["pose"])})
        self.metrics = copy.deepcopy(getattr(self.renderer, "metrics", {}))
        if "rendered_coverage" in captured:
            self.metrics["captured_coverage"] = {
                "reference": "site1_full_ground/body_reference/paired_narrow_wide",
                "map_pose": captured["map_pose"],
                "outside_reviewed": not captured["rendered_coverage"]["supported"],
                "measurement": captured["rendered_coverage"],
            }
        return frames

    def render(self, pose):
        frame = self.renderer.render(self._pose(pose))
        self.metrics = copy.deepcopy(getattr(self.renderer, "metrics", {}))
        self.metrics.pop("captured_coverage", None)
        return frame


def registered_starts(registration):
    """Select corresponding primary first-command poses, not later path corrections."""
    starts = {key: [] for key in MODELS}
    names = {
        "gpt-6-astra": "astra",
        "gpt-5.6-sol": "sol",
        "claude-fable-5.1": "fable",
        "grok-4.6": "grok",
    }
    for entry in registration["attempts"]:
        if entry["scope"] != "primary":
            continue
        key = names[entry["model"]]
        number = int(entry["label"].split("--", 1)[0].removeprefix("attempt-"))
        pose = entry["start_pose"]
        starts[key].append(
            {
                "attempt": number,
                "real_segment": entry["segment"],
                "start": {
                    "x": pose["x"],
                    "y": pose["y"],
                    "heading_deg": math.degrees(pose["heading"]),
                },
                "reference": "registered first accepted motion; first-observe/engagement pose may differ",
                "registration": entry["registration"],
            }
        )
    for rows in starts.values():
        rows.sort(key=lambda row: row["attempt"])
        if [row["attempt"] for row in rows] != [1, 2, 3]:
            raise ValueError("Each model needs exactly three registered primary attempts")
    return starts


def start_pose(start):
    """Keep the measured initial pose separate from the immutable render-scene origin."""
    return tuple(start["start"][key] for key in ("x", "y", "heading_deg"))


def preflight_starts(scene, starts, coverage, *, runtime_builder=runtime_factory):
    """Construct every native start before renderer, credentials or model dispatch."""
    course = Course(scene)
    evidence = []
    for model, rows in starts.items():
        for start in rows:
            factory = runtime_builder(
                scene,
                coverage,
                registered_start=start_pose(start),
                native_factory=lambda *, time, initial=start["native_state"]: SimNative(
                    time=time, **initial
                ),
            )
            runtime = factory(course)
            try:
                pose = runtime.pose_snapshot()
                if pose["lot_safety_backstop"] is not None:
                    raise ValueError(f"{model} attempt {start['attempt']}: lot safety backstop")
                if pose["obstacle_collision_ids"] or pose["boundary_violation"]:
                    raise ValueError(f"{model} attempt {start['attempt']}: terminal start geometry")
                if not pose["held"] or abs(pose["speed_mps"]) >= 0.05:
                    raise ValueError(f"{model} attempt {start['attempt']}: start is not held")
                evidence.append(
                    dict(
                        model=model,
                        attempt=start["attempt"],
                        map_pose=pose["map_pose"],
                        held=pose["held"],
                        rendered_coverage=pose["rendered_coverage"],
                        strict_boundary_violation=pose["strict_boundary_violation"],
                    )
                )
            finally:
                runtime.close()
    return evidence


def checkpoint(reports):
    """At least half the assigned models need proven pairs; driving faults stop all."""
    problems = []
    received = 0
    for row in reports:
        status = row.get("status", "complete")
        if status == "infrastructure_failure":
            problems.append(f"{row['model']}: driving infrastructure")
        if row.get("paired_observations", 0) > 0:
            received += 1
        elif status == "complete":
            problems.append(f"{row['model']}: no received paired observation")
    if {row["model"] for row in reports} != set(MODELS) or len(reports) != len(MODELS):
        problems.append("first round does not account for all four assigned models")
    if received < 2:
        problems.append("fewer than half the assigned models received a verified pair")
    return {
        "passed": not problems,
        "problems": problems,
        "assigned_models": len(reports),
        "models_with_received_pairs": received,
    }


def producer_paired_observations(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return sum(
        row["kind"] == "sim_observe_delivery"
        and {frame["camera"] for frame in row.get("frames", [])} == {"narrow", "wide"}
        for row in rows
    )


def received_pairs(receipt):
    return sum(
        row.get("error") is None
        and len(row["frames"]) == 2
        and {frame["camera"] for frame in row["frames"] if frame.get("accepted")}
        == {"narrow", "wide"}
        for row in receipt["observations"]
    )


def validate_hashes(declaration):
    for name, expected in declaration["hashes"].items():
        if file_hash(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def driving_status(report):
    failure = report["episode"].get("infrastructure_failure")
    if failure is not None and failure.get("source") != "model_client":
        return "infrastructure_failure"
    return cell_status(report)


def attempt_application(
    declaration,
    directory,
    scene,
    coverage,
    policy,
    renderer,
    start,
    *,
    runtime_builder=runtime_factory,
    execution=None,
):
    """Use one exact producer construction for drives and own qualification."""
    attempt_scene = copy.deepcopy(scene)
    attempt_scene_path = directory / "scene.json"
    save_new(attempt_scene_path, attempt_scene)
    factory = runtime_builder(
        attempt_scene,
        coverage,
        registered_start=start_pose(start),
        native_factory=lambda *, time, initial=start["native_state"]: SimNative(
            time=time, **initial
        ),
    )
    if execution is not None:
        factory = partial(factory, paused=True)
    app = simulator_app(
        directory,
        attempt_scene_path,
        renderer=PoseRendererLease(renderer, Course(scene).start, start_pose(start)),
        policy=policy,
        runtime_factory=factory,
        declaration={
            **(
                {
                    "paused": True,
                    "camera_capture": "on_demand",
                    "render_lock_path": execution["render_lock_path"],
                    "course_variant": declaration["course_variant"],
                    "layout_is_measured": False,
                    "scene_sha256": declaration["hashes"][declaration["scene"]],
                    "authored_layout": copy.deepcopy(scene.get("authored_layout")),
                    "camera_derivation": copy.deepcopy(declaration.get("camera_derivation")),
                }
                if execution is not None
                else {}
            ),
            "time_scale": 1.0,
            "restrictions": {},
            "study": declaration["protocol"],
            "matched_start": start,
            "coverage": {
                "certificate": declaration["coverage_certificate"],
                "sha256": declaration["hashes"][declaration["coverage_certificate"]],
                "policy": declaration["coverage_policy"],
            },
        },
    )
    app.state.settings.patch(declaration["settings"])
    return app, attempt_scene


def run_attempt(
    declaration,
    model,
    attempt,
    state,
    folder,
    scene,
    coverage,
    policy,
    renderer,
    deadline,
    *,
    runtime_builder=runtime_factory,
    execution=None,
    before_dispatch=None,
    abort=None,
):
    """Own one model's attempt using the same serial and concurrent lifecycle."""
    directory = folder / f"attempt-{attempt}"
    directory.mkdir()
    audit = auth_preflight(
        CLIENTS[model],
        min(
            deadline - time.monotonic(),
            (ATTEMPTS - attempt + 1) * policy.time_limit_s + 600,
        ),
        recovery_output=folder / "auth-recovery",
    )
    save_new(directory / "auth-status.json", audit)
    client = (
        prepare_isolated(
            CLIENTS[model],
            folder / "private-client-state",
            isolated_command(state["gateway"]),
            directory,
            profile="normal",
        )
        if audit["eligible"]
        else None
    )
    if client is None:
        report = dict(
            model=model,
            attempt=attempt,
            status="failed_auth",
            paired_observations=0,
            auth=audit,
        )
        state["skip"] = "failed_auth"
    else:
        state["client"] = client
        start = declaration["starts"][model][attempt - 1]
        app, attempt_scene = attempt_application(
            declaration,
            directory,
            scene,
            coverage,
            policy,
            renderer,
            start,
            runtime_builder=runtime_builder,
            execution=execution,
        )
        text = (
            declaration["prompt"]
            if attempt == 1
            else declaration["continuation"].format(
                previous_outcome=state["previous"], attempt=attempt
            )
        )
        with http_server(app) as producer:
            wait_ready(app)
            with app.state.sim_cameras.lock:
                frames = app.state.sim_cameras.frames
                if (
                    set(frames) != {"narrow", "wide"}
                    or app.state.sim_cameras.errors
                    or frames["narrow"].captured_at != frames["wide"].captured_at
                ):
                    raise RuntimeError("Paired cameras unavailable before dispatch")
            if before_dispatch is not None:
                before_dispatch(app, directory)
            watcher = ReceiptWatcher(
                directory, attempt_scene, client_driving_servers(client), producer
            )

            def checked_poll():
                if abort.is_set():
                    raise RuntimeError("Concurrent wave aborted after a driving fault")
                watcher.poll()

            with state["switch"].connected(producer):
                report = drive(
                    client,
                    app,
                    directory,
                    text,
                    state["session_id"],
                    on_poll=watcher.poll if abort is None else checked_poll,
                    wall_deadline=deadline,
                )
                if execution is not None:
                    watcher.poll(final=True)
                save_new(
                    directory / "drive-calls.json",
                    copy.deepcopy(app.state.recording.history()),
                )
        report["episode"] = json.loads((directory / "episode/summary.json").read_text())
        exposure = audit_attempt(directory, client)
        report.update(
            model=model,
            attempt=attempt,
            matched_start=start,
            paired_observations=received_pairs(exposure["receipt"]),
            producer_paired_publications=producer_paired_observations(
                directory / "episode/events.jsonl"
            ),
            live_received_pairs=len(watcher.records),
            status=driving_status(report),
        )
        if report["status"] in {"failed_auth", "failed_capacity"}:
            state["skip"] = report["status"]
        elif report["status"] == "complete":
            session = report["session_id"]
            if not session or state["session_id"] is not None and session != state["session_id"]:
                report.update(
                    status="infrastructure_failure",
                    infrastructure_error="Continuing conversation lost",
                )
            elif report["paired_observations"] < 1:
                report.update(
                    status="infrastructure_failure",
                    infrastructure_error="No proven client-received narrow+wide pair",
                )
            else:
                state.update(session_id=session, previous=report["episode"]["outcome"])
    if execution is not None:
        report["concurrency"] = dict(
            requested_max_concurrency=execution["max_concurrency"],
            shared_renderer=True,
            camera_capture="on_demand",
            render_lock_path=execution["render_lock_path"],
            ready=json.loads((directory / "wave-ready.json").read_text())
            if (directory / "wave-ready.json").exists()
            else None,
            released=json.loads((directory / "wave-release.json").read_text())
            if (directory / "wave-release.json").exists()
            else None,
            terminal_receipt_drain="postdrive_stream_drain_upper_bound",
        )
    save_new(directory / "attempt.json", report)
    return report


def scientific_configuration_sha256(declaration):
    """Bind exact driving inputs; policy includes the physical attempt time cap."""
    keys = ("settings", "policy", "starts", "prompt", "continuation", "stop_after_success")
    configuration = {key: declaration[key] for key in keys}
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def execution_for(declaration, *, runtime_builder=None, policy_factory=None):
    """Only exact current root qualification may opt into concurrent captures."""
    execution = declaration.get("execution")
    if execution is None:
        return None
    if set(execution) != {"max_concurrency", "camera_capture", "render_lock_path", "qualification"}:
        raise ValueError("Unknown concurrent execution field")
    if not isinstance(declaration.get("course_variant"), str) or not declaration["course_variant"]:
        raise ValueError("Explicit authored course variant required")
    width = execution["max_concurrency"]
    lock = execution["render_lock_path"]
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or not 1 <= width <= 4
        or execution["camera_capture"] != "on_demand"
        or not isinstance(lock, str)
        or not Path(lock).is_absolute()
    ):
        raise ValueError("Expected qualified width 1..4, on-demand capture and absolute GPU lock")
    path = str(Path(execution["qualification"]).resolve())
    expected = declaration["hashes"].get(path)
    if expected is None or file_hash(path) != expected:
        raise ValueError("Current concurrency qualification missing/changed")
    proof = json.loads(Path(path).read_text())
    if (
        proof.get("approved") is not True
        or proof.get("reviewer") != "root"
        or proof.get("course_variant") != declaration.get("course_variant")
        or proof.get("scene_sha256") != declaration["hashes"][declaration["scene"]]
        or proof.get("camera_capture") != "on_demand"
        or proof.get("qualified_max_concurrency") != width
        or proof.get("render_lock_path") != lock
        or proof.get("scientific_configuration_sha256")
        != scientific_configuration_sha256(declaration)
    ):
        raise ValueError("Own scene/source/capture root qualification required")
    pins = proof.get("current_source_hashes", {})
    if not pins or str(Path(__file__).resolve()) not in pins:
        raise ValueError("Actually executing owner must belong to current qualification")
    for source, digest in pins.items():
        if declaration["hashes"].get(source) != digest or file_hash(source) != digest:
            raise ValueError("Concurrency source attribution changed")
    for label, hook in (("runtime_builder", runtime_builder), ("policy_factory", policy_factory)):
        if hook is None:
            continue
        source = str(Path(inspect.getsourcefile(hook)).resolve())
        identity = dict(path=source, sha256=file_hash(source), symbol=hook.__qualname__)
        if proof.get(label) != identity or pins.get(source) != identity["sha256"]:
            raise ValueError("Current qualified runtime/policy hook changed")
    return execution


def run_wave(models, execute, deadline, *, attempt):
    """Join one paused startup group; each worker owns only its model state."""
    release, abort = threading.Event(), threading.Event()
    ready = {model: threading.Event() for model in models}

    def worker(model):
        def before_dispatch(app, directory):
            native = app.state.sim_native
            pose = native.pose_snapshot()
            if native.paused is not True:
                raise ValueError("Native must remain paused until wave release")
            save_new(
                directory / "wave-ready.json",
                dict(
                    model=model,
                    ready_monotonic_s=time.monotonic(),
                    pose=pose,
                ),
            )
            ready[model].set()
            while not release.wait(0.05):
                if abort.is_set() or time.monotonic() >= deadline:
                    raise RuntimeError("Wave aborted/deadline before model dispatch")
                if not native.paused or native.pose_snapshot() != pose:
                    raise RuntimeError("Native advanced during startup barrier")
            if abort.is_set() or time.monotonic() >= deadline:
                raise RuntimeError("Wave aborted/deadline before model dispatch")
            resumed = time.monotonic()
            native.resume()
            save_new(
                directory / "wave-release.json",
                dict(
                    model=model,
                    release_monotonic_s=resumed,
                    ready_native_group_size=sum(event.is_set() for event in ready.values()),
                    paused_initial_pose=pose,
                    released_native_pose=native.pose_snapshot(),
                ),
            )
            app.state.sim_episode.record(
                "concurrent_wave_release", model=model, release_monotonic_s=resumed
            )

        try:
            return execute(model, before_dispatch=before_dispatch, abort=abort)
        except Exception as error:
            abort.set()
            return dict(
                model=model,
                attempt=attempt,
                status="infrastructure_failure",
                worker_error=f"{type(error).__name__}: {error}",
                attempt_report_missing=True,
            )

    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {model: pool.submit(worker, model) for model in models}
        while not all(ready[model].is_set() or futures[model].done() for model in models):
            if abort.is_set() or time.monotonic() >= deadline:
                abort.set()
                break
            time.sleep(0.02)
        release.set()
        while not all(future.done() for future in futures.values()):
            if time.monotonic() >= deadline:
                abort.set()
            for future in futures.values():
                if future.done() and future.result().get("status") == "infrastructure_failure":
                    abort.set()
            time.sleep(0.02)
        return [futures[model].result() for model in models]


def run(
    protocol_path,
    output,
    *,
    coverage_factory=CoverageCertificate,
    runtime_builder=runtime_factory,
    policy_factory=RealCoursePolicy,
    checkpoint_review=None,
):
    """Run fresh cells only; every attempt rechecks frozen inputs and the deadline."""
    from drivingbench.sim.logs import REPO

    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    declaration = json.loads(protocol_path.read_text())
    protocol_sha256 = file_hash(protocol_path)
    expected_protocol = TIDY_PROTOCOL if declaration.get("execution") is not None else PROTOCOL
    if (
        declaration.get("protocol") != expected_protocol
        or declaration.get("client_profile") != "normal"
    ):
        raise ValueError("Expected measured-course normal-client declaration")
    if (
        not declaration.get("release_passed")
        or declaration["models"] != [asdict(CLIENTS[k]) for k in MODELS]
        or not isinstance(declaration.get("stop_after_success"), bool)
        or declaration.get("client_done_parking_confirmation_s") != PARKING_CONFIRMATION_S
        or declaration.get("coverage_policy") != COVERAGE_POLICY
    ):
        raise ValueError("Release, model selection or success-attempt rule changed")
    validate_hashes(declaration)
    scene_path = Path(declaration["scene"])
    scene = json.loads(scene_path.read_text())
    if "wide_camera" not in scene or declaration["settings"]["speed_limit_mps"] != 3.5:
        raise ValueError("Matched course needs both cameras and 3.5 m/s ceiling")
    deadline = (
        time.monotonic()
        + (
            datetime.fromisoformat(declaration["deadline_utc"]) - datetime.now(timezone.utc)
        ).total_seconds()
    )
    if time.monotonic() >= deadline:
        raise ValueError("Frozen batch deadline has expired")
    coverage = coverage_factory(declaration["coverage_certificate"]).validate_assets(REPO)
    start_preflight = preflight_starts(
        scene, declaration["starts"], coverage, runtime_builder=runtime_builder
    )
    policy = policy_factory(**declaration.get("policy", {}))
    execution = execution_for(
        declaration, runtime_builder=runtime_builder, policy_factory=policy_factory
    )
    if execution is not None and checkpoint_review is None:
        raise ValueError("Concurrent execution requires root first-four image review")
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "protocol.json", declaration)
    save_new(output / "start-preflight.json", start_preflight)
    renderer = SceneRenderer(scene_path, output / "renderer")
    reports, states, batch_error = [], {}, None
    try:
        with render_slot(None if execution is None else execution["render_lock_path"]):
            renderer.start()
        with ExitStack() as stack:
            for model in MODELS:
                folder = output / model
                folder.mkdir()
                switch = ProducerSwitch()
                proxy = stack.enter_context(http_server(switch.app))
                gateway = stack.enter_context(http_server(gateway_app(proxy)))
                states[model] = dict(
                    switch=switch,
                    gateway=gateway,
                    client=None,
                    session_id=None,
                    previous=None,
                    skip=None,
                )
            for attempt in range(1, ATTEMPTS + 1):
                round_reports = []
                eligible = [
                    model
                    for model in MODELS
                    if not states[model]["skip"]
                    and not (
                        declaration["stop_after_success"] and states[model]["previous"] == "success"
                    )
                ]

                def execute(model, **options):
                    if file_hash(protocol_path) != protocol_sha256:
                        raise ValueError("Frozen declaration changed")
                    validate_hashes(declaration)
                    coverage.validate_assets(REPO)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Frozen batch safety deadline reached")
                    return run_attempt(
                        declaration,
                        model,
                        attempt,
                        states[model],
                        output / model,
                        scene,
                        coverage,
                        policy,
                        renderer,
                        deadline,
                        runtime_builder=runtime_builder,
                        execution=execution,
                        **options,
                    )

                def record(report):
                    reports.append(report)
                    round_reports.append(report)
                    save_new(output / f"closed-{len(reports):02d}.json", reports)

                if execution is None:
                    for model in eligible:
                        report = execute(model)
                        record(report)
                        if report["status"] == "infrastructure_failure":
                            raise RuntimeError(f"{model}: driving infrastructure failure")
                else:
                    width = execution["max_concurrency"]
                    for offset in range(0, len(eligible), width):
                        group = eligible[offset : offset + width]
                        for report in run_wave(group, execute, deadline, attempt=attempt):
                            record(report)
                        if any(r["status"] == "infrastructure_failure" for r in round_reports):
                            raise RuntimeError("Concurrent wave driving infrastructure failure")
                if attempt == 1:
                    result = checkpoint(round_reports)
                    if result["passed"] and checkpoint_review is not None:
                        result["root_image_review"] = checkpoint_review(
                            round_reports, output, protocol_path
                        )
                        result["passed"] = result["root_image_review"].get("approved") is True
                        if not result["passed"]:
                            result["problems"].append(
                                "Root first-four original-image review denied"
                            )
                    save_new(output / "checkpoint.json", result)
                    if not result["passed"]:
                        raise RuntimeError(
                            "First-round paired-receipt/infrastructure checkpoint failed"
                        )
    except Exception as error:
        batch_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        renderer.close()
        done = {(row["model"], row["attempt"]): row["status"] for row in reports}
        assignments = [
            dict(
                model=model,
                attempt=attempt,
                status=done.get((model, attempt), "not_run"),
                reason=None
                if (model, attempt) in done
                else states.get(model, {}).get("skip")
                or (
                    "prior_success"
                    if declaration["stop_after_success"]
                    and states.get(model, {}).get("previous") == "success"
                    else batch_error
                ),
            )
            for attempt in range(1, ATTEMPTS + 1)
            for model in MODELS
        ]
        save_new(
            output / "batch.json",
            {
                "attempts": reports,
                "assignments": assignments,
                "batch_error": batch_error,
                "maximum_drives": 12,
                "stop_after_success": declaration["stop_after_success"],
            },
        )
    return reports


def settings(prompt):
    objective = prompt.split("OBJECTIVE\n", 1)[1].split("\n\nROLE", 1)[0]
    return camera_settings(objective).model_dump()


def freeze(
    scene_path,
    coverage_path,
    strict_audit_path,
    output,
    *,
    stop_after_success,
    deadline_hours=24.0,
    root=None,
    runtime_builder=runtime_factory,
    policy_factory=RealCoursePolicy,
    execution=None,
    bound_inputs=(),
    course_variant=None,
):
    """Hash the release evidence and all camera/stack inputs before dispatch."""
    from drivingbench.sim.logs import REPO
    from drivingbench.sim.scene import scene_asset_paths
    from drivingbench.sim.visual_splat import RENDER, TOOLS

    if (
        not isinstance(stop_after_success, bool)
        or not math.isfinite(deadline_hours)
        or deadline_hours <= 0
    ):
        raise ValueError("Freeze needs an explicit success rule and positive deadline")
    root = Path(root or REPO).resolve()
    scene_path, coverage_path, strict_audit_path = (
        Path(p).resolve() for p in (scene_path, coverage_path, strict_audit_path)
    )
    scene = json.loads(scene_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    from drivingbench.sim.real_course_study.coverage import CoverageCertificate

    certificate = CoverageCertificate(coverage_path).validate_assets(root)
    audit = json.loads(strict_audit_path.read_text())
    if coverage.get("approved") is not True or audit.get("status") != "cpu_complete":
        raise ValueError("Rendered camera approval and completed strict audit are required")
    if "wide_camera" not in scene:
        raise ValueError("Both cameras must be declared")
    prompt_path = root / "docs/sim/real-course-original-prompt.json"
    registration_path = root / "docs/sim/real-course-registration.json"
    prompt_record = json.loads(prompt_path.read_text())
    registration = json.loads(registration_path.read_text())
    starts = registered_starts(registration)
    from drivingbench.sim.validate import loaded

    real = {r["segment"]: r for r in registration["attempts"] if r["scope"] == "primary"}
    series = {}
    for rows in starts.values():
        for row in rows:
            reference = real[row["real_segment"]]
            path = root / reference["series_path"]
            if path not in series:
                series[path] = loaded(path)[0]
            columns, k = series[path], reference["start_index"]
            if abs(columns["v"][k]) > 0.05 or not bool(columns["cruise_standstill"][k]):
                raise ValueError("Matched first-command start must be held at standstill")
            row["native_state"] = {
                "speed": 0.0,
                "steer_deg": float(columns["angle"][k]),
                "hold": True,
            }
            row["logged_initial_speed_mps"] = float(columns["v"][k])
    start_preflight = preflight_starts(scene, starts, certificate, runtime_builder=runtime_builder)
    inputs = [scene_path, coverage_path, strict_audit_path, prompt_path, registration_path, *series]
    inputs += [root / scene["calibration"], root / scene["background"]["checkpoint"]]
    inputs += [root / p for p in scene_asset_paths(scene)]
    inputs += [
        root / scene["wide_camera"][k] for k in ("checkpoint", "calibration", "vehicle_layer")
    ]
    inputs += [RENDER, TOOLS / "Blender.app/Contents/MacOS/Blender"]
    inputs += sorted(source_files())
    inputs += sorted((root / "sim/real_course_study").glob("*.py"))
    inputs += [root / "sim/real_course_parking_analysis.py"]
    inputs += [
        path
        for path in (root / "vendor/openpilot").rglob("*")
        if path.is_file() and path.suffix in {".py", ".capnp"}
    ]
    inputs += [
        root / "sim/params.json",
        root / "sim/corolla.carparams",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "docs/upstream-sources.json",
    ]
    inputs += [Path(p).resolve() for p in bound_inputs]
    if execution is not None:
        inputs.append(Path(execution["qualification"]).resolve())
    policy = policy_factory()
    declaration = {
        "protocol": PROTOCOL,
        "release_passed": True,
        "stop_after_success": stop_after_success,
        "deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=deadline_hours)).isoformat(),
        "deadline_hours": deadline_hours,
        "client_profile": "normal",
        "models": [asdict(CLIENTS[key]) for key in MODELS],
        "scene": str(scene_path),
        "coverage_certificate": str(coverage_path),
        "strict_audit": str(strict_audit_path),
        "prompt": prompt_record["prompt"],
        "continuation": "The previous attempt ended: {previous_outcome}. The car has stopped and been reset. You have another attempt on the same course; the objective and driving instructions are unchanged. This is attempt {attempt} of three.",
        "continuation_difference": "Standardized operator reset feedback; original common initial prompt remains byte-identical. Success-attempt ending is explicitly frozen separately.",
        "starts": starts,
        "start_preflight": start_preflight,
        "start_frame": "Immutable scene start is renderer origin; registered per-attempt pose initializes native map frame. No map or start relocation.",
        "settings": settings(prompt_record["prompt"]),
        "policy": asdict(policy),
        "client_done_parking_confirmation_s": PARKING_CONFIRMATION_S,
        "hashes": {str(p.resolve()): file_hash(p) for p in inputs},
        "clock": "real-time physics, wall observe timestamps, no scenario restrictions",
        "plant": "current live Params/Corolla; offline refit and uncertainty candidates not installed",
        "coverage_policy": copy.deepcopy(COVERAGE_POLICY),
        "checkpoint": "After four assigned first attempts: >=2 models with proven client-received narrow+wide pairs, no driving infrastructure failures; every healthy eligible drive needs paired receipt. Auth/capacity failures remain failed cells and skip later attempts for that model; no fresh conversation or replacement.",
        "post_drive": "No added debrief in matched real-trial replication; any later approved question uses one retry and records missing, without stopping driving allocation.",
    }
    if execution is not None:
        declaration["protocol"] = TIDY_PROTOCOL
        declaration["execution"] = copy.deepcopy(execution)
        declaration["course_variant"] = course_variant
        declaration["layout_is_measured"] = False
        declaration["matched_real_trial_scene"] = False
        declaration["camera_capture"] = "on_demand"
        declaration["terminal_receipt_drain"] = (
            "after client join; final upper bound only, no backdating"
        )
        declaration["checkpoint"] += (
            " Root hash-bound original received N/W image review after the first four is required before attempts 2 and 3; no behavior gate."
        )

        execution_for(declaration, runtime_builder=runtime_builder, policy_factory=policy_factory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "protocol.json", declaration)
    return declaration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare = commands.add_parser("freeze")
    for name in ("scene", "coverage", "strict-audit", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--success-rule", choices=("stop", "continue-three"), required=True)
    prepare.add_argument("--deadline-hours", type=float, default=24.0)
    execute = commands.add_parser("run")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(
            args.scene,
            args.coverage,
            args.strict_audit,
            args.output,
            stop_after_success=args.success_rule == "stop",
            deadline_hours=args.deadline_hours,
        )
    else:
        run(args.protocol, args.output)


if __name__ == "__main__":
    main()
