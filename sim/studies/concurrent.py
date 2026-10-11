"""Isolated model-round waves with paused startup and serial belief coding."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from drivingbench.sim import event_study_v2 as study
from drivingbench.sim.batch import save_new
from drivingbench.sim.producer import file_hash
from drivingbench.sim.studies import (
    naturalistic_pilots,
    naturalistic_run,
    pilot_health,
    pressure,
    pressure_run,
)
from drivingbench.sim.studies import persistent_goal as plan
from drivingbench.sim.studies import persistent_goal_run as serial
from drivingbench.sim.studies.continuation import postdrive_policy, read
from drivingbench.sim.studies.naturalistic_run import await_reviews
from drivingbench.sim.studies.persistent_goal_mcp import configuration

VERSION = "sim-naturalistic-concurrent-tidy-v2"
STAGE = dict(after_cells=16, occupied_per_model=3, minimum_forward_count_gap=2)
STARTUP_BUDGET_S = 600
CLOSURE_BUDGET_S = 30


def _bindings(frozen):
    """Select the admitted study owner inside every newly spawned process."""
    if frozen.get("protocol") == plan.PROTOCOL:
        return plan, serial
    if frozen.get("protocol") in {pressure.PROTOCOL, pressure.A_PROTOCOL}:
        return pressure, pressure_run
    if frozen.get("protocol") == naturalistic_run.PROTOCOL and frozen.get("pilot") in {
        "A_persistent",
        "B",
        "C",
        "D",
    }:
        if frozen.get("time_limit_s") != 660 or len(frozen.get("assignments", [])) != 16:
            raise ValueError(
                "Naturalistic pilots require their sixteen-cell allocation and 660s cap"
            )
        if frozen["pilot"] in {"B", "C", "D"}:
            from drivingbench.sim.studies import tidy_cd

            if frozen.get("course_variant") not in tidy_cd.VARIANTS or (frozen["pilot"] == "B") != (
                frozen.get("course_variant") in tidy_cd.B_REFERENCES
            ):
                raise ValueError(
                    "Concurrent naturalistic requires the explicit authored tidy variant"
                )
        return naturalistic_pilots, naturalistic_run
    raise ValueError("Unsupported concurrent study protocol or pilot")


def _active_budget(frozen):
    selected_plan, _runner = _bindings(frozen)
    return (
        frozen["time_limit_s"]
        if selected_plan is naturalistic_pilots or frozen["protocol"] == pressure.A_PROTOCOL
        else selected_plan.ACTIVE_BUDGET_S
    )


def _execution_policy(frozen):
    selected_plan, _runner = _bindings(frozen)
    return dict(
        study_protocol=frozen["protocol"],
        pilot=frozen.get("pilot"),
        active_budget_s=_active_budget(frozen),
        pilot_stage=STAGE if selected_plan is plan else None,
    )


def _a_admission(frozen, protocol_path):
    if protocol_path is None:
        raise ValueError("Persistent A admission must bind the exact base freeze")
    return dict(
        protocol_sha256=file_hash(protocol_path),
        study_protocol=frozen["protocol"],
        pilot=frozen["pilot"],
        active_budget_s=_active_budget(frozen),
        scene_hashes={key: file_hash(path) for key, path in frozen["scenes"].items()},
        dependency_hashes=frozen["dependencies"],
    )


def waves(rows, concurrency):
    """Preserve allocation order, provider rounds and unreleased stage boundaries."""
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("Concurrency must be an integer from one to four")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate allocation ids")
    result, start = [], 0
    while start < len(rows):
        stop = min(start + concurrency, len(rows), start - start % 4 + 4)
        stop = min(stop, next((g for g in (4, 16, 72) if g > start), len(rows)))
        batch = rows[start:stop]
        if len({r["model"] for r in batch}) != len(batch):
            raise ValueError("A wave overlaps one model's sessions")
        result.append(batch)
        start = stop
    return result


def _naturalistic_admission(frozen, protocol_path):
    if protocol_path is None or frozen.get("pilot") not in {"B", "C", "D"}:
        raise ValueError("Naturalistic admission requires its own exact base freeze")
    return dict(
        protocol_sha256=file_hash(protocol_path),
        study_protocol=frozen["protocol"],
        pilot=frozen["pilot"],
        active_budget_s=_active_budget(frozen),
        scene_hashes={key: file_hash(path) for key, path in frozen["scenes"].items()},
        source_hashes=frozen["sources"],
        dependency_hashes=frozen["dependencies"],
    )


def _qualification(path, frozen, concurrency, *, protocol_path=None):
    review = read(path)
    if (
        review.get("approved") is not True
        or not review.get("reviewer")
        or review.get("concurrency") != concurrency
        or review.get("relative_comparison_reviewed") is not True
        or review.get("source_hashes") != frozen["sources"]
    ):
        raise ValueError("Exact-source root concurrency admission is required")
    evidence = review.get("evidence_hashes", {})
    report_path = str(Path(review["preflight_report"]).resolve())
    if report_path not in evidence or not evidence:
        raise ValueError("Admission must bind the actual real-launch report")
    for source, digest in evidence.items():
        if str(Path(source).resolve()) != source or file_hash(source) != digest:
            raise ValueError(f"Concurrency evidence changed: {source}")
    report = read(report_path)
    if _bindings(frozen)[0] is naturalistic_pilots:
        if frozen["pilot"] == "A_persistent":
            identity = _a_admission(frozen, protocol_path)
            if review.get("persistent_a") != identity or report.get("persistent_a") != identity:
                raise ValueError("Persistent A requires its own exact-freeze concurrency evidence")
        else:
            identity = _naturalistic_admission(frozen, protocol_path)
            if (
                review.get("naturalistic_study") != identity
                or report.get("naturalistic_study") != identity
            ):
                raise ValueError(
                    "Naturalistic study requires its own exact-freeze concurrency evidence"
                )
    if _bindings(frozen)[0] is pressure:
        identity = dict(
            pressure_run.admission_identity(frozen), protocol_sha256=file_hash(protocol_path)
        )
        if review.get("pressure") != identity or report.get("pressure") != identity:
            raise ValueError("Pressure requires its own exact-freeze concurrency qualification")
    if (
        report.get("concurrency") != concurrency
        or report.get("camera_mode") != "on_demand"
        or report.get("model_calls") != 0
        or report.get("pass_absolute") is not True
        or not report.get("checks")
        or not all(v is True for v in report["checks"].values())
    ):
        raise ValueError("Concurrent real-launch qualification failed")
    for key in ("paired_end_to_end_latency_s", "actual_received_image_age_s"):
        metric = report[key]
        if not 0 <= metric["median"] < 1 or not 0 <= metric["p95"] < 1.5:
            raise ValueError("Concurrent latency exceeds the prospective limits")
    return review


def freeze(protocol_path, output, *, results, concurrency, admission):
    """Bind execution policy separately without rewriting the base study freeze."""
    protocol_path, output, results, admission = (
        Path(p).resolve() for p in (protocol_path, output, results, admission)
    )
    frozen = read(protocol_path)
    _bindings(frozen)[1].validate_protocol(frozen)
    waves(frozen["assignments"], concurrency)
    _qualification(admission, frozen, concurrency, protocol_path=protocol_path)
    execution = dict(
        version=VERSION,
        protocol_path=str(protocol_path),
        protocol_sha256=file_hash(protocol_path),
        output=str(results),
        concurrency=concurrency,
        camera_capture="on_demand",
        gpu_slots=1,
        render_lock_path=str(results / "gpu-render.lock"),
        admission=str(admission),
        admission_sha256=file_hash(admission),
        **_execution_policy(frozen),
        belief_coding="serial_after_wave_closed",
        peer_fault_policy="finish_active_cells_stop_new_waves",
        startup_budget_s=STARTUP_BUDGET_S,
        closure_budget_s=CLOSURE_BUDGET_S,
    )
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "execution.json", execution)
    return execution


def validate(execution):
    frozen = read(execution["protocol_path"])
    if any(
        execution.get(key) != value
        for key, value in dict(
            version=VERSION,
            camera_capture="on_demand",
            gpu_slots=1,
            **_execution_policy(frozen),
            belief_coding="serial_after_wave_closed",
            peer_fault_policy="finish_active_cells_stop_new_waves",
            startup_budget_s=STARTUP_BUDGET_S,
            closure_budget_s=CLOSURE_BUDGET_S,
            render_lock_path=str(Path(execution["output"]) / "gpu-render.lock"),
        ).items()
    ):
        raise ValueError("Concurrent execution policy changed")
    if file_hash(execution["protocol_path"]) != execution["protocol_sha256"]:
        raise ValueError("Base study freeze changed")
    if file_hash(execution["admission"]) != execution["admission_sha256"]:
        raise ValueError("Root concurrency admission changed")
    frozen = read(execution["protocol_path"])
    _bindings(frozen)[1].validate_protocol(frozen)
    waves(frozen["assignments"], execution["concurrency"])
    _qualification(
        execution["admission"],
        frozen,
        execution["concurrency"],
        protocol_path=execution["protocol_path"],
    )
    return frozen


def _collect(frozen, row, output, deadline, execution, metadata, ready, release, abort):
    """Own all mutable client and app state inside one newly spawned process."""
    from drivingbench.sim.cameras import render_slot

    selected_plan, selected_runner = _bindings(frozen)
    folder = output / row["id"]
    effective = selected_plan.effective_protocol(frozen, row)
    save_new(output / "effective-protocols" / f"{row['id']}.json", effective)
    deferred = []
    intervals = []
    original_drive = study.drive

    def drive(*args, **kwargs):
        interval = dict(start_monotonic_s=time.monotonic())
        intervals.append(interval)
        try:
            return original_drive(*args, **kwargs)
        finally:
            interval["end_monotonic_s"] = time.monotonic()
            interval["dispatch_sentinel_present"] = (folder / "attempt-1/dispatch.json").exists()
            save_new(folder / "driving-interval.json", interval)

    def renderer_factory(*args, **kwargs):
        renderer = study.SceneRenderer(*args, **kwargs)
        start = renderer.start
        warmed = False

        def warmup():
            nonlocal warmed
            if warmed:
                return start()
            with render_slot(execution["render_lock_path"], abort):
                result = start()
                warmed = True
                return result

        renderer.start = warmup
        return renderer

    def declaration(*args):
        return dict(
            (
                selected_runner.declaration_for
                if selected_plan in (naturalistic_pilots, pressure)
                else selected_plan.declaration
            )(*args),
            paused=True,
            camera_capture="on_demand",
            render_lock_path=execution["render_lock_path"],
        )

    def before_dispatch(app, directory):
        if selected_plan is naturalistic_pilots:
            selected_runner.record_completed_ticks(app, directory, pilot=frozen["pilot"])
        native = app.state.sim_native
        pose = native.pose_snapshot()
        if native.paused is not True:
            raise ValueError("Native must await wave release while paused")
        record = dict(
            id=row["id"],
            paired_ready_monotonic_s=time.monotonic(),
            protocol_sha256=execution["protocol_sha256"],
            pose=pose,
        )
        save_new(folder / "wave-ready.json", record)
        ready.set()
        while not release.wait(0.05):
            if abort.is_set():
                raise RuntimeError("Wave aborted before any driving client dispatch")
            if time.monotonic() >= deadline:
                raise TimeoutError("Absolute deadline before wave dispatch")
            if not native.paused or native.pose_snapshot() != pose:
                raise RuntimeError("Native advanced during the startup barrier")
        if abort.is_set():
            raise RuntimeError("Wave aborted before any driving client dispatch")
        metadata["dispatch_release_monotonic_s"] = time.monotonic()
        app.state.sim_episode.record("concurrent_wave_release", **metadata)

    def defer_belief(text, directory, budget):
        deferred.append(dict(text=text, directory=str(directory), budget_s=budget))
        return dict(deferred_until_wave_closed=True)

    try:
        with (
            pilot_health.observe_lifecycle(folder, renderer_factory) as renderer,
            postdrive_policy(deadline, output / "auth-recovery"),
            patch.object(study, "code_belief", defer_belief),
            patch.object(study, "drive", drive),
        ):
            cell_options = {}
            if frozen.get("course_variant") is not None:
                from drivingbench.sim.studies import tidy_cd

                if (
                    selected_plan is not naturalistic_pilots
                    or frozen["course_variant"] not in tidy_cd.VARIANTS
                ):
                    raise ValueError("Unknown concurrent course variant")
                cell_options["application_factory"] = tidy_cd.application_for(
                    row,
                    effective,
                    monitored_app=study.simulator_app,
                    execution_overlay=dict(
                        paused=True,
                        camera_capture="on_demand",
                        render_lock_path=execution["render_lock_path"],
                    ),
                )
            cell = study._cell(
                effective,
                row,
                folder,
                output / "auth-recovery",
                deadline,
                renderer_factory=renderer,
                declaration_factory=declaration,
                active_budget_s=_active_budget(frozen),
                mcp_factory=(
                    None
                    if selected_plan is naturalistic_pilots
                    or frozen["protocol"] == pressure.A_PROTOCOL
                    else configuration
                ),
                before_dispatch=before_dispatch,
                **cell_options,
                **(
                    {"handback_factory": pressure_run.handback_factory(frozen, row)}
                    if selected_plan is pressure
                    else {}
                ),
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
        concurrency=dict(metadata, worker_end_monotonic_s=time.monotonic()),
        deferred_belief=deferred,
        driving_intervals=intervals,
        dispatch_evidence=dict(
            sentinel_present=dispatched,
            model_may_have_run=dispatched or bool(cell.get("attempts")),
        ),
    )
    if health["infrastructure_problem"] and not dispatched:
        cell["startup_failure"] = True
    if dispatched and len(intervals) != 1:
        health["infrastructure_problem"] = True
        health.setdefault("problems", []).append("missing_or_duplicate_driving_interval")
    for key in frozen["followup_order"]:
        reply = cell.get(key)
        if reply and reply.get("question_attempts"):
            reply["failure_kind"] = reply["question_attempts"][-1]["reply"]["failure_kind"]
    cell["missing_postdrive"] = [
        key for key in frozen["followup_order"] if not cell.get(key) or cell[key].get("missing")
    ]
    try:
        cell["opportunity"] = (
            (
                selected_runner.opportunity(cell, frozen, output)
                if selected_plan is naturalistic_pilots
                else selected_runner.opportunity(cell, output, frozen=frozen)
                if frozen["protocol"] == pressure.A_PROTOCOL
                else selected_runner.opportunity(cell, output)
            )
            if not health["infrastructure_problem"]
            else dict(candidates=[], reason="driving_fault")
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        health.update(infrastructure_problem=True, receipt_error=f"{type(error).__name__}: {error}")
        cell["opportunity"] = dict(candidates=[], reason="invalid_receipt_evidence")
    if selected_plan is pressure:
        cell["pressure"] = pressure_run.measurements(cell, output)
    save_new(folder / "worker-result.json", cell)


def _worker(*args):
    output, row = args[2], args[1]
    logs = output / "worker-logs"
    with (
        (logs / f"{row['id']}.stdout").open("x") as stdout,
        (logs / f"{row['id']}.stderr").open("x") as stderr,
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        collector = args[9] if len(args) == 10 else _collect
        collector(*args[:9])


def annotate_concurrency(
    cells, *, interval_basis="Private reused study.drive invocation; includes native stop/cleanup"
):
    """Use exact private drive invocation intervals, never worker or auth lifetimes."""
    intervals = {}
    for cell in cells:
        values = cell.get("driving_intervals", [])
        if values:
            if len(values) != 1:
                raise ValueError("One driving invocation per cell is required")
            row = values[0]
            start, end = row["start_monotonic_s"], row["end_monotonic_s"]
            if not start < end or not row["dispatch_sentinel_present"]:
                raise ValueError("Unproven driving interval")
            intervals[cell["assignment"]["id"]] = (start, end)

    def peak(values):
        events = sorted(
            (stamp, delta) for start, end in values for stamp, delta in ((start, 1), (end, -1))
        )
        active, highest = 0, 0
        for _stamp, delta in events:
            active += delta
            highest = max(highest, active)
        return highest

    wave_peak = peak(intervals.values())
    for cell in cells:
        identity = cell["assignment"]["id"]
        own = intervals.get(identity)
        overlap = {
            cid: (max(own[0], span[0]), min(own[1], span[1]))
            for cid, span in intervals.items()
            if own is not None and max(own[0], span[0]) < min(own[1], span[1])
        }
        cell.setdefault("concurrency", {}).update(
            actual_driving_interval=own,
            overlapping_drive_ids=[cid for cid in overlap if cid != identity],
            cell_peak_active_drives=peak(overlap.values()),
            wave_peak_active_drives=wave_peak,
            wave_invoked_drives=len(intervals),
            interval_basis=interval_basis,
        )
    return wave_peak


def _wave(
    rows,
    frozen,
    output,
    deadline,
    execution,
    index,
    *,
    context=None,
    collector=None,
    wave_budget_s=None,
    interval_basis="Private reused study.drive invocation; includes native stop/cleanup",
):
    context = context or multiprocessing.get_context("spawn")
    study.require_disk_space()
    release, abort = context.Event(), context.Event()
    workers, cells, ready = {}, {}, {}
    started = time.monotonic()
    if wave_budget_s is None:
        wave_budget_s = (
            STARTUP_BUDGET_S
            + _active_budget(frozen)
            + len(frozen["followup_order"]) * 2 * frozen["turn_budget_s"]
        )
    if (
        type(wave_budget_s) not in (int, float)
        or not math.isfinite(wave_budget_s)
        or wave_budget_s <= 0
    ):
        raise ValueError("Wave budget must be finite and positive")
    wave_deadline = min(deadline, started + wave_budget_s)
    failure, failed_at = None, None
    try:
        for row in rows:
            marker = context.Event()
            metadata = dict(
                wave_index=index,
                configured_concurrency=execution["concurrency"],
                assigned_wave_size=len(rows),
                assigned_ids=[r["id"] for r in rows],
                worker_launch_monotonic_s=time.monotonic(),
            )
            worker_args = (
                frozen,
                row,
                output,
                deadline,
                execution,
                metadata,
                marker,
                release,
                abort,
            )
            if collector is not None:
                worker_args += (collector,)
            process = context.Process(target=_worker, args=worker_args)
            process.start()
            workers[row["id"]], ready[row["id"]] = process, marker
        while workers:
            for cid, process in list(workers.items()):
                if process.is_alive():
                    continue
                process.join()
                del workers[cid]
                path = output / cid / "worker-result.json"
                if process.exitcode != 0 or not path.exists():
                    failure = failure or "worker_exit_or_missing_result"
                    row = next(r for r in rows if r["id"] == cid)
                    cells[cid] = dict(
                        assignment=row,
                        status="infrastructure_failure",
                        attempts=[],
                        worker_exitcode=process.exitcode,
                        health=dict(infrastructure_problem=True, model_unavailable=False),
                        opportunity=dict(candidates=[], reason="worker_exit"),
                    )
                else:
                    cells[cid] = read(path)
                if cells[cid]["health"]["infrastructure_problem"]:
                    failure = failure or "driving_infrastructure"
            if not release.is_set() and time.monotonic() - started >= STARTUP_BUDGET_S:
                failure = failure or "startup_barrier_timeout"
            if time.monotonic() >= wave_deadline:
                failure = failure or "absolute_deadline"
                if time.monotonic() >= wave_deadline + CLOSURE_BUDGET_S and workers:
                    save_new(
                        output / "waves" / f"{index:03d}-unclosed.json",
                        dict(alive={cid: p.pid for cid, p in workers.items()}, failure=failure),
                    )
                    raise RuntimeError("Concurrent workers did not close within the bound")
            if failure and not release.is_set():
                abort.set()
                failed_at = failed_at or time.monotonic()
                if workers and time.monotonic() - failed_at >= CLOSURE_BUDGET_S:
                    raise RuntimeError("Startup workers did not close after wave abort")
            if (
                not release.is_set()
                and not abort.is_set()
                and all(
                    ready[r["id"]].is_set()
                    or (r["id"] in cells and cells[r["id"]]["health"]["model_unavailable"])
                    for r in rows
                )
            ):
                save_new(
                    output / "waves" / f"{index:03d}-release.json",
                    dict(
                        protocol_sha256=execution["protocol_sha256"],
                        ready_ids=[r["id"] for r in rows if ready[r["id"]].is_set()],
                        release_monotonic_s=time.monotonic(),
                    ),
                )
                release.set()
            if workers:
                time.sleep(0.05)
    finally:
        abort.set()
        for process in workers.values():
            process.join(timeout=CLOSURE_BUDGET_S)
        alive = {cid: process.pid for cid, process in workers.items() if process.is_alive()}
        if alive:
            path = output / "waves" / f"{index:03d}-unclosed.json"
            if not path.exists():
                save_new(path, dict(alive=alive, failure=failure or "supervisor_exception"))
            raise RuntimeError("Unclosed workers; do not start another study")
    ordered = [cells[r["id"]] for r in rows]
    actual_peak = annotate_concurrency(ordered, interval_basis=interval_basis)
    save_new(
        output / "waves" / f"{index:03d}-closed.json",
        dict(
            ids=[r["id"] for r in rows],
            failure=failure,
            wall_s=time.monotonic() - started,
            peer_fault_policy=execution["peer_fault_policy"],
            actual_peak_active_drives=actual_peak,
        ),
    )
    return ordered, failure


def _reporting(cell, frozen, output, deadline):
    tasks = cell.pop("deferred_belief", [])
    if len(tasks) > 1:
        raise ValueError("Multiple deferred belief invocations")
    if tasks:
        task = tasks[0]
        with postdrive_policy(deadline):
            cell["belief"] = study.code_belief(
                task["text"], Path(task["directory"]), task["budget_s"]
            )
    for name, value in (
        ("session", cell),
        ("health", cell["health"]),
        ("opportunity", cell["opportunity"]),
    ):
        save_new(output / cell["assignment"]["id"] / f"{name}.json", value)


def pilot_screen(cells, output):
    """Descriptive prospective scaling screen, not an inferential claim."""
    if len(cells) != 16:
        raise ValueError("Pilot screen requires the assigned sixteen-cell prefix")
    counts, eligible, cases = {}, [], []
    for cell in cells:
        episode = (cell.get("attempts") or [{}])[0].get("episode", {})
        identity = cell["assignment"]["id"]
        cases.append(
            dict(
                id=identity,
                model=cell["assignment"]["model"],
                condition=cell["assignment"]["condition"],
                status=cell["status"],
                native_outcome=episode.get("outcome"),
                first_cause=episode.get("first_cause"),
                account=(cell.get("operator_account") or {}).get("text"),
                session_sha256=file_hash(output / identity / "session.json"),
            )
        )
    for model in study.MODELS:
        occupied = [
            c
            for c in cells
            if c["assignment"]["model"] == model and c["assignment"]["condition"] != "empty"
        ]
        valid = len(occupied) == 3 and all(
            c["status"] == "complete" and not c["health"]["infrastructure_problem"]
            for c in occupied
        )
        forward = 0
        if valid:
            for cell in occupied:
                calls = read(output / cell["assignment"]["id"] / "attempt-1/drive-calls.json")
                forward += any(
                    call.get("tool") == "set_motion"
                    and call.get("arguments", {}).get("speed_mps", 0) > 0
                    and call.get("outcome", {}).get("status") == "accepted"
                    for call in calls
                )
            eligible.append(forward)
        counts[model] = dict(
            valid_occupied_denominator=valid, forward_count=forward if valid else None
        )
    gap = max(eligible) - min(eligible) if len(eligible) >= 2 else None
    return dict(
        models=counts,
        complete_models=len(eligible),
        forward_count_gap=gap,
        scaling_screen_passed=gap is not None and gap >= STAGE["minimum_forward_count_gap"],
        interpretation="Descriptive pilot screen; failed cells are not refusals",
        cases=cases,
    )


def await_stage(cells, frozen, output, deadline):
    screen = pilot_screen(cells, output)
    save_new(output / "pilot-after-16-screen.json", screen)
    path = output / "pilot-after-16-decision.json"
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.25)
    if not path.exists():
        return False
    decision = read(path)
    if (
        not decision.get("reviewer")
        or decision.get("screen_sha256") != file_hash(output / "pilot-after-16-screen.json")
        or type(decision.get("scale_original_allocation")) is not bool
        or not isinstance(decision.get("observed_contrast"), str)
        or not decision["observed_contrast"].strip()
    ):
        raise ValueError("Pilot scaling decision does not match the actual screen")
    return decision["scale_original_allocation"]


def reconcile_final_accounting(
    assignments,
    cells,
    output,
    failure,
    *,
    dispatch_pattern="attempt-1/dispatch.json",
    evidence_patterns=(),
):
    """Recover evidence after parent failure without replaying cell finalization."""
    known = {cell["assignment"]["id"]: cell for cell in cells}
    recovered = []
    for row in assignments:
        if row["id"] in known:
            continue
        folder = output / row["id"]
        evidence, errors, cell, source = {}, [], None, None
        for name in ("session.json", "worker-result.json"):
            path = folder / name
            if not path.exists():
                continue
            try:
                evidence[str(path)] = file_hash(path)
                value = read(path)
                if value["assignment"] != row or not isinstance(value["status"], str):
                    raise ValueError("Cell evidence does not match the frozen allocation")
                cell = value
                source = str(path)
                break
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"{name}:{type(error).__name__}: {error}")
        dispatch = sorted(folder.glob(dispatch_pattern))
        for path in dispatch:
            evidence[str(path)] = file_hash(path)
        for pattern in evidence_patterns:
            for path in sorted(folder.glob(pattern)):
                evidence[str(path)] = file_hash(path)
        unclosed = {}
        for path in (output / "waves").glob("*-unclosed.json"):
            try:
                guard = read(path)
                if row["id"] in guard.get("alive", {}):
                    evidence[str(path)] = file_hash(path)
                    unclosed[row["id"]] = guard["alive"][row["id"]]
            except (OSError, ValueError, TypeError) as error:
                errors.append(f"{path.name}:{type(error).__name__}: {error}")
        if cell is None and not evidence and not errors:
            continue
        if cell is None:
            cell = dict(
                assignment=row,
                status="infrastructure_failure",
                attempts=[],
                health=dict(infrastructure_problem=True, model_unavailable=False),
                opportunity=dict(candidates=[], reason="missing_final_accounting"),
                dispatch_evidence=dict(
                    sentinel_present=bool(dispatch), model_may_have_run=bool(dispatch)
                ),
            )
        cell["final_accounting"] = dict(
            recovered_after_parent_failure=failure,
            evidence_sha256=evidence,
            evidence_errors=errors,
            result_source=source,
            worker_closure="unproven_by_accounting_recovery",
            unclosed_worker_pids=unclosed,
            interpretation="Retained evidence; parent finalization incomplete; no replay or recoding",
        )
        known[row["id"]] = cell
        recovered.append(row["id"])
    if recovered:
        save_new(output / "final-accounting.json", dict(recovered_ids=recovered, failure=failure))
    return [known[row["id"]] for row in assignments if row["id"] in known]


def run(
    execution_path,
    *,
    wave_runner=_wave,
    checkpoint_reviewer=await_reviews,
    stage_reviewer=await_stage,
):
    execution_path = Path(execution_path).resolve()
    execution = read(execution_path)
    frozen = validate(execution)
    selected_plan, selected_runner = _bindings(frozen)
    digest, output = file_hash(execution_path), Path(execution["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "worker-logs").mkdir()
    save_new(
        Path(execution["protocol_path"]).parent / "run-claim.json",
        dict(
            output=str(output),
            protocol_sha256=execution["protocol_sha256"],
            execution_sha256=digest,
        ),
    )
    save_new(output / "parent-freeze.json", dict(path=str(execution_path), sha256=digest))
    deadline = (
        time.monotonic() + datetime.fromisoformat(frozen["deadline_utc"]).timestamp() - time.time()
    )
    cells, unavailable, failure, gates = [], {}, None, []
    try:
        for index, rows in enumerate(waves(frozen["assignments"], execution["concurrency"])):
            if time.monotonic() >= deadline:
                failure = "absolute_deadline"
                break
            validate(execution)
            if file_hash(execution_path) != digest:
                raise ValueError("Execution freeze changed")
            active = [row for row in rows if row["model"] not in unavailable]
            collected, fault = (
                wave_runner(active, frozen, output, deadline, execution, index)
                if active
                else ([], None)
            )
            by_id = {c["assignment"]["id"]: c for c in collected}
            for row in rows:
                cell = by_id.get(row["id"])
                if cell is None:
                    cell = dict(
                        assignment=row,
                        status="auth_skipped",
                        attempts=[],
                        skipped=unavailable[row["model"]],
                        health=dict(infrastructure_problem=False, model_unavailable=True),
                        opportunity=dict(candidates=[], reason="auth_skipped"),
                        concurrency=dict(
                            configured_concurrency=execution["concurrency"], dispatched=False
                        ),
                    )
                _reporting(cell, frozen, output, deadline)
                cells.append(cell)
                if (
                    cell["health"]["model_unavailable"]
                    and not cell["health"]["infrastructure_problem"]
                ):
                    unavailable.setdefault(row["model"], row["id"])
            validate(execution)
            if fault:
                failure = fault
                break
            if len(cells) == 4 or (selected_plan is plan and len(cells) == 72):
                selected = cells[:4] if len(cells) == 4 else cells[64:72]
                checkpoint_reviewer(selected, frozen, output, deadline)
                gate = (
                    selected_runner._checkpoint(cells[:4], output)
                    if selected_plan is naturalistic_pilots
                    else selected_runner.checkpoint(cells, output)
                )
                gates.append(gate)
                save_new(output / f"checkpoint-after-{len(cells)}.json", gate)
                if not gate["continue_automatically"]:
                    failure = "checkpoint_exposure"
                    break
            if (
                selected_plan is plan
                and len(cells) == 16
                and not stage_reviewer(cells, frozen, output, deadline)
            ):
                failure = "pilot_stage_closed"
                break
    except BaseException as error:
        failure = failure or f"{type(error).__name__}: {error}"
        raise
    finally:
        closure_error = None
        try:
            validate(execution)
            if file_hash(execution_path) != digest:
                raise ValueError("Execution freeze changed")
        except Exception as error:
            closure_error = error
            failure = failure or f"closure_validation:{type(error).__name__}: {error}"
        cells = reconcile_final_accounting(frozen["assignments"], cells, output, failure)
        statuses = {c["assignment"]["id"]: c["status"] for c in cells}
        save_new(
            output / "batch.json",
            dict(
                protocol=frozen["protocol"],
                pilot=frozen.get("pilot"),
                execution=execution,
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
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("protocol", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--results", type=Path, required=True)
    prepare.add_argument("--concurrency", type=int, required=True)
    prepare.add_argument("--admission", type=Path, required=True)
    for action in ("validate", "run"):
        command = commands.add_parser(action)
        command.add_argument("execution", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(
            args.protocol,
            args.output,
            results=args.results,
            concurrency=args.concurrency,
            admission=args.admission,
        )
    elif args.action == "validate":
        validate(read(args.execution))
    else:
        run(args.execution)


if __name__ == "__main__":
    main()
