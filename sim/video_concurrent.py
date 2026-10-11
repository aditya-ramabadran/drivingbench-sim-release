"""Original trolley48 cells in isolated waves; video semantics retain their own owner."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from drivingbench.sim import scenario_run, video_run
from drivingbench.sim.batch import save_new
from drivingbench.sim.cameras import render_slot
from drivingbench.sim.producer import file_hash
from drivingbench.sim.studies import concurrent

VERSION = "sim-trolley-concurrent-v1"
NATIVE_INTERVAL_BASIS = "First normal driving-tool takeover to actual scenario terminal; excludes paused model inference"


def read(path):
    return json.loads(Path(path).read_text())


def identity(manifest_path, manifest):
    return dict(
        manifest_sha256=file_hash(manifest_path),
        protocol="trolley48",
        source_hashes=manifest["source_hashes"],
        asset_hashes=manifest["asset_hashes"],
        active_budgets_s=manifest["active_budgets_s"],
        templates=sorted({(r["condition"], r["mirror"]) for r in manifest["assignments"]}),
    )


def qualification(path, manifest_path, manifest, concurrency):
    """Require fresh exact-freeze video evidence, rather than reusing paired-image admission."""
    review = read(path)
    used_sources = {
        str(Path(module.__file__).resolve()) for module in (video_run, scenario_run, concurrent)
    } | {str(Path(__file__).resolve())}
    if not used_sources <= manifest["source_hashes"].keys():
        raise ValueError("Trolley freeze must include its actual cell and concurrent owner sources")
    expected = identity(manifest_path, manifest)
    # JSON represents the canonical sorted template tuples as arrays.
    expected["templates"] = [list(t) for t in expected["templates"]]
    if (
        review.get("approved") is not True
        or not review.get("reviewer")
        or review.get("concurrency") != concurrency
        or review.get("relative_comparison_reviewed") is not True
        or review.get("trolley") != expected
    ):
        raise ValueError("Exact trolley freeze and root concurrency admission required")
    evidence = review.get("evidence_hashes", {})
    report_path = str(Path(review["preflight_report"]).resolve())
    if report_path not in evidence:
        raise ValueError("Admission must bind the actual trolley real-launch report")
    for source, digest in evidence.items():
        if str(Path(source).resolve()) != source or file_hash(source) != digest:
            raise ValueError("Trolley qualification evidence changed")
    report = read(report_path)
    if (
        report.get("trolley") != expected
        or report.get("concurrency") != concurrency
        or report.get("camera_mode") != "continuous_narrow_only_cpu_video"
        or type(report.get("model_calls")) is not int
        or report.get("model_calls") != 0
        or type(report.get("authentication_calls")) is not int
        or report.get("authentication_calls") != 0
        or type(report.get("gpu_calls")) is not int
        or report.get("gpu_calls") != 0
        or report.get("pass_absolute") is not True
        or not isinstance(report.get("checks"), dict)
        or not isinstance(report.get("cases"), list)
        or not {
            "public_contract",
            "received_source_correspondence",
            "native_tick_cadence",
            "resource_headroom",
            "cached_frame_age",
            "clean_teardown",
        }
        <= report.get("checks", {}).keys()
        or not all(v is True for v in report["checks"].values())
        or sorted((c["condition"], c["mirror"]) for c in report.get("cases", []))
        != sorted(tuple(t) for t in expected["templates"])
        or any(c.get("status") != "passed" for c in report.get("cases", []))
    ):
        raise ValueError("All twelve actual video templates and concurrent gates must pass")
    for key in ("end_to_end_latency_s", "actual_received_image_age_s"):
        metric = report[key]
        if not 0 <= metric["median"] < 1 or not 0 <= metric["p95"] < 1.5:
            raise ValueError("Trolley latency exceeds declared admission limits")
    return review


def freeze(manifest_path, output, *, results, concurrency, admission, deadline_utc):
    manifest_path, output, results, admission = (
        Path(p).resolve() for p in (manifest_path, output, results, admission)
    )
    manifest = video_run.validate_manifest(manifest_path)
    if manifest["protocol"] != "trolley48" or len(manifest["assignments"]) != 48:
        raise ValueError("Concurrent trolley requires the exact original48")
    concurrent.waves(manifest["assignments"], concurrency)
    qualification(admission, manifest_path, manifest, concurrency)
    deadline = datetime.fromisoformat(deadline_utc)
    if deadline.utcoffset() is None:
        raise ValueError("Absolute trolley deadline must include its timezone")
    execution = dict(
        version=VERSION,
        protocol_path=str(manifest_path),
        protocol_sha256=file_hash(manifest_path),
        output=str(results),
        concurrency=concurrency,
        admission=str(admission),
        admission_sha256=file_hash(admission),
        deadline_utc=deadline_utc,
        camera_mode="continuous_narrow_only_cpu_video",
        warmup_lock=str(results / "video-warmup.lock"),
        peer_fault_policy="finish_active_cells_stop_new_waves",
        coding="original_video_belief_separate_serial_after_all_waves_closed",
    )
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "execution.json", execution)
    return execution


def validate(execution):
    if any(
        execution.get(key) != value
        for key, value in dict(
            version=VERSION,
            camera_mode="continuous_narrow_only_cpu_video",
            warmup_lock=str(Path(execution["output"]) / "video-warmup.lock"),
            peer_fault_policy="finish_active_cells_stop_new_waves",
            coding="original_video_belief_separate_serial_after_all_waves_closed",
        ).items()
    ):
        raise ValueError("Trolley execution policy changed")
    if (
        file_hash(execution["protocol_path"]) != execution["protocol_sha256"]
        or file_hash(execution["admission"]) != execution["admission_sha256"]
    ):
        raise ValueError("Trolley base freeze or admission changed")
    manifest = video_run.validate_manifest(execution["protocol_path"])
    if manifest["protocol"] != "trolley48" or len(manifest["assignments"]) != 48:
        raise ValueError("Concurrent trolley requires the exact original48")
    concurrent.waves(manifest["assignments"], execution["concurrency"])
    qualification(
        execution["admission"], execution["protocol_path"], manifest, execution["concurrency"]
    )
    if datetime.fromisoformat(execution["deadline_utc"]).utcoffset() is None:
        raise ValueError("Absolute trolley deadline must include its timezone")
    return manifest


def wave_budget(rows):
    """Bound auth/decoder barrier, pre-tool inference, native cap and same-session questions."""
    return concurrent.STARTUP_BUDGET_S + max(
        r["startup_budget_s"]
        + r["active_budget_s"]
        + 2 * (1 + r["followup_retries"]) * r["followup_budget_s"]
        for r in rows
    )


def collect(manifest, row, output, deadline, execution, metadata, ready, release, abort):
    """Own one normal video cell, retaining run.json and separate concurrency evidence."""
    folder = output / row["id"]
    assessments, invocations = [], []
    original_turn = scenario_run.model_turn

    class Assessment(video_run.VideoAssessment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assessments.append(self)

    def model_turn(*args, **kwargs):
        interval = dict(start_monotonic_s=time.monotonic(), output=str(args[2]))
        invocations.append(interval)
        metadata.setdefault("provider_session_start_monotonic_s", interval["start_monotonic_s"])
        save_new(folder / "invocations" / f"{len(invocations):02d}-started.json", interval)
        try:
            return original_turn(*args, **kwargs)
        finally:
            interval["end_monotonic_s"] = time.monotonic()
            save_new(folder / "invocations" / f"{len(invocations):02d}-closed.json", interval)

    def renderer_factory(*args):
        renderer = video_run.VideoSnapshotRenderer(*args)
        start, warmed = renderer.start, False

        def warmup():
            nonlocal warmed
            if warmed:
                return start()
            with render_slot(execution["warmup_lock"], abort):
                result = start()
                warmed = True
                return result

        renderer.start = warmup
        return renderer

    def before_dispatch(app, directory):
        native = app.state.sim_native
        snapshot = native.pose_snapshot()
        if not native.paused:
            raise ValueError("Video must remain paused until its first driving tool")
        save_new(folder / "wave-ready.json", dict(metadata, native_snapshot=snapshot))
        ready.set()
        while not release.wait(0.05):
            if abort.is_set() or time.monotonic() >= deadline:
                raise RuntimeError("Video wave aborted before a model invocation")
            if not native.paused or native.pose_snapshot() != snapshot:
                raise RuntimeError("Video native advanced while awaiting wave release")
        if abort.is_set():
            raise RuntimeError("Video wave aborted before a model invocation")
        metadata["wave_client_release_monotonic_s"] = time.monotonic()
        app.state.sim_episode.record("concurrent_wave_release", **metadata)

    try:
        with patch.object(scenario_run, "model_turn", model_turn):
            report = video_run.run_cell(
                row,
                folder,
                recovery_output=output / "auth-recovery" / row["model"],
                before_dispatch=before_dispatch,
                renderer_factory=renderer_factory,
                assessment_factory=Assessment,
            )
    except Exception as error:
        path = folder / "run.json"
        report = (
            read(path)
            if path.exists()
            else dict(
                assignment=row["id"],
                status="infrastructure_failure",
                error=f"{type(error).__name__}: {error}",
            )
        )
    faults = video_run.driving_faults(report)
    if report.get("terminal", {}).get(
        "outcome"
    ) == "assigned_branch_outcome" and not video_run.received_drive(report):
        faults.append("moving_drive_without_received_image_proof")
    metadata["worker_end_monotonic_s"] = time.monotonic()
    if invocations:
        metadata["provider_session_end_monotonic_s"] = invocations[-1]["end_monotonic_s"]
    native_intervals = []
    for assessment in assessments:
        if assessment.started_wall is not None:
            terminal = assessment.terminal or {}
            if not isinstance(terminal.get("wall_monotonic"), (int, float)):
                faults.append("native_drive_interval_without_terminal")
            else:
                native_intervals.append(
                    dict(
                        start_monotonic_s=assessment.started_wall,
                        end_monotonic_s=terminal["wall_monotonic"],
                        dispatch_sentinel_present=bool(invocations),
                    )
                )
    if len(native_intervals) > 1:
        faults.append("duplicate_native_drive_interval")
    unavailable = (
        report.get("status") == "auth_skipped" or report.get("provider_failure") == "failed_auth"
    )
    cell = dict(
        assignment=row,
        status=report.get("status", "infrastructure_failure"),
        report=report,
        health=dict(
            infrastructure_problem=bool(faults), model_unavailable=unavailable, problems=faults
        ),
        concurrency=metadata,
        model_invocation_intervals=invocations,
        driving_intervals=native_intervals,
        dispatch_evidence=dict(model_may_have_run=bool(invocations)),
        report_sha256=file_hash(folder / "run.json") if (folder / "run.json").exists() else None,
    )
    save_new(folder / "worker-result.json", cell)


def annotate_model_sessions(cells):
    """Reuse interval counting separately for whole released provider sessions."""
    proxies = []
    for cell in cells:
        metadata = cell.get("concurrency", {})
        start, end = (
            metadata.get("provider_session_start_monotonic_s"),
            metadata.get("provider_session_end_monotonic_s"),
        )
        interval = (
            []
            if start is None
            else [
                dict(start_monotonic_s=start, end_monotonic_s=end, dispatch_sentinel_present=True)
            ]
        )
        proxies.append(dict(assignment=cell["assignment"], driving_intervals=interval))
    concurrent.annotate_concurrency(
        proxies,
        interval_basis="Released normal provider session, including paused inference and original followups; not native driving",
    )
    for cell, proxy in zip(cells, proxies, strict=True):
        measured = proxy["concurrency"]
        cell["model_session_concurrency"] = dict(
            actual_session_interval=measured["actual_driving_interval"],
            overlapping_session_ids=measured["overlapping_drive_ids"],
            cell_peak_active_sessions=measured["cell_peak_active_drives"],
            wave_peak_active_sessions=measured["wave_peak_active_drives"],
            interval_basis=measured["interval_basis"],
        )


def publish(cell, output):
    """Publish parent annotations separately; never mutate a completed cell's run.json."""
    save_new(
        output / cell["assignment"]["id"] / "concurrency.json",
        dict(
            concurrency=cell.get("concurrency", {}),
            model_invocation_intervals=cell.get("model_invocation_intervals", []),
            native_drive_intervals=cell.get("driving_intervals", []),
            native_interval_basis=NATIVE_INTERVAL_BASIS,
            model_session_concurrency=cell.get("model_session_concurrency", {}),
            report_sha256=cell.get("report_sha256"),
        ),
    )


def run(execution_path, *, wave_runner=concurrent._wave):
    execution_path = Path(execution_path).resolve()
    execution = read(execution_path)
    manifest = validate(execution)
    digest, output = file_hash(execution_path), Path(execution["output"])
    video_run.claim(execution["protocol_path"], output, execution_sha256=digest)
    output.mkdir(parents=True, exist_ok=False)
    (output / "worker-logs").mkdir()
    save_new(output / "manifest.json", manifest)
    save_new(output / "concurrent-execution.json", dict(path=str(execution_path), sha256=digest))
    deadline = (
        time.monotonic()
        + datetime.fromisoformat(execution["deadline_utc"]).timestamp()
        - time.time()
    )
    cells, unavailable, failure, gates = [], {}, None, []
    all_closed = True
    try:
        for index, rows in enumerate(
            concurrent.waves(manifest["assignments"], execution["concurrency"])
        ):
            validate(execution)
            if file_hash(execution_path) != digest:
                raise ValueError("Execution freeze changed")
            active = [r for r in rows if r["model"] not in unavailable]
            if active and deadline - time.monotonic() < wave_budget(active):
                failure = "insufficient_absolute_deadline_for_wave"
                break
            all_closed = False
            collected, fault = (
                wave_runner(
                    active,
                    manifest,
                    output,
                    deadline,
                    execution,
                    index,
                    collector=collect,
                    wave_budget_s=wave_budget(active),
                    interval_basis=NATIVE_INTERVAL_BASIS,
                )
                if active
                else ([], None)
            )
            all_closed = True
            validate(execution)
            if file_hash(execution_path) != digest:
                raise ValueError("Execution freeze changed after wave")
            annotate_model_sessions(collected)
            by_id = {c["assignment"]["id"]: c for c in collected}
            for row in rows:
                cell = by_id.get(row["id"])
                if cell is None:
                    report = dict(
                        assignment=row["id"],
                        status="auth_skipped",
                        reason="same_model_auth_unavailable",
                        source_assignment=unavailable[row["model"]],
                    )
                    save_new(output / row["id"] / "run.json", report)
                    cell = dict(
                        assignment=row,
                        status="auth_skipped",
                        report=report,
                        health=dict(infrastructure_problem=False, model_unavailable=True),
                        concurrency=dict(
                            configured_concurrency=execution["concurrency"], dispatched=False
                        ),
                        report_sha256=file_hash(output / row["id"] / "run.json"),
                    )
                publish(cell, output)
                cells.append(cell)
                if (
                    cell["health"]["model_unavailable"]
                    and not cell["health"]["infrastructure_problem"]
                ):
                    unavailable.setdefault(row["model"], row["id"])
            if fault:
                failure = fault
                break
            if len(cells) == 4:
                gate = video_run.checkpoint(
                    manifest["assignments"][:4], [c["report"] for c in cells]
                )
                gates.append(gate)
                save_new(output / "checkpoint-first-four.json", gate)
                if not gate["continue_automatically"]:
                    failure = "first_four_incomplete"
                    break
    except BaseException as error:
        failure = failure or f"{type(error).__name__}: {error}"
        raise
    finally:
        # Retain recorded invocations and unclosed guards without replaying or coding.
        cells = concurrent.reconcile_final_accounting(
            manifest["assignments"],
            cells,
            output,
            failure,
            dispatch_pattern="invocations/*-started.json",
            evidence_patterns=("run.json", "wave-ready.json", "invocations/*-closed.json"),
        )
        known = {c["assignment"]["id"]: c for c in cells}
        save_new(
            output / "summary.json",
            dict(
                assigned=48,
                results=[c.get("report", {}) for c in cells],
                stopped_reason=failure,
                cells=cells,
                checkpoints=gates,
                all_waves_closed=all_closed,
                retrospective_coder_may_run=all_closed,
                assignments=[
                    dict(r, status=known[r["id"]]["status"] if r["id"] in known else "not_run")
                    for r in manifest["assignments"]
                ],
            ),
        )
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--results", type=Path, required=True)
    prepare.add_argument("--concurrency", type=int, required=True)
    prepare.add_argument("--admission", type=Path, required=True)
    prepare.add_argument("--deadline-utc", required=True)
    for action in ("validate", "run"):
        command = commands.add_parser(action)
        command.add_argument("execution", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(
            args.manifest,
            args.output,
            results=args.results,
            concurrency=args.concurrency,
            admission=args.admission,
            deadline_utc=args.deadline_utc,
        )
    elif args.action == "validate":
        validate(read(args.execution))
    else:
        run(args.execution)


if __name__ == "__main__":
    main()
