import copy
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from drivingbench.sim import scenario_run, video_run
from drivingbench.sim import video_concurrent as run
from drivingbench.sim.batch import save_new
from drivingbench.sim.studies import concurrent
from test_sim_video_run import healthy_report, trolley_inputs


@pytest.fixture(autouse=True)
def prohibit_real_jobs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU fake test cannot launch a process, model/auth or native")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(scenario_run, "auth_preflight", forbidden)
    monkeypatch.setattr(video_run.VideoForkRuntime, "__init__", forbidden)
    monkeypatch.setattr(video_run.VideoSnapshotRenderer, "start", forbidden)


@pytest.fixture
def frozen(tmp_path):
    manifest = video_run.freeze(trolley_inputs(tmp_path), tmp_path / "base", protocol="trolley48")
    path = tmp_path / "base/manifest.json"
    # Retain the ordinary recursive source freeze; launch evidence below is synthetic.
    expected = run.identity(path, manifest)
    expected["templates"] = [list(t) for t in expected["templates"]]
    report = dict(
        trolley=expected,
        concurrency=4,
        camera_mode="continuous_narrow_only_cpu_video",
        model_calls=0,
        authentication_calls=0,
        gpu_calls=0,
        pass_absolute=True,
        checks={
            key: True
            for key in (
                "public_contract",
                "received_source_correspondence",
                "native_tick_cadence",
                "resource_headroom",
                "cached_frame_age",
                "clean_teardown",
            )
        },
        cases=[dict(condition=p, mirror=m, status="passed") for p, m in expected["templates"]],
        end_to_end_latency_s=dict(median=0.1, p95=0.2),
        actual_received_image_age_s=dict(median=0.1, p95=0.2),
    )
    evidence = tmp_path / "preflight.json"
    save_new(evidence, report)
    admission = tmp_path / "admission.json"
    review = dict(
        approved=True,
        reviewer="root",
        concurrency=4,
        relative_comparison_reviewed=True,
        trolley=expected,
        preflight_report=str(evidence),
        evidence_hashes={str(evidence): run.file_hash(evidence)},
    )
    save_new(admission, review)
    execution = run.freeze(
        path,
        tmp_path / "execution",
        results=tmp_path / "results",
        concurrency=4,
        admission=admission,
        deadline_utc="2099-01-01T00:00:00+00:00",
    )
    return SimpleNamespace(
        manifest=manifest,
        path=path,
        report=report,
        evidence=evidence,
        admission=admission,
        review=review,
        execution=execution,
        execution_path=tmp_path / "execution/execution.json",
        output=tmp_path / "results",
    )


def fake_wave(rows, manifest, output, deadline, execution, index, **kwargs):
    assert kwargs["collector"] is run.collect
    assert kwargs["wave_budget_s"] == run.wave_budget(rows)
    assert kwargs["interval_basis"] == run.NATIVE_INTERVAL_BASIS
    cells = []
    for row in rows:
        report = healthy_report(row)
        save_new(output / row["id"] / "run.json", report)
        cell = dict(
            assignment=row,
            status="complete",
            report=report,
            health=dict(infrastructure_problem=False, model_unavailable=False),
            concurrency={},
            driving_intervals=[],
            report_sha256=run.file_hash(output / row["id"] / "run.json"),
        )
        save_new(output / row["id"] / "worker-result.json", cell)
        cells.append(cell)
    return cells, None


def test_original48_dry_validation_and_exclusive_serial_concurrent_claim(
    frozen, monkeypatch, tmp_path
):
    assert video_run.validate_manifest(frozen.path) == frozen.manifest
    assert len(run.validate(frozen.execution)["assignments"]) == 48
    cells = run.run(frozen.execution_path, wave_runner=fake_wave)
    assert len(cells) == 48
    assert run.read(frozen.output / "summary.json")["all_waves_closed"] is True
    monkeypatch.setattr(video_run, "trial", lambda *a, **k: pytest.fail("second dispatch"))
    with pytest.raises(FileExistsError):
        video_run.run(frozen.path, tmp_path / "serial-second")
    assert not (tmp_path / "serial-second").exists()


def test_serial_claim_blocks_concurrent(frozen, monkeypatch, tmp_path):
    monkeypatch.setattr(video_run, "trial", lambda row, *a, **k: healthy_report(row))
    video_run.run(frozen.path, tmp_path / "serial-first")
    with pytest.raises(FileExistsError):
        run.run(frozen.execution_path, wave_runner=lambda *a, **k: pytest.fail("second dispatch"))
    assert not frozen.output.exists()


@pytest.mark.parametrize(
    "field",
    [
        "trolley",
        "concurrency",
        "camera_mode",
        "model_calls",
        "checks",
        "cases",
        "end_to_end_latency_s",
    ],
)
def test_actual_admission_tampering_rejected(frozen, field):
    report = copy.deepcopy(frozen.report)
    report[field] = (
        {"median": 1.1, "p95": 1.6}
        if field == "end_to_end_latency_s"
        else ([] if field == "cases" else False)
    )
    frozen.evidence.write_text(json.dumps(report))
    review = run.read(frozen.admission)
    review["evidence_hashes"][str(frozen.evidence)] = run.file_hash(frozen.evidence)
    frozen.admission.write_text(json.dumps(review))
    with pytest.raises((ValueError, TypeError)):
        run.qualification(frozen.admission, frozen.path, frozen.manifest, 4)


def test_exact_scene_and_cap_identity_cannot_use_another_admission(frozen):
    manifest = copy.deepcopy(frozen.manifest)
    manifest["active_budgets_s"][next(iter(manifest["active_budgets_s"]))] += 1
    with pytest.raises(ValueError, match="Exact trolley"):
        run.qualification(frozen.admission, frozen.path, manifest, 4)


def test_different_model_session_and_native_overlap():
    rows = video_run.assignments(protocol="trolley48")[:2]
    cells = [
        dict(
            assignment=rows[0],
            concurrency=dict(
                provider_session_start_monotonic_s=1, provider_session_end_monotonic_s=8
            ),
            driving_intervals=[
                dict(start_monotonic_s=4, end_monotonic_s=5, dispatch_sentinel_present=True)
            ],
        ),
        dict(
            assignment=rows[1],
            concurrency=dict(
                provider_session_start_monotonic_s=2, provider_session_end_monotonic_s=9
            ),
            driving_intervals=[
                dict(start_monotonic_s=6, end_monotonic_s=7, dispatch_sentinel_present=True)
            ],
        ),
    ]
    assert concurrent.annotate_concurrency(cells, interval_basis=run.NATIVE_INTERVAL_BASIS) == 1
    run.annotate_model_sessions(cells)
    assert all(c["model_session_concurrency"]["wave_peak_active_sessions"] == 2 for c in cells)
    assert all(c["concurrency"]["wave_peak_active_drives"] == 1 for c in cells)


def test_no_takeover_never_claims_native_driving():
    cell = dict(
        assignment=video_run.assignments(protocol="trolley48")[0],
        concurrency={},
        driving_intervals=[],
    )
    assert concurrent.annotate_concurrency([cell], interval_basis=run.NATIVE_INTERVAL_BASIS) == 0
    assert cell["concurrency"]["actual_driving_interval"] is None


def test_first_four_receipt_gate_stops_without_behavior_gate(frozen):
    calls = []

    def wave(*args, **kwargs):
        result, fault = fake_wave(*args, **kwargs)
        calls.append(args[5])
        result[0]["report"]["client_transport"] = []
        return result, fault

    assert len(run.run(frozen.execution_path, wave_runner=wave)) == 4
    assert calls == [0]
    assert run.read(frozen.output / "summary.json")["stopped_reason"] == "first_four_incomplete"


@pytest.mark.parametrize("provider", ["failed_auth", "failed_capacity"])
def test_provider_cells_retained_with_auth_only_future_skip(frozen, provider):
    failed_id, failed_model = (
        frozen.manifest["assignments"][0]["id"],
        frozen.manifest["assignments"][0]["model"],
    )
    active_ids = []

    def wave(*args, **kwargs):
        active_ids.extend(r["id"] for r in args[0])
        result, fault = fake_wave(*args, **kwargs)
        if args[5] == 0:
            cell = result[0]
            cell["report"] = dict(
                assignment=failed_id,
                status="infrastructure_failure",
                provider_failure=provider,
                errors=["client_exit_1"],
                terminal={},
            )
            cell["status"] = "infrastructure_failure"
            cell["health"]["model_unavailable"] = provider == "failed_auth"
        return result, fault

    cells = run.run(frozen.execution_path, wave_runner=wave)
    expected = 11 if provider == "failed_auth" else 0
    assert sum(c["status"] == "auth_skipped" for c in cells) == expected
    assert len(cells) == 48
    if provider == "failed_auth":
        assert all(
            r["id"] not in active_ids
            for r in frozen.manifest["assignments"][4:]
            if r["model"] == failed_model
        )


def test_mixed_provider_driving_fault_stops_after_current_wave(frozen):
    def wave(*args, **kwargs):
        result, _ = fake_wave(*args, **kwargs)
        result[0]["health"].update(infrastructure_problem=True, model_unavailable=True)
        return result, "driving_infrastructure"

    assert len(run.run(frozen.execution_path, wave_runner=wave)) == 4
    assert run.read(frozen.output / "summary.json")["stopped_reason"] == "driving_infrastructure"


def test_provisional_workers_recovered_without_rewriting_run_or_coding(frozen, monkeypatch):
    original = run.publish

    def broken(cell, output):
        raise OSError("parent publication failure")

    monkeypatch.setattr(run, "publish", broken)
    with pytest.raises(OSError):
        run.run(frozen.execution_path, wave_runner=fake_wave)
    summary = run.read(frozen.output / "summary.json")
    assert len(summary["cells"]) == 4
    assert summary["all_waves_closed"] is True
    for cell in summary["cells"]:
        assert cell["report_sha256"] == run.file_hash(
            frozen.output / cell["assignment"]["id"] / "run.json"
        )
    monkeypatch.setattr(run, "publish", original)


def test_unclosed_wave_never_admits_coding(frozen):
    def unclosed(*args, **kwargs):
        raise RuntimeError("workers did not close")

    with pytest.raises(RuntimeError):
        run.run(frozen.execution_path, wave_runner=unclosed)
    assert run.read(frozen.output / "summary.json")["retrospective_coder_may_run"] is False


def test_wave_reuses_injected_collector_without_binding_a_or_blue(tmp_path, monkeypatch):
    monkeypatch.setattr(concurrent.study, "require_disk_space", lambda: None)
    seen = []
    row = video_run.assignments(protocol="trolley48")[0]
    (tmp_path / "worker-logs").mkdir()

    def collector(manifest, row, output, deadline, execution, metadata, ready, release, abort):
        seen.append(row["id"])
        save_new(
            output / row["id"] / "worker-result.json",
            dict(
                assignment=row,
                status="auth_skipped",
                health=dict(infrastructure_problem=False, model_unavailable=True),
                driving_intervals=[],
            ),
        )

    class Process:
        exitcode = 0

        def __init__(self, target, args):
            self.target, self.args, self.pid = target, args, 1234

        def start(self):
            self.target(*self.args)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    context = SimpleNamespace(Event=threading.Event, Process=Process)
    cells, fault = concurrent._wave(
        [row],
        {},
        tmp_path,
        time.monotonic() + 10,
        {
            "concurrency": 4,
            "protocol_sha256": "fixture",
            "peer_fault_policy": "finish_active_cells_stop_new_waves",
        },
        0,
        context=context,
        collector=collector,
        wave_budget_s=5,
        interval_basis=run.NATIVE_INTERVAL_BASIS,
    )
    assert seen == [row["id"]] and fault is None and len(cells) == 1


def test_video_run_cell_uses_original_trial_owner_and_hook(tmp_path, monkeypatch):
    scene = tmp_path / "scene.json"
    scene.write_text(
        json.dumps(
            dict(
                video_physics=dict(
                    commit_m=8, endpoints_m=dict(left=22, right=25), branch_rule="command"
                )
            )
        )
    )
    row = dict(scene=str(scene))
    hook = object()

    def trial(*args, **kwargs):
        assert kwargs["before_dispatch"] is hook
        assert kwargs["app_factory"] is video_run.video_app
        assert kwargs["adapter_factory"] is video_run.VideoForkAdapter
        assert kwargs["renderer_factory"] is video_run.VideoSnapshotRenderer
        return "normal trial"

    monkeypatch.setattr(video_run, "trial", trial)
    assert video_run.run_cell(row, tmp_path / "cell", before_dispatch=hook) == "normal trial"


@pytest.mark.parametrize("takeover", [False, True])
def test_collector_barrier_retains_pause_and_records_only_real_takeover(
    frozen, monkeypatch, takeover
):
    row = frozen.manifest["assignments"][0]
    ready, abort = threading.Event(), threading.Event()
    checks = []
    native = SimpleNamespace(
        paused=True, pose_snapshot=lambda: {"pose": [0, 0, 0], "sim_time_s": 0}
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            sim_native=native, sim_episode=SimpleNamespace(record=lambda *a, **k: None)
        )
    )

    class Release:
        polls = 0

        def wait(self, timeout):
            assert ready.is_set() and native.paused
            self.polls += 1
            return self.polls == 2

    release = Release()

    def assessment_init(self, *args):
        self.started_wall = self.terminal = None

    monkeypatch.setattr(video_run.VideoAssessment, "__init__", assessment_init)

    def original_turn(*args, **kwargs):
        if takeover:
            kwargs["assessment"].started_wall = time.monotonic()
            native.paused = False
            kwargs["assessment"].terminal = dict(wall_monotonic=time.monotonic() + 0.001)
        return None, [], []

    monkeypatch.setattr(scenario_run, "model_turn", original_turn)

    def trial(row, folder, **kwargs):
        Path(folder).mkdir(parents=True)
        assessment = kwargs["assessment_factory"](app, {})
        kwargs["before_dispatch"](app, folder)
        checks.append(native.paused)
        scenario_run.model_turn(
            None, "drive", folder / "drive", time.monotonic() + 1, assessment=assessment
        )
        report = (
            healthy_report(row)
            if takeover
            else dict(
                assignment=row["id"],
                status="complete",
                terminal=dict(outcome="client_done"),
                errors=[],
            )
        )
        save_new(folder / "run.json", report)
        return report

    monkeypatch.setattr(video_run, "run_cell", trial)
    frozen.output.mkdir()
    run.collect(
        frozen.manifest,
        row,
        frozen.output,
        time.monotonic() + 10,
        frozen.execution,
        {},
        ready,
        release,
        abort,
    )
    cell = run.read(frozen.output / row["id"] / "worker-result.json")
    assert checks == [True] and release.polls == 2
    assert len(cell["model_invocation_intervals"]) == 1
    assert len(cell["driving_intervals"]) == int(takeover)
    assert cell["health"]["infrastructure_problem"] is False
    assert cell["report_sha256"] == run.file_hash(frozen.output / row["id"] / "run.json")


def test_barrier_abort_has_no_model_invocation(frozen, monkeypatch):
    row = frozen.manifest["assignments"][0]
    ready, release, abort = threading.Event(), threading.Event(), threading.Event()
    abort.set()
    native = SimpleNamespace(paused=True, pose_snapshot=lambda: {"pose": [0, 0, 0]})
    app = SimpleNamespace(state=SimpleNamespace(sim_native=native))

    def trial(row, folder, **kwargs):
        kwargs["before_dispatch"](app, folder)
        pytest.fail("model invocation after abort")

    monkeypatch.setattr(video_run, "run_cell", trial)
    frozen.output.mkdir()
    run.collect(
        frozen.manifest,
        row,
        frozen.output,
        time.monotonic() + 10,
        frozen.execution,
        {},
        ready,
        release,
        abort,
    )
    cell = run.read(frozen.output / row["id"] / "worker-result.json")
    assert cell["dispatch_evidence"]["model_may_have_run"] is False
    assert cell["health"]["infrastructure_problem"] is True
    assert cell["model_invocation_intervals"] == cell["driving_intervals"] == []


@pytest.mark.parametrize("summary", [None, {"all_waves_closed": False}])
def test_original_coder_rejects_missing_or_unclosed_concurrent_summary(frozen, summary):
    from drivingbench.sim import video_belief

    frozen.output.mkdir()
    save_new(frozen.output / "manifest.json", frozen.manifest)
    save_new(frozen.output / "concurrent-execution.json", {"sha256": "fixture"})
    if summary is not None:
        save_new(frozen.output / "summary.json", summary)
    with pytest.raises(ValueError, match="await all video waves"):
        video_belief.code_results(frozen.output, frozen.output / "codes")
    assert not (frozen.output / "codes").exists()


def test_actual_owner_sources_required(frozen):
    manifest = copy.deepcopy(frozen.manifest)
    del manifest["source_hashes"][str(Path(run.__file__).resolve())]
    with pytest.raises(ValueError, match="actual cell and concurrent owner"):
        run.qualification(frozen.admission, frozen.path, manifest, 4)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_smaller_concurrency_still_closes_exact_first_four_gate(frozen, count):
    for path in (frozen.evidence, frozen.admission, frozen.execution_path):
        value = run.read(path)
        value["concurrency"] = count
        if path == frozen.admission:
            value["evidence_hashes"][str(frozen.evidence)] = run.file_hash(frozen.evidence)
        if path == frozen.execution_path:
            value["admission_sha256"] = run.file_hash(frozen.admission)
        path.write_text(json.dumps(value))
    dispatched = []

    def wave(*args, **kwargs):
        rows, output = args[0], args[2]
        if len(dispatched) >= 4:
            assert run.read(output / "checkpoint-first-four.json")["continue_automatically"]
        dispatched.extend(row["id"] for row in rows)
        return fake_wave(*args, **kwargs)

    cells = run.run(frozen.execution_path, wave_runner=wave)
    assert len(cells) == 48
    assert dispatched == [row["id"] for row in frozen.manifest["assignments"]]


@pytest.mark.parametrize("invoked", [False, True])
def test_unclosed_worker_is_accounted_without_inventing_or_losing_dispatch(frozen, invoked):
    snapshots = {}
    missing = frozen.manifest["assignments"][3]

    def wave(rows, manifest, output, deadline, execution, index, **kwargs):
        fake_wave(rows, manifest, output, deadline, execution, index, **kwargs)
        for row in rows[:3]:
            path = output / row["id"] / "worker-result.json"
            snapshots[path] = path.read_bytes()
        folder = output / missing["id"]
        (folder / "worker-result.json").unlink()
        (folder / "run.json").unlink()
        if invoked:
            save_new(
                folder / "invocations/01-started.json",
                dict(start_monotonic_s=123.0, output=str(folder / "drive")),
            )
        save_new(
            output / "waves/000-unclosed.json",
            dict(alive={missing["id"]: 24680}, failure="absolute_deadline"),
        )
        raise RuntimeError("Unclosed workers; do not start another study")

    with pytest.raises(RuntimeError, match="Unclosed workers"):
        run.run(frozen.execution_path, wave_runner=wave)
    summary = run.read(frozen.output / "summary.json")
    assert len(summary["cells"]) == 4
    assert summary["all_waves_closed"] is False
    assert summary["retrospective_coder_may_run"] is False
    unknown = summary["cells"][3]
    assert unknown["assignment"] == missing
    assert unknown["status"] == "infrastructure_failure"
    assert unknown["dispatch_evidence"]["model_may_have_run"] is invoked
    assert unknown["final_accounting"]["unclosed_worker_pids"] == {missing["id"]: 24680}
    assert sum(row["status"] == "not_run" for row in summary["assignments"]) == 44
    assert all(path.read_bytes() == content for path, content in snapshots.items())


def test_finished_invocation_evidence_without_worker_result_remains_accounted(frozen):
    missing = frozen.manifest["assignments"][0]
    snapshots = {}

    def wave(rows, manifest, output, deadline, execution, index, **kwargs):
        fake_wave(rows, manifest, output, deadline, execution, index, **kwargs)
        folder = output / missing["id"]
        (folder / "worker-result.json").unlink()
        for name, value in (
            ("invocations/01-started.json", dict(start_monotonic_s=123.0)),
            ("invocations/01-closed.json", dict(start_monotonic_s=123.0, end_monotonic_s=124.0)),
            ("wave-ready.json", dict(native_snapshot={"sim_time_s": 0})),
        ):
            save_new(folder / name, value)
        for name in (
            "run.json",
            "invocations/01-started.json",
            "invocations/01-closed.json",
            "wave-ready.json",
        ):
            path = folder / name
            snapshots[path] = path.read_bytes()
        raise ValueError("parent annotation failed before returning wave results")

    with pytest.raises(ValueError, match="parent annotation"):
        run.run(frozen.execution_path, wave_runner=wave)
    summary = run.read(frozen.output / "summary.json")
    assert len(summary["cells"]) == 4
    unknown = summary["cells"][0]
    assert unknown["status"] == "infrastructure_failure"
    assert unknown["dispatch_evidence"]["model_may_have_run"] is True
    hashes = unknown["final_accounting"]["evidence_sha256"]
    assert set(map(str, snapshots)) <= hashes.keys()
    assert all(
        path.read_bytes() == content and hashes[str(path)] == run.file_hash(path)
        for path, content in snapshots.items()
    )
    assert summary["retrospective_coder_may_run"] is False
    assert sum(row["status"] == "not_run" for row in summary["assignments"]) == 44
