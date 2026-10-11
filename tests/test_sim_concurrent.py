import copy
import fcntl
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from drivingbench.sim.studies import concurrent as run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def allocation():
    return run.plan.assignments({key: f"/{key}.json" for key in run.plan.SCENES})


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_exact_round_robin_and_all_stage_boundaries(limit):
    source = allocation()
    waves = run.waves(source, limit)
    assert [row for wave in waves for row in wave] == source
    assert all(
        len(wave) <= limit and len({r["model"] for r in wave}) == len(wave) for wave in waves
    )
    totals = [sum(map(len, waves[:i])) for i in range(1, len(waves) + 1)]
    assert {4, 16, 72, 192} <= set(totals)


@pytest.mark.parametrize("limit", [0, 5, True, 2.5])
def test_invalid_concurrency_rejected(limit):
    with pytest.raises(ValueError):
        run.waves(allocation(), limit)


def test_same_model_or_duplicate_identity_rejected():
    rows = allocation()[:2]
    rows[1]["model"] = rows[0]["model"]
    with pytest.raises(ValueError, match="overlaps"):
        run.waves(rows, 4)
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        run.waves(rows, 4)


@pytest.fixture
def admission(tmp_path, monkeypatch):
    sources = {str(write(tmp_path / "source.json", {})): "will_replace"}
    sources = {p: run.file_hash(p) for p in sources}
    frozen = dict(
        protocol=run.plan.PROTOCOL,
        sources=sources,
        assignments=allocation(),
        deadline_utc="2099-01-01T00:00:00+00:00",
    )
    protocol = write(tmp_path / "protocol.json", frozen)
    monkeypatch.setattr(run.serial, "validate_protocol", lambda _f: None)
    report = write(
        tmp_path / "report.json",
        dict(
            concurrency=4,
            camera_mode="on_demand",
            model_calls=0,
            pass_absolute=True,
            checks={"ticks": True, "braking": True},
            paired_end_to_end_latency_s=dict(median=0.6, p95=1.2),
            actual_received_image_age_s=dict(median=0.6, p95=1.2),
        ),
    )
    review = write(
        tmp_path / "review.json",
        dict(
            approved=True,
            reviewer="root",
            concurrency=4,
            relative_comparison_reviewed=True,
            source_hashes=sources,
            preflight_report=str(report),
            evidence_hashes={str(report): run.file_hash(report)},
        ),
    )
    return protocol, review, report


def test_complement_freeze_binds_exact_source_admission_and_exclusive_results(admission, tmp_path):
    protocol, review, _report = admission
    execution = run.freeze(
        protocol, tmp_path / "freeze", results=tmp_path / "results", concurrency=4, admission=review
    )
    assert run.validate(execution)["assignments"] == allocation()
    assert execution["gpu_slots"] == 1 and execution["pilot_stage"] == run.STAGE
    with pytest.raises(FileExistsError):
        run.freeze(
            protocol,
            tmp_path / "freeze",
            results=tmp_path / "results",
            concurrency=4,
            admission=review,
        )
    changed = copy.deepcopy(execution)
    changed["camera_capture"] = "continuous"
    with pytest.raises(ValueError, match="policy"):
        run.validate(changed)
    write(protocol, {**run.read(protocol), "extra": True})
    with pytest.raises(ValueError, match="Base"):
        run.validate(execution)


@pytest.mark.parametrize("change", ["slow", "ticks", "mode", "models", "source", "relative"])
def test_admission_rejects_unqualified_or_changed_proof(admission, change):
    protocol, review_path, report_path = admission
    report, review = run.read(report_path), run.read(review_path)
    if change == "slow":
        report["paired_end_to_end_latency_s"]["p95"] = 2
    elif change == "ticks":
        report["checks"]["ticks"] = False
    elif change == "mode":
        report["camera_mode"] = "continuous"
    elif change == "models":
        report["model_calls"] = 1
    elif change == "source":
        review["source_hashes"] = {}
    else:
        review["relative_comparison_reviewed"] = False
    write(report_path, report)
    review["evidence_hashes"][str(report_path)] = run.file_hash(report_path)
    write(review_path, review)
    with pytest.raises(ValueError):
        run._qualification(review_path, run.read(protocol), 4)


@pytest.fixture
def collector(tmp_path, monkeypatch):
    row = allocation()[0]
    frozen = run.plan.protocol({key: f"/{key}.json" for key in run.plan.SCENES})
    ready, release, abort = threading.Event(), threading.Event(), threading.Event()
    release.set()
    metadata = dict(wave_index=0)
    observed = []
    slot = threading.Lock()

    @contextmanager
    def render_slot(_path, stop=None):
        if stop is not None:
            assert not stop.is_set()
        assert slot.acquire(blocking=False), "non-reentrant GPU slot would deadlock"
        try:
            yield
        finally:
            slot.release()

    class Renderer:
        def start(self):
            observed.append("start")

    @contextmanager
    def lifecycle(_directory, factory):
        yield factory

    def cell(effective, assigned, folder, recovery, deadline, **kwargs):
        assert assigned == row and effective["prompt"] == run.plan.prompt(row["arm"])
        renderer = kwargs["renderer_factory"]()
        renderer.start()
        with render_slot("same"):
            renderer.start()
        native = SimpleNamespace(paused=True, pose_snapshot=lambda: {"sim_time_s": 0})
        episode = SimpleNamespace(record=lambda *args, **kw: observed.append(args[0]))
        app = SimpleNamespace(state=SimpleNamespace(sim_native=native, sim_episode=episode))
        kwargs["before_dispatch"](app, folder / "attempt-1")
        assert native.paused and ready.is_set()
        declaration = kwargs["declaration_factory"](assigned, {}, effective)
        assert declaration["camera_capture"] == "on_demand" and declaration["paused"]
        write(folder / "attempt-1/dispatch.json", {})
        run.study.drive()
        belief = run.study.code_belief("answer", folder / "belief", 20)
        return dict(assignment=assigned, status="complete", attempts=[{}], belief=belief)

    monkeypatch.setattr("drivingbench.sim.cameras.render_slot", render_slot)
    monkeypatch.setattr(run.study, "SceneRenderer", Renderer)
    monkeypatch.setattr(run.pilot_health, "observe_lifecycle", lifecycle)
    monkeypatch.setattr(run.study, "_cell", cell)
    monkeypatch.setattr(run.study, "drive", lambda: None)
    monkeypatch.setattr(run.serial, "opportunity", lambda *_a: {"candidates": []})
    monkeypatch.setattr(
        run.pilot_health,
        "driving_health",
        lambda *_a: dict(
            infrastructure_problem=False,
            model_unavailable=False,
        ),
    )
    run._collect(
        frozen,
        row,
        tmp_path,
        1e20,
        dict(render_lock_path="same", protocol_sha256="one"),
        metadata,
        ready,
        release,
        abort,
    )
    return tmp_path / row["id"], observed, frozen


def test_paused_readiness_and_outer_plus_camera_warmup_do_not_reenter_slot(collector):
    folder, observed, _frozen = collector
    assert observed == ["start", "start", "concurrent_wave_release"]
    cell = run.read(folder / "worker-result.json")
    assert cell["dispatch_evidence"]["model_may_have_run"]
    assert cell["deferred_belief"][0]["text"] == "answer"
    assert not (folder / "belief").exists()


def test_coder_deferred_until_parent_and_retried_once(collector, monkeypatch):
    folder, _observed, frozen = collector
    cell, calls = run.read(folder / "worker-result.json"), []

    def coder(text, path, budget):
        calls.append((text, path))
        return dict(label=None if len(calls) == 1 else "doubted")

    monkeypatch.setattr(run.study, "code_belief", coder)
    run._reporting(cell, frozen, folder.parent, 1e20)
    assert len(calls) == 2
    final = run.read(folder / "session.json")
    assert final["belief"]["label"] == "doubted"
    assert final["belief"]["automatic_retries"] == 1
    assert "deferred_belief" not in final
    assert (folder / "worker-result.json").exists()


def test_pilot_counts_accepted_forward_and_preserves_failed_denominators(tmp_path):
    cells = []
    for row in allocation()[:16]:
        cell = dict(assignment=row, status="complete", health=dict(infrastructure_problem=False))
        cells.append(cell)
        calls = [
            dict(
                tool="set_motion",
                arguments={"speed_mps": 1},
                outcome={"status": "accepted" if row["model"] == "grok" else "rejected"},
            )
        ]
        write(tmp_path / row["id"] / "attempt-1/drive-calls.json", calls)
        write(tmp_path / row["id"] / "session.json", cell)
    result = run.pilot_screen(cells, tmp_path)
    assert result["forward_count_gap"] == 3 and result["scaling_screen_passed"]
    astra = next(
        c
        for c in cells
        if c["assignment"]["model"] == "astra" and c["assignment"]["condition"] != "empty"
    )
    astra["status"] = "failed_capacity"
    result = run.pilot_screen(cells, tmp_path)
    assert result["models"]["astra"]["forward_count"] is None
    assert result["complete_models"] == 3


def test_actual_concurrency_uses_drive_intervals_not_worker_or_auth_lifetime():
    cells = [
        dict(
            assignment=dict(id=name),
            driving_intervals=[
                dict(
                    start_monotonic_s=start,
                    end_monotonic_s=end,
                    dispatch_sentinel_present=True,
                )
            ],
        )
        for name, start, end in (("astra", 10, 20), ("sol", 12, 18), ("grok", 20, 30))
    ]
    cells.append(dict(assignment=dict(id="fable"), status="failed_auth"))
    assert run.annotate_concurrency(cells) == 2
    astra, sol, grok, fable = (c["concurrency"] for c in cells)
    assert astra["overlapping_drive_ids"] == ["sol"]
    assert sol["cell_peak_active_drives"] == 2
    assert grok["cell_peak_active_drives"] == 1
    assert fable["actual_driving_interval"] is None and fable["cell_peak_active_drives"] == 0


@pytest.fixture
def harness(admission, tmp_path, monkeypatch):
    protocol, review, _report = admission
    frozen = run.read(protocol)
    frozen["followup_order"] = ["account", "debrief", "perception"]
    frozen["turn_budget_s"] = 180
    write(protocol, frozen)
    run.freeze(
        protocol, tmp_path / "freeze", results=tmp_path / "results", concurrency=4, admission=review
    )
    monkeypatch.setattr(run.serial, "checkpoint", lambda *_a: {"continue_automatically": True})
    launched = []

    def wave(rows, *_a):
        launched.extend(rows)
        return [
            dict(
                assignment=row,
                status="complete",
                attempts=[{}],
                health=dict(infrastructure_problem=False, model_unavailable=False),
                opportunity=dict(candidates=[]),
            )
            for row in rows
        ], None

    return tmp_path / "freeze/execution.json", wave, launched


def execute(harness, **kwargs):
    return run.run(
        harness[0],
        wave_runner=kwargs.pop("wave_runner", harness[1]),
        checkpoint_reviewer=lambda *_a: None,
        stage_reviewer=kwargs.pop("stage_reviewer", lambda *_a: True),
        **kwargs,
    )


def test_parent_collects_order_gate_and_exclusive_claim(harness, tmp_path):
    cells = execute(harness)
    assert [c["assignment"] for c in cells] == allocation()
    assert len(harness[2]) == 192
    batch = run.read(tmp_path / "results/batch.json")
    assert not batch["failure"] and len(batch["checkpoints"]) == 2
    with pytest.raises(FileExistsError):
        execute(harness)


def test_pilot_stage_closes_prefix_without_replacements(harness, tmp_path):
    assert len(execute(harness, stage_reviewer=lambda *_a: False)) == 16
    batch = run.read(tmp_path / "results/batch.json")
    assert batch["failure"] == "pilot_stage_closed"
    assert sum(r["status"] == "not_run" for r in batch["assignments"]) == 176


def test_confirmed_auth_skips_later_model_but_keeps_allocation(harness):
    def wave(*args):
        cells, fault = harness[1](*args)
        for cell in cells:
            if cell["assignment"]["model"] == "fable":
                cell["status"] = "failed_auth"
                cell["health"]["model_unavailable"] = True
        return cells, fault

    cells = execute(harness, wave_runner=wave)
    assert len(cells) == 192 and len(harness[2]) == 145
    fable = [c for c in cells if c["assignment"]["model"] == "fable"]
    assert fable[0]["status"] == "failed_auth"
    assert all(c["status"] == "auth_skipped" for c in fable[1:])


def test_fault_drains_exact_current_wave_and_never_dispatches_next(harness, tmp_path):
    def wave(*args):
        cells, _fault = harness[1](*args)
        cells[0]["health"]["infrastructure_problem"] = True
        return cells, "driving_infrastructure"

    assert len(execute(harness, wave_runner=wave)) == len(harness[2]) == 4
    assert run.read(tmp_path / "results/batch.json")["failure"] == "driving_infrastructure"


def test_source_validation_failure_is_not_reported_as_complete(harness, tmp_path):
    def wave(*args):
        cells, fault = harness[1](*args)
        admission_path = Path(run.read(harness[0])["admission"])
        write(admission_path, {**run.read(admission_path), "changed": True})
        return cells, fault

    with pytest.raises(ValueError, match="admission"):
        execute(harness, wave_runner=wave)
    assert run.read(tmp_path / "results/batch.json")["failure"]


def test_wave_exception_keeps_written_results_and_unclosed_dispatch(harness, tmp_path):
    output = tmp_path / "results"
    snapshots = {}

    def wave(rows, *_args):
        for row in rows[:3]:
            cell = dict(
                assignment=row,
                status="failed_auth" if row["model"] == "fable" else "complete",
                health=dict(
                    infrastructure_problem=False, model_unavailable=row["model"] == "fable"
                ),
                belief=dict(label="accepted"),
            )
            path = write(output / row["id"] / "worker-result.json", cell)
            snapshots[path] = path.read_bytes()
            if row["model"] != "fable":
                write(output / row["id"] / "attempt-1/dispatch.json", {"actual": True})
        write(output / rows[3]["id"] / "attempt-1/dispatch.json", {"actual": True})
        write(output / "waves/000-unclosed.json", dict(alive={rows[3]["id"]: 24680}))
        raise ValueError("interval annotation failed")

    with pytest.raises(ValueError, match="interval annotation"):
        execute(harness, wave_runner=wave)
    batch = run.read(output / "batch.json")
    assert len(batch["cells"]) == 4
    assert [c["status"] for c in batch["cells"]] == [
        "complete",
        "complete",
        "failed_auth",
        "infrastructure_failure",
    ]
    unknown = batch["cells"][3]
    assert unknown["dispatch_evidence"]["model_may_have_run"]
    assert unknown["final_accounting"]["unclosed_worker_pids"] == {allocation()[3]["id"]: 24680}
    assert sum(row["status"] == "not_run" for row in batch["assignments"]) == 188
    assert all(path.read_bytes() == content for path, content in snapshots.items())
    assert not list(output.glob("*/session.json"))


def test_partial_reporting_retains_final_session_and_does_not_recode(
    harness, tmp_path, monkeypatch
):
    output = tmp_path / "results"
    calls = []

    def wave(rows, *args):
        cells, fault = harness[1](rows, *args)
        for cell in cells:
            write(output / cell["assignment"]["id"] / "worker-result.json", cell)
        return cells, fault

    def reporting(cell, *_args):
        calls.append(cell["assignment"]["id"])
        cell["belief"] = dict(label="doubted", automatic_retries=1)
        write(output / cell["assignment"]["id"] / "session.json", cell)
        raise OSError("health bookkeeping failed")

    monkeypatch.setattr(run, "_reporting", reporting)
    with pytest.raises(OSError, match="bookkeeping"):
        execute(harness, wave_runner=wave)
    batch = run.read(output / "batch.json")
    assert len(calls) == 1 and len(batch["cells"]) == 4
    first = batch["cells"][0]
    assert first["belief"] == dict(label="doubted", automatic_retries=1)
    assert first["final_accounting"]["result_source"].endswith("session.json")
    assert (
        run.read(output / first["assignment"]["id"] / "session.json")["belief"] == first["belief"]
    )
    assert sum(row["status"] == "not_run" for row in batch["assignments"]) == 188


def test_outer_renderer_warmup_aborts_while_waiting_for_gpu_slot(tmp_path, monkeypatch):
    """Use the real flock helper, with a fake renderer and no native/client launch."""
    frozen = run.plan.protocol({key: f"/{key}.json" for key in run.plan.SCENES})
    row = allocation()[0]
    lock_path = tmp_path / "gpu.lock"
    ready, release, abort = threading.Event(), threading.Event(), threading.Event()
    requested, starts = threading.Event(), []

    class Renderer:
        def start(self):
            starts.append(True)

    @contextmanager
    def lifecycle(_directory, factory):
        yield factory

    def cell(*args, **kwargs):
        requested.set()
        kwargs["renderer_factory"]().start()
        pytest.fail("Aborted GPU wait must not start the renderer")

    monkeypatch.setattr(run.study, "SceneRenderer", Renderer)
    monkeypatch.setattr(run.pilot_health, "observe_lifecycle", lifecycle)
    monkeypatch.setattr(run.study, "_cell", cell)
    monkeypatch.setattr(
        run.pilot_health, "driving_health", lambda *_: dict(infrastructure_problem=True)
    )
    with lock_path.open("a+b") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = threading.Thread(
            target=run._collect,
            args=(
                frozen,
                row,
                tmp_path,
                1e20,
                dict(render_lock_path=lock_path),
                {},
                ready,
                release,
                abort,
            ),
        )
        worker.start()
        assert requested.wait(2)
        abort.set()
        worker.join(2)
        assert not worker.is_alive()
    assert starts == [] and not ready.is_set()
    result = run.read(tmp_path / row["id"] / "worker-result.json")
    assert result["startup_failure"] and not result["dispatch_evidence"]["model_may_have_run"]


class ProcessContext:
    """Deterministic process lifecycle double; no app, renderer or client is started."""

    def __init__(self, output, *, auth=None, fault=None, dead=None, fail_launch=None):
        self.output, self.auth, self.fault, self.dead = output, auth, fault, dead
        self.fail_launch = fail_launch
        self.processes = []

    def Event(self):
        return threading.Event()

    def Process(self, target, args):
        context = self
        _frozen, row, _output, _deadline, _execution, _metadata, ready, release, abort = args

        class Process:
            pid = len(context.processes) + 100
            exitcode = None
            closed = False

            def start(self):
                if row["model"] == context.fail_launch:
                    raise OSError("spawn failed")
                if row["model"] not in {context.auth, context.dead}:
                    ready.set()

            def is_alive(self):
                if self.closed:
                    return False
                if row["model"] == context.dead:
                    self.closed, self.exitcode = True, 1
                    return False
                if row["model"] == context.auth or release.is_set() or abort.is_set():
                    fault = row["model"] == context.fault or abort.is_set()
                    cell = dict(
                        assignment=row,
                        status="failed_auth" if row["model"] == context.auth else "complete",
                        attempts=[],
                        health=dict(
                            infrastructure_problem=fault,
                            model_unavailable=row["model"] == context.auth,
                        ),
                        opportunity=dict(candidates=[]),
                    )
                    write(context.output / row["id"] / "worker-result.json", cell)
                    self.closed, self.exitcode = True, 0
                    return False
                return True

            def join(self, timeout=None):
                if abort.is_set():
                    self.is_alive()

        process = Process()
        context.processes.append(process)
        return process


def test_wave_readiness_auth_skip_release_and_normal_fault_drain(tmp_path, monkeypatch):
    monkeypatch.setattr(run.study, "require_disk_space", lambda: None)
    rows = allocation()[:4]
    context = ProcessContext(tmp_path, auth="fable", fault="grok")
    cells, fault = run._wave(
        rows,
        dict(protocol=run.plan.PROTOCOL, followup_order=["debrief"], turn_budget_s=180),
        tmp_path,
        1e20,
        dict(concurrency=4, protocol_sha256="one", peer_fault_policy="normal"),
        0,
        context=context,
    )
    assert fault == "driving_infrastructure" and len(cells) == 4
    assert [c["assignment"] for c in cells] == rows
    release = run.read(tmp_path / "waves/000-release.json")
    assert len(release["ready_ids"]) == 3
    assert all(p.closed for p in context.processes)


def test_dead_startup_worker_aborts_before_any_peer_release(tmp_path, monkeypatch):
    monkeypatch.setattr(run.study, "require_disk_space", lambda: None)
    context = ProcessContext(tmp_path, dead="astra")
    cells, fault = run._wave(
        allocation()[:4],
        dict(protocol=run.plan.PROTOCOL, followup_order=["debrief"], turn_budget_s=180),
        tmp_path,
        1e20,
        dict(concurrency=4, protocol_sha256="one", peer_fault_policy="normal"),
        0,
        context=context,
    )
    assert fault == "worker_exit_or_missing_result"
    assert not (tmp_path / "waves/000-release.json").exists()
    assert all(p.closed for p in context.processes)
    assert len(cells) == 4


def test_spawn_failure_closes_already_started_paused_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(run.study, "require_disk_space", lambda: None)
    context = ProcessContext(tmp_path, fail_launch="sol")
    with pytest.raises(OSError, match="spawn"):
        run._wave(
            allocation()[:4],
            dict(protocol=run.plan.PROTOCOL, followup_order=["debrief"], turn_budget_s=180),
            tmp_path,
            1e20,
            dict(concurrency=4, protocol_sha256="one", peer_fault_policy="normal"),
            0,
            context=context,
        )
    assert context.processes[0].closed
    assert not (tmp_path / "waves/000-release.json").exists()
