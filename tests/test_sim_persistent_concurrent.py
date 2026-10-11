"""Prospective A concurrency behavior using fake cells/processes, never real dispatch."""

import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from drivingbench.sim.studies import concurrent as run
from test_sim_concurrent import ProcessContext, write


def allocation(tmp_path):
    return run.naturalistic_pilots.assignments(
        "A_persistent",
        {
            key: str(tmp_path / f"{key}.json")
            for key in (
                "central-legal-left",
                "central-legal-right",
                "restrictive-left",
                "restrictive-right",
            )
        },
    )


@pytest.fixture
def a_admission(tmp_path, monkeypatch):
    rows = allocation(tmp_path)
    scenes = {r["scene_key"]: r["scene"] for r in rows}
    for path in scenes.values():
        write(Path(path), {})
    source = write(tmp_path / "source.json", {})
    dependency = write(tmp_path / "dependency.json", {})
    frozen = dict(
        protocol=run.naturalistic_run.PROTOCOL,
        pilot="A_persistent",
        time_limit_s=660,
        scenes=scenes,
        sources={str(source): run.file_hash(source)},
        dependencies={str(dependency): run.file_hash(dependency)},
        assignments=rows,
        deadline_utc="2099-01-01T00:00:00+00:00",
        followup_order=["account", "debrief", "perception"],
        turn_budget_s=180,
    )
    protocol = write(tmp_path / "protocol.json", frozen)
    monkeypatch.setattr(run.naturalistic_run, "validate_protocol", lambda _f: None)
    binding = run._a_admission(frozen, protocol)
    report = write(
        tmp_path / "report.json",
        dict(
            concurrency=4,
            camera_mode="on_demand",
            model_calls=0,
            pass_absolute=True,
            checks={"ticks": True, "braking": True, "paused_start": True},
            paired_end_to_end_latency_s=dict(median=0.6, p95=1.2),
            actual_received_image_age_s=dict(median=0.6, p95=1.2),
            persistent_a=binding,
        ),
    )
    review = write(
        tmp_path / "review.json",
        dict(
            approved=True,
            reviewer="root",
            concurrency=4,
            relative_comparison_reviewed=True,
            source_hashes=frozen["sources"],
            preflight_report=str(report),
            evidence_hashes={str(report): run.file_hash(report)},
            persistent_a=binding,
        ),
    )
    return protocol, review, report


def prepare(a_admission, tmp_path):
    protocol, review, _report = a_admission
    return run.freeze(
        protocol, tmp_path / "freeze", results=tmp_path / "results", concurrency=4, admission=review
    )


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_a_rounds_keep_exact_16_and_first4_boundary(tmp_path, limit):
    rows = allocation(tmp_path)
    waves = run.waves(rows, limit)
    assert [r for w in waves for r in w] == rows
    assert all(len({r["model"] for r in w}) == len(w) for w in waves)
    assert 4 in [sum(map(len, waves[:i])) for i in range(1, len(waves) + 1)]
    assert sum(map(len, waves)) == 16


def test_a_freeze_binds_cap_stage_and_complete_identity(a_admission, tmp_path):
    execution = prepare(a_admission, tmp_path)
    assert execution["active_budget_s"] == 660
    assert execution["pilot_stage"] is None
    assert execution["pilot"] == "A_persistent"
    assert run.validate(execution)["pilot"] == "A_persistent"
    changed = dict(execution, active_budget_s=180)
    with pytest.raises(ValueError, match="policy"):
        run.validate(changed)


@pytest.mark.parametrize("target", ["review", "report"])
@pytest.mark.parametrize(
    "field", ["protocol_sha256", "pilot", "active_budget_s", "scene_hashes", "dependency_hashes"]
)
def test_blue_or_other_a_admission_cannot_be_borrowed(a_admission, target, field):
    protocol, review_path, report_path = a_admission
    path = review_path if target == "review" else report_path
    value = run.read(path)
    value["persistent_a"][field] = "different"
    write(path, value)
    if target == "report":
        review = run.read(review_path)
        review["evidence_hashes"][str(report_path)] = run.file_hash(report_path)
        write(review_path, review)
    with pytest.raises(ValueError, match="own exact-freeze"):
        run._qualification(review_path, run.read(protocol), 4, protocol_path=protocol)


def test_unsupported_pilots_or_allocation_changes_rejected(tmp_path):
    frozen = dict(
        protocol=run.naturalistic_run.PROTOCOL,
        pilot="A_persistent",
        time_limit_s=660,
        assignments=allocation(tmp_path),
    )
    assert run._bindings(frozen)[1] is run.naturalistic_run
    for change in (
        dict(pilot="A"),
        dict(pilot="C"),
        dict(pilot="D"),
        dict(time_limit_s=180),
        dict(assignments=frozen["assignments"][:-1]),
        dict(protocol="unknown"),
    ):
        with pytest.raises(ValueError):
            run._bindings(dict(frozen, **change))


def test_parent_a_uses_first4_gate_no_blue_stage_and_exclusive_claim(
    a_admission, tmp_path, monkeypatch
):
    prepare(a_admission, tmp_path)
    launched, checkpoints = [], []

    def wave(rows, *_args):
        launched.extend(rows)
        return [
            dict(
                assignment=r,
                status="complete",
                attempts=[{}],
                health=dict(infrastructure_problem=False, model_unavailable=False),
                opportunity=dict(candidates=[]),
            )
            for r in rows
        ], None

    def checkpoint(cells, _output):
        checkpoints.append([c["assignment"]["id"] for c in cells])
        return dict(continue_automatically=True)

    monkeypatch.setattr(run.naturalistic_run, "_checkpoint", checkpoint)
    monkeypatch.setattr(run.serial, "checkpoint", lambda *_: pytest.fail("blue checkpoint used"))
    result = run.run(
        tmp_path / "freeze/execution.json",
        wave_runner=wave,
        checkpoint_reviewer=lambda *_: None,
        stage_reviewer=lambda *_: pytest.fail("blue stage used"),
    )
    assert len(result) == len(launched) == 16
    assert len(checkpoints) == 1 and len(checkpoints[0]) == 4
    batch = run.read(tmp_path / "results/batch.json")
    assert batch["protocol"] == run.naturalistic_run.PROTOCOL and batch["pilot"] == "A_persistent"
    assert not batch["failure"]
    # Serial and concurrent launches share the exact same base-freeze claim.
    with pytest.raises(FileExistsError):
        run.naturalistic_run.run(
            a_admission[0],
            tmp_path / "serial-reuse",
            cell_runner=lambda *_: pytest.fail("duplicate dispatch"),
        )


def test_a_failed_first4_stops_with_all_16_accounted(a_admission, tmp_path, monkeypatch):
    prepare(a_admission, tmp_path)
    monkeypatch.setattr(
        run.naturalistic_run, "_checkpoint", lambda *_: dict(continue_automatically=False)
    )

    def wave(rows, *_args):
        return [
            dict(
                assignment=r,
                status="complete",
                attempts=[],
                health=dict(infrastructure_problem=False, model_unavailable=False),
                opportunity=dict(candidates=[]),
            )
            for r in rows
        ], None

    cells = run.run(
        tmp_path / "freeze/execution.json", wave_runner=wave, checkpoint_reviewer=lambda *_: None
    )
    batch = run.read(tmp_path / "results/batch.json")
    assert len(cells) == 4 and batch["failure"] == "checkpoint_exposure"
    assert sum(r["status"] == "not_run" for r in batch["assignments"]) == 12


def test_a_worker_selects_local_bindings_and_defers_coder(tmp_path, monkeypatch):
    rows = allocation(tmp_path)
    frozen = dict(
        protocol=run.naturalistic_run.PROTOCOL,
        pilot="A_persistent",
        time_limit_s=660,
        assignments=rows,
        followup_order=[],
    )
    row = rows[0]
    ready, release, abort = threading.Event(), threading.Event(), threading.Event()
    release.set()
    seen = []

    @contextmanager
    def lifecycle(_folder, factory):
        yield factory

    monkeypatch.setattr(run.pilot_health, "observe_lifecycle", lifecycle)
    monkeypatch.setattr(
        run.pilot_health,
        "driving_health",
        lambda *_: dict(infrastructure_problem=False, model_unavailable=False),
    )
    monkeypatch.setattr(
        run.naturalistic_pilots, "effective_protocol", lambda f, r: dict(f, selected=r["id"])
    )
    monkeypatch.setattr(
        run.plan, "effective_protocol", lambda *_: pytest.fail("worker borrowed blue global")
    )
    monkeypatch.setattr(
        run.naturalistic_run,
        "declaration_for",
        lambda *_: dict(
            time_scale=1, observe_clock="wall", restrictions={}, camera_capture="on_demand"
        ),
    )
    monkeypatch.setattr(
        run.naturalistic_run,
        "opportunity",
        lambda c, f, o: dict(candidates=[], selected_pilot=f["pilot"]),
    )
    monkeypatch.setattr(run.study, "drive", lambda: None)

    def cell(effective, assigned, folder, recovery, deadline, **kwargs):
        assert effective["selected"] == assigned["id"]
        assert kwargs["active_budget_s"] == 660 and kwargs["mcp_factory"] is None
        declaration = kwargs["declaration_factory"](assigned, {}, effective)
        assert declaration["paused"] is True and declaration["restrictions"] == {}
        native = SimpleNamespace(paused=True, pose_snapshot=lambda: {"sim_time_s": 0})
        episode = SimpleNamespace(record=lambda *a, **kw: seen.append(a[0]))
        kwargs["before_dispatch"](
            SimpleNamespace(state=SimpleNamespace(sim_native=native, sim_episode=episode)), folder
        )
        assert ready.is_set() and native.paused
        write(folder / "attempt-1/dispatch.json", {})
        run.study.drive()
        belief = run.study.code_belief("account", folder / "belief", 20)
        assert belief == dict(deferred_until_wave_closed=True)
        return dict(assignment=assigned, status="complete", attempts=[{}])

    monkeypatch.setattr(run.study, "_cell", cell)
    run._collect(
        frozen,
        row,
        tmp_path,
        1e20,
        dict(render_lock_path="fake", protocol_sha256="identity"),
        {},
        ready,
        release,
        abort,
    )
    result = run.read(tmp_path / row["id"] / "worker-result.json")
    assert seen == ["concurrent_wave_release"]
    assert result["opportunity"]["selected_pilot"] == "A_persistent"
    assert len(result["deferred_belief"]) == 1 and not (tmp_path / row["id"] / "belief").exists()


def test_a_wave_reuses_auth_barrier_abort_and_peer_fault_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(run.study, "require_disk_space", lambda: None)
    rows = allocation(tmp_path)[:4]
    frozen = dict(
        protocol=run.naturalistic_run.PROTOCOL,
        pilot="A_persistent",
        assignments=allocation(tmp_path),
        time_limit_s=660,
        followup_order=[],
        turn_budget_s=180,
    )
    context = ProcessContext(tmp_path, auth="fable", fault="grok")
    cells, fault = run._wave(
        rows,
        frozen,
        tmp_path,
        1e20,
        dict(
            concurrency=4,
            protocol_sha256="id",
            peer_fault_policy="finish_active_cells_stop_new_waves",
        ),
        0,
        context=context,
    )
    assert fault == "driving_infrastructure" and [c["assignment"] for c in cells] == rows
    assert all(p.closed for p in context.processes)
    assert len(run.read(tmp_path / "waves/000-release.json")["ready_ids"]) == 3


def test_a_parent_auth_failure_is_retained_then_provider_skipped(
    a_admission, tmp_path, monkeypatch
):
    prepare(a_admission, tmp_path)
    monkeypatch.setattr(
        run.naturalistic_run, "_checkpoint", lambda *_: dict(continue_automatically=True)
    )
    calls = []

    def wave(rows, *_args):
        calls.extend(rows)
        return [
            dict(
                assignment=r,
                status="failed_auth" if r["model"] == "fable" else "complete",
                attempts=[],
                health=dict(infrastructure_problem=False, model_unavailable=r["model"] == "fable"),
                opportunity=dict(candidates=[]),
            )
            for r in rows
        ], None

    cells = run.run(
        tmp_path / "freeze/execution.json", wave_runner=wave, checkpoint_reviewer=lambda *_: None
    )
    fable = [c for c in cells if c["assignment"]["model"] == "fable"]
    assert len(cells) == 16 and len(calls) == 13
    assert [c["status"] for c in fable] == [
        "failed_auth",
        "auth_skipped",
        "auth_skipped",
        "auth_skipped",
    ]


def test_a_fault_stops_new_waves_and_preserves_assigned_denominator(a_admission, tmp_path):
    prepare(a_admission, tmp_path)

    def wave(rows, *_args):
        return [
            dict(
                assignment=r,
                status="infrastructure_failure",
                attempts=[],
                health=dict(infrastructure_problem=True, model_unavailable=False),
                opportunity=dict(candidates=[]),
            )
            for r in rows
        ], "driving_infrastructure"

    cells = run.run(
        tmp_path / "freeze/execution.json",
        wave_runner=wave,
        checkpoint_reviewer=lambda *_: pytest.fail("fault bypassed"),
    )
    batch = run.read(tmp_path / "results/batch.json")
    assert len(cells) == 4 and batch["failure"] == "driving_infrastructure"
    assert len(batch["assignments"]) == 16
    assert sum(r["status"] == "not_run" for r in batch["assignments"]) == 12


def test_a_dead_peer_aborts_before_release_without_rewriting_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(run.study, "require_disk_space", lambda: None)
    rows = allocation(tmp_path)
    frozen = dict(
        protocol=run.naturalistic_run.PROTOCOL,
        pilot="A_persistent",
        assignments=rows,
        time_limit_s=660,
        followup_order=[],
        turn_budget_s=180,
    )
    context = ProcessContext(tmp_path, dead="astra")
    cells, fault = run._wave(
        rows[:4],
        frozen,
        tmp_path,
        1e20,
        dict(
            concurrency=4,
            protocol_sha256="id",
            peer_fault_policy="finish_active_cells_stop_new_waves",
        ),
        0,
        context=context,
    )
    assert fault == "worker_exit_or_missing_result" and len(cells) == 4
    assert not (tmp_path / "waves/000-release.json").exists()
    assert all(p.closed for p in context.processes)
