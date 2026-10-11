"""Private tidy construction and genuine handback bounds; no model/GPU clients."""

import copy
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from drivingbench.sim import event_study_v2 as study
from drivingbench.sim.course import Course
from drivingbench.sim.naturalistic_analysis import decision_cutoff, decision_ticks
from drivingbench.sim.real_course_study.refined import RefinedPolicy, runtime_factory
from drivingbench.sim.studies import concurrent, naturalistic_run, pilot_exposure, tidy_cd
from test_sim_pilot_exposure import fixture as exposure_fixture
from test_sim_real_course_refined import body_for_front, refined_scene
from test_sim_real_course_study import Coverage, western
from test_sim_runtime import Clock


@pytest.fixture
def tidy_inputs(tmp_path, monkeypatch):
    # Pin a complete small authored descendant, using the production selector.
    base = refined_scene()
    parent = tmp_path / "base.json"
    parent.write_text(json.dumps(base))
    monkeypatch.setattr(tidy_cd, "BASE_SHA", tidy_cd.file_hash(parent))
    entries = {}
    for pilot, treatments in (("C", ("casual", "stakes")), ("D", ("neutral", "pressure"))):
        entries[pilot] = {}
        for side in ("left", "right"):
            scene = copy.deepcopy(base)
            identity = f"{pilot}-{side}"
            scene["authored_derivation"] = dict(id=identity, base_scene_sha256=tidy_cd.BASE_SHA)
            path = tmp_path / f"{identity}.json"
            path.write_text(json.dumps(scene))
            for treatment in treatments:
                entries[pilot][f"{side}-{treatment}"] = dict(
                    path=str(path), sha256=tidy_cd.file_hash(path), derivation_id=identity
                )
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(dict(base_scene=str(parent), entries=entries, inherited_keys=list(base)))
    )
    monkeypatch.setattr(tidy_cd, "BINDINGS", bindings)
    monkeypatch.setattr(tidy_cd, "BINDINGS_SHA", tidy_cd.file_hash(bindings))
    return entries


@pytest.mark.parametrize("pilot", ["C", "D"])
def test_tidy_constructor_uses_monitored_factory_and_real_refined_runtime(tidy_inputs, pilot):
    key = next(iter(tidy_inputs[pilot]))
    row = dict(study=pilot, scene_key=key)
    protocol = dict(course_variant=tidy_cd.VARIANT)
    pinned, scene = tidy_cd.scenes(pilot)[key]
    calls = []

    def monitored(*args, **kwargs):
        calls.append((args, kwargs))
        return "monitored-app"

    construct = tidy_cd.bind_application(
        row, protocol, monitored_app=monitored, coverage=Coverage()
    )
    declaration = tidy_cd.declaration_for(row, scene, protocol)
    assert (
        construct(
            "run",
            pinned["path"],
            renderer="renderer",
            policy=RefinedPolicy(),
            declaration=declaration,
        )
        == "monitored-app"
    )
    args, options = calls[0]
    assert args == ("run", pinned["path"])
    assert type(options["policy"]) is RefinedPolicy
    runtime = options["runtime_factory"](Course(scene), clock=Clock())
    try:
        assert runtime.paused is (pilot == "D")
        assert runtime.course.western_score((0, 0, 0), scene)["applicable"] is False
        with pytest.raises(ValueError, match="pinned authored"):
            runtime.course.western_score((0, 0, 0), {})
    finally:
        runtime.close()


@pytest.mark.parametrize("damage", ["path", "bytes", "policy", "declaration", "overlay"])
def test_tidy_constructor_rejects_wrong_identity_before_monitored_factory(
    tidy_inputs, tmp_path, damage
):
    row = dict(study="C", scene_key="left-casual")
    protocol = dict(course_variant=tidy_cd.VARIANT)
    pinned, scene = tidy_cd.scenes("C")[row["scene_key"]]
    calls = []
    overlay = dict(
        paused=True, camera_capture="on_demand", render_lock_path=str(tmp_path / "gpu.lock")
    )
    if damage == "overlay":
        overlay["paused"] = False
    try:
        construct = tidy_cd.bind_application(
            row,
            protocol,
            monitored_app=lambda *a, **kw: calls.append((a, kw)),
            coverage=Coverage(),
            execution_overlay=overlay,
        )
        path, policy = pinned["path"], RefinedPolicy()
        declaration = dict(tidy_cd.declaration_for(row, scene, protocol), **overlay)
        if damage == "path":
            path = tmp_path / "same-bytes.json"
            path.write_text(json.dumps(scene))
        elif damage == "bytes":
            with open(path, "a") as stream:
                stream.write(" ")
        elif damage == "policy":
            policy = RefinedPolicy(boundary_tolerance_m=1)
        elif damage == "declaration":
            declaration["paused"] = False
        with pytest.raises(ValueError):
            construct("run", path, renderer=None, policy=policy, declaration=declaration)
    except ValueError:
        assert damage == "overlay"
    assert calls == []


def test_model_constructor_rejects_missing_certificate_before_shared_binding(
    tidy_inputs, tmp_path, monkeypatch
):
    certificate = tmp_path / "incomplete-certificate.json"
    certificate.write_text("{}")
    calls = []
    monkeypatch.setattr(tidy_cd, "bind_application", lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(ValueError, match="certificate"):
        tidy_cd.application_for(
            dict(study="C", scene_key="left-casual"),
            dict(
                course_variant=tidy_cd.VARIANT,
                preflight=str(certificate),
                scenes={k: v["path"] for k, v in tidy_inputs["C"].items()},
            ),
            monitored_app=lambda *a, **kw: None,
        )
    assert calls == []


@pytest.mark.parametrize("custom", [False, True])
def test_private_cell_constructor_seam_preserves_default_restriction_factory(
    tmp_path, monkeypatch, custom
):
    from drivingbench.sim import restrictions

    @contextmanager
    def server(_app):
        yield "http://127.0.0.1:54321"

    class Renderer:
        closed = False

        def start(self):
            pass

        def close(self):
            self.closed = True

    class Constructed(Exception):
        pass

    renderer, seen, restrictions_used = Renderer(), [], []

    def construct(*args, **kwargs):
        seen.append((args, kwargs))
        raise Constructed

    marker = object()
    monkeypatch.setattr(
        restrictions, "runtime_for", lambda d: restrictions_used.append(d) or marker
    )
    monkeypatch.setattr(study, "simulator_app", construct)
    monkeypatch.setattr(study, "http_server", server)
    monkeypatch.setattr(study, "gateway_app", lambda _proxy: None)
    monkeypatch.setattr(study, "auth_preflight", lambda *a, **kw: dict(eligible=True))
    monkeypatch.setattr(study, "require_disk_space", lambda: {})
    monkeypatch.setattr(study, "prepare_isolated", lambda *a, **kw: object())
    scene = tmp_path / "scene.json"
    scene.write_text("{}")
    protocol = dict(scenes={"one": str(scene)}, episode_policy={}, settings={})
    row = dict(id="one", model="sol", arm="real_time", condition="child", scene_key="one")
    with pytest.raises(Constructed):
        study._cell(
            protocol,
            row,
            tmp_path / "cell",
            tmp_path / "recovery",
            time.monotonic() + 30,
            renderer_factory=lambda *a: renderer,
            declaration_factory=lambda *a: dict(time_scale=1, restrictions={}),
            **(dict(application_factory=construct) if custom else {}),
        )
    assert len(seen) == 1 and renderer.closed
    assert restrictions_used == ([] if custom else [dict(time_scale=1, restrictions={})])
    if custom:
        assert "runtime_factory" not in seen[0][1]
    else:
        assert seen[0][1]["runtime_factory"] is marker


@pytest.mark.parametrize("fault", [None, "pose", "record"])
def test_D_evidence_fault_still_engages_original_physical_finish(tmp_path, monkeypatch, fault):
    clock = Clock()
    native = runtime_factory(
        refined_scene(), Coverage(), registered_start=body_for_front(0.5), western_score=western
    )(Course(refined_scene()), clock=clock)
    native.native.brake = False
    events = []
    native.on_evidence = lambda kind, **data: events.append(dict(kind=kind, **data))
    evidence_error = RuntimeError("injected evidence failure")

    def record(kind, **data):
        if fault == "record":
            raise evidence_error
        events.append(dict(kind=kind, **data))

    app = SimpleNamespace(
        state=SimpleNamespace(sim_native=native, sim_episode=SimpleNamespace(record=record))
    )
    try:
        if fault == "pose":
            monkeypatch.setattr(
                native, "pose_snapshot", lambda: (_ for _ in ()).throw(evidence_error)
            )
        naturalistic_run.record_completed_ticks(app, tmp_path, pilot="D")
        if fault:
            with pytest.raises(RuntimeError) as raised:
                native.finish("client_done")
            assert raised.value is evidence_error
        else:
            assert native.finish("client_done") is None
        assert native.native.brake is True
        assert native._parking_deadline == pytest.approx(clock.now + 5)
        assert native.end_reason is None
        assert [e["kind"] for e in events][-1] == "sim_parking_confirmation"
        if not fault:
            assert events[0]["kind"] == "sim_client_handback_requested"
            assert events[0]["clock_basis"] == "actual_host_finish_entry"
    finally:
        native.close()


@pytest.mark.parametrize("handback,eligible", [(7, False), (8, False), (8.01, True)])
def test_D_received_horizon_cannot_borrow_parking_cleanup(handback, eligible):
    data = exposure_fixture("D", receipt_s=6)
    data["rule"] = pilot_exposure.rule_for("D")
    for tick in data["events"]:
        if tick["kind"] == "sim_tick":
            tick["native_completed_monotonic_s"] = 100 + tick["sim_time_s"]
            tick["pose"] = [tick["sim_time_s"] * 2, 0, 0]
    frame = next(e for e in data["events"] if e["kind"] == "sim_frame")
    frame["pose"] = [12, 0, 0]
    request = dict(
        kind="sim_client_handback_requested",
        sim_time_s=handback,
        requested_at=100 + handback,
        requested_reason="client_done",
        clock_basis="actual_host_finish_entry",
    )
    data["events"].extend([request, dict(kind="sim_parking_confirmation", sim_time_s=handback)])
    before = copy.deepcopy(data)
    report = pilot_exposure.qualify(**data)
    candidate = report["candidates"][0]
    assert candidate["normal"]["eligible"] is eligible
    assert not candidate["safe_held"]["eligible"]
    assert report["actual_decision_bound"]["monotonic_s"] == 100 + handback
    cutoff = decision_cutoff(data["events"], {})
    assert cutoff["basis"] == "sim_client_handback_requested" and not cutoff["inclusive"]
    assert all(t["sim_time_s"] < handback for t in decision_ticks(data["events"], cutoff))
    assert data == before
    data["events"].remove(request)
    missing = pilot_exposure.qualify(**data)
    assert not missing["candidates"][0]["eligible"]
    assert missing["actual_decision_bound"] is None


@pytest.mark.parametrize("kind", ["blue", "A_persistent", "pressure", "pressure_A", "C", "D"])
def test_concurrent_default_dispatch_keeps_original_owner_hooks(tmp_path, monkeypatch, kind):
    if kind == "blue":
        plan, runner = concurrent.plan, concurrent.serial
        frozen = dict(protocol=plan.PROTOCOL)
    elif kind in {"pressure", "pressure_A"}:
        plan, runner = concurrent.pressure, concurrent.pressure_run
        frozen = dict(protocol=plan.A_PROTOCOL if kind == "pressure_A" else plan.PROTOCOL)
        if kind == "pressure_A":
            frozen.update(time_limit_s=660, pilot="A_persistent")
    else:
        plan, runner = concurrent.naturalistic_pilots, concurrent.naturalistic_run
        frozen = dict(protocol=runner.PROTOCOL, pilot=kind, time_limit_s=660, assignments=[{}] * 16)
    if kind in {"C", "D"}:
        frozen["course_variant"] = tidy_cd.VARIANT
    frozen["followup_order"] = []
    row = dict(id="one", arm="straight_sandbox", condition="child", model="sol", study=kind)
    effective = dict(frozen, prompt="unchanged assigned prompt")
    seen, handback, factory_calls = [], object(), []
    monitored, factory = object(), object()

    def application(assigned, actual, **kwargs):
        factory_calls.append(kwargs)
        assert assigned is row and actual is effective
        assert kwargs["monitored_app"] is monitored
        return factory

    monkeypatch.setattr(tidy_cd, "application_for", application)
    monkeypatch.setattr(plan, "effective_protocol", lambda *_: effective)
    opportunity_calls = []

    def opportunity(*args, **kwargs):
        opportunity_calls.append((args, kwargs))
        return dict(candidates=[])

    monkeypatch.setattr(runner, "opportunity", opportunity)
    monkeypatch.setattr(concurrent.pressure_run, "handback_factory", lambda *_: handback)
    monkeypatch.setattr(concurrent.pressure_run, "measurements", lambda *_: dict(triggered=False))
    monkeypatch.setattr(
        concurrent.pilot_health,
        "driving_health",
        lambda *_: dict(infrastructure_problem=False, model_unavailable=False),
    )

    @contextmanager
    def lifecycle(_folder, renderer):
        with monkeypatch.context() as active:
            active.setattr(concurrent.study, "simulator_app", monitored)
            yield renderer

    monkeypatch.setattr(concurrent.pilot_health, "observe_lifecycle", lifecycle)

    def cell(actual, assigned, folder, *_args, **kwargs):
        seen.append(kwargs)
        assert actual is effective and assigned is row
        return dict(assignment=assigned, status="complete", attempts=[])

    monkeypatch.setattr(concurrent.study, "_cell", cell)
    ready, release, abort = (threading.Event() for _ in range(3))
    release.set()
    concurrent._collect(
        frozen,
        row,
        tmp_path,
        time.monotonic() + 30,
        dict(render_lock_path=str(tmp_path / "gpu.lock"), protocol_sha256="test"),
        {},
        ready,
        release,
        abort,
    )
    assert len(seen) == 1
    if kind in {"C", "D"}:
        assert seen[0]["application_factory"] is factory
        assert factory_calls[0]["execution_overlay"] == dict(
            paused=True, camera_capture="on_demand", render_lock_path=str(tmp_path / "gpu.lock")
        )
    else:
        assert "application_factory" not in seen[0] and factory_calls == []
    if kind in {"pressure", "pressure_A"}:
        assert seen[0]["handback_factory"] is handback
    else:
        assert "handback_factory" not in seen[0]
    assert seen[0]["active_budget_s"] == (
        660 if kind in {"A_persistent", "pressure_A", "C", "D"} else plan.ACTIVE_BUDGET_S
    )
    assert len(opportunity_calls) == 1
    assert opportunity_calls[0][1] == ({"frozen": frozen} if kind == "pressure_A" else {})
    if kind == "pressure_A":
        assert seen[0]["mcp_factory"] is None
    result = concurrent.read(tmp_path / "one/worker-result.json")
    assert result["status"] == "complete" and not result["health"]["infrastructure_problem"]


@pytest.mark.parametrize(
    "record_fault,finish_fault", [(False, False), (True, False), (False, True), (True, True)]
)
def test_D_finish_result_and_original_exception_are_preserved(tmp_path, record_fault, finish_fault):
    evidence_error, finish_error = ValueError("record failed"), OSError("finish failed")
    token, finished, ticks = object(), [], []

    def original_finish(reason):
        finished.append(reason)
        if finish_fault:
            raise finish_error
        return token

    def original_evidence(kind, **data):
        ticks.append((kind, data))
        return token

    def record(*args, **kwargs):
        if record_fault:
            raise evidence_error

    native = SimpleNamespace(
        finish=original_finish,
        on_evidence=original_evidence,
        pose_snapshot=lambda: dict(sim_time_s=2),
    )
    app = SimpleNamespace(
        state=SimpleNamespace(sim_native=native, sim_episode=SimpleNamespace(record=record))
    )
    naturalistic_run.record_completed_ticks(app, tmp_path, pilot="D")
    if finish_fault or record_fault:
        with pytest.raises(OSError if finish_fault else ValueError) as raised:
            native.finish("client_done")
        assert raised.value is (finish_error if finish_fault else evidence_error)
        if finish_fault and record_fault:
            assert raised.value.__context__ is evidence_error
    else:
        assert native.finish("client_done") is token
    assert finished == ["client_done"]
    assert native.on_evidence("sim_tick", sim_time_s=2) is token
    assert ticks[0][1]["native_completed_monotonic_s"] > 0
