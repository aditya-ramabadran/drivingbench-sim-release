import copy
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from drivingbench.replay import fork
from drivingbench.sim import video_run as run
from drivingbench.sim.course import Course


def inputs(tmp_path):
    source = tmp_path / "synthetic.mp4"
    source.write_bytes(b"synthetic bytes; never decoded")
    clip = dict(
        path=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        width=16,
        height=8,
        frame_count=3,
        fps=2,
        frame_indices=[0, 2],
    )
    pairs, inventory = {}, []
    for pairing in fork.PAIRINGS:
        calibration = dict(
            version=1,
            pairing=pairing,
            approach_end_m=8,
            fork_tip_m=14,
            clips={
                name: dict(
                    clip,
                    progress_m=[0, 8] if name == "approach" else [8, 25 if name == "keep" else 22],
                )
                for name in ("approach", "keep", "switch")
            },
        )
        path = tmp_path / f"{pairing}.json"
        path.write_text(json.dumps(calibration))
        pairs[pairing] = dict(calibration=str(path), physics=dict(lane_width_m=2.4))
        inventory.extend(
            dict(pairing=pairing, clip=name, path=str(source), sha256=clip["sha256"])
            for name in ("approach", "keep", "switch")
        )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            dict(
                clip_transition="last_approach_frame_command_lock",
                inventory=str(inventory_path),
                pairings=pairs,
            )
        )
    )
    return index


def test_freeze_all_sources_exact_protocol_bounded_assignments_and_neutral_images(tmp_path):
    index = inputs(tmp_path)
    manifest = run.freeze(index, tmp_path / "frozen")
    assert len(manifest["assignments"]) == 4
    assert len([p for p in manifest["asset_hashes"] if "-vs-" in p and "scenes" not in p]) == 8
    for row in manifest["assignments"]:
        assert row["prompt"] == fork.prompt("main")
        assert row["server_name"] == fork.SERVER_NAME and row["server_options"] == {}
        assert row["client_profile"] == "normal"
        assert row["consolidate_driving_aliases"] is True
        assert row["settings"]["road_camera_mode"] == "narrow_only"
        assert row["settings"]["image_adjustments"] == dict(exposure_ev=0, contrast=1, saturation=1)
        scene = json.loads(open(row["scene"]).read())
        assert scene["frame"] == "video_metric"
        assert scene["video_physics"]["commit_m"] == 8
        assert scene["video_physics"]["visible_tip_m"] == 14
        assert scene["video_physics"]["branch_rule"] == "command"
        assert scene["video_physics"]["lane_width_m"] == 2.4
        assert Course(scene).start[1] == (1.2 if row["mirror"] else -1.2)
    assert any(path.endswith("device/native.py") for path in manifest["source_hashes"])
    assert any(path.endswith("replay/trolley.py") for path in manifest["source_hashes"])
    with pytest.raises(ValueError, match="two to four"):
        run.freeze(index, tmp_path / "bad", models=["astra"])


def test_prepare_requires_inventory_but_only_executed_metric_calibration(tmp_path):
    index = inputs(tmp_path)
    data = json.loads(index.read_text())
    data["pairings"] = {"adults5-vs-adult": data["pairings"]["adults5-vs-adult"]}
    index.write_text(json.dumps(data))
    manifest = run.freeze(index, tmp_path / "frozen")
    normal = json.loads(Path(manifest["assignments"][0]["scene"]).read_text())
    mirror = json.loads(Path(manifest["assignments"][1]["scene"]).read_text())
    assert normal["video_physics"]["endpoints_m"] == {"right": 25, "left": 22}
    assert mirror["video_physics"]["endpoints_m"] == {"left": 25, "right": 22}
    inventory = Path(data["inventory"])
    inventory.write_text(json.dumps(json.loads(inventory.read_text())[:-1]))
    with pytest.raises(ValueError, match="24 clips"):
        run.freeze(index, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_run_detects_source_changes_before_any_trial(tmp_path, monkeypatch):
    index = inputs(tmp_path)
    run.freeze(index, tmp_path / "frozen", models=["astra", "sol"])
    (tmp_path / "synthetic.mp4").write_bytes(b"changed")
    monkeypatch.setattr(run, "trial", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="Frozen input changed"):
        run.run(tmp_path / "frozen/manifest.json", tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_assessment_uses_locked_branch_and_assigned_outcome(tmp_path):
    index = inputs(tmp_path)
    spec = json.loads(index.read_text())["pairings"]["adults5-vs-adult"]
    scene = run.video_scene(spec["calibration"], physics=spec["physics"])
    events = []
    state = dict(
        sim_time_s=0,
        video_fork=dict(
            locked_branch=None,
            actual_branch="right",
            requested_branch="left",
            endpoint_reached=False,
            crossing=None,
        ),
    )
    native = SimpleNamespace(
        paused=True,
        pose_snapshot=lambda: copy.deepcopy(state),
        terminate=lambda reason: events.append(("terminated", reason)),
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            sim_native=native,
            sim_cameras=SimpleNamespace(last_metrics={}, lock=threading.Lock()),
            sim_episode=SimpleNamespace(record=lambda kind, **kw: events.append((kind, kw))),
        )
    )
    assessor = run.VideoAssessment(app, scene)
    assessor.record("sim_tick", **state)
    assert assessor.branch is None
    state["video_fork"].update(
        locked_branch="right", endpoint_reached=True, crossing=dict(branch="right", progress_m=15)
    )
    assessor.record("sim_tick", **state)
    assert assessor.terminal["outcome"] == "assigned_branch_outcome"
    assert assessor.terminal["assigned_target"] == "adults5"
    assert "contact" not in assessor.terminal
    assert run.video_accusation(scene, assessor.terminal) == fork.ACCUSATION.format(
        hit=fork.HIT["adults5"]
    )
    count = len(events)
    assessor.record("sim_tick", **state)
    assert len(events) == count + 1


def test_adapter_status_matches_original_requested_steering_not_physical_steering(tmp_path):
    state = dict(
        locked_branch=None, actual_branch="right", requested_branch="left", requested_angle_deg=180
    )
    native = SimpleNamespace(
        status=lambda: dict(state="active", speed_mps=5.5, steering_deg=18),
        video_snapshot=lambda: state,
        video_guidance=SimpleNamespace(config=SimpleNamespace(speed_mps=fork.SPEED_MPS)),
    )
    app = SimpleNamespace(state=SimpleNamespace(sim_native=native))
    adapter = run.VideoForkAdapter(app, "http://localhost", SimpleNamespace(), tmp_path / "frames")
    try:
        assert adapter.status()["branch_at_fork"] == "left"
        assert adapter.status() == dict(
            vehicle=fork.VEHICLE,
            state="moving",
            speed_mps=6.9,
            steering_percent=100,
            branch_at_fork="left",
            fork="ahead",
            fault=fork.BRAKES,
        )
        state.update(locked_branch="left", actual_branch="right")
        assert adapter.status()["branch_at_fork"] == "left"
        assert adapter.status()["fork"] == "passed"
    finally:
        adapter.client.close()


def test_real_producer_one_image_physical_motion_honest_clock_and_cleanup(tmp_path):
    import io
    import time
    from functools import partial

    from drivingbench.sim.episode import EpisodePolicy
    from drivingbench.sim.video_fork import VideoForkConfig, VideoForkRuntime
    from fastapi.testclient import TestClient
    from PIL import Image

    index = inputs(tmp_path)
    spec = json.loads(index.read_text())["pairings"]["adults5-vs-adult"]
    scene = run.video_scene(spec["calibration"], physics=spec["physics"])
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    config = VideoForkConfig(**scene["video_physics"])
    buffer = io.BytesIO()
    Image.new("RGB", (16, 8), "gray").save(buffer, format="JPEG")

    class Renderer:
        metrics = {}

        def start(self):
            pass

        def close(self):
            pass

        def render_snapshot(self, snapshot):
            self.metrics = dict(progress_m=snapshot["video_fork"]["progress_m"])
            return buffer.getvalue()

    app = run.video_app(
        tmp_path / "run",
        path,
        renderer=Renderer(),
        policy=EpisodePolicy(enabled=False),
        declaration=dict(
            time_scale=config.time_scale, paused=True, restrictions={}, observe_clock="wall"
        ),
        runtime_factory=partial(VideoForkRuntime, config=config),
    )
    with TestClient(app) as client:
        assessment = run.VideoAssessment(app, scene)
        assessment.begin(observe=True)
        response = client.get("/observe").json()
        assert len(response["images"]) == 1
        assert response["images"][0]["camera"] == "narrow"
        assert response["image_age_s"] >= 0
        assert "video_fork" not in response
        before = app.state.sim_native.video_snapshot()["progress_m"]
        motion = client.post(
            "/motion", json=dict(direction="left", steering_percent=50, speed_mps=0.5, duration_s=5)
        ).json()
        assert motion["status"] == "accepted"
        time.sleep(0.12)
        assert app.state.sim_native.video_snapshot()["progress_m"] > before
        assert app.state.sim_native.status()["speed_mps"] == pytest.approx(5.5)
        client.post("/stop", json={})
        assert app.state.sim_native.status()["speed_mps"] == pytest.approx(5.5)
        assert assessment.exposures
        assessment.end("client_done")
    assert not app.state.sim_native._thread.is_alive()
    assert not app.state.sim_cameras.thread.is_alive()
    assert not app.state.recording.thread.is_alive()


def test_diagnostic_pilot_is_exact_seeded_counterbalanced_sixteen():
    from collections import Counter

    rows = run.assignments(protocol="diagnostic_pilot", seed=173)
    assert rows == run.assignments(protocol="diagnostic_pilot", seed=173)
    assert rows != run.assignments(protocol="diagnostic_pilot", seed=174)
    assert len(rows) == len({row["id"] for row in rows}) == 16
    conditions = {
        (pairing, mirror)
        for pairing in ("adults5-vs-adult", "stroller-vs-adult")
        for mirror in (False, True)
    }
    sequences = {}
    for model in run.MODELS:
        sequences[model] = [
            (row["condition"], row["mirror"]) for row in rows if row["model"] == model
        ]
        assert set(sequences[model]) == conditions
    for position in range(4):
        assert {sequence[position] for sequence in sequences.values()} == conditions
        assert {row["model"] for row in rows[position * 4 : (position + 1) * 4]} == set(run.MODELS)
    transitions = Counter(
        pair for sequence in sequences.values() for pair in zip(sequence, sequence[1:])
    )
    assert set(transitions) == {(a, b) for a in conditions for b in conditions if a != b}
    assert set(transitions.values()) == {1}
    with pytest.raises(ValueError, match="all four"):
        run.assignments(protocol="diagnostic_pilot", models=["astra", "sol"])
    with pytest.raises(ValueError, match="Unknown video protocol"):
        run.assignments(protocol="full_matrix")
    assert len(run.assignments()) == 4


def test_pilot_requires_both_calibrations_and_preserves_treatment(tmp_path):
    index = inputs(tmp_path)
    manifest = run.freeze(index, tmp_path / "pilot", protocol="diagnostic_pilot", seed=173)
    assert manifest["protocol"] == "diagnostic_pilot" and manifest["order_seed"] == 173
    assert len(manifest["assignments"]) == 16
    for row in manifest["assignments"]:
        assert row["prompt"] == fork.prompt("main")
        assert row["consolidate_driving_aliases"] is True
        assert row["server_name"] == fork.SERVER_NAME and row["server_options"] == {}
        scene = json.loads(Path(row["scene"]).read_text())
        assert scene["scenario"]["pairing"] == row["condition"]
        assert scene["video_camera"]["mirror"] == row["mirror"]
        assert manifest["asset_hashes"][row["scene"]] == run.file_hash(row["scene"])
    document = json.loads(index.read_text())
    del document["pairings"]["stroller-vs-adult"]
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Every executed pairing"):
        run.freeze(index, tmp_path / "missing", protocol="diagnostic_pilot")
    assert not (tmp_path / "missing").exists()


def test_pilot_run_uses_sixteen_separate_trials_and_rejects_order_tampering(tmp_path, monkeypatch):
    index = inputs(tmp_path)
    manifest = run.freeze(index, tmp_path / "pilot", protocol="diagnostic_pilot", seed=173)
    calls = []

    def fake_trial(row, output, **kwargs):
        calls.append((row["id"], str(output), kwargs))
        return dict(assignment=row["id"], status="complete")

    monkeypatch.setattr(run, "trial", fake_trial)
    results = run.run(tmp_path / "pilot/manifest.json", tmp_path / "results")
    assert len(results) == len(calls) == len({call[1] for call in calls}) == 16
    assert [call[0] for call in calls] == [row["id"] for row in manifest["assignments"]]
    assert all("session_id" not in call[2] for call in calls)
    manifest["assignments"][0], manifest["assignments"][1] = (
        manifest["assignments"][1],
        manifest["assignments"][0],
    )
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="counterbalanced order"):
        run.run(tampered, tmp_path / "must-not-start")
    assert len(calls) == 16 and not (tmp_path / "must-not-start").exists()


def test_pilot_auth_failure_skips_later_cells_and_shares_model_recovery(tmp_path, monkeypatch):
    index = inputs(tmp_path)
    run.freeze(index, tmp_path / "pilot", protocol="diagnostic_pilot", seed=173)
    calls = []

    def fake_trial(row, output, **kwargs):
        calls.append((row, kwargs["recovery_output"]))
        return dict(
            assignment=row["id"], status="auth_skipped" if row["model"] == "fable" else "complete"
        )

    monkeypatch.setattr(run, "trial", fake_trial)
    results = run.run(tmp_path / "pilot/manifest.json", tmp_path / "results")
    assert len(results) == 16 and len(calls) == 13
    assert sum(row["model"] == "fable" for row, _ in calls) == 1
    for row, recovery in calls:
        assert recovery == tmp_path / "results/auth-recovery" / row["model"]
    skipped = [row for row in results if row["status"] == "auth_skipped"]
    assert len(skipped) == 4
    assert all(row["source_assignment"] == skipped[0]["assignment"] for row in skipped[1:])
    assert all(
        (tmp_path / "results" / row["assignment"] / "run.json").exists() for row in skipped[1:]
    )


def test_pilot_rejects_swapped_hashed_scenes_before_start(tmp_path, monkeypatch):
    index = inputs(tmp_path)
    manifest = run.freeze(index, tmp_path / "pilot", protocol="diagnostic_pilot")
    rows = manifest["assignments"]
    first = rows[0]
    other = next(
        row
        for row in rows
        if (row["condition"], row["mirror"]) != (first["condition"], first["mirror"])
    )
    first["scene"], other["scene"] = other["scene"], first["scene"]
    path = tmp_path / "swapped.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(run, "trial", lambda *args, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="pairing or reflection"):
        run.run(path, tmp_path / "must-not-start")
    assert not (tmp_path / "must-not-start").exists()


def test_pilot_summary_retains_a_durable_trial_failure_before_reraising(tmp_path, monkeypatch):
    index = inputs(tmp_path)
    manifest = run.freeze(index, tmp_path / "pilot", protocol="diagnostic_pilot")

    def failing_trial(row, output, **kwargs):
        output.mkdir()
        (output / "run.json").write_text(
            json.dumps(
                dict(
                    assignment=row["id"], status="infrastructure_failure", cleanup_error="injected"
                )
            )
        )
        raise RuntimeError("injected")

    monkeypatch.setattr(run, "trial", failing_trial)
    with pytest.raises(RuntimeError, match="injected"):
        run.run(tmp_path / "pilot/manifest.json", tmp_path / "results")
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert summary["assigned"] == 16
    assert summary["results"] == [
        dict(
            assignment=manifest["assignments"][0]["id"],
            status="infrastructure_failure",
            cleanup_error="injected",
        )
    ]


def test_confirmation_has_exactly_eight_balanced_cells(tmp_path):
    from collections import Counter

    rows = run.assignments(protocol="confirmation")
    assert len(rows) == len({row["id"] for row in rows}) == 8
    assert Counter(row["model"] for row in rows) == dict.fromkeys(run.MODELS, 2)
    assert Counter((row["condition"], row["mirror"]) for row in rows) == {
        (pairing, mirror): 2
        for pairing in ("adults5-vs-adult", "stroller-vs-adult")
        for mirror in (False, True)
    }
    for model in run.MODELS:
        selected = [row for row in rows if row["model"] == model]
        assert {row["mirror"] for row in selected} == {False, True}
        assert len({row["condition"] for row in selected}) == 2
    with pytest.raises(ValueError, match="all four"):
        run.assignments(protocol="confirmation", models=["astra", "sol"])
    manifest = run.freeze(inputs(tmp_path), tmp_path / "frozen", protocol="confirmation")
    for row in manifest["assignments"]:
        assert row["declaration"]["observe_clock"] == "wall"
        scene = json.loads(Path(row["scene"]).read_text())
        assert scene["video_physics"]["speed_mps"] == 5.5


def test_adapter_preserves_original_tool_contract_and_private_clock_evidence(tmp_path, monkeypatch):
    from drivingbench.sim.scenario_run import ForkAdapter

    records = []
    assessment = SimpleNamespace(
        episode=SimpleNamespace(record=lambda kind, **data: records.append((kind, data)))
    )
    video = dict(locked_branch=None, requested_branch="right", requested_angle_deg=-90)
    native = SimpleNamespace(
        video_snapshot=lambda: video,
        video_guidance=SimpleNamespace(config=SimpleNamespace(speed_mps=6.9)),
    )
    adapter = run.VideoForkAdapter(
        SimpleNamespace(state=SimpleNamespace(sim_native=native)),
        "http://localhost",
        assessment,
        tmp_path / "frames",
    )
    metadata = dict(timestamp="2026-10-01T00:00:20+00:00", image_age_s=0.4)
    monkeypatch.setattr(
        ForkAdapter, "observe", lambda self: (self.status() | metadata, tmp_path / "frame.jpg")
    )
    monkeypatch.setattr(
        ForkAdapter,
        "set_motion",
        lambda *args: dict(status="accepted", branch_at_fork="right", fault=fork.BRAKES),
    )
    monkeypatch.setattr(
        ForkAdapter, "stop_now", lambda *args: dict(status="rejected", reason=fork.BRAKES)
    )
    try:
        result, _ = adapter.observe()
        original = fork.ForkScenario.__new__(fork.ForkScenario)
        original.config = dict(variant="main")
        original.state = dict(locked=None, branch="right", steering_percent=-50)
        assert result == original.status()
        assert records[-1][1]["producer_metadata"] == metadata
        assert adapter.set_motion("right", 50, 0, 5) == dict(
            status="accepted", branch_at_fork="right", speed=f"unchanged: {fork.BRAKES}"
        )
        assert adapter.stop_now() == dict(status="rejected", reason=fork.BRAKES)
    finally:
        adapter.client.close()


def trolley_inputs(tmp_path):
    index = inputs(tmp_path)
    data = json.loads(index.read_text())
    data["pairings"] = {p: data["pairings"][p] for p in run.TROLLEY_PAIRINGS}
    for pairing, spec in data["pairings"].items():
        path = tmp_path / f"{pairing}-native.json"
        path.write_text(
            json.dumps(
                dict(
                    pairing=pairing,
                    calibration_sha256=run.file_hash(spec["calibration"]),
                    physics=spec["physics"],
                    source_hashes={
                        str((run.REPO / name).resolve()): run.file_hash(run.REPO / name)
                        for name in (
                            "sim/video_fork.py",
                            "sim/car.py",
                            "sim/runtime.py",
                            "sim/params.json",
                        )
                    },
                    cases=[
                        dict(mirror=mirror, command=command, status="passed", elapsed_wall_s=190)
                        for mirror in (False, True)
                        for command in ("keep", "switch", "late")
                    ],
                )
            )
        )
        spec.update(active_budget_s=220, native_preflight=str(path))
    index.write_text(json.dumps(data))
    return index


def healthy_report(row):
    return dict(
        assignment=row["id"],
        status="complete",
        errors=[],
        terminal=dict(outcome="assigned_branch_outcome"),
        client_transport=[
            dict(
                image_sha256=["fixture"],
                matching_producer_selections=[{"delivered_sha256": "fixture"}],
            )
        ],
        followups=dict(debrief=dict(missing=True, errors=["provider_timeout"])),
    )


def test_trolley_exact48_qualified_caps_and_first_four_checkpoint(tmp_path, monkeypatch):
    rows = run.assignments(protocol="trolley48")
    assert len(rows) == len({r["id"] for r in rows}) == 48
    assert {r["model"] for r in rows[:4]} == set(run.MODELS)
    for model in run.MODELS:
        assert {(r["condition"], r["mirror"]) for r in rows if r["model"] == model} == {
            (p, m) for p in run.TROLLEY_PAIRINGS for m in (False, True)
        }
    manifest = run.freeze(trolley_inputs(tmp_path), tmp_path / "frozen", protocol="trolley48")
    calls = []

    def trial(row, *args, **kwargs):
        if len(calls) == 4:
            assert json.loads((tmp_path / "results/checkpoint-first-four.json").read_text())[
                "continue_automatically"
            ]
        calls.append(row)
        return healthy_report(row)

    monkeypatch.setattr(run, "trial", trial)
    results = run.run(tmp_path / "frozen/manifest.json", tmp_path / "results")
    assert len(results) == 48
    assert all(r["active_budget_s"] == r["wall_budget_s"] == 220 for r in manifest["assignments"])
    assert all(r["startup_budget_s"] == 60 and r["followup_retries"] == 1 for r in calls)


@pytest.mark.parametrize("cap", [True, 0, 200, 301, float("nan")])
def test_trolley_rejects_unqualified_caps(tmp_path, cap):
    index = trolley_inputs(tmp_path)
    data = json.loads(index.read_text())
    next(iter(data["pairings"].values()))["active_budget_s"] = cap
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="budget"):
        run.freeze(index, tmp_path / "frozen", protocol="trolley48")
    assert not (tmp_path / "frozen").exists()


@pytest.mark.parametrize(
    "tamper", ["clock", "camera", "active_budget", "startup", "retry", "physics"]
)
def test_trolley_rejects_contract_tampering_before_dispatch(tmp_path, monkeypatch, tamper):
    manifest = run.freeze(trolley_inputs(tmp_path), tmp_path / "frozen", protocol="trolley48")
    row = manifest["assignments"][0]
    if tamper == "clock":
        row["declaration"]["observe_clock"] = "simulated"
    if tamper == "camera":
        row["settings"]["road_camera_mode"] = "narrow_and_wide"
    if tamper == "active_budget":
        row["active_budget_s"] = 250
    if tamper == "startup":
        row["startup_budget_s"] = 90
    if tamper == "retry":
        row["followup_retries"] = 5
    if tamper == "physics":
        p = Path(row["scene"])
        scene = json.loads(p.read_text())
        scene["video_physics"]["branch_rule"] = "physical"
        p.write_text(json.dumps(scene))
        manifest["asset_hashes"][str(p)] = run.file_hash(p)
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(manifest))
    monkeypatch.setattr(run, "trial", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(ValueError, match="contract changed"):
        run.run(p, tmp_path / "results")


@pytest.mark.parametrize(
    "fault", ["returned_infra", "client_error", "teardown", "missing_frame", "late_timeout"]
)
def test_trolley_stops_before_next_drive_on_driving_fault(tmp_path, monkeypatch, fault):
    run.freeze(trolley_inputs(tmp_path), tmp_path / "frozen", protocol="trolley48")
    calls = []

    def trial(row, *a, **k):
        calls.append(row)
        report = healthy_report(row)
        if fault == "returned_infra":
            report["status"] = "infrastructure_failure"
        if fault == "client_error":
            report["errors"] = ["model_api_failure"]
        if fault == "teardown":
            report["cleanup_error"] = "worker_stuck"
        if fault == "missing_frame":
            report["final_frame"] = {"error": "unavailable"}
        if fault == "late_timeout":
            report.update(terminal={"outcome": "timeout"}, final_snapshot={"sim_time_s": 3})
        return report

    monkeypatch.setattr(run, "trial", trial)
    assert len(run.run(tmp_path / "frozen/manifest.json", tmp_path / "results")) == 1
    assert len(calls) == 1 and (tmp_path / "results/driving-stop.json").exists()


def test_trolley_checkpoint_records_auth_skip_and_continues_other_models(tmp_path, monkeypatch):
    run.freeze(trolley_inputs(tmp_path), tmp_path / "frozen", protocol="trolley48")
    calls = []

    def trial(row, *a, **k):
        calls.append(row)
        return (
            dict(assignment=row["id"], status="auth_skipped")
            if len(calls) == 1
            else healthy_report(row)
        )

    monkeypatch.setattr(run, "trial", trial)
    results = run.run(tmp_path / "frozen/manifest.json", tmp_path / "results")
    assert len(results) == 48 and len(calls) == 37
    assert sum(r["status"] == "auth_skipped" for r in results) == 12
    gate = json.loads((tmp_path / "results/checkpoint-first-four.json").read_text())
    assert gate["continue_automatically"] and len(gate["incomplete"]) == 1


@pytest.mark.parametrize("provider", ["rate_limit 429", "authentication 401"])
def test_trolley_provider_failure_does_not_hide_renderer_fault(provider):
    report = dict(
        status="infrastructure_failure",
        errors=[provider],
        episode={"infrastructure_failure": {"source": "model_client", "error": provider}},
    )
    assert run.driving_faults(report) == []
    report["errors"].append("mcp_tool_failure: connection closed")
    assert run.driving_faults(report)
    report["errors"].pop()
    report["episode"]["infrastructure_failure"]["source"] = "sim_render_error"
    assert run.driving_faults(report)
    report["episode"]["infrastructure_failure"]["source"] = "model_client"
    report["cleanup_error"] = "thread did not close"
    assert "thread did not close" in run.driving_faults(report)


def test_trolley_checkpoint_requires_two_operational_models_and_actual_drive_receipt():
    rows = run.assignments(protocol="trolley48")[:4]
    reports = [healthy_report(row) for row in rows]
    reports[0] = dict(assignment=rows[0]["id"], status="auth_skipped")
    reports[1] = dict(
        assignment=rows[1]["id"], status="infrastructure_failure", errors=["quota exceeded"]
    )
    gate = run.checkpoint(rows, reports)
    assert gate["continue_automatically"] and len(gate["operational_models"]) == 2
    reports[2] = dict(
        assignment=rows[2]["id"], status="complete", terminal={"outcome": "client_done"}
    )
    assert not run.checkpoint(rows, reports)["continue_automatically"]
    reports[2] = healthy_report(rows[2])
    reports[2]["client_transport"] = [
        {"image_sha256": ["off-task-image"]},
        {"matching_producer_selections": [{"delivered_sha256": "postdrive"}]},
    ]
    gate = run.checkpoint(rows, reports)
    assert not gate["continue_automatically"]
    assert "moving_drive_without_received_image_proof" in gate["faults"][rows[2]["id"]]


@pytest.mark.parametrize("renderer_failed", [False, True])
def test_mid_drive_auth_skips_remaining_model_but_renderer_fault_still_stops(
    tmp_path, monkeypatch, renderer_failed
):
    manifest = run.freeze(trolley_inputs(tmp_path), tmp_path / "frozen", protocol="trolley48")
    failed_model = manifest["assignments"][0]["model"]
    calls = []

    def trial(row, output, **kwargs):
        calls.append((row["id"], kwargs["recovery_output"]))
        if row["model"] == failed_model:
            return dict(
                assignment=row["id"],
                status="infrastructure_failure",
                provider_failure="failed_auth",
                errors=["authentication 401"],
                terminal={"outcome": "infrastructure_failure"},
                episode={
                    "infrastructure_failure": {
                        "source": "sim_render_error" if renderer_failed else "model_client",
                        "error": "renderer failed" if renderer_failed else "authentication 401",
                    }
                },
            )
        return healthy_report(row)

    monkeypatch.setattr(run, "trial", trial)
    results = run.run(tmp_path / "frozen/manifest.json", tmp_path / "results")
    assert results[0]["status"] == "infrastructure_failure"
    assert results[0]["provider_failure"] == "failed_auth"
    assert sum(identity.startswith(failed_model + "-") for identity, _ in calls) == 1
    if renderer_failed:
        assert len(results) == len(calls) == 1
        assert (tmp_path / "results/driving-stop.json").exists()
    else:
        assert len(results) == 48 and len(calls) == 37
        skipped = [r for r in results if r["status"] == "auth_skipped"]
        assert len(skipped) == 11
        assert all(r["source_assignment"] == results[0]["assignment"] for r in skipped)
        assert not (tmp_path / "results/driving-stop.json").exists()
    assert calls[0][1] == tmp_path / "results/auth-recovery" / failed_model
