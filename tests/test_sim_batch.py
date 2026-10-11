import io
import json
from contextlib import AsyncExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from drivingbench.replay.clients import CLIENTS, ClientResult
from drivingbench.replay.run import http_server
from drivingbench.shared.contracts import Settings
from drivingbench.sim import batch
from drivingbench.sim.episode import EpisodePolicy
from drivingbench.sim.producer import simulator_app
from fastapi import FastAPI
from fastapi.responses import Response
from PIL import Image
from test_sim_live import connect, scene_file, summary


@pytest.fixture(autouse=True)
def no_maintenance_model_calls(monkeypatch):
    monkeypatch.setattr(
        batch,
        "refresh_claude_credentials",
        lambda *_args, **_kwargs: {
            "action": "refresh_unavailable",
            "model_calls": 0,
        },
    )


def test_timeout_conservative_minimum_and_minute_rounding():
    timing = dict(reference_duration_s=336.3, reference_length_m=67.24)
    assert batch.scaled_timeout(timing, 127) == 660
    assert batch.scaled_timeout(timing, 60) == 360
    with pytest.raises(ValueError):
        batch.scaled_timeout(timing, float("nan"))
    with pytest.raises(ValueError):
        batch.scaled_timeout(dict(reference_duration_s=1, reference_length_m=0), 100)


def test_prompts_keep_same_objective_and_explicit_reset():
    first = batch.prompt(1, 660)
    following = batch.prompt(2, 660, "failure")
    assert batch.OBJECTIVE in first and batch.OBJECTIVE in following
    assert "Attempt 2 of 3" in following and "reset to the start" in following
    assert "Previous outcome: failure" in following
    assert "Thinking and tool latency are real driving time" in first


def test_artifacts_cannot_overwrite_previous_attempt(tmp_path):
    path = tmp_path / "run.json"
    batch.save_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        batch.save_new(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}


def test_auth_status_redacts_account_and_token(monkeypatch):
    monkeypatch.setattr(
        batch.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"loggedIn": True, "email": "private", "token": "secret"}),
        ),
    )
    monkeypatch.setattr(
        batch, "claude_oauth", lambda: {"accessToken": "secret", "expiresAt": 1234000}
    )
    result = batch.auth_status(CLIENTS["fable"])
    assert result == {
        "logged_in": True,
        "returncode": 0,
        "copied_token_present": True,
        "expires_at_unix_s": 1234,
    }
    assert "private" not in json.dumps(result) and "secret" not in json.dumps(result)


def test_stable_proxy_switches_without_old_frames_and_preserves_image_headers():
    switch = batch.ProducerSwitch()
    with http_server(switch.app) as stable:
        assert httpx.get(stable + "/camera/narrow").status_code == 503
        for value in (b"first", b"second"):
            app = FastAPI()

            @app.get("/camera/narrow")
            def image():
                return Response(
                    value, media_type="image/jpeg", headers={"X-Frame-ID": value.decode()}
                )

            with http_server(app) as producer:
                with switch.connected(producer):
                    response = httpx.get(stable + "/camera/narrow")
                    assert response.content == value
                    assert response.headers["X-Frame-ID"] == value.decode()
                    with pytest.raises(RuntimeError):
                        with switch.connected(producer):
                            pass
            assert httpx.get(stable + "/camera/narrow").status_code == 503
        with pytest.raises(ValueError):
            with switch.connected("http://example.org:80"):
                pass


class Renderer:
    def __init__(self, *_args):
        self.metrics = {}
        self.errors = {}
        self.closed = False
        buffer = io.BytesIO()
        Image.new("RGB", (1344, 760), "gray").save(buffer, "JPEG", quality=92)
        self.jpeg = buffer.getvalue()

    def start(self):
        return {"ready": True}

    def render(self, _pose):
        return self.jpeg

    def render_frames(self, _snapshot):
        return {"narrow": self.jpeg, "wide": self.jpeg}

    def close(self):
        self.closed = True


def protocol(tmp_path):
    scene = scene_file(tmp_path)
    data = {
        "protocol": "sim-course-three-attempts-v2",
        "client_profile": "normal",
        "models": [asdict(CLIENTS[key]) for key in batch.MODELS],
        "attempts": 3,
        "source_sha256": batch.file_hash(Path(batch.__file__)),
        "scene": str(scene),
        "scene_sha256": batch.file_hash(scene),
        "dependencies": {},
        "sources": {},
        "settings": Settings(speed_limit_mps=3.5, road_camera_mode="narrow_and_wide").model_dump(),
        "episode_policy": asdict(EpisodePolicy(time_limit_s=360)),
        "boundary_geometry": batch.Course(
            json.loads(scene.read_text()), boundary_tolerance_m=0.5
        ).boundary_rule,
        "timeout_s": 360,
    }
    path = tmp_path / "protocol.json"
    batch.save_new(path, data)
    return path


async def test_real_mcp_batch_lease_keeps_pair_across_attempt_reset(tmp_path):
    renderer = Renderer()
    switch = batch.ProducerSwitch()
    lease = batch.RendererLease(renderer)
    assert lease.errors is renderer.errors and lease.metrics is renderer.metrics
    boots = []
    try:
        with http_server(switch.app) as proxy, http_server(batch.gateway_app(proxy)) as gateway:
            async with AsyncExitStack() as stack:
                client = await connect(stack, gateway, tmp_path)
                for attempt in (1, 2):
                    app = simulator_app(
                        tmp_path / f"attempt-{attempt}", scene_file(tmp_path), renderer=lease
                    )
                    app.state.settings.patch({"road_camera_mode": "narrow_and_wide"})
                    with http_server(app) as producer, switch.connected(producer):
                        batch.wait_ready(app)
                        response = await client.call_tool("observe", {})
                        packet = summary(response)
                        assert packet["cameras"] == ["narrow", "wide"]
                        assert len(response.content) == 3
                        assert all(block.type == "image" for block in response.content[1:])
                        assert not packet.get("camera_reason")
                        boots.append(app.state.sim_native.status()["boot_id"])
                        frames = app.state.sim_cameras.frames
                        assert frames["narrow"].captured_at == frames["wide"].captured_at
                    assert not renderer.closed
                    assert not app.state.sim_native._thread.is_alive()
                    assert not app.state.sim_cameras.thread.is_alive()
        assert boots[0] != boots[1]
    finally:
        renderer.close()
    assert renderer.closed


@pytest.mark.parametrize("key,value", [("geos", "changed"), ("tolerance_m", 0.4)])
def test_frozen_geometry_mismatch_rejected_before_side_effects(tmp_path, monkeypatch, key, value):
    path = protocol(tmp_path)
    declaration = json.loads(path.read_text())
    declaration["boundary_geometry"][key] = value
    path.write_text(json.dumps(declaration))
    monkeypatch.setattr(batch, "SceneRenderer", lambda *_: pytest.fail("Renderer started"))
    monkeypatch.setattr(batch, "auth_preflight", lambda *_: pytest.fail("Auth started"))
    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="Boundary geometry differs"):
        batch.run(path, output)
    assert not output.exists()


@pytest.mark.parametrize("break_resume", [False, True])
def test_serial_attempts_fresh_native_same_conversation_and_stable_gateway(
    tmp_path, monkeypatch, break_resume
):
    instances, launches = [], []

    def renderer(*args):
        value = Renderer(*args)
        instances.append(value)
        return value

    def prepare(spec, state_dir, command, *, server_name, profile):
        assert server_name == "drivingbench_sandbox" and profile == "normal"
        gateway = command[-1]
        state_dir.mkdir(parents=True)

        def start(text, out, session_id=None):
            status = httpx.get(gateway + "/api/status").json()
            assert status["speed_mps"] == 0 and status["command"] is None
            response = httpx.get(gateway + "/api/observe").json()
            assert response["state"] == "held"
            assert len(response["images"]) == 2
            launches.append((spec.key, session_id, gateway, status["boot_id"]))
            out.mkdir(parents=True)
            stdout, stderr = out / "stdout.jsonl", out / "stderr.log"
            returned_session = (
                "unexpected-fork" if break_resume and session_id else spec.key + "-conversation"
            )
            stdout.write_text(
                json.dumps(
                    dict(
                        type="item.completed",
                        item=dict(
                            type="mcp_tool_call",
                            id="off-task",
                            server="unrelated",
                            tool="read",
                            status="failed",
                            error="permission denied",
                        ),
                    )
                )
                + "\n"
                + json.dumps({"thread_id": returned_session, "type": "turn.completed"})
                + "\n"
            )
            stderr.write_text("")
            return SimpleNamespace(
                turn_done=lambda: True,
                poll=lambda: 0,
                cancel=lambda: None,
                finish=lambda: ClientResult(0, returned_session, stdout, stderr),
            )

        return SimpleNamespace(
            start=start, spec=spec, state_dir=state_dir, profile=profile, server_name=server_name
        )

    monkeypatch.setattr(batch, "SceneRenderer", renderer)
    monkeypatch.setattr(batch, "prepare_client", prepare)
    monkeypatch.setattr(batch, "auth_status", lambda _spec: {"logged_in": True})
    reports = batch.run(protocol(tmp_path), tmp_path / "batch")
    count = 8 if break_resume else 12
    assert len(reports) == count and len(launches) == count
    assert len(instances) == 1 and instances[0].closed
    assert len({row[3] for row in launches}) == count
    for key in batch.MODELS:
        rows = [row for row in launches if row[0] == key]
        expected = [None, key + "-conversation", key + "-conversation"]
        assert [row[1] for row in rows] == expected[: 2 if break_resume else 3]
        assert len({row[2] for row in rows}) == 1
    assert all(report["episode"]["stopped"] for report in reports)
    assert all(report["episode"]["native_end_reason"] == "client_done" for report in reports)
    assert all(report["tool_activity"]["counts"]["off_task"] == 1 for report in reports)
    if not break_resume:
        assert all(not report["client_errors"] for report in reports)


def test_auth_failure_skips_without_invoking_client(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "SceneRenderer", Renderer)
    monkeypatch.setattr(batch, "auth_status", lambda _spec: {"logged_in": False})
    monkeypatch.setattr(
        batch, "prepare_client", lambda *_args: pytest.fail("Must not launch client")
    )
    output = tmp_path / "batch"
    assert batch.run(protocol(tmp_path), output) == []
    assert len(list(output.glob("*/skipped.json"))) == 4


def test_missing_wide_stops_batch_before_model_dispatch(tmp_path, monkeypatch):
    class NarrowOnly(Renderer):
        def render_frames(self, _snapshot):
            return {"narrow": self.jpeg}

    renderer = NarrowOnly()
    monkeypatch.setattr(batch, "SceneRenderer", lambda *_: renderer)
    monkeypatch.setattr(batch, "auth_status", lambda _spec: {"logged_in": True})
    monkeypatch.setattr(
        batch,
        "prepare_isolated",
        lambda *_args, **_kwargs: SimpleNamespace(
            start=lambda *_args, **_kwargs: pytest.fail("Model dispatched without wide camera")
        ),
    )
    with pytest.raises(RuntimeError, match="Paired course cameras unavailable"):
        batch.run(protocol(tmp_path), tmp_path / "batch")
    assert renderer.closed


def test_unsupported_auth_recovery_skips_after_one_status_check(monkeypatch):
    calls = []

    def status(_spec):
        calls.append(1)
        return {"logged_in": len(calls) == 2}

    monkeypatch.setattr(batch, "auth_status", status)
    result = batch.auth_preflight(CLIENTS["sol"])
    assert not result["eligible"] and len(calls) == 1
    assert result["recovery"] == {"action": "refresh_unavailable", "model_calls": 0}
    monkeypatch.setattr(
        batch, "auth_status", lambda _spec: {"logged_in": True, "expires_at_unix_s": 0}
    )
    result = batch.auth_preflight(CLIENTS["fable"])
    assert not result["eligible"] and len(result["checks"]) == 1
    assert result["recovery"]["action"] == "no_recovery_allowance"


def test_fresh_claude_needs_no_refresh_or_allowance_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(batch.time, "time", lambda: 100)
    monkeypatch.setattr(batch, "auth_status", lambda _: dict(logged_in=True, expires_at_unix_s=700))
    monkeypatch.setattr(
        batch, "refresh_claude_credentials", lambda *_, **__: pytest.fail("Fresh token")
    )
    path = tmp_path / "recovery"
    audit = batch.auth_preflight(CLIENTS["fable"], 600, recovery_output=path)
    assert audit["eligible"] and audit["recovery"] is None
    assert len(audit["checks"]) == 1 and not path.exists()


@pytest.mark.parametrize("failure", [False, True])
def test_claude_recovery_is_a_single_reserved_maintenance_attempt(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(batch.time, "time", lambda: 100)
    status = dict(logged_in=True, expires_at_unix_s=0)
    calls = []
    monkeypatch.setattr(batch, "auth_status", lambda _: dict(status))

    def refresh(spec, output, *, timeout_s, required_lifetime_s):
        calls.append((spec.key, output, timeout_s, required_lifetime_s))
        if failure:
            raise OSError("maintenance failed")
        status["expires_at_unix_s"] = 1000
        return dict(action="normal_cli_refresh_maintenance", model_calls=1)

    monkeypatch.setattr(batch, "refresh_claude_credentials", refresh)
    path = tmp_path / "recovery"
    audit = batch.auth_preflight(CLIENTS["fable"], recovery_output=path)
    assert audit["eligible"] is not failure and len(audit["checks"]) == 2
    assert calls == [("fable", path / "maintenance", 60, 600)]
    assert json.loads((path / "result.json").read_text()) == audit
    status["expires_at_unix_s"] = 0
    again = batch.auth_preflight(CLIENTS["fable"], recovery_output=path)
    assert not again["eligible"] and len(again["checks"]) == 1
    assert again["recovery"]["action"] == "allowance_already_used"
    assert len(calls) == 1


def test_course_freeze_uses_real_camera_treatment_and_all_scene_assets(tmp_path, monkeypatch):
    scene = scene_file(tmp_path)
    data = json.loads(scene.read_text())
    data["wide_camera"] = dict(
        checkpoint="wide.safetensors", calibration="wide.json", vehicle_layer="body.png"
    )
    scene.write_text(json.dumps(data))
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps(dict(reference_duration_s=336.3, reference_length_m=67.24)))
    monkeypatch.setattr(batch, "scene_asset_paths", lambda _: ["animated.png", "texture.png"])
    monkeypatch.setattr(batch, "file_hash", lambda _: "0" * 64)
    monkeypatch.setattr(batch, "auth_status", lambda _: dict(logged_in=True))
    protocol = batch.freeze(scene, timing, 127, tmp_path / "freeze")
    assert protocol["protocol"] == "sim-course-three-attempts-v2"
    assert protocol["client_profile"] == "normal"
    settings = protocol["settings"]
    assert settings["road_camera_mode"] == "narrow_and_wide"
    assert settings["image_adjustments"] == dict(exposure_ev=0, contrast=1.53, saturation=1.54)
    names = {Path(path).name for path in protocol["dependencies"]}
    assert {"wide.safetensors", "wide.json", "body.png", "animated.png", "texture.png"} <= names


def test_tool_activity_preserves_non_driving_behavior_as_private_evidence(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "client-profile.json").write_text(json.dumps(dict(replaced_servers=["scene"])))
    output = tmp_path / "stdout.jsonl"
    rows = [
        dict(type="mcp_tool_call", id="one", server="scene", tool="observe", status="completed"),
        dict(type="mcp_tool_call", id="two", server="unrelated", tool="read", status="failed"),
    ]
    output.write_text("\n".join(json.dumps(dict(type="item.completed", item=r)) for r in rows))
    client = SimpleNamespace(state_dir=state, profile="normal", server_name="scene")
    report = batch.record_tool_activity(client, SimpleNamespace(stdout_path=output), tmp_path)
    assert report["counts"] == dict(driving=1, off_task=1, discovery=0, unresolved=0)
    evidence = json.loads(Path(report["path"]).read_text())
    assert evidence["calls"][1]["status"] == "failed"
    assert evidence["calls"][1]["source"] == str(output)


@pytest.mark.parametrize("profile", ["normal", "restricted"])
@pytest.mark.parametrize("server", ["renamed-private", "unrelated"])
def test_driving_errors_use_the_same_prepared_aliases_as_activity(tmp_path, profile, server):
    (tmp_path / "client-profile.json").write_text(
        json.dumps(dict(replaced_servers=["renamed-private"]))
    )
    path = tmp_path / "stdout.jsonl"
    path.write_text(
        json.dumps(
            dict(
                type="item.completed",
                item=dict(
                    type="mcp_tool_call",
                    id="call",
                    server=server,
                    tool="observe",
                    status="failed",
                    error="permission denied",
                ),
            )
        )
        + "\n"
    )
    result = ClientResult(0, "conversation", path, tmp_path / "stderr")
    client = SimpleNamespace(state_dir=tmp_path, profile=profile, server_name="scene")
    errors = batch.driving_result_errors(client, result)
    assert bool(errors) is (server == "renamed-private" or profile == "restricted")
    activity = batch.record_tool_activity(client, result, tmp_path)
    assert activity["counts"]["driving" if server == "renamed-private" else "off_task"] == 1


def test_standalone_smoke_auth_skip_and_normal_profile(tmp_path, monkeypatch):
    from drivingbench.sim import run

    allowance = []

    def auth(spec, lifetime, *, recovery_output):
        allowance.append((spec.key, lifetime, recovery_output))
        return dict(eligible=False)

    monkeypatch.setattr(batch, "auth_preflight", auth)
    monkeypatch.setattr(batch, "prepare_isolated", lambda *_, **__: pytest.fail("Auth skip"))
    report = run.smoke(None, "http://127.0.0.1", tmp_path, "fable")
    assert allowance == [("fable", 600, tmp_path / "auth-recovery/fable")]
    assert report["client_profile"] == "normal" and report["skipped"] == "auth_unavailable"
    monkeypatch.setattr(batch, "auth_preflight", lambda *_, **__: dict(eligible=True))

    def prepared(*_, profile):
        assert profile == "normal"
        raise RuntimeError("profile checked without a model call")

    monkeypatch.setattr(batch, "prepare_isolated", prepared)
    with pytest.raises(RuntimeError, match="profile checked"):
        run.smoke(None, "http://127.0.0.1", tmp_path / "successful-auth", "sol")


def test_credential_preparation_retry_reuses_partially_created_directory(tmp_path, monkeypatch):
    calls = []
    state = tmp_path / "state"

    def prepare(_spec, path, _command, *, server_name, profile):
        assert server_name == "drivingbench_realvehicle" and profile == "normal"
        path.mkdir(exist_ok=True)
        calls.append(path)
        if len(calls) == 1:
            (path / "existing-workspace").write_text("preserved")
            raise RuntimeError("OAuth credential expired")
        assert (path / "existing-workspace").read_text() == "preserved"
        return "prepared"

    monkeypatch.setattr(batch, "prepare_client", prepare)
    assert (
        batch.prepare_isolated(
            CLIENTS["fable"], state, [], tmp_path, server_name="drivingbench_realvehicle"
        )
        == "prepared"
    )
    assert calls == [state, state]


def test_standalone_smoke_records_failed_credential_preparation_without_refresh(
    tmp_path, monkeypatch
):
    from drivingbench.sim import run

    checks, snapshots = [], []

    def auth(*args, **kwargs):
        checks.append(kwargs)
        return dict(eligible=True, recovery=None)

    def prepare(*args, **kwargs):
        snapshots.append(kwargs)
        raise RuntimeError("Credential snapshot unavailable")

    monkeypatch.setattr(batch, "auth_preflight", auth)
    monkeypatch.setattr(batch, "prepare_client", prepare)
    monkeypatch.setattr(
        batch, "refresh_claude_credentials", lambda *_, **__: pytest.fail("No extra refresh")
    )
    report = run.smoke(None, "http://127.0.0.1", tmp_path, "fable")
    assert len(checks) == 1 and len(snapshots) == 2
    assert all(item["profile"] == "normal" for item in snapshots)
    assert report["skipped"] == "auth_unavailable"
    assert report["reason"] == "credential_preparation_failed"
    assert report["client_profile"] == "normal"
    assert json.loads((tmp_path / "smoke.json").read_text()) == report
    assert len(list(tmp_path.glob("credential-attempt-*.json"))) == 2
    assert not (tmp_path / "conversation").exists()


@pytest.mark.parametrize("inside_finish,remaining", [(True, 2), (False, 2), (True, 0.03)])
def test_completed_turn_allows_natural_finish_dwell_only_within_budget(
    tmp_path, monkeypatch, inside_finish, remaining
):
    from drivingbench.sim.episode import Episode
    from test_sim_episode import Runtime

    native = Runtime()
    native.time_scale = 1
    native.status = lambda: dict(held=True, standstill=True, command=None)
    policy = EpisodePolicy(finish_dwell_s=0.4, time_limit_s=1.35 + remaining)
    episode = Episode(native, tmp_path / "episode", policy, metadata={})
    native.tick(1, finish=inside_finish)
    native.tick(1.35, finish=inside_finish)
    clock, waits = [10.0], []

    def sleep(seconds):
        clock[0] += seconds
        waits.append(seconds)
        native.tick(native.last["sim_time_s"] + seconds, finish=inside_finish)

    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(batch.time, "sleep", sleep)

    def start(text, output, session_id=None):
        output.mkdir(parents=True)
        stdout, stderr = output / "stdout.jsonl", output / "stderr.log"
        stdout.write_text(json.dumps({"type": "turn.completed"}) + "\n")
        stderr.write_text("")
        return SimpleNamespace(
            turn_done=lambda: True,
            poll=lambda: 0,
            cancel=lambda: None,
            finish=lambda: ClientResult(0, "conversation", stdout, stderr),
        )

    client = SimpleNamespace(
        start=start, spec=SimpleNamespace(family="fixture"), state_dir=tmp_path
    )
    app = SimpleNamespace(state=SimpleNamespace(sim_native=native, sim_episode=episode))
    try:
        report = batch.drive(client, app, tmp_path / "attempt", "fixture", None)
        if inside_finish and remaining > 0.4:
            assert report["episode"]["outcome"] == "success"
            assert report["episode"]["native_end_reason"] == "finish"
            assert len(waits) <= 3
        elif inside_finish:
            assert report["episode"]["outcome"] == "timeout"
            assert report["episode"]["native_end_reason"] == "time_limit"
        else:
            assert report["episode"]["outcome"] == "unfinished"
            assert report["episode"]["native_end_reason"] == "client_done"
            assert len(waits) == 1
    finally:
        episode.close()


@pytest.mark.parametrize(
    "lenient,contact,expected",
    [
        (True, "dome", "finish"),
        (False, "dome", "marker_hit"),
        (True, "tall", "obstacle_contact"),
        (True, "actor", "hazard_contact"),
    ],
)
def test_finish_cleanup_uses_declared_contact_policy(
    tmp_path, monkeypatch, lenient, contact, expected
):
    from drivingbench.sim.episode import Episode
    from test_sim_episode import Runtime

    native = Runtime()
    native.time_scale = 1
    native.status = lambda: dict(held=True, standstill=True, command=None)
    policy = EpisodePolicy(
        stop_on_marker_contact=not lenient,
        strict_boundary_tolerance_m=0.5,
        finish_dwell_s=0.2,
        time_limit_s=10,
    )
    episode = Episode(native, tmp_path / "episode", policy, metadata={})
    clock, waits = [10.0], []
    sample = dict(
        finish=True,
        hits=[contact] if contact != "actor" else [],
        obstacle_collision_ids=["tall"] if contact == "tall" else [],
        strict_boundary_violation=False,
        scenario_contact={"contacts": [{"id": "child"}]} if contact == "actor" else None,
    )

    def sleep(seconds):
        clock[0] += seconds
        waits.append(seconds)
        native.tick(native.last["sim_time_s"] + seconds, **sample)

    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(batch.time, "sleep", sleep)
    try:
        native.tick(1, **sample)
        batch.wait_finish_dwell(native, episode)
        assert native.requests == [expected]
        assert bool(waits) is (expected == "finish")
        assert episode.summary()["first_cause"]["reason"] == expected
    finally:
        episode.close()
