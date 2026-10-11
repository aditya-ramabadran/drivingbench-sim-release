import base64
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import test_sim_event_study_v2 as fixtures
from drivingbench.replay.run import http_server
from drivingbench.sim.hazard_evidence import audit_receipt
from drivingbench.sim.live_receipt import ReceiptWatcher, selected_exposures
from drivingbench.sim.studies import pilot_health as health
from fastapi import FastAPI
from test_sim_event_study_v2 import live_fixture
from test_sim_hazard_run import jpeg


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def clean_lifecycle():
    row = dict(app_created=True, entered=True, exited=True)
    row.update(
        {
            k: dict(present=True, alive=False, error=None)
            for k in ("native", "cameras", "recording", "episode")
        }
    )
    row["renderers"] = [
        dict(close_called=True, closed=True, process_alive=False, reader_alive=False)
    ]
    return row


@pytest.fixture
def drive(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "jpeg", jpeg)
    folder = tmp_path / "cell"
    folder.mkdir()
    directory, scene, frame, delivery = live_fixture(folder)
    directory = directory.rename(folder / "attempt-1")
    events = [frame, delivery]
    (directory / "episode/events.jsonl").write_text("\n".join(map(json.dumps, events)))
    watcher = ReceiptWatcher(
        directory, scene, {"drivingbench_sandbox"}, "http://unused", clock=lambda: 20
    )
    watcher.poll()
    exposures = [
        dict(
            r,
            frame_id=frame["frame_ids"][r["camera"]],
            delivered_frame_id=frame["frame_ids"][r["camera"]] + "-" + r["sha256"][:12],
            delivered_timestamp=r["timestamp"],
            delivered_sha256=r["sha256"],
        )
        for r in selected_exposures(events)
    ]
    audit = audit_receipt(
        exposures,
        directory / "client/stdout.jsonl",
        directory / "producer/recordings",
        directory / "received-frames",
        {"drivingbench_sandbox"},
    )
    save(directory / "hazard-evidence.json", dict(receipt=audit))
    save(directory / "exposure-pending.json", [])
    save(directory / "receipt-supervisor.json", dict(censored=[], confirmed_calls=1))
    save(directory / "dispatch.json", dict(prompt="neutral"))
    attempt = dict(
        session_id="session",
        returncode=-15,
        supervisor_cancelled=True,
        client_errors=[],
        stop_error=None,
    )
    episode = dict(
        stopped=True,
        native_end_reason="finish",
        unqueued_after_writer_error=0,
        infrastructure_failure=None,
    )
    attempt.update(assignment=dict(id="cell"), status="complete", episode=episode)
    save(directory / "attempt.json", attempt)
    save(directory / "episode/summary.json", episode)
    save(folder / "lifecycle.json", clean_lifecycle())
    cell = dict(assignment=dict(id="cell"), status="complete", attempts=[attempt])
    return cell, tmp_path, directory


def check(drive):
    cell, root, _ = drive
    return health.driving_health(cell, root)


def change(path, **values):
    data = json.loads(path.read_text())
    data.update(values)
    save(path, data)


def provider(drive, status="failed_auth", *, code=None):
    cell, _, directory = drive
    cell["status"] = status
    code = code or ("authentication_error" if status == "failed_auth" else "rate_limit_error")
    change(directory / "attempt.json", client_errors=[code], failure_kind="authentication")
    change(
        directory / "episode/summary.json",
        infrastructure_failure=dict(source="model_client", error=code),
    )
    attempt = json.loads((directory / "attempt.json").read_text())
    attempt.update(
        status=status, episode=json.loads((directory / "episode/summary.json").read_text())
    )
    save(directory / "attempt.json", attempt)
    cell["attempts"] = [attempt]


def test_complete_cancelled_drive_and_postdrive_missing_are_healthy(drive):
    drive[0].update(reporting_failure="failed_auth", operator_account=None, belief=None)
    result = check(drive)
    assert not result["infrastructure_problem"] and result["verified_pairs"] == 1
    assert result["model_unavailable"] is False


@pytest.mark.parametrize("status", ["failed_auth", "failed_capacity"])
def test_provider_only_failure_retains_clean_physical_stop(drive, status):
    provider(drive, status)
    result = check(drive)
    assert not result["infrastructure_problem"] and result["provider_failure"] == status
    assert result["model_unavailable"] is (status == "failed_auth")


@pytest.mark.parametrize("kind", sorted(health.FAULT_KINDS))
def test_later_structured_fault_overrides_first_auth_latch(drive, kind):
    provider(drive)
    path = drive[2] / "episode/events.jsonl"
    with path.open("a") as stream:
        stream.write("\n" + json.dumps(dict(kind=kind, error="actual fault")))
    assert check(drive)["infrastructure_problem"]


def test_camera_error_overrides_auth_and_returned_empty_attempts(drive):
    provider(drive)
    drive[0]["attempts"] = []
    path = drive[2] / "episode/events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["camera_errors"] = {"wide": "render_failed"}
    path.write_text("\n".join(map(json.dumps, rows)))
    assert "structured_driving_fault:sim_frame" in check(drive)["problems"]


@pytest.mark.parametrize(
    "field,value",
    [("stopped", False), ("native_end_reason", None), ("unqueued_after_writer_error", 1)],
)
def test_stop_or_writer_failure_overrides_provider(drive, field, value):
    provider(drive, "failed_capacity")
    change(drive[2] / "episode/summary.json", **{field: value})
    assert check(drive)["infrastructure_problem"]


def test_failed_braking_after_provider_latch(drive):
    provider(drive)
    change(drive[2] / "attempt.json", stop_error="attempt_failed_to_stop")
    assert "native_stop_error" in check(drive)["problems"]


@pytest.mark.parametrize(
    "file",
    [
        "episode/summary.json",
        "attempt.json",
        "dispatch.json",
        "live-receipts.jsonl",
        "client/stdout.jsonl",
    ],
)
def test_missing_required_evidence_fails_closed(drive, file):
    (drive[2] / file).unlink()
    assert check(drive)["infrastructure_problem"]


def test_missing_lifecycle_and_corrupt_evidence_fail_closed(drive):
    (drive[2].parent / "lifecycle.json").unlink()
    assert check(drive)["infrastructure_problem"]
    save(drive[2].parent / "lifecycle.json", clean_lifecycle())
    with (drive[2] / "episode/events.jsonl").open("a") as stream:
        stream.write("\n{incomplete")
    assert check(drive)["infrastructure_problem"]


@pytest.mark.parametrize("resource", ["native", "cameras", "recording", "episode"])
def test_worker_leak_overrides_auth(drive, resource):
    provider(drive)
    state = clean_lifecycle()
    state[resource]["alive"] = True
    save(drive[2].parent / "lifecycle.json", state)
    assert f"{resource}_closure_fault" in check(drive)["problems"]


def test_public_receipt_bytes_cannot_be_replaced_by_saved_accepted_flag(drive):
    path = drive[2] / "client/stdout.jsonl"
    event = json.loads(path.read_text())
    event["item"]["result"]["content"][1]["data"] = base64.b64encode(jpeg("red")).decode()
    save(path, event)
    assert "public_receipt_bytes_or_timestamp_mismatch" in check(drive)["problems"]


def test_receipt_endpoint_failure_overrides_provider_label(drive):
    provider(drive)
    path = drive[2] / "live-receipts.jsonl"
    change(path, supervisor_error="ConnectError: private receipt endpoint unavailable")
    assert "receipt_supervisor_error" in check(drive)["problems"]


def test_provider_preobserve_allows_absent_receipt_file_but_not_delivery(drive):
    provider(drive)
    directory = drive[2]
    (directory / "live-receipts.jsonl").unlink()
    (directory / "episode/events.jsonl").write_text("")
    (directory / "client/stdout.jsonl").write_text("")
    save(directory / "hazard-evidence.json", dict(receipt=dict(observations=[])))
    change(directory / "receipt-supervisor.json", confirmed_calls=0)
    assert not check(drive)["infrastructure_problem"]
    assert check(drive)["verified_pairs"] == 0


def test_fuzzy_auth_word_is_not_verified_provider_failure(drive):
    provider(drive)
    change(drive[2] / "attempt.json", client_errors=["login page was mentioned"])
    assert check(drive)["infrastructure_problem"]


@pytest.mark.parametrize(
    "errors", [{"error": "authentication_error"}, [dict(error="rate_limit_error")]]
)
def test_malformed_provider_errors_fail_closed(drive, errors):
    provider(drive)
    change(drive[2] / "attempt.json", client_errors=errors)
    assert check(drive)["infrastructure_problem"]


def test_preflight_auth_skip_is_distinct_from_startup_failure(tmp_path):
    folder = tmp_path / "cell"
    save(
        folder / "lifecycle.json",
        dict(app_created=False, entered=False, exited=False, renderers=[]),
    )
    save(folder / "auth-status.json", dict(eligible=False))
    cell = dict(
        assignment=dict(id="cell"),
        status="failed_auth",
        attempts=[],
        skipped="auth_or_token_lifetime_unavailable",
    )
    assert health.driving_health(cell, tmp_path)["model_unavailable"]
    (folder / "attempt-1").mkdir()
    assert health.driving_health(cell, tmp_path)["infrastructure_problem"]


def test_credential_preparation_failure_requires_both_saved_failures(tmp_path):
    folder = tmp_path / "cell"
    save(
        folder / "lifecycle.json",
        dict(app_created=False, entered=False, exited=False, renderers=[]),
    )
    save(folder / "auth-status.json", dict(eligible=True))
    cell = dict(
        assignment=dict(id="cell"),
        status="failed_auth",
        attempts=[],
        skipped="credential_preparation_failed",
    )
    save(
        folder / "credential-attempt-1.json",
        dict(error="RuntimeError: Claude OAuth token expires soon; run claude to refresh"),
    )
    assert health.driving_health(cell, tmp_path)["infrastructure_problem"]
    save(
        folder / "credential-attempt-2.json",
        dict(error="RuntimeError: Claude OAuth token expires soon; run claude to refresh"),
    )
    assert health.driving_health(cell, tmp_path)["model_unavailable"]


class Renderer:
    def __init__(self, *, leak=False):
        self.process = SimpleNamespace(pid=123, poll=lambda: None if leak else 0)
        self.worker = SimpleNamespace(
            process=self.process, reader=SimpleNamespace(is_alive=lambda: leak)
        )
        self.calls = 0

    def close(self):
        self.calls += 1
        self.worker.process = None
        self.worker.reader = None


def fake_app(*, fail=False):
    @asynccontextmanager
    async def lifespan(app):
        yield
        if fail:
            raise RuntimeError("actual_shutdown_failure")

    app = FastAPI(lifespan=lifespan)
    for name, thread in (
        ("sim_native", "_thread"),
        ("sim_cameras", "thread"),
        ("recording", "thread"),
        ("sim_episode", "_worker"),
    ):
        setattr(app.state, name, SimpleNamespace(**{thread: None}, error=None))
    return app


def test_genuine_http_shutdown_exception_survives_uvicorn_exit(tmp_path, monkeypatch):
    app = fake_app(fail=True)
    monkeypatch.setattr(health.study, "simulator_app", lambda *a, **kw: app)
    renderer = Renderer()
    with health.observe_lifecycle(tmp_path, lambda *a: renderer) as factory:
        assert health.study.simulator_app() is app
        observed = factory()
        with http_server(app) as url:
            assert httpx.get(url + "/openapi.json", trust_env=False).status_code == 200
        observed.close()
    evidence = json.loads((tmp_path / "lifecycle.json").read_text())
    assert evidence["entered"] and evidence["exited"]
    assert evidence["lifespan_error"] == "RuntimeError: actual_shutdown_failure"
    assert "producer_cleanup_or_cell_error" in health._closed(evidence)
    assert renderer.calls == 1


def test_repeated_close_keeps_first_leaked_process_and_restores_method(tmp_path, monkeypatch):
    app = fake_app()
    monkeypatch.setattr(health.study, "simulator_app", lambda: app)
    renderer = Renderer(leak=True)
    original = renderer.close
    with health.observe_lifecycle(tmp_path, lambda: renderer) as factory:
        factory().close()
        renderer.close()
    row = json.loads((tmp_path / "lifecycle.json").read_text())["renderers"][0]
    assert len(row["close_calls"]) == 2
    assert row["close_calls"][0]["pid"] == 123 and row["close_calls"][1]["pid"] is None
    assert row["process_alive"] and row["reader_alive"]
    assert "renderer_closure_fault" in health._closed(dict(clean_lifecycle(), renderers=[row]))
    assert renderer.close == original and renderer.calls == 2


def test_outer_exception_is_preserved_and_recorded_without_dispatch(tmp_path):
    with pytest.raises(RuntimeError, match="startup_failed"):
        with health.observe_lifecycle(tmp_path, lambda: Renderer()):
            raise RuntimeError("startup_failed")
    evidence = json.loads((tmp_path / "lifecycle.json").read_text())
    assert evidence["cell_error"] == "RuntimeError: startup_failed"
    assert not evidence["app_created"] and not (tmp_path / "attempt-1/dispatch.json").exists()


@pytest.mark.parametrize(
    "change_kind", ["returned_empty", "returned_fields", "closed_episode", "assignment", "status"]
)
def test_returned_and_persisted_attempt_must_agree(drive, change_kind):
    cell, _, directory = drive
    if change_kind == "returned_empty":
        cell["attempts"] = []
    elif change_kind == "returned_fields":
        cell["attempts"][0]["session_id"] = "other-session"
    elif change_kind == "closed_episode":
        change(directory / "episode/summary.json", native_end_reason="different_end")
    else:
        change(directory / "attempt.json", **{change_kind: "different"})
    assert "returned_or_closed_attempt_mismatch" in check(drive)["problems"]


def test_driving_mcp_failure_cannot_be_hidden_by_auth_code(drive):
    provider(drive)
    path = drive[2] / "attempt.json"
    change(path, client_errors=["authentication_error", "mcp_tool_failure: HTTP 401"])
    assert "driving_mcp_failure" in check(drive)["problems"]


def test_mismatched_provider_label_cannot_skip_model(drive):
    provider(drive)
    change(drive[2] / "attempt.json", client_errors=["rate_limit_error"])
    result = check(drive)
    assert result["infrastructure_problem"] and not result["model_unavailable"]


def test_unclassified_credential_failure_is_not_pure_auth(tmp_path):
    folder = tmp_path / "cell"
    save(
        folder / "lifecycle.json",
        dict(app_created=False, entered=False, exited=False, renderers=[]),
    )
    save(folder / "auth-status.json", dict(eligible=True))
    for i in (1, 2):
        save(folder / f"credential-attempt-{i}.json", dict(error="private MCP config invalid"))
    cell = dict(
        assignment=dict(id="cell"),
        status="failed_auth",
        attempts=[],
        skipped="credential_preparation_failed",
    )
    assert health.driving_health(cell, tmp_path)["infrastructure_problem"]


def test_close_exception_is_rethrown_and_latched_through_second_close(tmp_path):
    renderer = Renderer()
    calls = []

    def close():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("actual_renderer_close_error")
        renderer.worker.process = None

    renderer.close = close
    with health.observe_lifecycle(tmp_path, lambda: renderer) as factory:
        with pytest.raises(RuntimeError, match="actual_renderer_close_error"):
            factory().close()
        renderer.close()
    record = json.loads((tmp_path / "lifecycle.json").read_text())["renderers"][0]
    assert len(calls) == 2 and record["error"] == "RuntimeError: actual_renderer_close_error"
    assert "renderer_closure_fault" in health._closed(dict(clean_lifecycle(), renderers=[record]))


@pytest.mark.parametrize("where", ["summary", "raw"])
@pytest.mark.parametrize(
    "error",
    [
        "ConnectError: private receipt endpoint unavailable",
        ["ConnectError: private receipt endpoint unavailable"],
        "rate_limit_error",
        "authentication_error; ConnectError: private receipt endpoint unavailable",
        ["authentication_error", "ConnectError: private receipt endpoint unavailable"],
    ],
)
def test_each_model_client_failure_must_independently_prove_provider(drive, where, error):
    provider(drive)
    failure = dict(source="model_client", error=error)
    if where == "summary":
        change(drive[2] / "episode/summary.json", infrastructure_failure=failure)
        attempt = json.loads((drive[2] / "attempt.json").read_text())
        attempt["episode"] = json.loads((drive[2] / "episode/summary.json").read_text())
        save(drive[2] / "attempt.json", attempt)
        drive[0]["attempts"] = [attempt]
    else:
        with (drive[2] / "episode/events.jsonl").open("a") as stream:
            stream.write("\n" + json.dumps(dict(kind="infrastructure_failure", **failure)))
    assert "structured_driving_fault:infrastructure_failure" in check(drive)["problems"]


@pytest.mark.parametrize("error", ["authentication_error", ["authentication_error"]])
def test_raw_provider_list_or_string_is_allowed_only_with_same_class(drive, error):
    provider(drive)
    with (drive[2] / "episode/events.jsonl").open("a") as stream:
        stream.write(
            "\n"
            + json.dumps(dict(kind="infrastructure_failure", source="model_client", error=error))
        )
    assert not check(drive)["infrastructure_problem"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("confirmed_calls", 0),
        ("confirmed_calls", 2),
        ("confirmed_calls", True),
        ("confirmed_calls", "1"),
        ("censored", None),
        ("censored", {}),
        ("censored", ""),
        ("censored", [dict(reason="timeout")]),
    ],
)
def test_receipt_supervisor_counts_and_types_fail_closed(drive, field, value):
    change(drive[2] / "receipt-supervisor.json", **{field: value})
    assert check(drive)["infrastructure_problem"]


def test_explicit_receipt_rejection_stops_even_with_auth_label(drive):
    provider(drive)
    save(
        drive[2] / "live-receipt-rejections.jsonl",
        dict(camera="narrow", correspondence=dict(accepted=False)),
    )
    assert "receipt_correspondence_rejected" in check(drive)["problems"]


@pytest.mark.parametrize("error", ["disk failed", dict(unexpected="shape"), {}, [], 0])
def test_actual_recorder_error_overrides_provider_label(drive, error):
    provider(drive)
    state = clean_lifecycle()
    state["recording"]["error"] = error
    save(drive[2].parent / "lifecycle.json", state)
    assert "recording_closure_fault" in check(drive)["problems"]


@pytest.mark.parametrize("kind", [None, *sorted(health.FAULT_KINDS)])
def test_observed_sol_capacity_signature_retains_slot_unless_engine_fault(drive, kind):
    message = "Selected model is at capacity. Please try a different model."
    provider(drive, "failed_capacity", code=message)
    path = drive[2] / "episode/events.jsonl"
    with path.open("a") as stream:
        stream.write(
            "\n"
            + json.dumps(dict(kind="infrastructure_failure", source="model_client", error=message))
        )
        if kind:
            stream.write("\n" + json.dumps(dict(kind=kind, error="renderer failed")))
    result = check(drive)
    assert result["provider_failure"] == "failed_capacity"
    assert result["infrastructure_problem"] is bool(kind)
    assert result["model_unavailable"] is False


def test_generic_capacity_mention_is_not_a_provider_signature(drive):
    provider(drive, "failed_capacity", code="Renderer capacity exhausted")
    assert check(drive)["infrastructure_problem"]
