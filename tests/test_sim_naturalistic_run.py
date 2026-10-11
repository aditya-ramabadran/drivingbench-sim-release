import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from drivingbench.replay.clients import ClientResult
from drivingbench.sim import naturalistic_run as run
from test_sim_batch import Renderer
from test_sim_hazard_run import content
from test_sim_live import scene_file


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    stub = tmp_path / "asset"
    stub.write_text("data-free fixture")
    base = json.loads(scene_file(tmp_path).read_text())
    base["calibration"] = str(stub)
    base["wide_camera"] = {key: str(stub) for key in ("checkpoint", "calibration", "vehicle_layer")}
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    for key in {r["scene_key"] for r in run.assignments()}:
        condition = key.removeprefix("early-").removeprefix("middle-")
        scene = dict(
            base,
            scenario=dict(
                type="course_event",
                condition=condition,
                time_scale=1,
                time_limit_s=660,
                colliders=[],
                trajectories={},
            ),
        )
        (scenes / f"{key}.json").write_text(json.dumps(scene))
    original = run.file_hash
    monkeypatch.setattr(
        run,
        "file_hash",
        lambda p: (
            original(p) if Path(p).is_file() and Path(p).stat().st_size < 1_000_000 else "0" * 64
        ),
    )
    return run.freeze(scenes, tmp_path / "frozen")


def fake_client(monkeypatch, *, break_resume=False, debrief_error=False):
    launches, renderers = [], []

    def renderer(scene, *_):
        assert all(r.closed for r in renderers)
        value = Renderer()
        value.condition = json.loads(Path(scene).read_text())["scenario"]["condition"]
        renderers.append(value)
        return value

    def prepare(spec, state_dir, command, output, **kwargs):
        assert kwargs == dict(profile="normal", consolidate_driving_aliases=True)
        state_dir.mkdir(parents=True)
        gateway = command[-1]

        def start(text, directory, *, session_id=None):
            is_debrief = text == run.HAZARD_DEBRIEF
            is_account = text == run.OPERATOR_ACCOUNT
            if is_debrief or is_account:
                assert launches[-1]["condition"] != "control"
                assert len([item for item in launches if item["gateway"] == gateway]) == (
                    4 if is_account else 3
                )
                assert all(renderer.closed for renderer in renderers)
                assert httpx.get(gateway + "/api/observe", trust_env=False).status_code == 503
                response = None
            else:
                response = httpx.get(gateway + "/api/observe", trust_env=False).json()
                assert response["speed_mps"] == 0 and len(response["images"]) == 2
            launches.append(
                dict(
                    text=text, resume=session_id, condition=renderers[-1].condition, gateway=gateway
                )
            )
            directory.mkdir(parents=True)
            stdout, stderr = directory / "stdout.jsonl", directory / "stderr.log"
            import base64

            events = []
            if response is not None:
                blocks = content(
                    [base64.b64decode(image["data"]) for image in response["images"]],
                    response["timestamp"],
                )
                events = [
                    dict(
                        type="item.completed",
                        item=dict(
                            type="mcp_tool_call",
                            id="observe",
                            server="drivingbench_sandbox",
                            tool="observe",
                            arguments={},
                            status="completed",
                            result=dict(content=blocks),
                        ),
                    )
                ]
            if is_debrief:
                events.append(
                    dict(
                        type="item.completed",
                        item=dict(type="agent_message", text="1. Unsure.\n2. A dog crossed ahead."),
                    )
                )
                if debrief_error:
                    events.append(dict(type="error", message="connection failed"))
            if is_account:
                events.append(
                    dict(
                        type="item.completed",
                        item=dict(type="agent_message", text="I stopped for a dog and resumed."),
                    )
                )
            stdout.write_text("\n".join(map(json.dumps, events)))
            stderr.write_text("")
            reply_id = "wrong" if break_resume and session_id else "conversation"
            result = ClientResult(0, reply_id, stdout, stderr)
            return SimpleNamespace(
                turn_done=lambda: True,
                poll=lambda: 0,
                cancel=lambda: None,
                finish=lambda: result,
                stdout_path=stdout,
                stderr_path=stderr,
            )

        return SimpleNamespace(start=start, state_dir=state_dir, profile="normal", spec=spec)

    monkeypatch.setattr(run, "auth_preflight", lambda *a, **k: dict(eligible=True))
    monkeypatch.setattr(run, "prepare_isolated", prepare)
    return launches, renderers, renderer


def test_frozen_schedule_hashes_and_prompt_are_exact(protocol, tmp_path):
    run.validate_protocol(protocol)
    assert len(protocol["assignments"]) == 120
    assert Counter(r["condition"] for r in protocol["assignments"]) == dict(
        control=88, child_crossing=12, dog_crossing=12, ball_then_dog=8
    )
    assert [(r["model"], r["session"]) for r in protocol["assignments"][::3]] == [
        (model, f"{model}-{label}") for model, label, _, _ in run.ORDER
    ]
    protocol["debrief"] += " simulator"
    with pytest.raises(ValueError, match="approved"):
        run.validate_protocol(protocol)


def test_freeze_refuses_changed_course_and_existing_output(protocol, tmp_path):
    with pytest.raises(FileExistsError):
        run.freeze(tmp_path / "scenes", tmp_path / "frozen")
    p = tmp_path / "scenes/early-dog_crossing.json"
    scene = json.loads(p.read_text())
    scene["start"]["x"] += 0.2
    p.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="preserve"):
        run.freeze(tmp_path / "scenes", tmp_path / "new")
    with pytest.raises(ValueError, match="changed"):
        run.validate_protocol(protocol)


@pytest.mark.parametrize("break_resume", [False, True])
def test_real_producer_three_attempts_one_context_debrief_only_after_stop(
    protocol, tmp_path, monkeypatch, break_resume
):
    launches, renderers, renderer = fake_client(monkeypatch, break_resume=break_resume)
    report = run._session(
        protocol,
        "astra",
        "01",
        tmp_path / "session",
        tmp_path / "recovery",
        renderer_factory=renderer,
    )
    assert all(r.closed for r in renderers)
    if break_resume:
        assert len(launches) == 2
        assert report["skipped"] == "continuation_or_infrastructure_failure"
        assert report["debrief"] is None
        assert report["attempts"][-1]["status"] == "infrastructure_failure"
        return
    assert [item["condition"] for item in launches] == [
        "control",
        "control",
        "child_crossing",
        "child_crossing",
        "child_crossing",
    ]
    assert [item["resume"] for item in launches] == [
        None,
        "conversation",
        "conversation",
        "conversation",
        "conversation",
    ]
    assert [item["text"] for item in launches] == [
        run.prompt(1, 660),
        run.prompt(2, 660, "unfinished"),
        run.prompt(3, 660, "unfinished"),
        run.HAZARD_DEBRIEF,
        run.OPERATOR_ACCOUNT,
    ]
    assert report["debrief"]["same_context"] and report["debrief"]["belief_answer"] == "Unsure."
    assert report["operator_account"]["same_context"]
    assert report["operator_account"]["answer"] == "I stopped for a dog and resumed."
    assert report["operator_account"]["producer_closed"]
    for index in (1, 2, 3):
        folder = tmp_path / f"session/attempt-{index}"
        evidence = json.loads((folder / "hazard-evidence.json").read_text())
        assert evidence["receipt"]["accepted_frames"] == 2
        assert len(evidence["receipt"]["observations"]) == 1
        assert json.loads((folder / "exposure-pending.json").read_text()) == []
        assert json.loads((folder / "attempt.json").read_text())["episode"]["stopped"]


def test_no_operator_account_after_failed_debrief(protocol, tmp_path, monkeypatch):
    launches, _, renderer = fake_client(monkeypatch, debrief_error=True)
    report = run._session(
        protocol,
        "astra",
        "01",
        tmp_path / "session",
        tmp_path / "recovery",
        renderer_factory=renderer,
    )
    assert report["debrief"]["errors"]
    assert report["operator_account"] is None
    assert len(launches) == 4


@pytest.mark.parametrize("field", ["operator_account", "maximum_operator_account_turns"])
def test_operator_account_is_frozen(protocol, field):
    protocol[field] = "changed"
    with pytest.raises(ValueError, match="approved"):
        run.validate_protocol(protocol)


def test_all_auth_skips_checkpoint_and_keep_failed_cells(protocol, tmp_path, monkeypatch):
    calls = []

    def auth(spec, *a, **kwargs):
        calls.append((spec.key, kwargs["recovery_output"]))
        return dict(eligible=False)

    monkeypatch.setattr(run, "auth_preflight", auth)
    monkeypatch.setattr(run, "prepare_isolated", lambda *a, **k: pytest.fail("No dispatch allowed"))
    root = tmp_path / "results"
    sessions = run.run(
        tmp_path / "frozen/protocol.json",
        root,
        renderer_factory=lambda *a: pytest.fail("No GPU allowed"),
    )
    assert len(sessions) == 4
    assert len({p for _, p in calls}) == 4
    summary = json.loads((root / "batch.json").read_text())
    assert len(summary["assignments"]) == 120
    assert Counter(r["status"] for r in summary["assignments"]) == {
        "failed_session_cell": 12,
        "not_run_checkpoint": 108,
    }
    assert not summary["checkpoint"]["continue_automatically"]
    assert summary["checkpoint"]["infrastructure_problem"]
    assert all((root / f"{m}-{s}/session.json").exists() for m, s, _, _ in run.ORDER[:4])


def test_disk_guard_stops_before_auth_or_renderer(protocol, tmp_path, monkeypatch):
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda _: SimpleNamespace(free=run.MINIMUM_FREE_BYTES - 1)
    )
    monkeypatch.setattr(
        run, "auth_preflight", lambda *a, **k: pytest.fail("No auth maintenance allowed")
    )
    with pytest.raises(OSError, match="30 GiB"):
        run.run(
            tmp_path / "frozen/protocol.json",
            tmp_path / "full-disk",
            renderer_factory=lambda *a: pytest.fail("No GPU allowed"),
        )
    summary = json.loads((tmp_path / "full-disk/batch.json").read_text())
    assert len(summary["assignments"]) == 120
    assert all(row["status"] == "not_run" for row in summary["assignments"])


def test_round_robin_site_and_condition_balance():
    rows = run.assignments()
    for offset in range(0, len(run.ORDER), 4):
        assert tuple(r[0] for r in run.ORDER[offset : offset + 4]) == run.MODELS
    events = [row for row in rows if row["condition"] != "control"]
    assert Counter(row["site_id"] for row in events) == {"early": 16, "middle": 16}
    assert all(row["attempt"] == 3 for row in events)
    assert all(site == "early" for _, _, _, site in run.ORDER[:4])
    for model in run.MODELS:
        own = [row for row in events if row["model"] == model]
        assert Counter(row["site_id"] for row in own) == {"early": 4, "middle": 4}


@pytest.mark.parametrize(
    "reached,problem,continues", [(1, False, False), (2, False, True), (4, True, False)]
)
def test_checkpoint_requires_active_site_reach_and_no_infrastructure(
    tmp_path, reached, problem, continues
):
    scene = tmp_path / "scene.json"
    scene.write_text(
        json.dumps(
            dict(
                design=dict(
                    path=[[0, 0, 0], [30, 0, 0]],
                    event_sites=[dict(id="early", crossing_station_m=20)],
                )
            )
        )
    )
    sessions = []
    for index, model in enumerate(run.MODELS):
        session = f"{model}-01"
        folder = tmp_path / session / "attempt-3/episode"
        folder.mkdir(parents=True)
        ticks = [
            dict(kind="sim_tick", sim_time_s=t, map_pose=[x, 0, 0], actor_onsets={"hazard": 0})
            for t, x in [(1, 20 if index < reached else 19), (3, 21)]
        ]
        (folder / "events.jsonl").write_text("\n".join(map(json.dumps, ticks)))
        sessions.append(
            dict(
                session=session,
                skipped=None,
                attempts=[
                    dict(
                        status="infrastructure_failure" if problem and index == 0 else "complete",
                        assignment=dict(
                            attempt=3,
                            id=session + "-attempt-3",
                            scene_key="early-child_crossing",
                            site_id="early",
                        ),
                        episode=dict(first_cause=dict(evidence=dict(sim_time_s=2))),
                    )
                ],
            )
        )
    report = run.checkpoint(sessions, dict(scenes={"early-child_crossing": str(scene)}), tmp_path)
    assert report["event_sites_reached"] == reached
    assert report["continue_automatically"] is continues
    assert all(row["maximum_through_physical_stop_m"] == 21 for row in report["rows"])
