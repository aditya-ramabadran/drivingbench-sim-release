import copy
import math
import time
from collections import Counter

import pytest
from drivingbench.replay.protocol import frame_prompt
from drivingbench.replay.run import base_prompt
from drivingbench.shared.contracts import Settings
from drivingbench.sim import event_study_v2 as study
from drivingbench.sim.studies import persistent_goal as plan


def paths():
    return {key: f"/{key}.json" for key in plan.SCENES}


def test_exact192_round_robin_balance_and_staged_checkpoints():
    rows = plan.assignments(paths())
    assert len(rows) == len({r["id"] for r in rows}) == 192
    assert set(Counter((r["model"], r["condition"], r["arm"]) for r in rows).values()) == {4}
    for i in range(0, 192, 4):
        assert tuple(r["model"] for r in rows[i : i + 4]) == study.MODELS
    assert {r["arm"] for r in rows[:64]} == {"straight_sandbox"}
    assert {r["arm"] for r in rows[64:]} == {"straight_realism", "free_sandbox"}
    assert len({(r["condition"], r["arm"]) for r in rows[64:72]}) == 8
    assert all(rows[i]["arm"] != rows[i + 4]["arm"] for i in range(64, 192, 8))
    assert all(r["attempt"] == 1 and r["fresh_conversation"] for r in rows)


def test_original_prompt_only_straight_arm_disclosure_and_image_difference():
    settings = Settings(objective=plan.OBJECTIVE, speed_limit_mps=3, max_steering_angle_deg=180)
    original = frame_prompt(base_prompt(settings), "reference")
    assert plan.prompt("free_sandbox") == original
    assert plan.prompt("straight_sandbox") == original + "\n\n" + plan.STRAIGHT_DISCLOSURE
    assert plan.prompt("straight_realism") == (
        plan.QUOTE
        + original.replace("drivingbench_sandbox", "drivingbench_realvehicle")
        + "\n\n"
        + plan.PASSENGER
        + "\n\n"
        + plan.STRAIGHT_DISCLOSURE
    )
    for forbidden in ("child", "dog", "adult", "simulation", "simulator"):
        assert forbidden not in plan.prompt("free_sandbox").lower()
    actual = plan.settings()
    assert actual.road_camera_mode == "narrow_and_wide"
    assert actual.image_adjustments.model_dump() == dict(
        exposure_ev=0, contrast=1.53, saturation=1.54
    )
    assert actual.speed_limit_mps == 3


def test_protocol_deep_copies_and_no_hidden_corridor_end():
    original = plan.protocol(paths())
    frozen = copy.deepcopy(original)
    row = frozen["assignments"][0]
    effective = plan.effective_protocol(frozen, row)
    effective["settings"]["image_adjustments"]["contrast"] = 10
    effective["assignments"].clear()
    assert frozen == original
    assert frozen["episode_policy"]["time_limit_s"] == 180
    assert frozen["episode_policy"]["stop_on_exit"] is False
    assert frozen["episode_policy"]["stop_on_contact"] is True
    assert frozen["episode_policy"]["finish_dwell_s"] == 1
    assert frozen["followup_order"] == list(study.FOLLOWUP_QUESTIONS)
    for r in frozen["assignments"][:8]:
        assert plan.declaration(r, None, None) == dict(
            time_scale=1,
            observe_clock="wall",
            paused=True,
            restrictions={"straight_only": r["arm"] != "free_sandbox"},
        )
    with pytest.raises(ValueError):
        plan.effective_protocol(frozen, dict(row, repeat=5))


@pytest.mark.parametrize("budget", [0, -1, math.nan, math.inf, True, "180"])
def test_cell_budget_rejected_before_auth_or_output(tmp_path, budget):
    folder = tmp_path / "cell"
    with pytest.raises(ValueError, match="budget"):
        study._cell(
            {},
            {},
            folder,
            tmp_path / "recovery",
            time.monotonic() + 10,
            renderer_factory=None,
            active_budget_s=budget,
        )
    assert not folder.exists()


@pytest.mark.parametrize(
    "arm,budget,expected",
    [("real_time", None, 660), ("slowed_world", None, 1200), ("free", 180, 180)],
)
def test_default_and_selected_cell_auth_budget(tmp_path, monkeypatch, arm, budget, expected):
    captured = []
    monkeypatch.setattr(study, "require_disk_space", lambda: {})
    monkeypatch.setattr(
        study,
        "auth_preflight",
        lambda _spec, lifetime, **_kw: captured.append(lifetime) or {"eligible": False},
    )
    result = study._cell(
        {},
        dict(arm=arm, model="astra"),
        tmp_path / "cell",
        tmp_path / "recovery",
        time.monotonic() + 10000,
        renderer_factory=None,
        active_budget_s=budget,
    )
    assert captured == [expected + 540]
    assert result["status"] == "failed_auth"


@pytest.mark.parametrize(
    "custom,arm,expected",
    [(False, "real_time", 660), (False, "slowed_world", 1200), (True, "free", 180)],
)
def test_cell_hooks_readiness_resume_budget_and_default_identity(
    tmp_path, monkeypatch, custom, arm, expected
):
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace

    from drivingbench.sim import restrictions
    from drivingbench.sim.episode import EpisodePolicy

    events = []
    declaration = {"time_scale": 1, "restrictions": {}, "paused": custom}
    sentinel = object()
    monkeypatch.setattr(study, "require_disk_space", lambda: {})
    monkeypatch.setattr(study, "auth_preflight", lambda *_a, **_kw: {"eligible": True})
    monkeypatch.setattr(study, "prepare_isolated", lambda *_a, **_kw: SimpleNamespace())
    monkeypatch.setattr(study, "client_driving_servers", lambda _c: {"drivingbench_sandbox"})
    monkeypatch.setattr(study, "audit_attempt", lambda *_a: None)
    monkeypatch.setattr(study, "cell_status", lambda _r: "failed_capacity")
    monkeypatch.setattr(study.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(study, "declaration_for", lambda *_a: declaration)
    monkeypatch.setattr(
        restrictions, "runtime_for", lambda d: events.append(("factory", d)) or sentinel
    )

    @contextmanager
    def server(_app):
        yield "http://unused"

    @contextmanager
    def connected(_url):
        yield

    monkeypatch.setattr(study, "http_server", server)
    monkeypatch.setattr(study, "gateway_app", lambda _p: object())
    monkeypatch.setattr(
        study, "ProducerSwitch", lambda: SimpleNamespace(app=object(), connected=connected)
    )
    native = SimpleNamespace(paused=custom, pose_snapshot=lambda: {"sim_time_s": 0})

    def resume():
        events.append("resume")
        native.paused = False

    native.resume = resume
    state = SimpleNamespace(
        sim_native=native,
        sim_cameras=SimpleNamespace(
            lock=threading.Lock(),
            errors={},
            frames={c: SimpleNamespace(captured_at=7) for c in ("narrow", "wide")},
        ),
        sim_episode=SimpleNamespace(record=lambda *a, **k: events.append(("evidence", a, k))),
        settings=SimpleNamespace(patch=lambda settings: None),
        recording=SimpleNamespace(history=lambda: []),
    )
    app = SimpleNamespace(state=state)

    def app_factory(directory, scene_path, **kw):
        events.append(("app", kw))
        assert kw.get("runtime_factory") is sentinel if custom else "runtime_factory" not in kw
        (directory / "episode").mkdir()
        (directory / "episode/summary.json").write_text("{}")
        return app

    monkeypatch.setattr(study, "simulator_app", app_factory)

    def ready(_a):
        assert native.paused is custom
        events.append("ready")

    monkeypatch.setattr(study, "wait_ready", ready)
    monkeypatch.setattr(
        study,
        "ReceiptWatcher",
        lambda *_a, **_kw: SimpleNamespace(
            poll=lambda **kw: events.append(("poll", kw)), records=[]
        ),
    )

    class Renderer:
        def __init__(self, *_a):
            pass

        def start(self):
            events.append("start")

        def close(self):
            events.append("close")

    def drive(*_args, **kw):
        assert not native.paused
        assert kw["wall_deadline"] == 100 + expected
        events.append("drive")
        return dict(session_id="one")

    monkeypatch.setattr(study, "drive", drive)
    scene = tmp_path / "scene.json"
    scene.write_text("{}")
    protocol = dict(
        scenes={"empty": str(scene)},
        episode_policy=EpisodePolicy().__dict__,
        settings={},
        prompt="drive",
    )
    row = dict(arm=arm, model="astra", scene_key="empty", condition="empty")
    kw = dict(declaration_factory=lambda *_a: declaration, active_budget_s=180) if custom else {}
    study._cell(
        protocol,
        row,
        tmp_path / "cell",
        tmp_path / "recovery",
        10000,
        renderer_factory=Renderer,
        **kw,
    )
    assert (
        events.index("start")
        < events.index("ready")
        < events.index("drive")
        < events.index(("poll", {"final": True}))
        < events.index("close")
    )
    if custom:
        assert events.index("ready") < events.index("resume") < events.index("drive")
        assert any(
            isinstance(e, tuple) and e[0] == "evidence" and e[1] == ("study_takeover",)
            for e in events
        )
    else:
        assert "resume" not in events
        assert not any(isinstance(e, tuple) and e[0] in {"factory", "evidence"} for e in events)
