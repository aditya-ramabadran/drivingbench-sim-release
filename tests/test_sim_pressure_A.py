"""Prospective A pressure allocation and stopped-handback eligibility checks."""

import copy
import json
import math
from collections import Counter
from types import SimpleNamespace

import pytest
from drivingbench.sim.course import Course
from drivingbench.sim.studies import concurrent, naturalistic_run, pressure, pressure_run, tidy_cd


def scenes():
    return {
        key: f"/{key}.json"
        for key in (
            "central-legal-left",
            "restrictive-left",
            "central-legal-right",
            "restrictive-right",
        )
    }


def test_A_pressure_reuses_neutral_A_contract_with_explicit_balanced_selection():
    paths = scenes()
    arms = ["neutral", "order", "escalation"]  # Test fixture, never a selected study.
    frozen = pressure_run.a_protocol(paths, arms=arms, repeats=2)
    base = naturalistic_run._base("A_persistent", paths)
    for key, value in base.items():
        if key not in {"protocol", "assignments"}:
            assert frozen[key] == value
    assert frozen["protocol"] == pressure.A_PROTOCOL
    rows = frozen["assignments"]
    assert len(rows) == len({row["id"] for row in rows}) == 96
    assert frozen["maximum_operator_messages"] == sum(len(r["pressure_prompts"]) for r in rows)
    assert frozen["maximum_drive_turns"] == len(rows) + frozen["maximum_operator_messages"]
    for model in pressure.MODELS:
        for arm in arms:
            chosen = [r for r in rows if r["model"] == model and r["pressure_arm"] == arm]
            assert Counter(r["scene_key"] for r in chosen) == {key: 2 for key in paths}
    assert all(r["physical_mode"] == "free" and r["active_budget_s"] == 660 for r in rows)
    assert all(r["fresh_conversation"] and r["fresh_initial_conversation"] for r in rows)
    assert all(
        [r["model"] for r in rows[i : i + 4]] == list(pressure.MODELS)
        for i in range(0, len(rows), 4)
    )
    assert pressure.effective_protocol(frozen, rows[0])["prompt"] == base["prompt"]
    with pytest.raises(ValueError):
        pressure.effective_protocol(frozen, dict(rows[0], session="changed"))


@pytest.mark.parametrize(
    "arms,repeats",
    [
        (None, None),
        (["neutral"], 1),
        (["order", "prediction"], 1),
        (["neutral", "order", "order"], 1),
        (["neutral", "unknown"], 1),
        (["neutral", "order"], True),
        (["neutral", "order"], 0),
    ],
)
def test_A_pressure_has_no_default_or_invalid_selection(arms, repeats):
    with pytest.raises(ValueError):
        pressure_run.a_protocol(scenes(), arms=arms, repeats=repeats)


def course_scene(heading):
    angle = math.radians(heading)

    def point(x, y):
        return [
            x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle),
        ]

    return dict(
        frame="pilot_ground",
        start=dict(x=0, y=0, heading_deg=heading),
        boundaries=dict(
            left=[point(-10, 10), point(60, 10)],
            right=[point(-10, -10), point(60, -10)],
        ),
        markers=[],
        finish=dict(polygon=[point(30, -5), point(38, -5), point(38, 5), point(30, 5)]),
        design=dict(path=[[0, 0, heading], [*point(38, 0), heading]]),
    ), point


@pytest.mark.parametrize("heading", [0, 90, 180, 230])
@pytest.mark.parametrize(
    "station,expected", [(20, True), (30 - 4.63 / 2, False), (29, False), (34, False), (45, False)]
)
def test_A_receiver_plane_is_conservative_for_rotations_and_reversed_car(
    heading, station, expected
):
    scene, point = course_scene(heading)
    course = Course(scene)
    pose = dict(map_pose=[*point(station, 0), heading], finish_reached=False)
    assert pressure_run.a_goal_geometry(scene, pose, course)[0] is expected
    pose["map_pose"][2] = heading + 180
    assert pressure_run.a_goal_geometry(scene, pose, course)[0] is expected
    pose["finish_reached"] = True
    assert pressure_run.a_goal_geometry(scene, pose, course)[0] is False


@pytest.mark.parametrize(
    "change",
    ["held", "standstill", "command", "sample", "hazard", "finish", "session", "deadline"],
)
def test_A_actual_handback_factory_rejects_unproven_physical_or_session_state(
    tmp_path, monkeypatch, change
):
    scene, point = course_scene(0)
    scene.update(objects=[dict(id="hazard")], scenario=dict(trajectories={}))
    row = pressure.a_assignments(scenes(), arms=["neutral", "escalation"], repeats=1)[16]
    pose = dict(sim_time_s=2, monotonic_at=10, map_pose=[*point(20, 0), 0], finish_reached=False)
    status = dict(sim_time_s=2, held=True, standstill=True, command=None)
    session, deadline = "original", 660
    if change in {"held", "standstill"}:
        status[change] = False
    elif change == "command":
        status["command"] = {"active": True}
    elif change == "sample":
        status["sim_time_s"] = 3
    elif change == "hazard":
        scene["objects"] = []
    elif change == "finish":
        pose["finish_reached"] = True
    elif change == "deadline":
        deadline = 11
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    native = SimpleNamespace(
        pose_snapshot=lambda: dict(pose),
        status=lambda: dict(status),
        end_reason=None,
        pending_end=None,
        error=None,
    )
    episode = SimpleNamespace(
        summary=lambda: dict(first_cause=None, infrastructure_failure=None),
        record=lambda *_a, **_k: None,
    )
    app = SimpleNamespace(state=SimpleNamespace(sim_native=native, sim_episode=episode))
    monkeypatch.setattr(pressure_run.time, "monotonic", lambda: 11)
    frozen = dict(protocol=pressure.A_PROTOCOL, scenes={row["scene_key"]: str(path)})
    callback = pressure_run.handback_factory(frozen, row)(None, app, tmp_path)
    if change == "session":
        assert callback(SimpleNamespace(session_id=session), deadline, 0) is not None
        session = "different"
        sent = 1
    else:
        sent = 0
    assert callback(SimpleNamespace(session_id=session), deadline, sent) is None
    record = json.loads((tmp_path / f"pressure-handback-{sent}.json").read_text())
    assert record["decision"]["eligible"] is False
    assert record["decision"]["original_drive_deadline"] == deadline


def test_A_pressure_identity_changes_with_prospective_selection_and_current_sources(monkeypatch):
    monkeypatch.setattr(pressure_run, "file_hash", lambda path: "hash:" + path)
    frozen = pressure_run.a_protocol(scenes(), arms=["neutral", "order"], repeats=1)
    frozen.update(sources={"source": "current"}, dependencies={"dependency": "current"})
    identity = pressure_run.admission_identity(frozen)
    assert identity["study_protocol"] == pressure.A_PROTOCOL
    assert identity["active_budget_s"] == 660 and identity["pilot"] == "A_persistent"
    for field, value in (
        ("pressure_arms", ["neutral", "prediction"]),
        ("pressure_repeats", 2),
        ("sources", {"source": "changed"}),
        ("dependencies", {"dependency": "changed"}),
    ):
        changed = copy.deepcopy(frozen)
        changed[field] = value
        assert pressure_run.admission_identity(changed) != identity
    assert concurrent._bindings(frozen) == (pressure, pressure_run)
    assert concurrent._active_budget(frozen) == 660
    assert concurrent._execution_policy(frozen)["pilot_stage"] is None


@pytest.mark.parametrize("pilot", ["C", "D"])
def test_A_pressure_routing_preserves_explicit_tidy_CD_guard(pilot):
    frozen = dict(
        protocol=naturalistic_run.PROTOCOL,
        pilot=pilot,
        assignments=[{}] * 16,
        time_limit_s=660,
    )
    with pytest.raises(ValueError, match="explicit authored tidy variant"):
        concurrent._bindings(frozen)
    frozen["course_variant"] = tidy_cd.VARIANT
    assert concurrent._bindings(frozen) == (concurrent.naturalistic_pilots, naturalistic_run)
