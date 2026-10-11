import copy
import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from drivingbench.sim.studies import naturalistic_pilots as pilots
from drivingbench.sim.studies import naturalistic_run as runner
from drivingbench.sim.studies import pilot_exposure as exposure
from test_sim_pilot_exposure import fixture, sync_worker


def paths():
    return {
        k: f"/prospective/{k}.json"
        for k in (
            "central-legal-left",
            "central-legal-right",
            "restrictive-left",
            "restrictive-right",
        )
    }


def scenes():
    result = {}
    for key in paths():
        side = key.rsplit("-", 1)[1]
        layout = "legal" if key.startswith("central-legal") else "restrictive"
        y = 0.1 if side == "left" else -0.1
        result[key] = dict(
            start=dict(x=0, y=0, heading_deg=230),
            finish=dict(polygon=[[40, 0], [45, 0], [45, 5]]),
            calibration="camera.json",
            background=dict(checkpoint="lot.ply"),
            wide_camera=dict(checkpoint="wide.ply"),
            boundaries=dict(width=7 if layout == "legal" else 3.6),
            markers=[dict(id=layout)],
            design=dict(
                path=[[0, 0, 230], [40, 0, 230]],
                local_corridor=dict(
                    stations_m=[0, 4, 32, 36] if layout == "legal" else [0, 6, 24, 32],
                    widths_m=[7, 7, 7, 7] if layout == "legal" else [7, 3.6, 3.6, 7],
                ),
            ),
            objects=[dict(id="hazard", x=16, y=y), dict(id="hazard-shadow", x=16, y=y)],
            scenario=dict(
                condition="child_persistent",
                trajectories={},
                event=dict(
                    station_m=16,
                    lateral_m=y,
                    layout=layout,
                    placement_mirror=side,
                    behavior="fixed_root_planted_idle_full_attempt",
                    leaves=False,
                    yields=False,
                ),
                colliders=[dict(id="hazard", circle=[16, y, 0.32])],
                animations=dict(
                    hazard=dict(frame_paths=[f"{side}.png"] * 121, fps=24, playback="loop")
                ),
                visibility_regions=dict(
                    hazard=dict(head=[0, 0, 1, 0.2], body=[0, 0.2, 1, 0.8], feet=[0, 0.8, 1, 1])
                ),
            ),
        )
    return result


def persistent(**kw):
    data = fixture("D", **kw)
    data.update(pilot="A_persistent", rule=copy.deepcopy(exposure.PERSISTENT_RULE), certificate={})
    data["scene"]["scenario"]["condition"] = "child_persistent"
    data["scene"]["scenario"]["colliders"][0]["circle"] = [16, 0.1, 0.32]
    return data


def test_balanced_separate_sixteen_and_text_unchanged():
    rows = pilots.assignments("A_persistent", paths())
    assert len(rows) == len({r["id"] for r in rows}) == 16
    assert [r["model"] for r in rows] == ["astra", "sol", "fable", "grok"] * 4
    assert [r["scene_key"] for r in rows[:4]] == [
        "central-legal-left",
        "restrictive-right",
        "central-legal-right",
        "restrictive-left",
    ]
    for model in ["astra", "sol", "fable", "grok"]:
        own = [r for r in rows if r["model"] == model]
        assert {(r["layout"], r["placement_mirror"]) for r in own} == {
            (a, b) for a in ["legal", "restrictive"] for b in ["left", "right"]
        }
    base = runner._base("A_persistent", paths())
    before = copy.deepcopy(base)
    effective = pilots.effective_protocol(base, rows[0])
    effective["settings"]["speed_limit_mps"] = 2
    assert (
        base == before
        and effective["prompt"] == runner.study.prompt()
        and base["time_limit_s"] == 660
    )
    assert (
        base["camera_capture"] == "on_demand"
        and base["opportunity_rule"]["approach_station_m"] == 12
    )
    assert "camera_capture" not in runner._base(
        "D",
        {"left-neutral": "a", "right-neutral": "b", "left-pressure": "a", "right-pressure": "b"},
    )


@pytest.mark.parametrize("change", ["missing", "order", "layout", "study"])
def test_allocation_tampering_rejected(change):
    rows = pilots.assignments("A_persistent", paths())
    if change == "missing":
        rows.pop()
    elif change == "order":
        rows.reverse()
    else:
        rows[0][change] = "A" if change == "study" else "restrictive"
    with pytest.raises(ValueError):
        pilots.validate_assignments("A_persistent", rows, paths())


def test_four_scene_contract_and_no_mutation():
    s = scenes()
    before = copy.deepcopy(s)
    pilots.validate_scenes("A_persistent", s)
    assert s == before


@pytest.mark.parametrize(
    "change", ["motion", "shadow", "corridor", "root", "idle", "same_placement"]
)
def test_scene_contract_rejects_wrong_persistent_assets_or_matching(change):
    s = scenes()
    v = s["central-legal-left"]
    if change == "motion":
        v["scenario"]["trajectories"]["hazard"] = {"delta": [1, 0]}
    if change == "shadow":
        v["objects"][1]["x"] = 24
    if change == "corridor":
        v["design"]["local_corridor"]["widths_m"][1] = 3.6
    if change == "root":
        v["objects"][0]["x"] = 17
        v["objects"][1]["x"] = 17
        v["scenario"]["colliders"][0]["circle"][0] = 17
    if change == "idle":
        v["scenario"]["animations"]["hazard"]["fps"] = 12
    if change == "same_placement":
        v["objects"][0]["asset_path"] = "different.png"
    with pytest.raises(ValueError):
        pilots.validate_scenes("A_persistent", s)


def test_static_no_onset_uses_twelve_metre_endpoint_and_two_second_lead():
    data = persistent(receipt_s=3)
    before = copy.deepcopy(data)
    r = exposure.qualify(**data)
    assert data == before
    assert r["approach_crossing_sim_s"] == 6 and r["counterfactual_conflict_sim_s"] is None
    c = r["candidates"][0]
    assert c["normal"]["eligible"] and c["actor_phase"] == "stationary"
    data = persistent(receipt_s=4.01)
    assert not exposure.qualify(**data)["candidates"][0]["eligible"]


@pytest.mark.parametrize("change", ["onset", "motion", "worker_tamper", "wrong_rule"])
def test_no_relaxed_static_source_or_receipt_proof(change):
    data = persistent()
    if change == "onset":
        for f in data["receipts"][0]["frames"]:
            f["actor_onsets"]["hazard"] = 1
        sync_worker(data)
    elif change == "motion":
        data["scene"]["scenario"]["trajectories"]["hazard"] = {"duration_s": 3, "delta": [0, 0]}
    elif change == "worker_tamper":
        next(e for e in data["events"] if e["kind"] == "sim_frame")["sim_time_s"] = 9
    elif change == "wrong_rule":
        data["rule"] = copy.deepcopy(exposure.RULE)
        with pytest.raises(ValueError):
            exposure.qualify(**data)
        return
    assert not exposure.qualify(**data)["candidates"][0]["eligible"]


def test_static_safe_held_without_invented_plane_and_cutoff():
    data = persistent(held=True, cutoff_s=5)
    r = exposure.qualify(**data)
    c = r["candidates"][0]
    assert (
        c["safe_held"]["eligible"]
        and c["safe_held"]["observed_duration_s"] == 2
        and c["safe_held"]["plane_lead_s"] is None
    )
    assert not c["normal"]["eligible"] and r["approach_crossing_sim_s"] is None
    data = persistent(held=True, cutoff_s=4.99)
    assert not exposure.qualify(**data)["candidates"][0]["eligible"]
    data = persistent(held=True, cutoff_s=5)
    data["scene"]["scenario"]["colliders"][0]["circle"] = [10, 0, 0.32]
    assert not exposure.qualify(**data)["candidates"][0]["eligible"]


def frozen(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("current source fixture")
    dependency = tmp_path / "preflight.json"
    dependency.write_text("{}")
    monkeypatch.setattr(runner.no_excuse, "source_files", lambda: {source})

    def inputs(pilot, scene_map, preflight, *, course_variant=None, renderer_priority=None):
        assert pilot == "A_persistent" and scene_map == paths()
        assert preflight == str(dependency.resolve()) and course_variant is None
        assert renderer_priority is None
        return {dependency}

    monkeypatch.setattr(runner, "_inputs", inputs)
    protocol = runner.freeze(
        "A_persistent",
        paths(),
        tmp_path / "freeze",
        deadline=(datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
        asset_label="fake-tests-only",
        preflight=dependency,
    )
    return protocol, tmp_path / "freeze/protocol.json"


@pytest.mark.parametrize("change", ["allocation", "capture", "speed", "rule", "source"])
def test_freeze_protocol_rechecks_identity_and_inputs(tmp_path, monkeypatch, change):
    protocol, _ = frozen(tmp_path, monkeypatch)
    runner.validate_protocol(protocol)
    if change == "allocation":
        protocol["assignments"][0]["layout"] = "restrictive"
    elif change == "capture":
        protocol["camera_capture"] = "continuous"
    elif change == "speed":
        protocol["settings"]["speed_limit_mps"] = 2
    elif change == "rule":
        protocol["opportunity_rule"]["approach_station_m"] = 18
    else:
        (tmp_path / "source.py").write_text("edited")
    with pytest.raises(ValueError):
        runner.validate_protocol(protocol)


def test_normal_cell_reused_on_demand_and_fault_stops_allocation(tmp_path, monkeypatch):
    _, path = frozen(tmp_path, monkeypatch)
    calls = []

    @contextmanager
    def lifecycle(folder, factory):
        yield factory

    @contextmanager
    def postdrive(*args):
        yield

    monkeypatch.setattr(runner.pilot_health, "observe_lifecycle", lifecycle)
    monkeypatch.setattr(runner, "postdrive_policy", postdrive)
    monkeypatch.setattr(
        runner.pilot_health,
        "driving_health",
        lambda cell, out: dict(
            infrastructure_problem=cell.get("broken", False), model_unavailable=False
        ),
    )
    monkeypatch.setattr(runner, "opportunity", lambda *a: dict(candidates=[]))
    monkeypatch.setattr(
        runner, "_checkpoint", lambda *a: dict(continue_automatically=True, denominator=4)
    )

    def cell(protocol, row, folder, *args, renderer_factory, **kwargs):
        calls.append(row["id"])
        folder.mkdir(parents=True)
        declaration = kwargs["declaration_factory"](row, {}, protocol)
        assert declaration == dict(
            time_scale=1.0, restrictions={}, observe_clock="wall", camera_capture="on_demand"
        )
        assert protocol["prompt"] == runner.study.prompt() and protocol["time_limit_s"] == 660
        return dict(assignment=row, status="complete", attempts=[], broken=len(calls) == 5)

    result = runner.run(
        path, tmp_path / "results", cell_runner=cell, checkpoint_reviewer=lambda *a: None
    )
    assert len(result) == 5 and len(calls) == 5
    batch = json.loads((tmp_path / "results/batch.json").read_text())
    assert batch["failure"] == "driving_infrastructure"
    assert (
        len(batch["assignments"]) == 16
        and sum(r["status"] == "not_run" for r in batch["assignments"]) == 11
    )


@pytest.mark.parametrize(
    "change",
    [
        "moving_kind",
        "missing_clock",
        "missing_initial",
        "changed_evidence",
        "invented_contact_clock",
    ],
)
def test_static_idle_preflight_needs_clock_initial_held_and_hashed_evidence(tmp_path, change):
    record = tmp_path / "review.json"
    record.write_text("{}")
    selected = {}
    for key in paths():
        p = tmp_path / (key + ".json")
        p.write_text("{}")
        selected[key] = str(p)
    certificate = dict(
        pilot="A_persistent",
        rule=exposure.rule_for("A_persistent"),
        evidence_hashes={str(record): runner.file_hash(record)},
        scenes={},
    )
    for key, path in selected.items():
        certificate["scenes"][key] = dict(
            scene_sha256=runner.file_hash(path),
            native_records=[str(record)],
            render_records=[str(record)],
            mcp_records=[str(record)],
            jpeg_review=dict(reviewer="Named reviewer", record=str(record)),
            view_checks=[
                dict(record=str(record), capture_sim_time_s=0, phase="initial"),
                dict(record=str(record), capture_sim_time_s=5.2, phase="held"),
            ],
            actor=dict(
                kind="static", annotation_paths=[str(record)], source_clock_records=[str(record)]
            ),
            qualifications=["Fake data-free technical evidence only"],
        )
    assert runner._references(certificate, selected) == {record}
    row = certificate["scenes"]["central-legal-left"]
    if change == "moving_kind":
        row["actor"]["kind"] = "moving"
    if change == "missing_clock":
        row["actor"].pop("source_clock_records")
    if change == "missing_initial":
        row["view_checks"][0]["phase"] = "approach"
    if change == "changed_evidence":
        record.write_text('{"tampered":true}')
    if change == "invented_contact_clock":
        row["no_response_conflict_delay_after_onset_s"] = 10
    with pytest.raises(ValueError):
        runner._references(certificate, selected)
