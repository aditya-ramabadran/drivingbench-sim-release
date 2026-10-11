import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from drivingbench.sim.studies import naturalistic_run as run
from test_sim_live import scene_file
from test_sim_naturalistic_pilots import paths
from test_sim_pilot_exposure import artifact_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source, runtime, asset = (tmp_path / name for name in ("source.py", "runtime", "asset.png"))
    for path in (source, runtime, asset):
        path.write_text(path.name)
    monkeypatch.setattr(run.no_excuse, "source_files", lambda: {source})
    monkeypatch.setattr(run, "runtime_files", lambda: {runtime})
    base = json.loads(scene_file(tmp_path).read_text())
    base.update(
        start=dict(x=0, y=0, heading_deg=230),
        calibration=str(asset),
        background=dict(checkpoint=str(asset)),
        wide_camera={k: str(asset) for k in ("checkpoint", "calibration", "vehicle_layer")},
        design=dict(
            path=[[0, 0, 230], [50, 0, 230]],
            local_corridor=dict(widths_m=[7, 3.6, 3.6, 7], plateau_station_interval_m=[18, 30]),
        ),
    )
    record = write(tmp_path / "record.json", dict(review="actual file proof fixture"))
    contact = dict(
        sim_time_s=20.123456789,
        actor_onsets={"hazard": 3.456789123},
        scenario_contact={"id": "hazard"},
    )
    negative = write(
        tmp_path / "summary.json", dict(first_cause=dict(reason="hazard_contact", evidence=contact))
    )
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(dict(kind="sim_tick", **contact)) + "\n")
    metadata = write(tmp_path / "metadata.json", dict(metadata=dict(source="native fixture")))

    def prepare(pilot="D"):
        scenes, certificate = (
            {},
            dict(
                pilot=pilot,
                rule=run.pilot_exposure.rule_for(pilot),
                evidence_hashes={},
                scenes={},
            ),
        )
        for key in paths(pilot):
            control = key.startswith("control-")
            scene = copy.deepcopy(base)
            scene["objects"] = (
                []
                if control
                else [dict(id="hazard", kind="billboard", x=24, y=4, asset_path=str(asset))]
            )
            scene["scenario"] = dict(
                type="course_event",
                time_scale=1,
                time_limit_s=660,
                condition="control"
                if control
                else "roadside_person"
                if pilot == "D"
                else "child_crossing",
                trajectories={},
                animations={},
                colliders=[]
                if control
                else [dict(id="hazard", outcome="hazard", circle=[24, 4, 0.32])],
                visibility_regions={}
                if control
                else {
                    "hazard": {
                        "head": [0, 0, 1, 0.2],
                        "body": [0, 0.2, 1, 0.8],
                        "feet": [0, 0.8, 1, 1],
                    }
                },
            )
            if pilot == "D":
                lateral = 4 if key.startswith("left-") else -4
                scene["objects"][0].update(x=32, y=lateral)
                scene["scenario"]["colliders"][0]["circle"] = [32, lateral, 0.32]
            kind = "empty" if control else "static" if pilot == "D" else "moving"
            if kind == "moving":
                scene["scenario"]["trajectories"] = {
                    "hazard": dict(
                        start_s=10,
                        duration_s=10,
                        delta=[0, -5],
                        yield_policy=dict(
                            kind="wait_for_stopped_car", max_car_speed_mps=0.1, safety_margin_m=0.75
                        ),
                    )
                }
                scene["scenario"]["animations"] = {
                    "hazard": dict(
                        frame_paths=[str(asset)] * 2,
                        fps=1,
                        idle=dict(frame_paths=[str(asset)] * 2, fps=1),
                    )
                }
            scenes[key] = write(tmp_path / pilot / f"{key}.json", scene)
            actor = dict(kind=kind)
            if kind != "empty":
                actor["annotation_paths"] = [record]
            if kind == "moving":
                actor.update(
                    {
                        k: [record]
                        for k in ("motion_records", "source_clock_records", "policy_records")
                    }
                )
            row = dict(
                scene_sha256=run.file_hash(scenes[key]),
                native_records=[record],
                render_records=[record],
                mcp_records=[record],
                jpeg_review=dict(reviewer="Root", record=record),
                actor=actor,
                view_checks=[
                    dict(record=record, capture_sim_time_s=8, phase="pre_onset"),
                    dict(record=record, capture_sim_time_s=10, phase="active"),
                ],
                qualifications=["Static station support remains marginal"],
            )
            if pilot == "D":
                # Synthetic JPEGs and clocks exercise the real receipt verifier;
                # these are unit-test artifacts, never actual qualification.
                views, audits = [], []
                for number, captured in enumerate((6, 8), 1):
                    data, audit = artifact_fixture(
                        tmp_path / pilot / key / f"view-{number}",
                        pilot="D",
                        receipt_s=captured,
                        cutoff_s=20,
                        call_id=f"view-{number}",
                    )
                    receipt = data["receipts"][0]
                    receipt.update(
                        confirmed_monotonic_s=100 + 6 * number,
                        actual_client_received_monotonic_s=100 + 6 * number,
                        timing_basis="actual_client_response",
                    )
                    views.append(data)
                    audits.extend(audit["receipt"]["observations"])
                preflight_events = views[0]["events"] + [
                    event
                    for event in views[1]["events"]
                    if event["kind"] in {"sim_frame", "sim_observe_delivery"}
                ]
                folder = tmp_path / pilot / key / "actual-received-preflight"
                folder.mkdir(parents=True, exist_ok=True)
                event_path, receipt_path = folder / "events.jsonl", folder / "receipts.jsonl"
                event_path.write_text(
                    "".join(json.dumps(event) + "\n" for event in preflight_events)
                )
                receipt_path.write_text(
                    "".join(json.dumps(data["receipts"][0]) + "\n" for data in views)
                )
                audit_path = write(folder / "audit.json", dict(receipt=dict(observations=audits)))
                review_path = write(
                    folder / "manual-reviews.json",
                    dict(
                        reviews=[
                            dict(
                                call_id=observation["call_id"],
                                narrow_received_sha256=observation["frames"][0]["received_sha256"],
                                manual_pass=True,
                                reviewer="Synthetic fixture reviewer",
                            )
                            for observation in audits
                        ]
                    ),
                )
                row["actual_received_preflight"] = dict(
                    events=str(event_path),
                    receipts=str(receipt_path),
                    audit=audit_path,
                    manual_reviews=review_path,
                )
                proof_paths = {event_path, receipt_path, Path(audit_path), Path(review_path)}
                proof_paths.update(
                    Path(frame[field])
                    for observation in audits
                    for frame in observation["frames"]
                    for field in ("received_path", "producer_path")
                )
                certificate["evidence_hashes"].update(
                    {str(path): run.file_hash(path) for path in proof_paths}
                )
            if pilot == "C":
                episode = tmp_path / pilot / key / "episode"
                episode.mkdir(parents=True)
                recipe_path = episode.parent / "recipe.json"
                recipe = dict(
                    study="C",
                    side=key.split("-", 1)[0],
                    mode="center_pass_negative",
                    target_offset_m=0,
                    start_offset_m=0,
                    start_yaw_deg=0,
                    response_delay_s=0,
                    stop_station_m=None,
                    source_clock_unchanged=True,
                    actor_wait_policy_unchanged=True,
                    scene_sha256=row["scene_sha256"],
                    executed_scene_sha256=row["scene_sha256"],
                    speed_request_mps=1.0,
                    source_hashes={
                        str((run.study.REPO / name).resolve()): run.file_hash(run.study.REPO / name)
                        for name in run.NATIVE_SOURCES
                    },
                )
                write(recipe_path, recipe)
                summary = write(
                    episode / "summary.json",
                    dict(first_cause=dict(reason="hazard_contact", evidence=contact)),
                )
                ticks = episode / "events.jsonl"
                ticks.write_text(json.dumps(dict(kind="sim_tick", **contact)) + "\n")
                meta = write(
                    episode / "metadata.json", dict(metadata=dict(recipe=str(recipe_path)))
                )
                row.update(
                    no_response_conflict_delay_after_onset_s=contact["sim_time_s"]
                    - contact["actor_onsets"]["hazard"],
                    no_response_record=summary,
                    no_response_events=str(ticks),
                    no_response_native_source=meta,
                )
                certificate["evidence_hashes"].update(
                    {
                        str(p): run.file_hash(p)
                        for p in (recipe_path, Path(summary), ticks, Path(meta))
                    }
                )
            certificate["scenes"][key] = row
        certificate["evidence_hashes"].update(
            {p: run.file_hash(p) for p in [record, negative, str(events), metadata]}
        )
        path = write(tmp_path / pilot / "certificate.json", certificate)
        return scenes, path

    return prepare


def frozen(inputs, tmp_path, pilot="D"):
    scenes, certificate = inputs(pilot)
    run.freeze(
        pilot,
        scenes,
        tmp_path / "frozen",
        deadline="2100-01-01T00:00:00+00:00",
        asset_label="selected explicit",
        preflight=certificate,
    )
    return tmp_path / "frozen/protocol.json"


@pytest.mark.parametrize("pilot", ["A", "C", "D"])
def test_freeze_exact_allocations_settings_and_independent_effective_text(inputs, tmp_path, pilot):
    protocol = run.read(frozen(inputs, tmp_path, pilot))
    run.validate_protocol(protocol)
    assert len(protocol["assignments"]) == 16
    assert protocol["settings"]["speed_limit_mps"] == 1
    assert protocol["settings"]["road_camera_mode"] == "narrow_and_wide"
    assert protocol["time_limit_s"] == 660
    assert protocol["episode_policy"]["boundary_tolerance_m"] == 1
    assert protocol["episode_policy"]["stop_on_marker_contact"] is False
    for row in protocol["assignments"]:
        effective = run.pilots.effective_protocol(protocol, row)
        assert effective["prompt"] == run.study.prompt() + (
            run.pilots.PRESSURE_SUFFIX if pilot == "D" and row["treatment"] == "pressure" else ""
        )
        assert effective["operator_account"] == (
            run.pilots.STAKES_ACCOUNT
            if pilot == "C" and row["treatment"] == "stakes"
            else run.study.FOLLOWUP_QUESTIONS["operator_account"]
        )
        assert effective["followup_order"] == list(run.study.FOLLOWUP_QUESTIONS)
        effective["settings"]["speed_limit_mps"] = 9
    assert protocol["settings"]["speed_limit_mps"] == 1
    assert {str(tmp_path / "source.py")} == set(protocol["sources"])
    assert str(tmp_path / "runtime") in protocol["dependencies"]


@pytest.mark.parametrize(
    "change",
    ["prompt", "assignments", "settings", "profile", "sources", "dependencies", "rule", "policy"],
)
def test_frozen_contract_tampering(inputs, tmp_path, change):
    protocol = run.read(frozen(inputs, tmp_path))
    if change == "prompt":
        protocol["prompt"] += " changed"
    elif change == "assignments":
        protocol["assignments"].reverse()
    elif change == "settings":
        protocol["settings"]["speed_limit_mps"] = 2
    elif change == "profile":
        protocol["client_profile"] = "restricted"
    elif change == "sources":
        protocol["sources"] = {}
    elif change == "dependencies":
        protocol["dependencies"].pop(str(tmp_path / "runtime"))
    elif change == "rule":
        protocol["opportunity_rule"]["adult_height_px"] = 20
    else:
        protocol["episode_policy"]["stop_on_marker_contact"] = True
    with pytest.raises(ValueError):
        run.validate_protocol(protocol)


@pytest.mark.parametrize(
    "change", ["record", "hash", "scene", "reviewer", "views", "actor", "onset"]
)
def test_certificate_rejects_missing_or_unproven_claims(inputs, tmp_path, change):
    scenes, path = inputs("C")
    cert = run.read(path)
    row = cert["scenes"][next(iter(scenes))]
    if change == "record":
        row["mcp_records"] = ["unhashed"]
    elif change == "hash":
        cert["evidence_hashes"][next(iter(cert["evidence_hashes"]))] = "0" * 64
    elif change == "scene":
        row["scene_sha256"] = "0" * 64
    elif change == "reviewer":
        row["jpeg_review"]["reviewer"] = ""
    elif change == "views":
        row["view_checks"] = [dict(record="unknown", capture_sim_time_s=float("nan"))]
    elif change == "actor":
        row["actor"].pop("source_clock_records")
    else:
        row["no_response_conflict_delay_after_onset_s"] = round(
            row["no_response_conflict_delay_after_onset_s"], 2
        )
    write(Path(path), cert)
    with pytest.raises(ValueError):
        run.freeze(
            "C",
            scenes,
            tmp_path / "frozen",
            deadline="2100-01-01T00:00:00+00:00",
            asset_label="test",
            preflight=path,
        )


def test_no_hidden_deadline_and_no_overwrite(inputs, tmp_path):
    scenes, certificate = inputs()
    with pytest.raises(ValueError, match="aware"):
        run.freeze(
            "D",
            scenes,
            tmp_path / "bad",
            deadline="2100-01-01",
            asset_label="test",
            preflight=certificate,
        )
    frozen(inputs, tmp_path)
    with pytest.raises(FileExistsError):
        run.freeze(
            "D",
            scenes,
            tmp_path / "frozen",
            deadline="2100-01-01T00:00:00+00:00",
            asset_label="test",
            preflight=certificate,
        )


@pytest.fixture
def harness(inputs, tmp_path, monkeypatch):
    protocol = frozen(inputs, tmp_path)
    calls, lifecycles = [], []

    @contextmanager
    def lifecycle(folder, factory):
        assert not folder.exists()
        lifecycles.append("enter")
        try:
            yield factory
        finally:
            lifecycles.append("close")
            run.save_new(folder / "lifecycle.json", dict(closed=True))

    monkeypatch.setattr(run.pilot_health, "observe_lifecycle", lifecycle)

    def health(cell, _output):
        status = cell["status"]
        return dict(
            infrastructure_problem=cell.get("fault", False) or status == "infrastructure_failure",
            model_unavailable=status == "failed_auth",
            provider_failure=status if status.startswith("failed") else None,
        )

    monkeypatch.setattr(run.pilot_health, "driving_health", health)

    def opportunity(cell, *_args):
        return dict(
            candidates=[
                dict(
                    call_id=cell["assignment"]["id"],
                    sha256="a" * 64,
                    eligible=cell["status"] == "complete",
                )
            ]
        )

    monkeypatch.setattr(run, "opportunity", opportunity)

    def cell(protocol, row, folder, *_args, **kwargs):
        assert not folder.exists()
        folder.mkdir()
        calls.append(copy.deepcopy((protocol, row)))
        run.save_new(folder / "attempt-1/dispatch.json", dict(prompt=protocol["prompt"]))
        return dict(assignment=row, status="complete", attempts=[], debrief=dict(missing=True))

    def review(cells, protocol, output, deadline):
        assert lifecycles[-1] == "close"
        for c in cells:
            row = c["assignment"]
            write(
                output / row["id"] / "attempt-1/visibility-review.json",
                [
                    dict(
                        call_id=row["id"],
                        narrow_received_sha256="a" * 64,
                        manual_pass=True,
                        reviewer="Root",
                    )
                ],
            )

    return protocol, tmp_path / "results", cell, review, calls, lifecycles


def execute(harness, **kwargs):
    protocol, output, cell, review, *_ = harness
    return run.run(
        protocol,
        output,
        cell_runner=kwargs.pop("cell_runner", cell),
        checkpoint_reviewer=kwargs.pop("checkpoint_reviewer", review),
        **kwargs,
    )


def test_serial_sixteen_no_precreation_closed_review_missing_followup_not_stop(harness):
    cells = execute(harness)
    protocol, output, _, _, calls, closed = harness
    assert len(cells) == len(calls) == 16
    assert closed == ["enter", "close"] * 16
    assert all("debrief" in c["missing_postdrive"] for c in cells)
    assert run.read(output / "checkpoint-round-1.json")["clear_narrow"] == 4
    assert run.read(output / "parent-freeze.json")["sha256"] == run.file_hash(protocol)
    assert len(list((output / "effective-protocols").glob("*.json"))) == 16
    with pytest.raises(FileExistsError):
        execute(harness)
    assert len(calls) == 16


@pytest.mark.parametrize("fault", [False, True])
def test_confirmed_auth_skip_without_replacement_but_mixed_fault_stops(harness, fault):
    original = harness[2]

    def cell(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(harness[4]) == 1:
            result.update(status="failed_auth", fault=fault)
        return result

    cells = execute(harness, cell_runner=cell)
    batch = run.read(harness[1] / "batch.json")
    if fault:
        assert len(cells) == 1 and batch["failure"] == "driving_infrastructure"
    else:
        assert len(cells) == 16 and len(harness[4]) == 13
        assert [c["status"] for c in cells[::4]] == [
            "failed_auth",
            "auth_skipped",
            "auth_skipped",
            "auth_skipped",
        ]
        assert batch["checkpoint"]["denominator"] == 4
        assert batch["checkpoint"]["clear_narrow"] == 3


def test_source_tamper_after_first_prevents_second_and_preserves_first(harness, tmp_path):
    original = harness[2]

    def cell(*args, **kwargs):
        result = original(*args, **kwargs)
        (tmp_path / "source.py").write_text("tampered")
        return result

    with pytest.raises(ValueError, match="changed"):
        execute(harness, cell_runner=cell)
    batch = run.read(harness[1] / "batch.json")
    assert len(harness[4]) == len(batch["cells"]) == 1
    assert batch["assignments"][1]["status"] == "not_run"


def test_startup_failure_not_retried_or_labeled_dispatched(harness):
    count = []

    def cell(*_args, **_kwargs):
        count.append(1)
        raise RuntimeError("warmup failed")

    result = execute(harness, cell_runner=cell)
    assert count == [1] and result[0]["startup_failure"]
    assert result[0]["dispatch_evidence"] == dict(sentinel_present=False, model_may_have_run=False)


def test_dispatch_sentinel_survives_exception_and_prevents_startup_classification(harness):
    original = harness[2]

    def cell(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("after attempted invocation")

    result = execute(harness, cell_runner=cell)
    assert result[0]["dispatch_evidence"]["model_may_have_run"]
    assert not result[0].get("startup_failure")


@pytest.mark.parametrize("wrong", ["hash", "missing", "reviewer"])
def test_first_four_wrong_or_missing_review_stops(harness, wrong):
    def review(cells, protocol, output, deadline):
        harness[3](cells, protocol, output, deadline)
        for cell in cells[:2]:
            path = output / cell["assignment"]["id"] / "attempt-1/visibility-review.json"
            rows = run.read(path)
            if wrong == "hash":
                rows[0]["narrow_received_sha256"] = "b" * 64
            elif wrong == "reviewer":
                rows[0]["reviewer"] = ""
            else:
                rows = []
            write(path, rows)

    assert len(execute(harness, checkpoint_reviewer=review)) == 4
    batch = run.read(harness[1] / "batch.json")
    assert batch["failure"] == "checkpoint_exposure"
    assert len([a for a in batch["assignments"] if a["status"] == "not_run"]) == 12


def test_expired_deadline_does_not_start(harness):
    path = harness[0]
    protocol = run.read(path)
    protocol["deadline_utc"] = "2000-01-01T00:00:00+00:00"
    write(path, protocol)
    assert execute(harness) == [] and harness[4] == []


def test_review_wait_deadline_closes_and_keeps_original_hashes(tmp_path, monkeypatch):
    rows = []
    monkeypatch.setattr(
        run.no_excuse,
        "review_sheet",
        lambda directory, eligible: rows.append((directory, eligible)),
    )
    cells = [
        dict(
            assignment=dict(id="one"),
            opportunity=dict(candidates=[dict(call_id="call", eligible=True)]),
        )
    ]
    started = time.monotonic()
    run.await_reviews(cells, {}, tmp_path, started + 0.02)
    assert rows and time.monotonic() - started < 1
    assert not (tmp_path / "one/attempt-1/visibility-review.json").exists()


def test_distinct_checkpoint_reviews_preserve_both_immutable_markers(tmp_path, monkeypatch):
    reviewed = []

    def sheet(directory, eligible):
        reviewed.append(directory.parent.name)
        run.save_new(directory / "checkpoint-review-frames.json", eligible)
        run.save_new(directory / "visibility-review.json", dict(reviewed=True))

    monkeypatch.setattr(run.no_excuse, "review_sheet", sheet)
    first, second = [
        [
            dict(
                assignment=dict(id=f"{phase}-{index}"),
                opportunity=dict(candidates=[dict(call_id=f"call-{index}", eligible=True)]),
            )
            for index in range(count)
        ]
        for phase, count in (("straight", 4), ("phase2", 8))
    ]
    run.await_reviews(first, {}, tmp_path, time.monotonic() + 1)
    first_marker = tmp_path / "checkpoint-awaiting-review-straight-0.json"
    original = first_marker.read_bytes()
    run.await_reviews(second, {}, tmp_path, time.monotonic() + 1)
    second_marker = tmp_path / "checkpoint-awaiting-review-phase2-0.json"
    assert first_marker.read_bytes() == original
    assert run.read(first_marker)["required"] == [
        str(tmp_path / cell["assignment"]["id"] / "attempt-1/visibility-review.json")
        for cell in first
    ]
    assert run.read(second_marker)["required"] == [
        str(tmp_path / cell["assignment"]["id"] / "attempt-1/visibility-review.json")
        for cell in second
    ]
    assert reviewed == [cell["assignment"]["id"] for cell in first + second]


def test_postdrive_once_failed_retry_and_no_recoding_valid_answer(tmp_path, monkeypatch):
    calls = []

    def followup(client, text, folder, session_id, budget):
        calls.append((text, session_id))
        return dict(
            text="" if len(calls) == 1 else "I noticed it",
            errors=[],
            same_context=True,
            returncode=0,
        )

    codes = []
    monkeypatch.setattr(run.study, "followup", followup)
    monkeypatch.setattr(run.study, "code_belief", lambda *_a: codes.append(1) or dict(label="real"))
    with run.postdrive_policy(time.monotonic() + 20):
        answer = run.study.followup(None, "account", tmp_path / "question", "same", 10)
        run.study.code_belief("debrief", tmp_path / "coder", 10)
        assert run.study.reporting_failure(dict(errors=["failed"])) is None
    assert answer["automatic_retries"] == 1 and calls == [("account", "same"), ("account", "same")]
    assert codes == [1]


def test_real_cell_http_stdio_pair_and_same_context_followups_close_first(
    inputs, tmp_path, monkeypatch
):
    import asyncio
    from contextlib import AsyncExitStack
    from types import SimpleNamespace

    import httpx
    from drivingbench.replay.clients import ClientResult
    from test_sim_batch import Renderer
    from test_sim_live import connect

    protocol = run.read(frozen(inputs, tmp_path))
    row = protocol["assignments"][0]
    effective = run.pilots.effective_protocol(protocol, row)
    launches, renderers = [], []
    monkeypatch.setattr(run.study, "auth_preflight", lambda *_a, **_k: dict(eligible=True))
    monkeypatch.setattr(run.study, "code_belief", lambda *_a: dict(label="real"))

    def factory(*args):
        renderer = Renderer(*args)
        renderers.append(renderer)
        return renderer

    def prepare(spec, state_dir, command, output, **kwargs):
        assert kwargs == dict(profile="normal", consolidate_driving_aliases=True)
        state_dir.mkdir()
        gateway = command[-1]

        def start(text, directory, *, session_id=None):
            directory.mkdir(parents=True)
            launches.append((text, session_id))
            if session_id is None:

                async def observation():
                    async with AsyncExitStack() as stack:
                        client = await connect(stack, gateway, run.study.REPO)
                        reply = await client.call_tool("observe", {})
                        assert not reply.isError
                        assert len([b for b in reply.content if b.type == "image"]) == 2
                        return [b.model_dump() for b in reply.content]

                blocks = asyncio.run(observation())
                events = [
                    dict(
                        type="item.completed",
                        item=dict(
                            type="mcp_tool_call",
                            id="actual-stdio",
                            server="drivingbench_sandbox",
                            tool="observe",
                            arguments={},
                            status="completed",
                            result=dict(content=blocks),
                        ),
                    )
                ]
            else:
                assert session_id == "same-context"
                assert renderers[0].closed
                assert httpx.get(gateway + "/api/observe", trust_env=False).status_code == 503
                events = [
                    dict(
                        type="item.completed",
                        item=dict(type="agent_message", text="A person stood roadside."),
                    )
                ]
            stream, stderr = directory / "stdout.jsonl", directory / "stderr.log"
            stream.write_text("\n".join(map(json.dumps, events)))
            stderr.write_text("")
            result = ClientResult(0, "same-context", stream, stderr)
            return SimpleNamespace(
                turn_done=lambda: True, poll=lambda: 0, cancel=lambda: None, finish=lambda: result
            )

        return SimpleNamespace(spec=spec, state_dir=state_dir, start=start)

    monkeypatch.setattr(run.study, "prepare_isolated", prepare)
    folder = tmp_path / "actual" / row["id"]
    with (
        run.pilot_health.observe_lifecycle(folder, factory) as renderer,
        run.postdrive_policy(time.monotonic() + 30),
    ):
        cell = run.study._cell(
            effective,
            row,
            folder,
            tmp_path / "recovery",
            time.monotonic() + 30,
            renderer_factory=renderer,
        )
    assert cell["status"] == "complete"
    assert [p for p, _ in launches] == [
        effective["prompt"],
        *[effective[k] for k in effective["followup_order"]],
    ]
    assert all(session == "same-context" for _, session in launches[1:])
    health = run.pilot_health.driving_health(cell, folder.parent)
    assert health["problems"] == []
    assert health["verified_pairs"] == 1
    lifecycle = run.read(folder / "lifecycle.json")
    assert lifecycle["exited"] and all(
        not lifecycle[k]["alive"] for k in ("native", "cameras", "recording", "episode")
    )


def test_opportunity_evidence_failure_preserves_completed_invocation_and_stops(
    harness, monkeypatch
):
    def fail(*_args):
        raise ValueError("mismatched source clock")

    monkeypatch.setattr(run, "opportunity", fail)
    cells = execute(harness)
    assert len(cells) == len(harness[4]) == 1
    assert cells[0]["health"]["infrastructure_problem"]
    assert cells[0]["opportunity"]["reason"] == "invalid_opportunity_evidence"
    assert run.read(harness[1] / "batch.json")["failure"] == "driving_infrastructure"


def test_closure_revalidation_after_review_preserves_gate_and_reports_tamper(harness, tmp_path):
    def review(*args):
        harness[3](*args)
        (tmp_path / "runtime").write_text("changed during closed manual review")

    with pytest.raises(ValueError, match="changed"):
        execute(harness, checkpoint_reviewer=review)
    batch = run.read(harness[1] / "batch.json")
    assert len(batch["cells"]) == len(harness[4]) == 4
    assert batch["checkpoint"]["continue_automatically"]
    assert "changed" in batch["failure"]
    assert all(a["status"] == "not_run" for a in batch["assignments"][4:])


def test_capacity_is_allocated_without_retry_or_later_model_skip(harness):
    original = harness[2]

    def cell(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(harness[4]) == 1:
            result["status"] = "failed_capacity"
        return result

    cells = execute(harness, cell_runner=cell)
    assert len(cells) == len(harness[4]) == 16
    assert [c["status"] for c in cells[::4]] == [
        "failed_capacity",
        "complete",
        "complete",
        "complete",
    ]


def test_command_line_accepts_nested_release_inputs_without_hidden_defaults(
    inputs, tmp_path, monkeypatch
):
    scenes, certificate = inputs()
    document = write(tmp_path / "release.json", dict(scene_map=scenes))
    output = tmp_path / "cli-freeze"
    monkeypatch.setattr(
        run.sys,
        "argv",
        [
            "naturalistic_run",
            "freeze",
            "D",
            document,
            str(output),
            "--deadline",
            "2100-01-01T00:00:00+00:00",
            "--asset-label",
            "explicit",
            "--preflight",
            certificate,
        ],
    )
    run.main()
    monkeypatch.setattr(
        run.sys, "argv", ["naturalistic_run", "validate", str(output / "protocol.json")]
    )
    run.main()
    assert run.read(output / "protocol.json")["scenes"] == scenes


@pytest.mark.parametrize(
    "regions", [{}, {"head": [0, 0, 1, 1]}, {"default": {"head": [0, 0, 1, 1]}}, {"frames": []}]
)
def test_selected_source_annotations_reject_malformed_regions(regions):
    scene = dict(
        objects=[dict(id="hazard", asset_path="still.png")],
        scenario=dict(visibility_regions=dict(hazard=regions)),
    )
    with pytest.raises(ValueError):
        run._annotations(scene)


@pytest.mark.parametrize("missing", ["moving.png", "idle.png", "hold.png", "still.png"])
def test_annotations_require_every_selected_moving_idle_hold_and_static_source(tmp_path, missing):
    boxes = dict(head=[0, 0, 1, 0.2], body=[0, 0.2, 1, 0.8], feet=[0, 0.8, 1, 1])
    paths = {
        key: str(tmp_path / key) for key in ("moving.png", "idle.png", "hold.png", "still.png")
    }
    scene = dict(
        objects=[dict(id="hazard", asset_path=paths["still.png"])],
        scenario=dict(
            animations=dict(
                hazard=dict(
                    frame_paths=[paths["moving.png"]],
                    idle=dict(frame_paths=[paths["idle.png"]]),
                    hold_clip=dict(frame_paths=[paths["hold.png"]]),
                )
            ),
            visibility_regions=dict(hazard=dict(frames={path: boxes for path in paths.values()})),
        ),
    )
    run._annotations(scene)
    del scene["scenario"]["visibility_regions"]["hazard"]["frames"][paths[missing]]
    with pytest.raises(ValueError, match="Every selected"):
        run._annotations(scene)
    scene["scenario"]["visibility_regions"]["hazard"]["default"] = boxes
    run._annotations(scene)


@pytest.mark.parametrize("qualifications", [[], [""], ["   "], ["valid", None]])
def test_empty_or_blank_qualifications_are_rejected(inputs, tmp_path, qualifications):
    scenes, path = inputs()
    certificate = run.read(path)
    certificate["scenes"][next(iter(scenes))]["qualifications"] = qualifications
    write(Path(path), certificate)
    with pytest.raises(ValueError, match="qualifications"):
        run.freeze(
            "D",
            scenes,
            tmp_path / "rejected",
            deadline="2100-01-01T00:00:00+00:00",
            asset_label="explicit",
            preflight=path,
        )


@pytest.mark.parametrize("status", ["failed_auth", "failed_capacity"])
def test_dispatched_provider_failure_with_zero_receipts_has_no_opportunity_fault(
    tmp_path, monkeypatch, status
):
    import test_sim_pilot_health as fixtures

    cell, output, directory = evidence = fixtures.drive.__wrapped__(tmp_path, monkeypatch)
    fixtures.provider(evidence, status)
    (directory / "live-receipts.jsonl").unlink()
    (directory / "client/stdout.jsonl").write_text("")
    (directory / "episode/events.jsonl").write_text("")
    write(directory / "hazard-evidence.json", dict(receipt=dict(observations=[])))
    write(directory / "receipt-supervisor.json", dict(censored=[], confirmed_calls=0))
    assert (directory / "dispatch.json").exists()
    health = run.pilot_health.driving_health(cell, output)
    assert not health["infrastructure_problem"]
    assert health["verified_pairs"] == 0 and health["provider_failure"] == status
    scene = dict(design=dict(path=[[0, 0, 0], [50, 0, 0]]), scenario=dict(trajectories={}))
    cell["assignment"].update(
        scene=write(tmp_path / "selected.json", scene),
        scene_key="selected",
        condition="roadside_person",
    )
    protocol = dict(
        pilot="D",
        opportunity_rule=run.pilot_exposure.rule_for("D"),
        preflight=write(tmp_path / "certificate.json", dict(scenes=dict(selected={}))),
    )
    result = run.opportunity(cell, protocol, output)
    assert result["candidates"] == []


@pytest.mark.parametrize("regions", [{"head": [0, 0, 1, 1]}, {"frames": {}}])
def test_freeze_rejects_unusable_annotation_even_with_reviewed_scene_hash(
    inputs, tmp_path, regions
):
    scenes, path = inputs()
    certificate = run.read(path)
    for key, scene_path in scenes.items():
        scene = run.read(scene_path)
        scene["scenario"]["visibility_regions"]["hazard"] = regions
        write(Path(scene_path), scene)
        certificate["scenes"][key]["scene_sha256"] = run.file_hash(scene_path)
    write(Path(path), certificate)
    with pytest.raises(ValueError, match="source rectangles|Every selected"):
        run.freeze(
            "D",
            scenes,
            tmp_path / "rejected",
            deadline="2100-01-01T00:00:00+00:00",
            asset_label="explicit",
            preflight=path,
        )


@pytest.mark.parametrize(
    "change",
    [
        "scene",
        "executed_scene",
        "side",
        "mode",
        "offset",
        "yaw",
        "delay",
        "stop",
        "clock",
        "wait",
        "source",
        "recipe_unhashed",
    ],
)
def test_c_native_recipe_must_prove_selected_no_response_and_current_physics(
    inputs, tmp_path, change
):
    scenes, path = inputs("C")
    certificate = run.read(path)
    row = certificate["scenes"][next(iter(scenes))]
    metadata = run.read(row["no_response_native_source"])
    recipe_path = Path(metadata["metadata"]["recipe"])
    recipe = run.read(recipe_path)
    fields = dict(
        scene="scene_sha256",
        executed_scene="executed_scene_sha256",
        side="side",
        mode="mode",
        offset="start_offset_m",
        yaw="start_yaw_deg",
        delay="response_delay_s",
        stop="stop_station_m",
        clock="source_clock_unchanged",
        wait="actor_wait_policy_unchanged",
    )
    if change == "source":
        recipe["source_hashes"][str((run.study.REPO / "sim/plant.py").resolve())] = "0" * 64
    elif change != "recipe_unhashed":
        values = dict(
            scene="0" * 64,
            executed_scene="0" * 64,
            side="wrong",
            mode="timely_stop",
            offset=0.5,
            yaw=5,
            delay=2,
            stop=12,
            clock=False,
            wait=False,
        )
        recipe[fields[change]] = values[change]
    write(recipe_path, recipe)
    certificate["evidence_hashes"][str(recipe_path)] = run.file_hash(recipe_path)
    if change == "recipe_unhashed":
        del certificate["evidence_hashes"][str(recipe_path)]
    write(Path(path), certificate)
    with pytest.raises(ValueError, match="C native"):
        run.freeze(
            "C",
            scenes,
            tmp_path / "rejected",
            deadline="2100-01-01T00:00:00+00:00",
            asset_label="explicit",
            preflight=path,
        )


def test_c_freeze_includes_native_recipe_and_exact_relevant_sources(inputs, tmp_path):
    protocol = run.read(frozen(inputs, tmp_path, "C"))
    for name in run.NATIVE_SOURCES:
        path = str((run.study.REPO / name).resolve())
        assert protocol["dependencies"][path] == run.file_hash(path)
    assert len([p for p in protocol["dependencies"] if p.endswith("recipe.json")]) == 4


@pytest.mark.parametrize(
    "change", ["missing_bundle", "receipt_clock", "manual_hash", "gap", "capture_pose"]
)
def test_D_freeze_requires_exact_two_view_received_proof(inputs, tmp_path, change):
    scenes, path = inputs("D")
    cert = run.read(path)
    row = cert["scenes"][next(iter(scenes))]
    bundle = row["actual_received_preflight"]
    if change == "missing_bundle":
        row.pop("actual_received_preflight")
    elif change in {"receipt_clock", "gap"}:
        receipts = run.study.read_lines(Path(bundle["receipts"]))
        if change == "receipt_clock":
            receipts[0].pop("actual_client_received_monotonic_s")
        else:
            receipts[1].update(
                confirmed_monotonic_s=111.99, actual_client_received_monotonic_s=111.99
            )
        Path(bundle["receipts"]).write_text("".join(json.dumps(row) + "\n" for row in receipts))
        cert["evidence_hashes"][bundle["receipts"]] = run.file_hash(bundle["receipts"])
    elif change == "manual_hash":
        reviews = run.read(bundle["manual_reviews"])
        reviews["reviews"][0]["narrow_received_sha256"] = "0" * 64
        write(Path(bundle["manual_reviews"]), reviews)
        cert["evidence_hashes"][bundle["manual_reviews"]] = run.file_hash(bundle["manual_reviews"])
    else:
        events = run.study.read_lines(Path(bundle["events"]))
        next(event for event in events if event["kind"] == "sim_frame")["pose"] = [999, 0, 0]
        Path(bundle["events"]).write_text("".join(json.dumps(row) + "\n" for row in events))
        cert["evidence_hashes"][bundle["events"]] = run.file_hash(bundle["events"])
    write(Path(path), cert)
    with pytest.raises(ValueError):
        run.freeze(
            "D",
            scenes,
            tmp_path / "rejected",
            deadline="2100-01-01T00:00:00+00:00",
            asset_label="explicit",
            preflight=path,
        )
