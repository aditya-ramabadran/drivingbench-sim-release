import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from drivingbench.sim.course import finish_before_line
from drivingbench.sim.studies import persistent_goal as plan
from drivingbench.sim.studies import persistent_goal_run as run
from test_sim_live import scene_file
from test_sim_pilot_exposure import artifact_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path.resolve())


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source, runtime, asset = (tmp_path / n for n in ("source.py", "runtime", "image.png"))
    for p in (source, runtime, asset):
        p.write_text(p.name)
    monkeypatch.setattr(run, "source_files", lambda: {source})
    monkeypatch.setattr(run, "runtime_files", lambda: {runtime})
    monkeypatch.setattr(plan, "base_prompt", lambda _s: "Original prompt")
    base = json.loads(scene_file(tmp_path).read_text())
    base.update(
        calibration=str(asset),
        background={"checkpoint": str(asset)},
        wide_camera={k: str(asset) for k in ("checkpoint", "calibration", "vehicle_layer")},
        finish=finish_before_line([13.85, 3.5], [13.85, -3.5]),
    )
    # This fixture isolates orchestration from the semantic validators, exercised separately.
    monkeypatch.setattr(run.proof, "backstop", lambda *_a, **_kw: None)
    monkeypatch.setattr(run.proof, "native", lambda p, *_a, **_kw: (run.read(p)["mode"], set()))
    monkeypatch.setattr(run.proof, "render", lambda p, *_a, **_kw: run.read(p)["time"])
    monkeypatch.setattr(run.proof, "mcp", lambda p, *_a: run.read(p)["arm"])
    monkeypatch.setattr(run.proof, "idle", lambda *_a, **_kw: None)
    record = write(tmp_path / "evidence.json", {"review": "orchestration fixture"})
    native = {m: write(tmp_path / f"native-{m}.json", {"mode": m}) for m in plan.PHYSICAL_MODES}
    renders = [write(tmp_path / f"render-{t}.json", {"time": t}) for t in (0, 3)]
    mcp = [write(tmp_path / f"mcp-{arm}.json", {"arm": arm}) for arm in plan.ARMS]
    refs = [record, *native.values(), *renders, *mcp]
    certificate = dict(
        protocol=plan.PROTOCOL,
        visibility_rule=plan.VISIBILITY,
        evidence_hashes={p: run.file_hash(p) for p in refs},
        scenes={},
    )
    mapping = {}
    for key in plan.SCENES:
        scene = copy.deepcopy(base)
        scene["scenario"] = dict(
            type="course_event",
            time_scale=1,
            time_limit_s=180,
            condition=key,
            finish_line=dict(center=[14, 0], width_m=7, depth_m=0.30, heading_deg=0),
            trajectories={},
            colliders=[],
            animations={},
            coverage=dict(
                polygon=[[-10, -10], [30, -10], [30, 10], [-10, 10]],
                heading_deg=0,
                max_heading_deviation_deg=180,
            ),
        )
        if key != "empty":
            scene["objects"] = [
                dict(id="hazard", kind="billboard", x=14, y=0, asset_path=str(asset))
            ]
            scene["scenario"].update(
                colliders=[
                    dict(
                        id="hazard",
                        outcome="hazard",
                        circle=[14, 0, 0.76 if key == "dog" else 0.32],
                    )
                ],
                animations={"hazard": dict(frame_paths=[str(asset)] * 2, fps=24, playback="loop")},
                visibility_regions={
                    "hazard": dict(head=[0, 0, 1, 0.2], body=[0, 0.2, 1, 0.8], feet=[0, 0.8, 1, 1])
                },
            )
        path = write(tmp_path / f"{key}.json", scene)
        mapping[key] = path
        required = (
            run.NATIVE_CHECKS
            if key != "empty"
            else tuple(k for k in run.NATIVE_CHECKS if k not in {"contact", "persistent_wait"})
        )
        certificate["scenes"][key] = dict(
            scene_sha256=run.file_hash(path),
            native_checks={
                k: [native["free"]] if k.startswith("free_") else list(native.values())
                for k in required
            },
            render_records=renders,
            mcp_records=mcp,
            jpeg_review={"reviewer": "root", "record": record},
            view_checks=[
                dict(record=p, capture_sim_time_s=t, phase="persistent")
                for p, t in zip(renders, (0, 3))
            ],
            idle_records=[record],
            annotation_records=[record],
            source_clock_records=[record],
            qualifications=["Fixture; no real evidence claimed"],
        )
    certificate_path = write(tmp_path / "preflight.json", certificate)
    return mapping, certificate_path


def freeze(inputs, tmp_path):
    folder = tmp_path / "frozen"
    run.freeze(
        inputs[0],
        folder,
        deadline="2099-01-01T00:00:00+00:00",
        asset_label="fixture",
        preflight=inputs[1],
    )
    return folder / "protocol.json"


def test_freeze_exact_membership_and_exclusive_outputs(inputs, tmp_path):
    path = freeze(inputs, tmp_path)
    frozen = run.read(path)
    run.validate_protocol(frozen)
    assert len(frozen["assignments"]) == 192
    assert frozen["active_budget_s"] == 180
    with pytest.raises(FileExistsError):
        freeze(inputs, tmp_path)
    frozen["settings"]["speed_limit_mps"] = 3.5
    with pytest.raises(ValueError, match="differs"):
        run.validate_protocol(frozen)


@pytest.mark.parametrize(
    "change",
    ["moved", "once", "static", "trajectory", "wide", "finish", "background", "annotation"],
)
def test_scene_rejects_accidental_treatment_deltas(inputs, change):
    path = Path(inputs[0]["adult"])
    scene = run.read(path)
    if change == "moved":
        scene["objects"][0]["x"] += 0.01
    elif change == "once":
        scene["scenario"]["animations"]["hazard"]["playback"] = "once"
    elif change == "static":
        scene["scenario"]["animations"] = {}
    elif change == "trajectory":
        scene["scenario"]["trajectories"]["hazard"] = dict(start_s=0, duration_s=2, delta=[0, 1])
    elif change == "wide":
        scene.pop("wide_camera")
    elif change == "finish":
        scene["finish"]["polygon"][0][0] += 0.01
    elif change == "background":
        scene["start"]["x"] += 0.1
    else:
        scene["scenario"]["visibility_regions"]["hazard"]["head"] = [0, 0, 0, 0]
    write(path, scene)
    with pytest.raises((ValueError, KeyError)):
        run._scenes(inputs[0])


@pytest.mark.parametrize(
    "change",
    [
        "bool_only",
        "scene_hash",
        "empty_review",
        "missing_native",
        "missing_idle",
        "qualification",
        "wrong_rule",
        "bad_stamp",
    ],
)
def test_preflight_requires_actual_referenced_records(inputs, change):
    certificate = run.read(inputs[1])
    row = certificate["scenes"]["adult"]
    if change == "bool_only":
        row["render_records"] = True
    elif change == "scene_hash":
        row["scene_sha256"] = "a" * 64
    elif change == "empty_review":
        row["jpeg_review"]["reviewer"] = " "
    elif change == "missing_native":
        row["native_checks"].pop("stop_ceiling_request")
    elif change == "missing_idle":
        row["idle_records"] = []
    elif change == "qualification":
        row["qualifications"] = [" "]
    elif change == "wrong_rule":
        certificate["visibility_rule"] = {}
    else:
        row["view_checks"][0]["capture_sim_time_s"] = float("nan")
    write(Path(inputs[1]), certificate)
    with pytest.raises(ValueError):
        run._inputs(*inputs)


@pytest.fixture
def harness(inputs, tmp_path, monkeypatch):
    path = freeze(inputs, tmp_path)
    output = tmp_path / "results"
    calls = []

    @contextmanager
    def lifecycle(folder, renderer):
        yield renderer

    @contextmanager
    def postdrive(*a):
        yield

    monkeypatch.setattr(run.pilot_health, "observe_lifecycle", lifecycle)
    monkeypatch.setattr(run, "postdrive_policy", postdrive)
    monkeypatch.setattr(
        run.pilot_health,
        "driving_health",
        lambda c, _o: dict(
            infrastructure_problem=c["status"] == "infrastructure_failure",
            model_unavailable=c["status"] == "failed_auth",
        ),
    )
    monkeypatch.setattr(
        run,
        "opportunity",
        lambda c, _o: dict(
            candidates=[dict(call_id="one", sha256="a" * 64, eligible=True)]
            if c["status"] == "complete"
            else []
        ),
    )

    def cell(protocol, row, folder, recovery, deadline, **kwargs):
        assert not folder.exists()
        assert (
            kwargs["active_budget_s"] == 180 and kwargs["declaration_factory"] is plan.declaration
        )
        assert kwargs["mcp_factory"] is run.configuration
        assert protocol["prompt"] == plan.prompt(row["arm"])
        write(folder / "attempt-1/dispatch.json", {})
        calls.append(row)
        return dict(assignment=row, status="complete", attempts=[{}])

    def review(cells, protocol, output, deadline):
        for c in cells:
            write(
                output / c["assignment"]["id"] / "attempt-1/visibility-review.json",
                [
                    dict(
                        call_id="one",
                        narrow_received_sha256="a" * 64,
                        manual_pass=True,
                        reviewer="root",
                    )
                ],
            )

    return path, output, cell, review, calls


def execute(harness, **kw):
    path, output, cell, review, _ = harness
    return run.run(
        path,
        output,
        cell_runner=kw.pop("cell_runner", cell),
        checkpoint_reviewer=kw.pop("checkpoint_reviewer", review),
        **kw,
    )


def test_serial192_no_precreated_cell_and_claim_blocks_repeat(harness, tmp_path):
    cells = execute(harness)
    assert len(cells) == len(harness[4]) == 192
    assert all(c["missing_postdrive"] == list(run.study.FOLLOWUP_QUESTIONS) for c in cells)
    batch = run.read(harness[1] / "batch.json")
    assert len(batch["checkpoints"]) == 2
    assert all(g["continue_automatically"] for g in batch["checkpoints"]) and not batch["failure"]
    with pytest.raises(FileExistsError):
        execute(harness)
    with pytest.raises(FileExistsError):
        run.run(harness[0], tmp_path / "duplicate", cell_runner=harness[2])
    assert len(harness[4]) == 192


def test_source_tamper_before_second_stops_and_preserves_all192(harness, tmp_path):
    def cell(*args, **kwargs):
        result = harness[2](*args, **kwargs)
        (tmp_path / "source.py").write_text("changed")
        return result

    with pytest.raises(ValueError, match="changed"):
        execute(harness, cell_runner=cell)
    batch = run.read(harness[1] / "batch.json")
    assert len(batch["cells"]) == 1 and len(batch["assignments"]) == 192
    assert batch["assignments"][1]["status"] == "not_run"


@pytest.mark.parametrize("mixed", [False, True])
def test_confirmed_auth_preserved_remaining_model_skipped_unless_driving_fault(
    harness, mixed, monkeypatch
):
    def cell(*args, **kwargs):
        c = harness[2](*args, **kwargs)
        if c["assignment"]["model"] == "astra":
            c["status"] = "failed_auth"
        return c

    if mixed:
        monkeypatch.setattr(
            run.pilot_health,
            "driving_health",
            lambda *_a: dict(infrastructure_problem=True, model_unavailable=True),
        )
    cells = execute(harness, cell_runner=cell)
    if mixed:
        assert len(cells) == 1
        assert run.read(harness[1] / "batch.json")["failure"] == "driving_infrastructure"
    else:
        assert len(cells) == 192 and len(harness[4]) == 145
        assert cells[0]["status"] == "failed_auth"
        assert all(c["status"] == "auth_skipped" for c in cells[4::4])
        assert all(
            a["denominator"] == 4 and a["approved"] == 3
            for gate in run.read(harness[1] / "batch.json")["checkpoints"]
            for a in gate["arms"].values()
        )


def test_missing_or_wrong_manual_hash_fails_assigned_first4(harness):
    def review(cells, *args):
        harness[3](cells, *args)
        for c in cells[:4]:
            write(
                harness[1] / c["assignment"]["id"] / "attempt-1/visibility-review.json",
                [
                    dict(
                        call_id="one",
                        narrow_received_sha256="b" * 64,
                        manual_pass=True,
                        reviewer="root",
                    )
                ],
            )

    assert len(execute(harness, checkpoint_reviewer=review)) == 4
    batch = run.read(harness[1] / "batch.json")
    assert batch["failure"] == "checkpoint_exposure"
    assert sum(r["status"] == "not_run" for r in batch["assignments"]) == 188


def test_startup_fault_is_separate_and_never_retried(harness):
    count = []

    def cell(*_a, **_kw):
        count.append(1)
        raise RuntimeError("startup")

    cells = execute(harness, cell_runner=cell)
    assert count == [1] and cells[0]["startup_failure"]
    assert not cells[0]["dispatch_evidence"]["model_may_have_run"]


def receipt_files(tmp_path, kind="adult", transcode=False):
    data, audit = artifact_fixture(tmp_path, transcode=transcode)
    for frame in data["receipts"][0]["frames"]:
        frame["actor_onsets"] = {}
        frame.pop("actor_states", None)
    for proof in audit["receipt"]["observations"][0]["frames"]:
        proof["actor_onsets"] = {}
    for event in data["events"]:
        if "actor_onsets" in event:
            event["actor_onsets"] = {}
        event.pop("actor_states", None)
    row = dict(id="one", condition=kind)
    directory = tmp_path / "one/attempt-1"

    def lines(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(map(json.dumps, rows)) + "\n")

    lines(directory / "episode/events.jsonl", data["events"])
    lines(directory / "live-receipts.jsonl", data["receipts"])
    write(directory / "episode/summary.json", data["episode"])
    write(directory / "hazard-evidence.json", audit)
    return dict(assignment=row, attempts=[{}]), directory, data, audit


@pytest.mark.parametrize("kind", ["empty", "adult", "child", "dog"])
@pytest.mark.parametrize("transcode", [False, True])
def test_actual_artifact_join_sheet_manual_review_and_empty_control(tmp_path, kind, transcode):
    cell, directory, data, audit = receipt_files(tmp_path, kind, transcode)
    opportunity = run.opportunity(cell, tmp_path)
    assert opportunity["candidates"][0]["eligible"]
    run.no_excuse.review_sheet(directory, opportunity["candidates"])
    assert (directory / "checkpoint-received.jpg").exists()
    review = dict(
        call_id="call-one",
        narrow_received_sha256=opportunity["candidates"][0]["sha256"],
        manual_pass=True,
        reviewer="root",
    )
    assert run.pilot_exposure.manual_matches(opportunity, [review])
    assert not run.pilot_exposure.manual_matches(
        opportunity, [dict(review, narrow_received_sha256="b" * 64)]
    )
    assert (
        run.read(directory / "checkpoint-review-frames.json")[0]["frames"][0]["received_path"]
        == audit["receipt"]["observations"][0]["frames"][0]["received_path"]
    )


@pytest.mark.parametrize(
    "change", ["late", "nan", "future", "cropped", "wideonly", "bytes", "clock"]
)
def test_received_exposure_adversarial(tmp_path, change):
    cell, directory, data, audit = receipt_files(tmp_path)
    if change in {"late", "nan"}:
        data["receipts"][0]["confirmed_monotonic_s"] = 111 if change == "late" else float("nan")
        (directory / "live-receipts.jsonl").write_text(json.dumps(data["receipts"][0]) + "\n")
    elif change == "bytes":
        Path(audit["receipt"]["observations"][0]["frames"][0]["received_path"]).write_bytes(
            b"changed"
        )
    else:
        for frame in data["receipts"][0]["frames"]:
            if change == "future":
                frame["sim_time_s"] = 50
            elif change == "clock":
                frame["actor_onsets"] = {"hazard": 1}
            elif frame["camera"] == "narrow":
                frame["billboards"]["hazard"]["whole_actor_visibility"]["region_survival"][
                    "head"
                ] = 0
        (directory / "live-receipts.jsonl").write_text(json.dumps(data["receipts"][0]) + "\n")
    assert not run.opportunity(cell, tmp_path)["candidates"][0]["eligible"]


def test_genuine_cropped_narrow_with_clear_wide_does_not_pass(tmp_path):
    cell, directory, data, audit = receipt_files(tmp_path)
    frame = data["receipts"][0]["frames"][0]
    frame["billboards"]["hazard"]["whole_actor_visibility"]["region_survival"]["head"] = 0.2
    audit["receipt"]["observations"][0]["frames"][0]["billboards"] = copy.deepcopy(
        frame["billboards"]
    )
    worker = next(e for e in data["events"] if e["kind"] == "sim_frame")
    worker["worker"]["billboards"] = copy.deepcopy(frame["billboards"])
    (directory / "episode/events.jsonl").write_text(
        "\n".join(map(json.dumps, data["events"])) + "\n"
    )
    (directory / "live-receipts.jsonl").write_text(json.dumps(data["receipts"][0]) + "\n")
    write(directory / "hazard-evidence.json", audit)
    candidate = run.opportunity(cell, tmp_path)["candidates"][0]
    assert candidate["reasons"] == ["whole_narrow_actor_screen_failed"]


def test_manual_wait_uses_actual_sheet_and_stops_at_deadline(tmp_path):
    import time

    cell, directory, _data, _audit = receipt_files(tmp_path)
    cell["opportunity"] = run.opportunity(cell, tmp_path)
    now = time.monotonic()
    run.await_reviews([cell], {}, tmp_path, now + 0.01)
    assert time.monotonic() - now < 1
    assert (directory / "checkpoint-received.jpg").exists()
    assert not (directory / "visibility-review.json").exists()


def test_budget_expired_before_first_cell_keeps_all192_unrun(harness):
    frozen = run.read(harness[0])
    frozen["deadline_utc"] = "2000-01-01T00:00:00+00:00"
    write(harness[0], frozen)
    assert execute(harness) == []
    assert not harness[4]
    batch = run.read(harness[1] / "batch.json")
    assert batch["failure"] == "absolute_deadline"
    assert all(r["status"] == "not_run" for r in batch["assignments"])


def test_failed_postdrive_retry_keeps_failure_kind_and_collection_continues(harness):
    def cell(*args, **kwargs):
        result = harness[2](*args, **kwargs)
        result["operator_account"] = dict(
            missing=True,
            failure_kind=None,
            question_attempts=[
                dict(reply=dict(failure_kind="failed_capacity")),
                dict(reply=dict(failure_kind="failed_capacity")),
            ],
        )
        return result

    cells = execute(harness, cell_runner=cell)
    assert len(cells) == 192
    assert all(
        c["operator_account"]["failure_kind"] == "failed_capacity"
        and "operator_account" in c["missing_postdrive"]
        for c in cells
    )
