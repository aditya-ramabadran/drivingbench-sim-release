import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from drivingbench.sim import event_study_v2 as study
from drivingbench.sim.batch import save_new
from drivingbench.sim.studies import continuation as resume


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "validate_protocol", lambda protocol: None)
    rows = [
        dict(id=f"cell-{i}", condition="control", scene_key="control", arm="real_time")
        for i in range(4)
    ]
    protocol = dict(
        assignments=rows,
        followup_order=list(study.FOLLOWUP_QUESTIONS),
        deadline_utc=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        **study.FOLLOWUP_QUESTIONS,
    )
    protocol_path = tmp_path / "protocol.json"
    save_new(protocol_path, protocol)
    old = tmp_path / "old"
    cell = dict(assignment=rows[0], status="complete", attempts=[])
    save_new(old / rows[0]["id"] / "session.json", cell)
    save_new(old / "batch.json", dict(cells=[cell]))
    save_new(old / "checkpoint-round-1.json", dict(continue_automatically=True))
    output = tmp_path / "new"
    resume.freeze_addendum(protocol_path, old, output)
    return protocol, old, output, output / "addendum.json"


def good(text="answer"):
    return dict(text=text, errors=[], same_context=True, session_id="original")


def fake_cell(protocol, row, folder, recovery, deadline, *, renderer_factory):
    folder.mkdir(parents=True)
    result = dict(assignment=row, status="complete", attempts=[])
    for key in protocol["followup_order"]:
        result[key] = study.followup(None, protocol[key], folder / key, "original", 180)
        failure = study.reporting_failure(result[key])
        result[key]["failure_kind"] = failure
        if failure:
            result["reporting_failure"] = failure
            break
    if not result.get("debrief", {}).get("errors"):
        result["belief"] = study.code_belief(result["debrief"]["text"], folder / "belief", 180)
    return result


def test_retries_only_failed_question_and_preserves_raw(tmp_path, monkeypatch):
    calls = []

    def followup(client, text, path, session, budget):
        calls.append((text, Path(path).name, session, budget))
        return (
            dict(errors=["websocket disconnected"], same_context=False, text="flagged")
            if len(calls) == 1
            else good()
        )

    monkeypatch.setattr(study, "followup", followup)
    with resume.postdrive_policy(study.time.monotonic() + 60):
        answer = study.followup(None, "exact prompt", tmp_path / "q", "original", 180)
        assert study.reporting_failure(answer) is None
    assert calls[0][:3] == ("exact prompt", "first", "original")
    assert calls[1][:3] == ("exact prompt", "retry", "original")
    assert all(0 < call[3] <= 60 for call in calls)
    assert answer["automatic_retries"] == 1 and not answer["missing"]
    first = resume.read(tmp_path / "q/first/supervisor-reply.json")
    assert first["text"] == "flagged" and first["failure_kind"] == "infrastructure_failure"
    assert study.followup is followup
    assert study.reporting_failure(first) == "infrastructure_failure"


def test_missing_question_continues_same_context_to_next_question(frozen, monkeypatch):
    protocol, old, output, addendum = frozen
    before = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    calls = []

    def followup(client, text, folder, session, budget):
        calls.append((text, session))
        return (
            dict(errors=[], same_context=False, text="new chat")
            if text == protocol["operator_account"]
            else good()
        )

    monkeypatch.setattr(study, "followup", followup)
    monkeypatch.setattr(study, "code_belief", lambda *args: dict(label="doubted", returncode=0))
    cells = resume.run(addendum, cell_runner=fake_cell)
    assert len(cells) == 3
    assert [text for text, _ in calls[:4]] == [protocol["operator_account"]] * 2 + [
        protocol["debrief"],
        protocol["perception"],
    ]
    assert all(session == "original" for _, session in calls)
    assert all(cell["operator_account"]["missing"] for cell in cells)
    assert all(
        cell["operator_account"]["failure_kind"] == "infrastructure_failure" for cell in cells
    )
    assert all(cell["belief"]["label"] == "doubted" for cell in cells)
    assert {p: p.read_bytes() for p in before} == before
    manifest = resume.read(output / "union-manifest.json")
    assert manifest["completed"] == 4
    assert manifest["cells"][0]["path"] == str(old / "cell-0/session.json")
    assert resume.read(output / "batch.json")["failure"] is None
    with pytest.raises(ValueError, match="already dispatched"):
        resume.run(addendum, cell_runner=fake_cell)


def test_coder_retries_failure_only_and_preserves_both(tmp_path, monkeypatch):
    calls = []

    def coder(text, folder, budget):
        calls.append(folder)
        return (
            dict(label=None, returncode=1) if len(calls) == 1 else dict(label="real", returncode=0)
        )

    monkeypatch.setattr(study, "code_belief", coder)
    with resume.postdrive_policy(study.time.monotonic() + 60):
        answer = study.code_belief("same text", tmp_path / "belief", 180)
    assert len(calls) == 2 and answer["label"] == "real"
    assert resume.read(tmp_path / "belief/first/supervisor-reply.json")["missing"]
    with resume.postdrive_policy(study.time.monotonic() + 60):
        answer = study.code_belief("same text", tmp_path / "valid", 180)
    assert len(calls) == 3 and answer["automatic_retries"] == 0


def test_driving_failure_stops_without_next_dispatch(frozen):
    _, _, output, addendum = frozen
    calls = []

    def fault(protocol, row, *args, **kwargs):
        calls.append(row["id"])
        raise RuntimeError("camera unavailable")

    cells = resume.run(addendum, cell_runner=fault)
    assert calls == ["cell-1"] and cells[0]["status"] == "infrastructure_failure"
    batch = resume.read(output / "batch.json")
    assert batch["failure"] == "driving_infrastructure_health"
    assert len(batch["not_dispatched"]) == 2


def test_clock_health_fault_still_stops(frozen, monkeypatch):
    _, _, output, addendum = frozen
    monkeypatch.setattr(resume, "amended_health", lambda *args: dict(infrastructure_problem=True))

    def cell(protocol, row, *args, **kwargs):
        return dict(assignment=row, status="complete", attempts=[])

    assert len(resume.run(addendum, cell_runner=cell)) == 1
    assert resume.read(output / "batch.json")["failure"] == "driving_infrastructure_health"


def test_expired_deadline_invokes_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study, "followup", lambda *args: calls.append(args))
    with resume.postdrive_policy(study.time.monotonic() - 1):
        answer = study.followup(None, "question", tmp_path / "q", "original", 180)
    assert not calls and answer["missing"] and answer["failure_kind"] == "not_run_deadline"


def test_pending_rejects_duplicate_or_previously_dispatched(frozen):
    protocol, old, _, _ = frozen
    previous = resume.read(old / "batch.json")
    (old / "cell-1").mkdir()
    with pytest.raises(ValueError, match="already has evidence"):
        resume.pending(protocol, previous, old)
    protocol["assignments"][2] = dict(protocol["assignments"][1])
    with pytest.raises(ValueError, match="Duplicate"):
        resume.pending(protocol, previous, old)


def test_gate_and_source_mutations_rejected(frozen):
    _, old, output, addendum = frozen
    value = resume.read(addendum)
    value["policy"] = dict(value["policy"], postdrive_retries=2)
    with pytest.raises(ValueError, match="policy changed"):
        resume.validate_addendum(value)
    (old / "batch.json").write_text("{}")
    with pytest.raises(ValueError, match="dependency changed"):
        resume.validate_addendum(resume.read(addendum))
    assert not (output / "batch.json").exists()


def test_auth_recovery_allowance_not_reset(tmp_path, monkeypatch):
    (tmp_path / "previous/fable").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        study, "auth_preflight", lambda spec, lifetime, **kwargs: calls.append(kwargs)
    )
    with resume.postdrive_policy(study.time.monotonic() + 60, tmp_path / "previous"):
        study.auth_preflight(SimpleNamespace(key="fable"), recovery_output=tmp_path / "new")
    assert calls == [dict(recovery_output=None)]


def test_empty_answer_retried_then_missing(tmp_path, monkeypatch):
    calls = []

    def followup(*args):
        calls.append(args)
        return good("   ")

    monkeypatch.setattr(study, "followup", followup)
    with resume.postdrive_policy(study.time.monotonic() + 60):
        answer = study.followup(None, "question", tmp_path / "empty", "original", 180)
    assert len(calls) == 2 and answer["missing"]
    assert answer["failure_kind"] == "empty_answer"


def test_failed_coder_does_not_stop_cells(frozen, monkeypatch):
    _, _, output, addendum = frozen
    calls = []
    monkeypatch.setattr(study, "followup", lambda *args: good())

    def coder(*args):
        calls.append(args)
        raise OSError("coder unavailable")

    monkeypatch.setattr(study, "code_belief", coder)
    cells = resume.run(addendum, cell_runner=fake_cell)
    assert len(cells) == 3 and len(calls) == 6
    assert all(cell["belief"]["missing"] for cell in cells)
    assert resume.read(output / "batch.json")["failure"] is None


def test_exclusive_dispatch_marker_blocks_duplicate(frozen):
    _, _, output, addendum = frozen
    save_new(output / "dispatch-start.json", dict(started=True))
    with pytest.raises(ValueError, match="already dispatched"):
        resume.run(addendum, cell_runner=fake_cell)
    assert not (output / "batch.json").exists()


def test_absolute_expired_run_saves_final_batch(frozen, monkeypatch):
    protocol, old, output, addendum = frozen
    protocol["deadline_utc"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    protocol_path = Path(resume.read(addendum)["protocol_path"])
    protocol_path.write_text(json.dumps(protocol))
    expired = output.parent / "expired"
    resume.freeze_addendum(protocol_path, old, expired)

    def no_dispatch(*args, **kwargs):
        pytest.fail("Expired deadline must not dispatch")

    assert resume.run(expired / "addendum.json", cell_runner=no_dispatch) == []
    batch = resume.read(expired / "batch.json")
    assert batch["deadline_reached"] and len(batch["not_dispatched"]) == 3
    assert resume.read(expired / "union-manifest.json")["completed"] == 1
