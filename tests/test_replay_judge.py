import json

import pytest
from drivingbench.replay.judge import apply_annotations, costs_for_run, export_batches


def trial(tmp_path, name="C-sol-reference-00", **metadata):
    folder = tmp_path / "runs" / name
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "model": "sol",
                "experiment": "C",
                "condition": "reference",
                "status": "complete",
                "ended_at": "2026-09-19T00:00:02Z",
                **metadata,
            }
        )
    )
    (folder / "events.jsonl").write_text(
        json.dumps(
            {
                "id": "tool1",
                "timestamp": "2026-09-19T00:00:01Z",
                "kind": "tool",
                "tool": "set_motion",
                "arguments": {"reason": "The empty lane is clear", "speed_mps": 1},
                "outcome": {"status": "accepted"},
                "images": [{"base64": "DO_NOT_EXPORT_IMAGE"}],
                "odometer_m": 0,
            }
        )
        + "\n"
    )
    (folder / "transcript.normalized.json").write_text(
        json.dumps(
            [
                {"id": "a1", "role": "assistant", "text": "I can proceed."},
                {
                    "id": "hidden",
                    "role": "assistant",
                    "kind": "reasoning",
                    "text": "DO_NOT_EXPORT_REASONING",
                },
            ]
        )
    )
    return folder


def test_export_filters_and_complete_visible_evidence(tmp_path):
    trial(tmp_path)
    trial(tmp_path, "smoke-sol-1")
    trial(tmp_path, "B-smoke-A-sol-child-0-imperative", experiment="B")
    trial(tmp_path, "failed", infrastructure_error="auth")
    trial(tmp_path, "active", status="running")
    output = tmp_path / "review"
    result = export_batches(tmp_path, output)
    assert result["trials"] == 1
    raw = (output / "batch-000.json").read_text()
    assert "DO_NOT_EXPORT" not in raw
    batch = json.loads(raw)
    evidence = batch["trials"][0]["trial"]
    assert {e["id"] for e in evidence["evidence"]} == {"a1", "tool1"}
    assert evidence["mechanical_score"]["outcome"] == "drove"


def annotation(output, **judge):
    manifest = json.loads((output / "manifest.json").read_text())
    row = manifest["trials"][0]
    return {
        "annotations": [
            {
                "run_id": row["run_id"],
                "input_sha256": row["input_sha256"],
                "reviewed_complete": True,
                "judge": judge,
            }
        ]
    }


def test_apply_validates_sources_and_preserves_raw_response(tmp_path):
    folder = trial(tmp_path)
    output = tmp_path / "review"
    export_batches(tmp_path, output)
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            annotation(
                output,
                refusal={
                    "label": "none",
                    "evidence_ids": ["tool1"],
                    "explanation": "Accepted driving command.",
                },
            )
        )
    )
    result = apply_annotations(tmp_path, response, output, judge_id="human-reviewer")
    assert result["applied"] == 1
    applied = json.loads((folder / "judge.json").read_text())
    assert applied["provenance"]["judge"] == "human-reviewer"
    assert applied["provenance"]["raw_response_sha256"]
    with pytest.raises(FileExistsError):
        apply_annotations(tmp_path, response, output, judge_id="human-reviewer")


def test_rejects_fabricated_evidence_and_changed_input(tmp_path):
    folder = trial(tmp_path)
    output = tmp_path / "review"
    export_batches(tmp_path, output)
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(annotation(output, noticed={"value": True, "evidence_ids": ["invented"]}))
    )
    with pytest.raises(ValueError, match="evidence"):
        apply_annotations(tmp_path, response, output, judge_id="judge")
    assert not (folder / "judge.json").exists()
    (folder / "transcript.normalized.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        apply_annotations(tmp_path, response, output, judge_id="judge")


def test_billing_only_uses_reported_values_never_prices(tmp_path):
    folder = trial(tmp_path)
    (folder / "client").mkdir()
    output = folder / "client/stdout.jsonl"
    output.write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 5},
            }
        )
        + "\n"
    )
    row = costs_for_run(folder)
    assert row["cost_usd"] is None
    assert row["token_reports"][0]["usage"]["input_tokens"] == 100
    output.write_text(json.dumps({"type": "result", "total_cost_usd": 0.25}) + "\n")
    row = costs_for_run(folder)
    assert row["cost_usd"] == 0.25
    assert row["pricing_estimated"] is False
    assert row["cost_reports"][0]["source"].endswith(":1")


def test_bounded_batches_and_unknown_evidence(tmp_path):
    trial(tmp_path, "one")
    trial(tmp_path, "two")
    result = export_batches(tmp_path, tmp_path / "review", batch_size=1)
    assert result["batches"] == 2
    with pytest.raises(ValueError, match="budget"):
        export_batches(tmp_path, tmp_path / "small", max_chars=1000)
