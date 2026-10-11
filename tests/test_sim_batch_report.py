import hashlib
import json

import pytest
from drivingbench.sim import batch_report


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def stream(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / "batch"
    scene = tmp_path / "scene.json"
    save(
        scene,
        {
            "name": "fixture",
            "boundaries": {"left": [[0, -2], [10, -2]], "right": [[0, 2], [10, 2]]},
        },
    )
    assignments = [
        dict(model=model, attempt=i, status="not_run")
        for model in ("fable", "astra", "sol", "grok")
        for i in (1, 2, 3)
    ]
    assignments[0]["status"] = "attempted"
    save(root / "batch.json", dict(assignments=assignments, attempts=[], batch_error=None))
    save(root / "protocol.json", {"scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest()})
    folder = root / "fable/attempt-1"
    summary = dict(
        outcome="failure",
        elapsed_s=2,
        stopped=True,
        native_end_reason="marker_hit",
        first_cause={"reason": "marker_hit"},
        infrastructure_failure=None,
    )
    save(folder / "attempt.json", {"episode": summary, "wall_s": 3, "client_errors": []})
    save(folder / "episode/summary.json", summary)
    ticks = [
        dict(
            kind="sim_tick",
            sequence=i + 1,
            sim_time_s=i,
            map_pose=pose,
            speed_mps=0 if i == 2 else 1,
            collision_ids=["cone"] if i else [],
            outside_course=i == 2,
            finish_reached=i == 2,
            valid_finish=False,
        )
        for i, pose in enumerate(([0, 0, 0], [3, 4, 0], [3, 4, 0]))
    ]
    stream(
        folder / "episode/events.jsonl",
        [*ticks, dict(kind="episode_trigger", sequence=4, reason="marker_hit", evidence=ticks[1])],
    )
    rows = [
        dict(kind="telemetry", speed_mps=0, held=True, standstill=True),
        dict(kind="tool", tool="observe", client="MCP", outcome={"state": "held"}),
        dict(kind="tool", tool="set_motion", client="MCP", outcome={"error": "native_disengaged"}),
        dict(
            kind="tool",
            tool="stop_now",
            client="MCP",
            arguments={"reason": "already stopped"},
            outcome={"status": "stopping"},
        ),
        dict(kind="tool", tool="stop_now", client="operator", outcome={"status": "stopping"}),
        dict(kind="segment_end", reason="operator_brake"),
    ]
    for i, row in enumerate(rows):
        row.update(id=f"e{i}", segment="a", timestamp=f"2026-10-01T00:00:0{i}+00:00")
    stream(folder / "producer/recordings/segments/a/events.jsonl", rows)
    save(root / "grok/skipped.json", {"reason": "auth_unavailable"})
    save(root / "fable/remaining-skipped.json", {"after_attempt": 1, "reason": "client_error"})
    return root, scene


def test_reports_all_assignments_actual_path_and_separate_stopping(batch):
    root, scene = batch
    report = batch_report.summarize(root, scene)
    assert len(report["attempts"]) == 12
    row = report["attempts"][0]
    assert row["distance_m"] == 5 and row["max_progress_m"] is None
    assert row["first_contact"]["sim_time_s"] == 1
    assert row["first_outside"]["sim_time_s"] == 2
    assert row["first_finish_region"]["sim_time_s"] == 2
    assert row["outcome"] == "failure"
    assert row["first_cause"]["reason"] == "marker_hit"
    assert row["tools"]["outcomes"]["set_motion:native_disengaged"] == 1
    stops = row["tools"]["stop_calls"]
    assert [stop["actor"] for stop in stops] == ["model_transport", "operator"]
    assert stops[0]["preceding_telemetry"]["standstill"]
    assert stops[0]["arguments"]["reason"] == "already stopped"
    assert stops[0]["source"]["line"] == 4
    assert row["private_termination_events"][0]["reason"] == "marker_hit"
    assert report["attempts"][1]["skip"]["reason"] == "client_error"
    assert report["attempts"][-1]["skip"]["reason"] == "auth_unavailable"
    assert report["attempts"][-1]["distance_m"] is None
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256(batch_report.Path(path).read_bytes()).hexdigest() == digest


def test_infrastructure_and_missing_evidence_are_not_course_failure(batch):
    root, scene = batch
    path = root / "batch.json"
    value = json.loads(path.read_text())
    value["assignments"][3]["status"] = "attempted"
    value["attempts"] = [dict(model="astra", attempt=1, infrastructure_error="renderer crashed")]
    save(path, value)
    row = batch_report.summarize(root, scene)["attempts"][3]
    assert row["status"] == "attempted" and row["outcome"] == "incomplete_evidence"
    assert row["supervisor_error"] == "renderer crashed"
    assert row["distance_m"] is None and len(row["missing_sources"]) == 3


def test_centerline_is_explicit_and_invalid_data_fails():
    points = [[0, 0, 0], [1, 3, 2], [2, 3, 1]]
    assert batch_report.maximum_progress(points, None) is None
    assert batch_report.maximum_progress(points, [[0, 0], [3, 0], [3, 4]]) == 5
    with pytest.raises(ValueError, match="zero-length"):
        batch_report.maximum_progress(points, [[0, 0], [0, 0]])


def test_frozen_scene_and_trace_integrity(batch):
    root, scene = batch
    with pytest.raises(ValueError, match="Completed batch"):
        batch_report.summarize(root / "missing", scene)
    original = scene.read_text()
    save(scene, {"name": "wrong scene"})
    with pytest.raises(ValueError, match="frozen batch"):
        batch_report.summarize(root, scene)
    scene.write_text(original)
    events = root / "fable/attempt-1/episode/events.jsonl"
    stream(events, [dict(kind="sim_tick", sim_time_s=1, map_pose=[float("nan"), 0, 0])])
    with pytest.raises(ValueError, match="Nonfinite"):
        batch_report.summarize(root, scene)


def test_cli_retains_reference_and_refuses_overwrite(batch, tmp_path, monkeypatch):
    root, scene = batch
    real = tmp_path / "real.json"
    save(real, {"primary_models": [{"model": "astra", "attempts": 2, "curated_successes": 1}]})
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv",
        [
            "batch_report",
            str(root),
            "--scene",
            str(scene),
            "--output",
            str(output),
            "--real-reference",
            str(real),
        ],
    )
    batch_report.main()
    report = json.loads((output / "report.json").read_text())
    assert report["real_reference"]["models"][0]["attempts"] == 2
    with pytest.raises(FileExistsError):
        batch_report.main()
