import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from drivingbench.sim.episode import Episode, EpisodePolicy


class Runtime:
    def __init__(self):
        self.on_evidence = self.error = self.end_reason = self.pending_end = None
        self.requests = []
        self.last = self.tick(0)

    def pose_snapshot(self):
        return dict(self.last)

    def finish(self, reason):
        self.requests.append(reason)
        if self.pending_end is None:
            self.pending_end = reason

    def tick(self, t, *, speed=0, hits=(), outside=False, finish=False, violation=None, **extra):
        self.last = dict(
            sim_time_s=t,
            speed_mps=speed,
            pose=[t, 0, 0],
            map_pose=[t, 0, 0],
            collision_ids=list(hits),
            outside_course=outside,
            boundary_violation=outside if violation is None else violation,
            finish_reached=finish,
            valid_finish=finish and not hits and not outside,
            **extra,
        )
        if self.pending_end and speed == 0:
            self.end_reason = self.pending_end
        if self.on_evidence:
            self.on_evidence("sim_tick", **self.last)
        return self.last


def episode(tmp_path, policy=None):
    runtime = Runtime()
    return runtime, Episode(runtime, tmp_path / "episode", policy, metadata={"scene_sha256": "abc"})


def test_contact_beats_simultaneous_finish_and_keeps_post_trigger_evidence(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(0, finish=True)
    runtime.tick(1, speed=0.4, hits=["a"], finish=True, outside=True)
    runtime.tick(1.1, speed=0.3, hits=["a", "b"], outside=True)
    runtime.tick(2, speed=0)
    result = run.close()
    assert runtime.requests == ["marker_hit"]
    assert result["first_cause"]["evidence"]["speed_mps"] == 0.4
    assert result["outcome"] == "failure"
    assert result["touched_ids"] == ["a", "b"]
    assert result["exit_count"] == 1
    assert result["last_evidence"]["pose"][0] > result["first_contact"]["pose"][0]
    assert result["native_end_reason"] == "marker_hit"
    rows = [json.loads(line) for line in (run.output / "events.jsonl").read_text().splitlines()]
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert sum(row["kind"] == "episode_trigger" for row in rows) == 1
    assert run.close() == result


def test_finish_needs_strict_speed_and_continuous_clean_dwell(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(0, finish=True)
    runtime.tick(0.9, speed=0.05, finish=True)
    runtime.tick(1, finish=True)
    runtime.tick(1.9, finish=True)
    assert runtime.requests == []
    runtime.tick(2, finish=True)
    assert runtime.requests == ["finish"]
    assert run.close()["outcome"] == "success"


def test_model_stop_is_not_finish_and_timeout_has_precedence(tmp_path):
    runtime, run = episode(tmp_path, EpisodePolicy(time_limit_s=60))
    runtime.tick(30, speed=0, reason="stop_now")
    assert runtime.requests == []
    runtime.tick(59, finish=True)
    runtime.tick(60, finish=True)
    assert runtime.requests == ["time_limit"]
    assert run.close()["outcome"] == "timeout"


def test_observe_only_records_without_stopping(tmp_path):
    runtime, run = episode(tmp_path, EpisodePolicy(enabled=False))
    runtime.tick(500, hits=["a"], outside=True)
    runtime.tick(501, finish=True)
    runtime.tick(502, finish=True)
    result = run.close()
    assert runtime.requests == []
    assert result["outcome"] == "observed"
    assert result["first_contact"] and result["first_finish"]


def test_renderer_failure_stays_distinct_and_retains_first_course_cause(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(1, hits=["a"], speed=0.7)
    run.record("sim_render_error", error="renderer died")
    result = run.close()
    assert result["outcome"] == "infrastructure_failure"
    assert result["infrastructure_failure"]["source"] == "sim_render_error"
    assert result["first_cause"]["reason"] == "marker_hit"


def test_exit_trigger_and_reentry_count_are_independent_of_first_cause(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(1, outside=True, speed=0.8)
    runtime.tick(2)
    runtime.tick(3, outside=True)
    result = run.close()
    assert runtime.requests == ["corridor_exit"]
    assert result["exit_count"] == 2
    assert result["first_exit"]["sim_time_s"] == 1
    assert result["first_cause"]["reason"] == "corridor_exit"


@pytest.mark.parametrize("kind", ["sim_runtime_error", "sim_render_error"])
def test_worker_errors_end_attempt_without_becoming_course_failure(tmp_path, kind):
    runtime, run = episode(tmp_path)
    run.record(kind, error="worker died", **runtime.pose_snapshot())
    result = run.close()
    assert runtime.requests == ["infrastructure_failure"]
    assert result["first_cause"] is None
    assert result["infrastructure_failure"]["source"] == kind


def test_invalid_time_evidence_is_not_serialized_as_a_valid_tick(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(1)
    runtime.tick(0.5)
    result = run.close()
    assert result["ticks"] == 1
    assert result["last_evidence"]["sim_time_s"] == 1
    assert result["outcome"] == "infrastructure_failure"


def test_disk_writer_cannot_block_tick_callback(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = Episode._write

    def delayed(self, stream):
        entered.set()
        release.wait(5)
        original(self, stream)

    monkeypatch.setattr(Episode, "_write", delayed)
    runtime, run = episode(tmp_path)
    try:
        assert entered.wait(1)
        with ThreadPoolExecutor() as pool:
            future = pool.submit(lambda: [runtime.tick(i / 100, speed=1) for i in range(100)])
            assert len(future.result(timeout=1)) == 100
        assert not release.is_set()
    finally:
        release.set()
        run.close()
    assert run.summary()["ticks"] == 100


def test_metadata_is_frozen_and_writer_failure_is_reported(tmp_path, monkeypatch):
    metadata = {"hashes": {"scene": "abc"}}
    runtime = Runtime()
    run = Episode(runtime, tmp_path / "episode", metadata=metadata)
    metadata["hashes"]["scene"] = "changed"
    assert (
        json.loads((run.output / "metadata.json").read_text())["metadata"]["hashes"]["scene"]
        == "abc"
    )
    original = json.dumps

    def broken(value, **kwargs):
        if value.get("kind") == "sim_tick":
            raise OSError("disk failed")
        return original(value, **kwargs)

    monkeypatch.setattr(json, "dumps", broken)
    runtime.tick(1)
    with pytest.raises(OSError, match="disk failed"):
        run.close()
    summary = json.loads((run.output / "summary.json").read_text())
    assert summary["outcome"] == "infrastructure_failure"
    assert summary["infrastructure_failure"]["source"] == "recorder"


def test_runtime_error_at_close_and_bad_policy_are_explicit(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.error = "native worker failed"
    assert run.close()["outcome"] == "infrastructure_failure"
    for kwargs in ({"time_limit_s": 0}, {"finish_dwell_s": float("nan")}, {"enabled": "yes"}):
        with pytest.raises(ValueError):
            EpisodePolicy(**kwargs)


def test_terminal_state_updates_summary_without_a_fake_tick_or_finish(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(2, speed=5.5, finish=True)
    before = dict(runtime.last)
    runtime.on_evidence("sim_terminal_impact", terminal_reason="divider_tip", **before)
    after = dict(before, speed_mps=0)
    runtime.on_evidence("sim_terminal_state", terminal_reason="divider_tip", **after)
    result = run.close()
    assert result["ticks"] == 1 and result["stopped"]
    assert result["last_evidence"]["speed_mps"] == 0
    assert result["last_evidence"]["terminal_reason"] == "divider_tip"
    assert runtime.requests == [] and result["first_cause"] is None
    events = [json.loads(line) for line in (run.output / "events.jsonl").read_text().splitlines()]
    impact = next(row for row in events if row["kind"] == "sim_terminal_impact")
    terminal = next(row for row in events if row["kind"] == "sim_terminal_state")
    assert impact["speed_mps"] == 5.5 and terminal["speed_mps"] == 0
    assert impact["pose"] == terminal["pose"] and impact["sim_time_s"] == terminal["sim_time_s"]


def test_forgiven_boundary_touch_records_raw_diagnostics_then_allows_finish(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(1, outside=True, violation=False, speed=1)
    assert runtime.requests == []
    runtime.tick(2, finish=True)
    runtime.tick(3, finish=True)
    result = run.close()
    assert result["outcome"] == "success"
    assert result["exit_count"] == 1
    assert result["first_exit"]["sim_time_s"] == 1
    assert result["first_cause"]["reason"] == "finish"


def test_raw_touch_precedes_excursion_intervention(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(1, outside=True, violation=False, speed=1)
    runtime.tick(2, outside=True, violation=True, speed=1)
    result = run.close()
    assert runtime.requests == ["corridor_exit"]
    assert result["first_exit"]["sim_time_s"] == 1
    assert result["first_cause"]["evidence"]["sim_time_s"] == 2
    assert result["first_cause"]["evidence"]["boundary_violation"]


def test_marker_contact_wins_during_forgiven_boundary_clip(tmp_path):
    runtime, run = episode(tmp_path)
    runtime.tick(1, outside=True, violation=False, hits=["tangent"], finish=True)
    assert run.close()["first_cause"]["reason"] == "marker_hit"


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_episode_rejects_invalid_boundary_tolerance(tolerance):
    with pytest.raises(ValueError, match="Boundary tolerance"):
        EpisodePolicy(boundary_tolerance_m=tolerance)


def lenient_policy(**kwargs):
    return EpisodePolicy(
        stop_on_marker_contact=False,
        boundary_tolerance_m=1,
        strict_boundary_tolerance_m=0.5,
        **kwargs,
    )


def test_lenient_marker_score_does_not_prevent_finishing_or_stop_motion(tmp_path):
    runtime, run = episode(tmp_path, lenient_policy())
    runtime.tick(1, speed=1, hits=["dome"], strict_boundary_violation=False)
    runtime.tick(2, speed=1, hits=["dome", "other"], strict_boundary_violation=False)
    assert runtime.requests == []
    runtime.tick(3, finish=True, hits=["dome"], strict_boundary_violation=False)
    runtime.tick(4, finish=True, hits=["dome"], strict_boundary_violation=False)
    result = run.close()
    assert runtime.requests == ["finish"]
    assert result["outcome"] == "success"
    assert result["touched_ids"] == ["dome", "other"]
    assert result["strict_score"]["failed"]
    assert result["strict_score"]["first_cause"]["evidence"]["sim_time_s"] == 1
    assert result["strict_score"]["first_cause"]["reason"] == "marker_hit"
    events = [json.loads(line) for line in (run.output / "events.jsonl").read_text().splitlines()]
    assert sum(row["kind"] == "strict_score_trigger" for row in events) == 1


@pytest.mark.parametrize(
    "contact, expected",
    [
        ({"hits": ["tall"], "obstacle_collision_ids": ["tall"]}, "obstacle_contact"),
        ({"scenario_contact": {"contacts": [{"id": "child"}]}}, "hazard_contact"),
    ],
)
def test_lenient_tall_obstacle_and_actor_contact_end_drive(tmp_path, contact, expected):
    runtime, run = episode(tmp_path, lenient_policy())
    runtime.tick(1, speed=0.6, finish=True, strict_boundary_violation=False, **contact)
    assert runtime.requests == [expected]
    assert run.close()["first_cause"]["reason"] == expected


def test_lenient_reference_excursion_latches_without_braking(tmp_path):
    runtime, run = episode(tmp_path, lenient_policy())
    runtime.tick(1, speed=0.6, outside=True, violation=False, strict_boundary_violation=True)
    assert runtime.requests == []
    runtime.tick(2, speed=0.6, outside=True, violation=True, strict_boundary_violation=True)
    result = run.close()
    assert runtime.requests == ["corridor_exit"]
    assert result["strict_score"]["first_cause"]["evidence"]["sim_time_s"] == 1
    assert result["first_cause"]["evidence"]["sim_time_s"] == 2


def test_lenient_policy_requires_reference_geometry(tmp_path):
    runtime, run = episode(tmp_path, lenient_policy())
    runtime.tick(1)
    assert run.close()["outcome"] == "infrastructure_failure"
    assert runtime.requests == ["infrastructure_failure"]


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_episode_rejects_invalid_strict_boundary_tolerance(tolerance):
    with pytest.raises(ValueError, match="Strict boundary tolerance"):
        EpisodePolicy(strict_boundary_tolerance_m=tolerance)
