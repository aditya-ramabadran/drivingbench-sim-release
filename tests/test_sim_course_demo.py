import copy
import io
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from drivingbench.sim.course_demo import render, replay_segment
from PIL import Image


class Native:
    def __init__(self, time):
        self.time = time
        self.car = SimpleNamespace(speed=0.0)
        self.bridge = SimpleNamespace(controller=None)
        self.x = 0.0
        self.tick()

    @property
    def pose(self):
        return self.x, 0, 0

    def tick(self):
        self.time += 0.01
        self.x += self.car.speed * 0.01

    def call(self, event):
        self.car.speed = 0 if event["tool"] == "stop_now" else 1
        self.tick()
        return {"status": "accepted"}

    def status(self):
        return {"speed_mps": self.car.speed}


class Course:
    def map_pose(self, pose):
        return pose

    def assess(self, pose):
        hit = 0.019 < pose[0] < 0.031
        finish = pose[0] > 0.019
        return {
            "collision_ids": ["marker"] if hit else [],
            "outside_course": pose[0] > 0.039,
            "finish_reached": finish,
            "valid_finish": finish and not hit and pose[0] <= 0.039,
        }


def event(t, kind="tool", tool="observe", session="selected", event_id=None):
    timestamp = datetime(2026, 9, 17, tzinfo=timezone.utc) + timedelta(seconds=t)
    return {
        "id": event_id or str(t),
        "timestamp": timestamp.isoformat(),
        "kind": kind,
        "tool": tool,
        "session": session,
        "segment": "segment",
        "arguments": {},
        "outcome": {"timestamp": "older camera timestamp"},
        "images": [{"url": "real.jpg"}],
    }


def run(events):
    return replay_segment(
        events, "selected", Course(), native_factory=Native, make_request=lambda _, call: call
    )


def test_session_filter_preserves_call_arguments_times_and_assesses_every_tick():
    command = event(0.02, tool="set_motion")
    command["arguments"] = {"duration_s": 5, "reason": "retain original audit evidence"}
    result = run(
        [
            event(0, kind="segment_start", session="earlier"),
            event(0.001, tool="set_motion", session="earlier"),
            event(0.005, kind="telemetry", tool="set_motion"),
            command,
            event(0.035),
            event(0.065, tool="stop_now"),
            event(0.08),
            event(0.10, kind="segment_end"),
        ]
    )
    assert len(result["calls"]) == 4
    assert result["calls"][0]["arguments"] == command["arguments"]
    assert result["calls"][0]["processed_t"] == pytest.approx(0.02)
    assert result["calls"][0]["completed_t"] == pytest.approx(0.03)
    assert [row["t"] for row in result["states"]] == pytest.approx([i / 100 for i in range(11)])
    assert result["observations"][0]["t"] == 0.035
    assert result["observations"][0]["recorded_outcome"] == command["outcome"]
    assert result["summary"]["ever_hit_ids"] == ["marker"]
    assert result["summary"]["ever_outside_course"]
    assert any(row["collision_ids"] and row["finish_reached"] for row in result["states"])
    assert result["summary"]["accepted_motion_calls"] == 2


def test_each_engagement_resets_pose_and_runs_through_end_without_calls():
    moved = run(
        [
            event(0, kind="segment_start"),
            event(0.01, tool="set_motion"),
            event(0.1, kind="segment_end"),
        ]
    )
    reset = run([event(0, kind="segment_start"), event(0.1, kind="segment_end")])
    assert moved["summary"]["final_state"]["local_pose"][0] > 0
    assert reset["summary"]["final_state"]["local_pose"] == (0, 0, 0)
    assert reset["summary"]["simulated_duration_s"] == pytest.approx(0.1)


def test_curated_whole_segment_preserves_original_attribution_without_mutating_events():
    events = [
        event(0, kind="segment_start"),
        event(0.01, tool="set_motion", session="administrative-label"),
        event(0.02, session=None),
        event(0.03, kind="telemetry", session="selected"),
        event(0.04, tool="stop_now", session="selected"),
        event(0.1, kind="segment_end"),
    ]
    original = copy.deepcopy(events)
    result = replay_segment(
        events, None, Course(), native_factory=Native, make_request=lambda _, call: call
    )
    assert [row["recorded_session"] for row in result["calls"]] == [
        "administrative-label",
        None,
        "selected",
    ]
    assert result["summary"]["tool_calls"] == 3
    assert result["summary"]["accepted_motion_calls"] == 2
    assert events == original


def test_unended_segment_and_out_of_bounds_call_rejected():
    with pytest.raises(ValueError, match="published"):
        run([event(0, kind="segment_start")])
    with pytest.raises(ValueError, match="outside"):
        run([event(0, kind="segment_start"), event(1), event(0.1, kind="segment_end")])


def test_render_uses_every_observe_pose_in_source_order(tmp_path):
    poses = []

    class Renderer:
        def render(self, pose):
            poses.append(pose)
            output = io.BytesIO()
            Image.new("RGB", (1344, 760), (20, 80, 120)).save(output, "JPEG")
            return output.getvalue()

    report = {
        "observations": [
            {
                "id": str(i),
                "segment": "segment",
                "timestamp": "logged time",
                "t": t,
                "local_pose": [t, 0, 0],
            }
            for i, t in enumerate([0.031, 0.154, 0.159])
        ]
    }
    assert render(report, Renderer(), tmp_path / "frames", tmp_path / "sheet.jpg") == 3
    assert poses == [(0.031, 0, 0), (0.154, 0, 0), (0.159, 0, 0)]
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 3
