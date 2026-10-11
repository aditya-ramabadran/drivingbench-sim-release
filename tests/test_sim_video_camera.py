import hashlib
import io
import json
import shutil
import subprocess

import pytest
from drivingbench.sim import video_camera as module
from PIL import Image


def calibration(tmp_path):
    clips = {}
    for name in module.CLIPS:
        source = tmp_path / f"{name}.mp4"
        source.write_bytes(name.encode())
        clips[name] = dict(
            path=str(source),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            width=8,
            height=4,
            frame_count=3,
            fps=2,
            progress_m=[0, 2] if name == "approach" else [2, 6],
            frame_indices=[0, 2],
        )
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            dict(
                version=1, approach_end_m=2, fork_tip_m=4, clips=clips, focal_px=10, warp_depth_m=10
            )
        )
    )
    return path


def camera(tmp_path, monkeypatch):
    calls = []

    def decode(clip):
        calls.append(clip["path"])
        return [Image.new("RGB", (8, 4), (index * 100, 0, 0)).tobytes() for index in range(3)]

    monkeypatch.setattr(module, "_decode", decode)
    return module.VideoCamera(calibration(tmp_path)), calls


def test_selection_uses_progress_and_decodes_once(tmp_path, monkeypatch):
    subject, calls = camera(tmp_path, monkeypatch)
    first = subject.render(1)
    assert subject.last_metadata["frame_index"] == 1
    assert subject.last_metadata["clip"] == "approach"
    subject.render(3, "switch")
    assert subject.last_metadata["clip"] == "switch"
    assert subject.last_metadata["before_tip"] is True
    assert subject.render(1) == first
    assert len(calls) == 3
    subject.render(2)
    assert subject.last_metadata["clip"] == "approach"
    subject.render(6, "keep")
    assert subject.last_metadata["frame_index"] == 2
    subject.close()
    subject.close()
    assert not subject.frames


@pytest.mark.parametrize("progress", [-1, 6.01, float("nan"), float("inf")])
def test_unsupported_progress_does_not_freeze_last_frame(tmp_path, monkeypatch, progress):
    subject, _ = camera(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        subject.render(progress)


@pytest.mark.parametrize(
    "update",
    [
        {"progress_m": [0, 0]},
        {"frame_indices": [0, 3]},
        {"fps": 0},
        {"frame_count": True},
        {"sha256": "a" * 64},
    ],
)
def test_bad_calibration_rejected_without_decode(tmp_path, update):
    path = calibration(tmp_path)
    data = json.loads(path.read_text())
    data["clips"]["approach"].update(update)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.load_calibration(path)


def test_mirror_changes_pixels_without_changing_source_frame(tmp_path, monkeypatch):
    subject, _ = camera(tmp_path, monkeypatch)
    image = Image.new("RGB", (8, 4))
    for x in range(4):
        for y in range(4):
            image.putpixel((x, y), (255, 255, 255))
    monkeypatch.setattr(module, "_decode", lambda clip: [image.tobytes()] * 3)
    normal = Image.open(io.BytesIO(subject.render(1)))
    mirrored = Image.open(io.BytesIO(subject.render(1, mirror=True)))
    assert normal.getpixel((0, 0))[0] > 200
    assert mirrored.getpixel((7, 0))[0] > 200
    assert subject.last_metadata["frame_index"] == 1


def test_warp_is_bounded_and_opt_in(tmp_path, monkeypatch):
    subject, _ = camera(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="limited"):
        subject.render(1, lateral_m=0.251)
    with pytest.raises(ValueError, match="limited"):
        subject.render(1, yaw_deg=2.01)
    assert Image.open(io.BytesIO(subject.render(1, lateral_m=0.1, yaw_deg=1))).size == (8, 4)


def test_snapshot_uses_physical_branch_and_preserves_single_image(tmp_path, monkeypatch):
    subject, _ = camera(tmp_path, monkeypatch)
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({"video_camera": {"calibration": str(calibration(tmp_path))}}))
    monkeypatch.setattr(module, "VideoCamera", lambda _: subject)
    renderer = module.VideoSnapshotRenderer(scene)
    renderer.render_snapshot(
        {"video_fork": {"progress_m": 3, "actual_branch": "left", "locked_branch": None}}
    )
    assert renderer.metrics["clip"] == "switch"
    renderer.render_snapshot(
        {"video_fork": {"progress_m": 5, "actual_branch": "right", "locked_branch": "left"}}
    )
    assert renderer.metrics["clip"] == "switch"
    renderer.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_real_cpu_decode_validates_frame_count(tmp_path):
    path = tmp_path / "tiny.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=8x4:r=2",
            "-frames:v",
            "3",
            "-c:v",
            "ffv1",
            str(path),
        ],
        check=True,
    )
    clip = dict(path=str(path), width=8, height=4, frame_count=3)
    frames = module._decode(clip)
    assert len(frames) == 3 and len(frames[0]) == 96
    assert frames[0][0] > 240
    with pytest.raises(ValueError, match="exceeds"):
        module._decode(dict(clip, frame_count=2))
    with pytest.raises(ValueError, match="ended"):
        module._decode(dict(clip, frame_count=4))


def test_incomplete_video_coverage_does_not_redefine_physical_tip(tmp_path):
    path = calibration(tmp_path)
    data = json.loads(path.read_text())
    data["clips"]["switch"]["progress_m"] = [2, 3]
    path.write_text(json.dumps(data))
    assert module.load_calibration(path)["fork_tip_m"] == 4


def test_decode_failure_does_not_publish_partial_cache(tmp_path, monkeypatch):
    subject, _ = camera(tmp_path, monkeypatch)

    def fail(clip):
        if "keep.mp4" in clip["path"]:
            raise RuntimeError("decode failed")
        return [bytes(96)] * 3

    monkeypatch.setattr(module, "_decode", fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        subject.start()
    assert subject.frames == {}


def terminal_renderer(tmp_path, monkeypatch):
    subject, _ = camera(tmp_path, monkeypatch)
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({"video_camera": {"calibration": str(calibration(tmp_path))}}))
    monkeypatch.setattr(module, "VideoCamera", lambda _: subject)
    return module.VideoSnapshotRenderer(scene)


def terminal_snapshot(overshoot=0.04):
    return dict(
        speed_mps=5.5,
        video_fork=dict(
            progress_m=6 + overshoot,
            requested_branch="left",
            actual_branch="right",
            locked_branch="left",
            endpoint_reached=True,
            phase="assigned_outcome",
            endpoint_m=6,
            endpoint_overshoot_m=overshoot,
            endpoint_step_m=0.055,
        ),
    )


def test_terminal_exact_last_frame_keeps_actual_overshoot(tmp_path, monkeypatch):
    renderer = terminal_renderer(tmp_path, monkeypatch)
    snapshot = terminal_snapshot()
    result = renderer.render_snapshot(snapshot)
    assert result == renderer.camera.render(6, "switch")
    assert renderer.metrics["frame_index"] == 2
    assert renderer.metrics["progress_m"] == 6
    assert renderer.metrics["actual_progress_m"] == 6.04
    assert renderer.metrics["endpoint_overshoot_m"] == pytest.approx(0.04)
    assert snapshot["video_fork"]["progress_m"] == 6.04


@pytest.mark.parametrize(
    "changes",
    [
        {"endpoint_reached": False},
        {"phase": "continuation"},
        {"locked_branch": None},
        {"endpoint_m": 5.9},
        {"endpoint_overshoot_m": 0.01},
        {"endpoint_step_m": 0.061},
        {"endpoint_step_m": None},
        {"progress_m": 6.056},
        {"progress_m": 5.99, "endpoint_overshoot_m": 0},
    ],
)
def test_no_ongoing_or_unbounded_endpoint_freeze(tmp_path, monkeypatch, changes):
    renderer = terminal_renderer(tmp_path, monkeypatch)
    snapshot = terminal_snapshot()
    snapshot["video_fork"].update(changes)
    with pytest.raises(ValueError):
        renderer.render_snapshot(snapshot)


def test_requested_branch_precedes_lateral_diagnostic(tmp_path, monkeypatch):
    renderer = terminal_renderer(tmp_path, monkeypatch)
    renderer.render_snapshot(
        dict(
            video_fork=dict(
                progress_m=3, requested_branch="left", actual_branch="right", locked_branch=None
            )
        )
    )
    assert renderer.metrics["clip"] == "switch"


def test_terminal_stop_uses_retained_physical_step_not_relabelled_speed(tmp_path, monkeypatch):
    renderer = terminal_renderer(tmp_path, monkeypatch)
    snapshot = terminal_snapshot()
    snapshot["speed_mps"] = 0
    renderer.render_snapshot(snapshot)
    assert renderer.metrics["terminal_frame"] is True
    assert snapshot["speed_mps"] == 0


def test_real_runtime_termination_preserves_terminal_frame(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from drivingbench.sim.video_fork import VideoForkConfig, VideoForkRuntime

    renderer = terminal_renderer(tmp_path, monkeypatch)
    clock = [1000.0]
    course = SimpleNamespace(map_pose=lambda pose: pose, assess=lambda _: {})
    config = VideoForkConfig(commit_m=2, endpoints_m={"left": 6, "right": 6}, branch_rule="command")
    runtime = VideoForkRuntime(
        course, config=config, clock=lambda: clock[0], frame_clock=lambda: clock[0]
    )
    try:
        while not runtime.video_snapshot()["endpoint_reached"]:
            clock[0] = runtime._wall_at_sim(runtime.native.time + module.DT)
            runtime.advance()
        before = runtime.pose_snapshot()
        runtime.terminate("assigned_branch_outcome")
        stopped = runtime.pose_snapshot()
        assert stopped["speed_mps"] == 0
        assert stopped["video_fork"]["progress_m"] == before["video_fork"]["progress_m"]
        assert renderer.render_snapshot(stopped) == renderer.camera.render(6, "keep")
        assert renderer.metrics["terminal_frame"]
        assert renderer.metrics["endpoint_overshoot_m"] > 0
    finally:
        runtime.close()
        renderer.close()
