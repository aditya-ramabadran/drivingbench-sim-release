"""Paired simulator captures keep their own acquisition time through failures."""

import base64
import json
import queue
import shlex
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from drivingbench.device.cameras import camera_clock
from drivingbench.sim.cameras import SceneRenderer, SimCameras
from drivingbench.sim.producer import simulator_app
from drivingbench.sim.scenarios import hazard_scene
from fastapi.testclient import TestClient
from test_sim_live import Renderer, scene_file


def test_study_renderer_launch_imports_actor_policy_without_parent_environment(
    tmp_path, monkeypatch
):
    interpreter = tmp_path / "runs/sim/visual/venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    interpreter.chmod(0o755)
    monkeypatch.setattr("drivingbench.sim.cameras.REPO", tmp_path)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, str(tmp_path / "invalid-parent-environment"))
    calibration = tmp_path / "deliberately-absent-calibration.json"
    scene = hazard_scene(
        {
            "frame": "pilot_ground",
            "calibration": str(calibration),
            "background": {"checkpoint": "unused"},
        },
        "child_entering",
        assets={},
    )
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene))
    renderer = SceneRenderer(scene_path, tmp_path / "rendered")
    with pytest.raises(RuntimeError, match="stream closed"):
        renderer.start()
    log = renderer.worker.log.read_text()
    assert "FileNotFoundError" in log and str(calibration) in log
    assert "ModuleNotFoundError" not in log
    assert renderer.worker.process is None


class PairedRenderer:
    def __init__(self):
        self.requests = queue.Queue()
        self.responses = queue.Queue()
        self.errors = {}

    def start(self):
        return {"ready": True}

    def render_frames(self, snapshot):
        self.requests.put(snapshot)
        value = self.responses.get(timeout=5)
        if isinstance(value, Exception):
            raise value
        frames, self.errors = value
        return frames

    def close(self):
        self.responses.put(RuntimeError("closed"))


def test_shutdown_cancels_inflight_pair_without_recording_infrastructure_failure():
    renderer = PairedRenderer()
    snapshot = dict(pose=(0, 0, 0), sim_time_s=1, captured_at=camera_clock())
    cameras = SimCameras(SimpleNamespace(pose_snapshot=lambda: snapshot), renderer)
    evidence = queue.Queue()
    cameras.on_evidence = lambda kind, **data: evidence.put((kind, data))
    cameras.start()
    renderer.requests.get(timeout=2)
    cameras.close()
    assert not cameras.thread.is_alive()
    assert evidence.get_nowait() == ("sim_render_shutdown", {"error": "closed"})
    assert evidence.empty() and not cameras.frames
    assert cameras.errors == {"wide": "camera_unavailable"}


def test_active_startup_failure_remains_an_infrastructure_error():
    class BrokenStartup(PairedRenderer):
        def start(self):
            raise RuntimeError("startup crashed")

    cameras = SimCameras(SimpleNamespace(), BrokenStartup())
    evidence = queue.Queue()
    cameras.on_evidence = lambda kind, **data: evidence.put((kind, data))
    cameras.start()
    assert evidence.get(timeout=2) == ("sim_render_error", {"error": "startup crashed"})
    cameras.close()
    assert all(
        "startup: RuntimeError: startup crashed" == value for value in cameras.errors.values()
    )


def test_scene_renderer_reads_paired_paths_and_preserves_snapshot(tmp_path):
    renderer = SceneRenderer.__new__(SceneRenderer)
    renderer.output = tmp_path
    received = []

    def request(**payload):
        received.append(payload)
        paths = {}
        for camera in ("narrow", "wide"):
            path = tmp_path / f"{camera}.jpg"
            path.write_bytes(camera.encode())
            paths[camera] = str(path)
        return {"paths": paths, "metrics": {"render_s": 0.2}}

    renderer.worker = SimpleNamespace(request=request)
    snapshot = dict(pose=(1, 2, 3), sim_time_s=12.3, captured_at=99.4, actor_onsets={"dog": 4.5})
    assert renderer.render_frames(snapshot) == {"narrow": b"narrow", "wide": b"wide"}
    assert renderer.errors == {}
    assert received[0]["pose"] == [1, 2, 3]
    assert received[0]["sim_time_s"] == 12.3
    assert received[0]["actor_onsets"] == {"dog": 4.5}
    assert snapshot["captured_at"] == 99.4
    assert renderer.render_snapshot(snapshot) == b"narrow"


def test_scene_renderer_retains_other_camera_when_one_file_is_missing(tmp_path):
    renderer = SceneRenderer.__new__(SceneRenderer)
    renderer.output = tmp_path
    path = tmp_path / "narrow.jpg"
    path.write_bytes(b"narrow")
    renderer.worker = SimpleNamespace(
        request=lambda **_: {
            "paths": {"narrow": str(path), "wide": str(tmp_path / "missing.jpg")},
            "metrics": {},
        }
    )
    assert renderer.render_frames(dict(pose=(0, 0, 0), sim_time_s=0)) == {"narrow": b"narrow"}
    assert "FileNotFoundError" in renderer.errors["wide"]


@pytest.mark.parametrize("failure", ["omitted", "reported", "exception", "invalid"])
def test_capture_pair_and_partial_failure_preserve_original_time(failure):
    renderer = PairedRenderer()
    snapshot = dict(
        pose=(1, 2, 3), sim_time_s=4, captured_at=camera_clock(), actor_onsets={"dog": 2}
    )
    captures = []

    def capture():
        value = dict(snapshot, captured_at=snapshot["captured_at"] + 0.01 * len(captures))
        captures.append(value)
        return value

    native = SimpleNamespace(pose_snapshot=capture)
    cameras = SimCameras(native, renderer, interval=0)
    evidence = queue.Queue()
    cameras.on_evidence = lambda kind, **data: evidence.put((kind, data))
    cameras.start()
    try:
        assert renderer.requests.get(timeout=2) == snapshot
        assert not cameras.frames
        renderer.responses.put(({"narrow": b"n1", "wide": b"w1"}, {}))
        kind, data = evidence.get(timeout=2)
        assert kind == "sim_frame"
        with cameras.lock:
            first = dict(cameras.frames)
            assert {frame.captured_at for frame in first.values()} == {snapshot["captured_at"]}
            assert set(data["frame_ids"]) == {"narrow", "wide"}
            assert not cameras.errors
        second = renderer.requests.get(timeout=2)
        if failure == "exception":
            renderer.responses.put(RuntimeError("worker lost"))
        elif failure == "invalid":
            renderer.responses.put(({"narrow": b"n2", "wide": b""}, {}))
        else:
            renderer.responses.put(
                ({"narrow": b"n2"}, {"wide": "wide render failed"} if failure == "reported" else {})
            )
        while evidence.get(timeout=2)[0] != "sim_render_error":
            pass
        with cameras.lock:
            assert cameras.frames["wide"] is first["wide"]
            assert cameras.errors["wide"]
            assert cameras.frames["wide"].captured_at == snapshot["captured_at"]
            if failure != "exception":
                assert cameras.frames["narrow"].jpeg == b"n2"
                assert cameras.frames["narrow"].captured_at == second["captured_at"]
                assert "narrow" not in cameras.errors
            else:
                assert cameras.frames["narrow"] is first["narrow"]
                assert cameras.errors["narrow"]
    finally:
        cameras.stop.set()
        renderer.close()
        cameras.close()


def test_normal_dual_observe_keeps_physics_running_and_stale_wide_honest(tmp_path):
    renderer = PairedRenderer()
    jpeg = Renderer().jpeg
    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=renderer)
    app.state.settings.patch({"road_camera_mode": "narrow_and_wide"})
    cameras = app.state.sim_cameras
    delivered = threading.Event()
    with TestClient(app) as client:
        first = renderer.requests.get(timeout=2)
        record = cameras.on_evidence

        def evidence(kind, **data):
            record(kind, **data)
            if kind == "sim_frame":
                delivered.set()

        cameras.on_evidence = evidence
        renderer.responses.put(({"narrow": jpeg, "wide": jpeg}, {}))
        assert delivered.wait(2)
        packet = client.get("/observe").json()
        assert [image["camera"] for image in packet["images"]] == ["narrow", "wide"]
        assert all(base64.b64decode(image["data"]) == jpeg for image in packet["images"])
        assert "camera_reason" not in packet
        for camera in ("narrow", "wide"):
            response = client.get(f"/camera/{camera}")
            assert float(response.headers["X-Captured-At"]) == first["captured_at"]
        renderer.requests.get(timeout=2)
        before = app.state.sim_native.pose_snapshot()["sim_time_s"]
        time.sleep(0.12)
        assert app.state.sim_native.pose_snapshot()["sim_time_s"] > before + 0.05
        # Aging the capture models an old cached pair without waiting the stale timeout.
        with cameras.lock:
            old = cameras.frames["wide"]
            cameras.frames["wide"] = type(old)(old.key + "-aged", camera_clock() - 4, old.jpeg)
        delivered.clear()
        renderer.responses.put(({"narrow": jpeg}, {"wide": "injected wide failure"}))
        assert delivered.wait(2)
        packet = client.get("/observe").json()
        assert packet["camera_reason"] == "camera_stale"
        images = {image["camera"]: image for image in packet["images"]}
        assert images["wide"]["age_s"] >= 4
        assert images["narrow"]["age_s"] < 2
        assert client.get("/status").json()["camera_errors"]["wide"] == "injected wide failure"
        cameras.stop.set()
        renderer.close()


def test_wide_only_success_does_not_relabel_cached_narrow_evidence():
    renderer = PairedRenderer()
    native = SimpleNamespace(
        pose_snapshot=lambda: dict(pose=(0, 0, 0), sim_time_s=2, captured_at=camera_clock())
    )
    cameras = SimCameras(native, renderer, interval=0)
    evidence = queue.Queue()
    cameras.on_evidence = lambda kind, **data: evidence.put((kind, data))
    cameras.start()
    try:
        renderer.requests.get(timeout=2)
        renderer.responses.put(({"narrow": b"n1", "wide": b"w1"}, {}))
        assert evidence.get(timeout=2)[0] == "sim_frame"
        old = cameras.frames["narrow"]
        renderer.requests.get(timeout=2)
        renderer.responses.put(({"wide": b"w2"}, {"narrow": "narrow failed"}))
        kind, data = evidence.get(timeout=2)
        assert kind == "sim_render_error" and data["camera"] == "narrow"
        assert cameras.frames["narrow"] is old
        assert cameras.frames["wide"].jpeg == b"w2"
        assert evidence.empty()
    finally:
        cameras.stop.set()
        renderer.close()
        cameras.close()


def test_delayed_wide_publishes_one_old_capture_pair_without_stalling_physics(tmp_path):
    jpeg = Renderer().jpeg

    class DelayedWide(PairedRenderer):
        def render_frames(self, snapshot):
            narrow = jpeg
            # The narrow render has completed before the wide render waits on its queue.
            wide = super().render_frames(snapshot)["wide"]
            return {"narrow": narrow, "wide": wide}

    renderer = DelayedWide()
    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=renderer)
    app.state.settings.patch({"road_camera_mode": "narrow_and_wide"})
    cameras = app.state.sim_cameras
    delivered = queue.Queue()
    with TestClient(app) as client:
        try:
            first = renderer.requests.get(timeout=2)
            record = cameras.on_evidence

            def evidence(kind, **data):
                record(kind, **data)
                if kind == "sim_frame":
                    delivered.put(data)

            cameras.on_evidence = evidence
            renderer.responses.put(({"wide": jpeg}, {}))
            delivered.get(timeout=2)
            second = renderer.requests.get(timeout=2)
            with cameras.lock:
                old_frames = dict(cameras.frames)
            assert all(frame.captured_at == first["captured_at"] for frame in old_frames.values())

            deadline = time.monotonic() + 2
            while app.state.sim_native.pose_snapshot()["sim_time_s"] < second["sim_time_s"] + 0.12:
                assert time.monotonic() < deadline, "Physics stopped during the wide render"
                time.sleep(0.005)
            assert camera_clock() - second["captured_at"] >= 0.1
            assert delivered.empty()
            with cameras.lock:
                assert all(cameras.frames[name] is frame for name, frame in old_frames.items())
            for camera in ("narrow", "wide"):
                response = client.get(f"/camera/{camera}")
                assert float(response.headers["X-Captured-At"]) == first["captured_at"]

            renderer.responses.put(({"wide": jpeg}, {}))
            metrics = delivered.get(timeout=2)
            with cameras.lock:
                for name, old in old_frames.items():
                    assert cameras.frames[name].key != old.key
                    assert cameras.frames[name].captured_at == second["captured_at"]
            assert metrics["sim_time_s"] == second["sim_time_s"]
            assert metrics["image_age_s"] >= 0.1
            assert set(metrics["frame_ids"]) == {"narrow", "wide"}
            packet = client.get("/observe").json()
            assert {image["camera"] for image in packet["images"]} == {"narrow", "wide"}
            assert all(image["age_s"] >= 0.1 for image in packet["images"])
            for camera in ("narrow", "wide"):
                response = client.get(f"/camera/{camera}")
                assert float(response.headers["X-Captured-At"]) == second["captured_at"]
        finally:
            cameras.stop.set()
            renderer.close()
    assert not cameras.thread.is_alive()


@pytest.mark.parametrize("render_duration, expected_wait", [(0.01, 0.04), (0.08, 0.0)])
def test_capture_interval_is_minimum_cycle_not_extra_frame_delay(
    monkeypatch, render_duration, expected_wait
):
    clock = [0.0]
    waits = []

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, duration):
            waits.append(duration)
            self.stopped = True

    class Backend:
        def start(self):
            return {}

        def render(self, pose):
            clock[0] += render_duration
            return b"jpeg"

        def close(self):
            pass

    monkeypatch.setattr(
        "drivingbench.sim.cameras.time", SimpleNamespace(perf_counter=lambda: clock[0])
    )
    snapshot = dict(pose=(0, 0, 0), sim_time_s=1, captured_at=camera_clock())
    cameras = SimCameras(SimpleNamespace(pose_snapshot=lambda: snapshot), Backend(), interval=0.05)
    cameras.stop = Stop()
    cameras.start()
    cameras.thread.join(timeout=1)
    assert not cameras.thread.is_alive()
    assert waits == pytest.approx([expected_wait])
    assert cameras.frames["narrow"].captured_at == snapshot["captured_at"]
