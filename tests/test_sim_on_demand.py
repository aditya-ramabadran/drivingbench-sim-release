"""Observe-demand captures stay paired and truthful without blocking physics."""

import fcntl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from drivingbench.device.cameras import camera_clock
from drivingbench.sim import cameras as module
from drivingbench.sim.cameras import OnDemandCameras


class Native:
    course = SimpleNamespace(start=(0, 0, 0))

    def __init__(self):
        self.captures = []

    def pose_snapshot(self):
        return {"pose": (0, 0, 0), "sim_time_s": 0, "captured_at": camera_clock()}

    def capture_snapshot(self):
        sampled = time.monotonic() - 0.02
        capture = {
            "pose": (len(self.captures), 0, 0),
            "sim_time_s": len(self.captures),
            "captured_at": camera_clock(),
            "monotonic_at": time.monotonic(),
            "pose_sample_monotonic_at": sampled,
        }
        self.captures.append(capture)
        return capture


class Renderer:
    def __init__(self):
        self.calls = []
        self.errors = {}
        self.failure = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.closed = False
        self.start_count = 0

    def start(self):
        self.start_count += 1
        return {"ready": True}

    def render_frames(self, snapshot):
        self.entered.set()
        assert self.release.wait(2)
        self.calls.append(snapshot)
        if self.failure:
            raise self.failure
        self.errors = {}
        return {
            camera: f"{camera}-{snapshot['sim_time_s']}".encode() for camera in ("narrow", "wide")
        }

    def close(self):
        self.closed = True
        self.release.set()


def start(native=None, renderer=None, **options):
    native, renderer = native or Native(), renderer or Renderer()
    cameras = OnDemandCameras(native, renderer, **options)
    cameras.start()
    assert cameras._startup_done.wait(2)
    return cameras, native, renderer


def test_no_idle_renders_and_each_observe_is_own_pair():
    cameras, native, renderer = start()
    try:
        first = dict(cameras.frames)
        assert renderer.start_count == 1 and len(renderer.calls) == 1
        cameras.start()
        assert renderer.start_count == 1
        with ThreadPoolExecutor(2) as pool:
            snapshots = list(pool.map(lambda _: cameras.observation_snapshot(), range(2)))
        assert len(renderer.calls) == len(native.captures) == 3
        assert first["narrow"].key != snapshots[0].frames["narrow"].key
        assert {snapshot.frames["narrow"].jpeg for snapshot in snapshots} == {
            b"narrow-1",
            b"narrow-2",
        }
        for snapshot in snapshots:
            index = snapshot.frames["narrow"].jpeg.decode().split("-")[-1]
            assert snapshot.frames["wide"].jpeg == f"wide-{index}".encode()
            assert snapshot.frames["narrow"].captured_at == snapshot.frames["wide"].captured_at
        assert cameras.last_metrics["pose_sample_age_s"] >= 0.02
    finally:
        cameras.close()
    assert renderer.closed and not cameras.thread.is_alive()


def test_failed_render_retains_old_identity_and_capture():
    cameras, _, renderer = start()
    try:
        before = dict(cameras.frames)
        renderer.failure = RuntimeError("injected render failure")
        snapshot = cameras.observation_snapshot()
        assert snapshot.frames == before
        assert snapshot.errors == {
            camera: "RuntimeError: injected render failure" for camera in ("narrow", "wide")
        }
        renderer.failure = None
        recovered = cameras.observation_snapshot()
        assert recovered.errors == {}
        assert recovered.frames["narrow"].key != before["narrow"].key
    finally:
        cameras.close()


def test_queue_wait_is_measured_and_capture_occurs_after_gpu_slot(tmp_path, monkeypatch):
    cameras, native, _ = start(render_lock_path=tmp_path / "gpu.lock")
    attempted = threading.Event()
    original_flock = fcntl.flock

    def observed_flock(fd, flags):
        if flags & fcntl.LOCK_NB:
            attempted.set()
        return original_flock(fd, flags)

    monkeypatch.setattr(module.fcntl, "flock", observed_flock)
    try:
        with (tmp_path / "gpu.lock").open("a+b") as lock, ThreadPoolExecutor(1) as pool:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            before = len(native.captures)
            requested = time.monotonic()
            waiting = pool.submit(cameras.observation_snapshot)
            assert attempted.wait(2)
            assert not waiting.done()
            assert len(native.captures) == before
            released = time.monotonic()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            snapshot = waiting.result(timeout=2)
        assert native.captures[-1]["monotonic_at"] >= released
        metrics = cameras.last_metrics
        assert metrics["queue_wait_s"] >= 0
        assert metrics["requested_at"] >= requested
        assert snapshot.frames["narrow"].captured_at == native.captures[-1]["captured_at"]
    finally:
        cameras.close()


def test_blocked_render_does_not_hold_publication_or_native_read_lock():
    cameras, native, renderer = start()
    try:
        renderer.entered.clear()
        renderer.release.clear()
        with ThreadPoolExecutor(1) as pool:
            waiting = pool.submit(cameras.observation_snapshot)
            assert renderer.entered.wait(2)
            assert cameras.lock.acquire(timeout=0.1)
            cameras.lock.release()
            assert native.pose_snapshot()["pose"] == (0, 0, 0)
            renderer.release.set()
            waiting.result(timeout=2)
    finally:
        cameras.close()


def test_close_cancels_startup_waiting_on_global_gpu_slot(tmp_path):
    path = tmp_path / "gpu.lock"
    with path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        renderer = Renderer()
        cameras = OnDemandCameras(Native(), renderer, render_lock_path=path)
        cameras.start()
        cameras.close()
        assert not cameras.thread.is_alive()
        assert renderer.start_count == 0 and renderer.closed
    with pytest.raises(RuntimeError, match="Camera closed"):
        cameras.observation_snapshot()


def test_two_producers_share_slot_without_mixing_source_pairs(tmp_path):
    first, first_native, first_renderer = start(render_lock_path=tmp_path / "gpu.lock")
    second, second_native, _ = start(render_lock_path=tmp_path / "gpu.lock")
    try:
        first_renderer.entered.clear()
        first_renderer.release.clear()
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(first.observation_snapshot)
            assert first_renderer.entered.wait(2)
            b = pool.submit(second.observation_snapshot)
            assert len(second_native.captures) == 1
            first_renderer.release.set()
            a_snapshot, b_snapshot = a.result(timeout=2), b.result(timeout=2)
        assert len(first_native.captures) == len(second_native.captures) == 2
        assert a_snapshot.frames["narrow"].key != b_snapshot.frames["narrow"].key
        assert a_snapshot.frames["narrow"].captured_at == first_native.captures[-1]["captured_at"]
        assert b_snapshot.frames["narrow"].captured_at == second_native.captures[-1]["captured_at"]
    finally:
        first.close()
        second.close()


def test_http_observe_renders_on_demand_while_native_motion_keeps_ticking(tmp_path):
    import base64
    import io

    from drivingbench.sim.producer import simulator_app
    from fastapi.testclient import TestClient
    from PIL import Image
    from test_sim_live import scene_file

    class HTTPRenderer(Renderer):
        def render_frames(self, snapshot):
            super().render_frames(snapshot)
            images = {}
            for camera, color in (("narrow", (90, 100, 110)), ("wide", (110, 100, 90))):
                stream = io.BytesIO()
                Image.new("RGB", (12, 8), color).save(stream, "JPEG", quality=92)
                images[camera] = stream.getvalue()
            return images

    renderer = HTTPRenderer()
    app = simulator_app(
        tmp_path / "run",
        scene_file(tmp_path),
        renderer=renderer,
        declaration={"time_scale": 1.0, "restrictions": {}, "camera_capture": "on_demand"},
    )
    app.state.settings.patch({"road_camera_mode": "narrow_and_wide"})
    deliveries = []
    app.state.sim_observe_delivery = lambda **event: deliveries.append(event)
    with TestClient(app) as client:
        cameras, native = app.state.sim_cameras, app.state.sim_native
        assert cameras._startup_done.wait(2)
        assert len(renderer.calls) == 1
        body = {"direction": "straight", "steering_percent": 0, "speed_mps": 0.8, "duration_s": 5}
        assert client.post("/motion", json=body).json()["status"] == "accepted"
        renderer.entered.clear()
        renderer.release.clear()
        with ThreadPoolExecutor(1) as pool:
            observation = pool.submit(client.get, "/observe")
            assert renderer.entered.wait(2)
            before = native.pose_snapshot()
            with native._condition:
                assert native._condition.wait_for(
                    lambda: native._pose["sim_time_s"] >= before["sim_time_s"] + 0.1,
                    timeout=1,
                )
            assert time.monotonic() - native.status()["received_at"] < 0.1
            renderer.release.set()
            packet = observation.result(timeout=2).json()
        assert len(renderer.calls) == 2
        assert len(packet["images"]) == len(deliveries[0]["frames"]) == 2
        assert {frame["captured_at"] for frame in deliveries[0]["frames"]} == {
            renderer.calls[-1]["captured_at"]
        }
        assert packet["image_age_s"] >= 0.05
        for image in packet["images"]:
            with Image.open(io.BytesIO(base64.b64decode(image["data"]))) as decoded:
                assert decoded.format == "JPEG"
        assert cameras.last_metrics["pose_sample_age_at_capture_s"] is None
        assert cameras.last_metrics["pose_sample_age_s"] is None
        assert cameras.last_metrics["pose_sample_monotonic_at"] is None
        assert native.pose_snapshot()["sim_time_s"] >= renderer.calls[-1]["sim_time_s"] + 0.1
        assert renderer.start_count == 1
    assert renderer.closed and not native._thread.is_alive()
