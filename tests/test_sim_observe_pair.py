"""Observe pins one paired publication without locking JPEG work or hiding failures."""

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from drivingbench.device import cameras as camera_module
from drivingbench.device.cameras import Frame, camera_clock
from drivingbench.sim.producer import simulator_app
from fastapi.testclient import TestClient
from PIL import Image
from test_sim_live import Renderer, scene_file


def jpeg(color):
    stream = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(stream, "JPEG", quality=92)
    return stream.getvalue()


def publication(name, captured_at):
    return {
        "narrow": Frame(name + "-n", captured_at, jpeg((50 if name == "old" else 150, 20, 40))),
        "wide": Frame(name + "-w", captured_at, jpeg((20, 50 if name == "old" else 150, 40))),
    }


def publish(cameras, frames, errors=None):
    assert cameras.lock.acquire(timeout=1), "Publication blocked by observation JPEG work"
    try:
        cameras.frames.update(frames)
        cameras.errors.clear()
        cameras.errors.update(errors or {})
    finally:
        cameras.lock.release()


@pytest.fixture
def app_client(tmp_path):
    renderer = Renderer()
    renderer.block.set()
    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=renderer)
    app.state.settings.patch({"road_camera_mode": "narrow_and_wide"})
    deliveries = []
    app.state.sim_observe_delivery = lambda **event: deliveries.append(event)
    with TestClient(app) as client:
        assert renderer.entered.wait(2)
        try:
            yield app, client, deliveries
        finally:
            renderer.release.set()


def test_http_observe_cannot_mix_publications_between_camera_reads(app_client, monkeypatch):
    app, client, deliveries = app_client
    cameras = app.state.sim_cameras
    old, new = publication("old", camera_clock() - 2), publication("new", camera_clock())
    publish(cameras, old)
    adjust = camera_module.adjust
    calls = []

    def interleave(data, adjustments):
        calls.append(data)
        if data == old["narrow"].jpeg:
            publish(cameras, new)
        return adjust(data, adjustments)

    monkeypatch.setattr(camera_module, "adjust", interleave)
    packet = client.get("/observe").json()
    assert calls == [old["narrow"].jpeg, old["wide"].jpeg]
    event = deliveries[0]
    assert [frame["captured_at"] for frame in event["frames"]] == [old["narrow"].captured_at] * 2
    assert all(frame["frame_id"].startswith("old-") for frame in event["frames"])
    assert all(image["age_s"] >= 2 for image in packet["images"])
    assert "camera_errors" not in packet
    assert cameras.frames == new
    assert client.get("/camera/wide").headers["X-Frame-ID"].startswith("new-w-")
    assert len(deliveries) == 1


def test_concurrent_observers_keep_independent_pairs_while_publication_and_physics_continue(
    app_client, monkeypatch
):
    app, client, deliveries = app_client
    cameras = app.state.sim_cameras
    old, new = publication("old", camera_clock() - 2), publication("new", camera_clock())
    publish(cameras, old)
    entered, release = Event(), Event()
    adjust = camera_module.adjust

    def interleave(data, adjustments):
        if data == old["narrow"].jpeg:
            entered.set()
            assert release.wait(3)
        return adjust(data, adjustments)

    monkeypatch.setattr(camera_module, "adjust", interleave)
    with ThreadPoolExecutor() as pool:
        first = pool.submit(client.get, "/observe")
        assert entered.wait(2)
        before = app.state.sim_native.pose_snapshot()["sim_time_s"]
        publish(cameras, new)
        second = pool.submit(client.get, "/observe")
        assert second.result(timeout=2).status_code == 200
        with app.state.sim_native._condition:
            assert app.state.sim_native._condition.wait_for(
                lambda: app.state.sim_native._pose["sim_time_s"] > before, timeout=1
            )
        release.set()
        assert first.result(timeout=2).status_code == 200
    assert {
        tuple(frame["frame_id"].split("-")[0] for frame in event["frames"]) for event in deliveries
    } == {("old", "old"), ("new", "new")}
    assert len({event["delivery_id"] for event in deliveries}) == 2


def test_partial_failure_keeps_original_age_and_snapshot_error_even_when_recovery_publishes(
    app_client, monkeypatch
):
    app, client, deliveries = app_client
    cameras = app.state.sim_cameras
    old, new = publication("old", camera_clock() - 8), publication("new", camera_clock())
    retained = {"narrow": new["narrow"], "wide": old["wide"]}
    publish(cameras, retained, {"wide": "injected_wide_failure"})
    adjust = camera_module.adjust

    def recover(data, adjustments):
        if data == new["narrow"].jpeg:
            publish(cameras, new)
        return adjust(data, adjustments)

    monkeypatch.setattr(camera_module, "adjust", recover)
    packet = client.get("/observe").json()
    assert len(packet["images"]) == 2
    assert packet["camera_reason"] == "camera_stale"
    assert packet["images"][1]["age_s"] >= 8
    assert deliveries[0]["camera_errors"] == {"wide": "injected_wide_failure"}
    assert deliveries[0]["frames"][1]["captured_at"] == old["wide"].captured_at
    assert base64.b64decode(packet["images"][1]["data"])
    with cameras.lock:
        cameras.frames.pop("wide")
        cameras.errors["wide"] = "missing_camera_frame"
    packet = client.get("/observe").json()
    assert len(packet["images"]) == 1
    assert packet["camera_reason"] == "camera_unavailable"
    assert deliveries[1]["camera_errors"] == {"wide": "missing_camera_frame"}


async def test_real_mcp_observe_keeps_both_old_frames_when_new_pair_publishes_between_reads(
    tmp_path, monkeypatch
):
    from contextlib import AsyncExitStack

    from drivingbench.gateway.app import create_app as gateway_app
    from test_e2e import http_server, summary
    from test_sim_live import connect

    renderer = Renderer()
    renderer.block.set()
    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=renderer)
    app.state.settings.patch({"road_camera_mode": "narrow_and_wide"})
    deliveries = []
    app.state.sim_observe_delivery = lambda **event: deliveries.append(event)
    cameras = app.state.sim_cameras
    old, new = publication("old", camera_clock() - 2), publication("new", camera_clock())
    adjust = camera_module.adjust

    def interleave(data, adjustments):
        if data == old["narrow"].jpeg:
            publish(cameras, new)
        return adjust(data, adjustments)

    monkeypatch.setattr(camera_module, "adjust", interleave)
    with http_server(app) as producer, http_server(gateway_app(producer)) as gateway:
        try:
            assert renderer.entered.wait(2)
            publish(cameras, old)
            async with AsyncExitStack() as stack:
                client = await connect(stack, gateway, tmp_path)
                result = await client.call_tool("observe", {})
                packet = summary(result)
                assert packet["cameras"] == ["narrow", "wide"]
                assert len([block for block in result.content if block.type == "image"]) == 2
                assert packet["image_age_s"] >= 2
                assert [frame["captured_at"] for frame in deliveries[0]["frames"]] == [
                    old["narrow"].captured_at
                ] * 2
                assert all(
                    frame["frame_id"].startswith("old-") for frame in deliveries[0]["frames"]
                )
                assert cameras.frames == new
        finally:
            renderer.release.set()
