"""Real stdio MCP and TCP producer/gateway, with only the image renderer replaced."""

import asyncio
import base64
import io
import json
import sys
import threading
import time
from contextlib import AsyncExitStack

import httpx
from drivingbench.gateway.app import create_app as gateway_app
from drivingbench.sim.producer import DEFAULT_SCENE, simulator_app
from mcp.client.stdio import stdio_client
from PIL import Image
from test_e2e import http_server, summary, wait_status

from mcp import ClientSession, StdioServerParameters


class Renderer:
    def __init__(self):
        output = io.BytesIO()
        Image.new("RGB", (1344, 760), (90, 100, 110)).save(output, "JPEG", quality=92)
        self.jpeg = output.getvalue()
        self.block = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = threading.Event()
        self.closed = threading.Event()

    def start(self):
        return {"ready": True, "renderer": "test"}

    def render(self, pose):
        if self.block.is_set():
            self.entered.set()
            if not self.release.wait(20):
                raise TimeoutError("test renderer was not released")
        if self.fail.is_set():
            raise RuntimeError("injected render failure")
        return self.jpeg

    def close(self):
        self.closed.set()
        self.release.set()


def scene_file(tmp_path):
    scene = json.loads(DEFAULT_SCENE.read_text())
    scene.pop("wide_camera", None)
    scene.update(
        name="Live test course",
        background={"checkpoint": "runs/sim/visual/nonexistent-test.npz"},
        start={"x": 0, "y": 0, "heading_deg": 0},
        boundaries={"left": [[-100, 10], [100, 10]], "right": [[-100, -10], [100, -10]]},
        finish={"polygon": [[80, -5], [95, -5], [95, 5], [80, 5]]},
        markers=[],
        objects=[],
    )
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    return path


async def connect(stack, gateway, cwd):
    streams = await stack.enter_async_context(
        stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m",
                    "drivingbench.replay.mcp",
                    "--gateway-url",
                    gateway,
                    "--variant",
                    "reference",
                    "--client",
                    "sim-live-test",
                ],
                cwd=str(cwd),
            )
        )
    )
    client = await stack.enter_async_context(ClientSession(*streams))
    initialized = await client.initialize()
    assert initialized.serverInfo.name == "drivingbench_sandbox"
    assert {tool.name for tool in (await client.list_tools()).tools} == {
        "observe",
        "set_motion",
        "stop_now",
    }
    return client


async def test_real_mcp_continuous_physics_expiry_stop_and_retained_camera_failure(tmp_path):
    renderer = Renderer()
    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=renderer)
    runtime = app.state.sim_native
    with http_server(app) as producer, http_server(gateway_app(producer)) as gateway:
        async with AsyncExitStack() as stack, httpx.AsyncClient(base_url=gateway) as http:
            client = await connect(stack, gateway, tmp_path)
            first = await client.call_tool("observe", {})
            initial = summary(first)
            assert initial["state"] == "held"
            assert initial["cameras"] == ["narrow"]
            assert len(first.content) == 2 and first.content[1].type == "image"
            with Image.open(io.BytesIO(base64.b64decode(first.content[1].data))) as image:
                assert image.format == "JPEG" and image.size == (1344, 760)
            assert not (
                {"pose", "collision_ids", "outside_course", "valid_finish"} & initial.keys()
            )
            command = dict(direction="straight", steering_percent=0, speed_mps=0.8, duration_s=5)
            assert summary(await client.call_tool("set_motion", command))["status"] == "accepted"
            await wait_status(http, lambda state: state.get("speed_mps", 0) > 0.2, timeout=4)

            renderer.block.set()
            assert await asyncio.to_thread(renderer.entered.wait, 2)
            before = runtime.pose_snapshot()
            started = time.monotonic()
            blocked = await client.call_tool("observe", {})
            assert time.monotonic() - started < 1
            assert len(blocked.content) == 2
            age = summary(blocked)["image_age_s"]
            await asyncio.sleep(0.15)
            after = runtime.pose_snapshot()
            assert after["sim_time_s"] - before["sim_time_s"] >= 0.1
            assert after["pose"][0] > before["pose"][0]
            assert time.monotonic() - runtime.status()["received_at"] < 0.2

            await asyncio.sleep(3.05)
            stale = await client.call_tool("observe", {})
            packet = summary(stale)
            assert packet["camera_reason"] == "camera_stale"
            assert packet["image_age_s"] > 3 and packet["image_age_s"] > age + 3
            assert len(stale.content) == 2
            assert stale.content[1].data == blocked.content[1].data
            expired = await wait_status(http, lambda state: state.get("command") is None, timeout=2)
            assert expired["enabled"] and runtime.end_reason is None

            assert summary(await client.call_tool("set_motion", command))["status"] == "accepted"
            await wait_status(http, lambda state: state.get("speed_mps", 0) > 0.2, timeout=4)
            rolling = runtime.pose_snapshot()
            stopped = await client.call_tool("stop_now", {"reason": "test"})
            assert summary(stopped)["status"] == "stopping"
            assert len(stopped.content) == 1
            await wait_status(http, lambda state: state.get("state") == "held", timeout=6)
            assert runtime.pose_snapshot()["pose"][0] > rolling["pose"][0]
            renderer.fail.set()
            renderer.block.clear()
            renderer.release.set()
            await wait_status(
                http, lambda state: state.get("end_reason") == "infrastructure_failure"
            )
            retained = await client.call_tool("observe", {})
            assert summary(retained)["camera_reason"] == "camera_stale"
            assert summary(retained)["image_age_s"] > packet["image_age_s"]
            assert retained.content[1].data == blocked.content[1].data
            assert app.state.sim_episode.summary()["outcome"] == "infrastructure_failure"
            assert runtime.error is None
            calls = (await http.get("/api/history")).json()["calls"]
            assert {row["tool"] for row in calls} == {"observe", "set_motion", "stop_now"}
    assert renderer.closed.is_set()
    assert not runtime._thread.is_alive()
    assert not app.state.sim_cameras.thread.is_alive()
    assert runtime.error is None


def test_blocked_renderer_is_released_during_clean_teardown(tmp_path):
    renderer = Renderer()
    renderer.block.set()
    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=renderer)
    with http_server(app):
        assert renderer.entered.wait(2)
        before = app.state.sim_native.pose_snapshot()["sim_time_s"]
        time.sleep(0.04)
        assert app.state.sim_native.pose_snapshot()["sim_time_s"] > before
    assert renderer.closed.is_set()
    assert not app.state.sim_cameras.thread.is_alive()
    assert not app.state.sim_native._thread.is_alive()
