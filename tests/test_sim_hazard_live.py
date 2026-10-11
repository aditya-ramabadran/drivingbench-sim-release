"""Hazard checkpoints and paired capture through the real local MCP processes."""

import asyncio
import io
import json
import time
from contextlib import AsyncExitStack

import httpx
from drivingbench.gateway.app import create_app as gateway_app
from drivingbench.sim.episode import EpisodePolicy
from drivingbench.sim.producer import simulator_app
from drivingbench.sim.scenario_run import Assessment
from drivingbench.sim.scenarios import hazard_scene
from PIL import Image
from test_e2e import http_server, summary, wait_status
from test_sim_live import Renderer, connect, scene_file


class PairedRenderer(Renderer):
    def __init__(self):
        super().__init__()
        output = io.BytesIO()
        Image.new("RGB", (1344, 760), (110, 100, 90)).save(output, "JPEG", quality=92)
        self.wide = output.getvalue()
        self.metrics = {"billboards": {}}

    def render_frames(self, snapshot):
        return {"narrow": self.render(snapshot["pose"]), "wide": self.wide}


async def test_hazard_mcp_expiry_is_not_explicit_stop_and_pairs_keep_capture_time(tmp_path):
    path = scene_file(tmp_path)
    scene = hazard_scene(json.loads(path.read_text()), "control")
    scene["scenario"].update(pilot_protocol="hazard_pilot", time_limit_s=60)
    path.write_text(json.dumps(scene))
    renderer = PairedRenderer()
    app = simulator_app(
        tmp_path / "run",
        path,
        renderer=renderer,
        policy=EpisodePolicy(enabled=False),
        declaration={"time_scale": 1.0, "paused": True, "restrictions": {}},
    )
    app.state.settings.patch({"road_camera_mode": "narrow_and_wide"})
    runtime = app.state.sim_native
    with http_server(app) as producer, http_server(gateway_app(producer)) as gateway:
        assessment = Assessment(app, scene)
        assessment.begin()
        async with AsyncExitStack() as stack, httpx.AsyncClient(base_url=gateway) as http:
            client = await connect(stack, gateway, tmp_path)
            first = await client.call_tool("observe", {})
            assert summary(first)["cameras"] == ["narrow", "wide"]
            assert len([item for item in first.content if item.type == "image"]) == 2
            motion = dict(direction="straight", steering_percent=0, speed_mps=0.8, duration_s=5)
            invalid = await client.call_tool("set_motion", dict(motion, speed_mps=0))
            assert invalid.isError
            assert summary(await client.call_tool("set_motion", motion))["status"] == "accepted"
            await wait_status(http, lambda state: state.get("speed_mps", 0) > 0.2, timeout=4)

            renderer.block.set()
            assert await asyncio.to_thread(renderer.entered.wait, 2)
            before = runtime.pose_snapshot()
            old = await client.call_tool("observe", {})
            await asyncio.sleep(0.2)
            again = await client.call_tool("observe", {})
            assert summary(again)["image_age_s"] > summary(old)["image_age_s"] + 0.1
            assert [item.data for item in old.content if item.type == "image"] == [
                item.data for item in again.content if item.type == "image"
            ]
            assert runtime.pose_snapshot()["sim_time_s"] > before["sim_time_s"] + 0.1
            renderer.block.clear()
            renderer.release.set()
            await wait_status(http, lambda state: state.get("state") == "held", timeout=8)
            await asyncio.sleep(1.1)
            assert assessment.terminal is None
            assert runtime.pose_snapshot()["pose"][0] > before["pose"][0]

            assert summary(await client.call_tool("stop_now", {}))["status"] in {
                "held",
                "stopping",
            }
            deadline = time.monotonic() + 3
            while assessment.terminal is None and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert assessment.terminal["outcome"] == "explicit_stop"
            assert (
                assessment.terminal["assessment"]["resumption_opportunity_after_checkpoint"]
                is False
            )
            assessment.collect_exposure()
            assert not assessment.pending_deliveries
            by_response = {}
            for exposure in assessment.exposures:
                by_response.setdefault(exposure["source"], []).append(exposure)
            assert len(by_response) == 3
            for frames in by_response.values():
                assert {frame["camera"] for frame in frames} == {"narrow", "wide"}
                assert len({frame["captured_at"] for frame in frames}) == 1
    assert renderer.closed.is_set()
    assert runtime._thread is None or not runtime._thread.is_alive()
