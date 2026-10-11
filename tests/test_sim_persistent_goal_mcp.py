import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from drivingbench.device.ipc import NativeClient
from drivingbench.device.service import create_app as producer_app
from drivingbench.gateway.app import create_app as gateway_app
from drivingbench.mcp.tools import make_server
from drivingbench.sim import event_study_v2 as run
from drivingbench.sim.run import isolated_command
from drivingbench.sim.studies import persistent_goal_mcp as private
from mcp.client.stdio import stdio_client
from test_e2e import MemoryCameras, NativeLoop, http_server, summary, wait_status

from mcp import ClientSession, StdioServerParameters


async def connect(stack, command, cwd):
    streams = await stack.enter_async_context(
        stdio_client(StdioServerParameters(command=command[0], args=command[1:], cwd=str(cwd)))
    )
    session = await stack.enter_async_context(ClientSession(*streams))
    return session, await session.initialize()


def digest(content):
    return [
        hashlib.sha256(base64.b64decode(block.data)).hexdigest()
        for block in content
        if block.type == "image"
    ]


async def test_realism_observe_preserves_every_existing_field_and_image():
    packet = dict(
        timestamp="2026-10-03T10:00:00+00:00",
        state="stopping",
        speed_mps=0.4,
        steering_percent=-20,
        remaining_s=0,
        image_age_s=1.7,
        camera_reason="wide_stale",
        reason="camera_unavailable",
        error="observation_unavailable",
        images=[dict(camera="narrow", data="aW1hZ2U=", mime_type="image/jpeg")],
    )
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=packet))
    async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
        original = make_server(client)
        wrapped = private.study_server(client, realism=True)
        before = await original.call_tool("observe", {})
        after = await wrapped.call_tool("observe", {})
        expected = json.loads(before[0].text) | {"vehicle": private.VEHICLE}
        assert json.loads(after[0].text) == expected
        assert after[1:] == before[1:]
        assert await wrapped.list_tools() == await original.list_tools()
        assert wrapped._mcp_server.instructions == original._mcp_server.instructions.replace(
            "drivingbench_sandbox", private.REALVEHICLE_SERVER
        )


async def test_private_stdio_uses_actual_producer_images_errors_and_braking(tmp_path):
    with tempfile.TemporaryDirectory(prefix="db-private-", dir="/tmp") as local:
        socket_path = Path(local) / "native.sock"
        cameras = MemoryCameras()
        with (
            NativeLoop(socket_path) as loop,
            http_server(
                producer_app(tmp_path, native=NativeClient(socket_path), cameras=cameras)
            ) as producer,
            http_server(gateway_app(producer)) as gateway,
        ):
            async with (
                httpx.AsyncClient(base_url=gateway, timeout=5) as ui,
                AsyncExitStack() as stack,
            ):
                await wait_status(ui, lambda state: state.get("gear") == "drive")
                loop.update(brake_pressed=False, enabled=True, longitudinal_active=True)
                await wait_status(ui, lambda state: state.get("enabled") is True)
                assert (
                    await ui.patch("/api/settings", json={"road_camera_mode": "narrow_and_wide"})
                ).status_code == 200
                sessions = {}
                catalogs = {}
                for arm in ("straight_sandbox", "straight_realism", "free_sandbox"):
                    command, name = private.configuration({"arm": arm}, gateway)
                    session, initialized = await connect(stack, command, tmp_path)
                    assert initialized.serverInfo.name == name
                    sessions[arm] = session
                    catalogs[arm] = (await session.list_tools()).tools
                assert catalogs["straight_realism"] == catalogs["straight_sandbox"]
                assert catalogs["free_sandbox"] == catalogs["straight_sandbox"]
                assert {tool.name for tool in catalogs["straight_realism"]} == {
                    "observe",
                    "set_motion",
                    "stop_now",
                }
                observations = {
                    arm: await session.call_tool("observe", {}) for arm, session in sessions.items()
                }
                source = (await ui.get("/api/observe")).json()
                expected_hashes = [
                    hashlib.sha256(base64.b64decode(image["data"])).hexdigest()
                    for image in source["images"]
                ]
                for arm, observed in observations.items():
                    fields = summary(observed)
                    assert fields["cameras"] == ["narrow", "wide"]
                    assert fields["state"] == "held" and fields["speed_mps"] == 0
                    assert digest(observed.content) == expected_hashes
                    assert ("vehicle" in fields) == (arm == "straight_realism")
                realism = sessions["straight_realism"]
                assert summary(observations["straight_realism"])["vehicle"] == private.VEHICLE
                later = summary(await realism.call_tool("observe", {}))
                assert datetime.fromisoformat(later["timestamp"]) > datetime.fromisoformat(
                    summary(observations["straight_realism"])["timestamp"]
                )
                assert (
                    later["image_age_s"] >= summary(observations["straight_realism"])["image_age_s"]
                )
                motion = dict(direction="left", steering_percent=20, speed_mps=0.6, duration_s=5)
                for arm, session in sessions.items():
                    invalid = await session.call_tool(
                        "set_motion", motion | {"steering_percent": -1}
                    )
                    assert invalid.isError
                    assert (await session.call_tool("stop_now", {"extra": True})).isError
                    assert (
                        summary(await session.call_tool("set_motion", motion))["status"]
                        == "accepted"
                    )
                    assert "vehicle" not in summary(await session.call_tool("stop_now", {}))
                await ui.patch("/api/settings", json={"speed_limit_mps": 0.9})
                rejected = summary(await realism.call_tool("set_motion", motion | {"speed_mps": 1}))
                assert rejected["error"] == "speed_limit_exceeded" and "vehicle" not in rejected
                assert (
                    summary(await realism.call_tool("set_motion", motion))["status"] == "accepted"
                )
                await wait_status(ui, lambda state: state.get("command") is not None)
                loop.update(speed_mps=0.6, standstill=False, lateral_active=True)
                expired = await wait_status(
                    ui, lambda state: state.get("command") is None, timeout=7
                )
                assert expired["state"] == "stopping"
                assert summary(await realism.call_tool("observe", {}))["state"] == "stopping"
                stopped = summary(await realism.call_tool("stop_now", {}))
                assert stopped["status"] == "stopping" and "vehicle" not in stopped
                loop.update(speed_mps=0, standstill=True, lateral_active=False)
                await wait_status(ui, lambda state: state.get("state") == "held")
                with cameras.lock:
                    cameras.frames.clear()
                missing = await realism.call_tool("observe", {})
                assert summary(missing)["reason"] == "camera_unavailable"
                assert summary(missing)["vehicle"] == private.VEHICLE
                assert len(missing.content) == 1


@pytest.mark.parametrize("arm", ["straight_sandbox", "free_sandbox"])
def test_sandbox_factory_is_existing_command(arm):
    gateway = "http://127.0.0.1:12345"
    assert private.configuration({"arm": arm}, gateway) == (
        isolated_command(gateway),
        "drivingbench_sandbox",
    )


def test_factory_rejects_unknown_arm():
    with pytest.raises(ValueError, match="Unknown"):
        private.configuration({"arm": "typo"}, "http://127.0.0.1:12345")


@pytest.mark.parametrize(
    "gateway",
    [
        "http://127.0.0.1:8766",
        "http://127.0.0.1:8876",
        "http://127.0.0.1:8877",
        "http://127.0.0.1",
        "http://localhost:12345",
        "https://127.0.0.1:12345",
        "http://127.0.0.1:bad",
        "http://user@127.0.0.1:12345",
    ],
)
def test_cli_rejects_public_or_nonlocal_gateway(gateway):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "drivingbench.sim.studies.persistent_goal_mcp",
            "--gateway-url",
            gateway,
            "--realism",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "dedicated 127.0.0.1 HTTP port" in result.stderr


@pytest.mark.parametrize("custom", [False, True])
def test_cell_factory_is_optional_and_preserves_default_preparation(tmp_path, monkeypatch, custom):
    prepared = []
    monkeypatch.setattr(run, "require_disk_space", lambda: {})
    monkeypatch.setattr(run, "auth_preflight", lambda *_a, **_k: {"eligible": True})

    def prepare(_spec, _state, command, _folder, **kwargs):
        prepared.append((command, kwargs))
        return None

    monkeypatch.setattr(run, "prepare_isolated", prepare)
    row = {"model": "astra", "arm": "straight_realism"}
    result = run._cell(
        {},
        row,
        tmp_path / "cell",
        tmp_path / "recovery",
        time.monotonic() + 60,
        renderer_factory=None,
        mcp_factory=private.configuration if custom else None,
    )
    assert result["status"] == "failed_auth"
    command, kwargs = prepared[0]
    assert kwargs == dict(
        profile="normal",
        consolidate_driving_aliases=True,
        **({"server_name": private.REALVEHICLE_SERVER} if custom else {}),
    )
    assert command[2] == (
        "drivingbench.sim.studies.persistent_goal_mcp" if custom else "drivingbench.replay.mcp"
    )
