"""Offline dynamics and the real gateway/MCP interface, without any device access."""

import json
from contextlib import AsyncExitStack
from types import SimpleNamespace

import httpx
from drivingbench.gateway.app import create_app as gateway_app
from drivingbench.replay.producer import VirtualNative, replay_app
from drivingbench.shared.contracts import PROTOCOL, Settings
from PIL import Image
from test_e2e import connect_mcp, http_server, summary


def parameters():
    return SimpleNamespace(
        launch_delay_s=0.2, accel_mps2=1, brake_mps2=2, steering_rate_deg_s=30, speed_noise_mps=0
    )


def motion(native, request_id="one", **fields):
    return native.call(
        dict(
            protocol=PROTOCOL,
            operation="motion",
            request_id=request_id,
            boot_id=native.controller.boot_id,
            command_epoch=native.controller.command_epoch,
            settings=Settings().model_dump(),
            motion=dict(direction="straight", steering_percent=0, speed_mps=1, duration_s=5)
            | fields,
        )
    )


def test_braking_expiry_replacement_and_distance():
    clock = [0.0]
    native = VirtualNative(100, parameters(), clock=lambda: clock[0])
    assert motion(native)["status"] == "accepted"
    clock[0] = 2
    assert native.status()["state"] == "executing"
    assert 1 < native.odometer_m < 2
    before = native.odometer_m
    result = native.call(
        dict(
            protocol=PROTOCOL,
            operation="stop",
            request_id="stop",
            boot_id=native.controller.boot_id,
            command_epoch=native.controller.command_epoch,
        )
    )
    assert result["status"] == "accepted"
    assert native.status()["state"] == "stopping"
    clock[0] = 3
    assert native.status()["state"] == "held"
    assert native.odometer_m > before
    stopped = native.odometer_m
    clock[0] = 4
    native.status()
    assert native.odometer_m == stopped
    assert motion(native, "next")["status"] == "accepted"
    clock[0] = 11
    assert native.status()["state"] == "held"
    assert native.odometer_m > stopped


class ColorClip:
    length_m = 100

    def frame_at(self, distance_m, camera="narrow"):
        return Image.new("RGB", (64, 48), (int(distance_m * 30) % 255, 60, 100))


async def test_real_mcp_replay_both_cameras_and_evidence(tmp_path):
    app = replay_app(
        tmp_path, ColorClip(), parameters(), settings=Settings(road_camera_mode="narrow_and_wide")
    )
    with http_server(app) as producer, http_server(gateway_app(producer)) as gateway:
        async with AsyncExitStack() as stack:
            client = await connect_mcp(stack, gateway, "study", tmp_path)
            result = await client.call_tool("observe", {})
            assert summary(result)["state"] == "held"
            assert summary(result)["cameras"] == ["narrow", "wide"]
            assert "odometer_m" not in summary(result)
            assert len(result.content) == 3
            move = await client.call_tool(
                "set_motion",
                dict(
                    direction="straight",
                    steering_percent=0,
                    speed_mps=1,
                    duration_s=5,
                    reason="fixture",
                ),
            )
            assert summary(move)["status"] == "accepted"
            await client.call_tool("stop_now", {})
            app.state.virtual_native.finish("fixture_complete")
            async with httpx.AsyncClient(base_url=gateway) as http:
                history = (await http.get("/api/history")).json()["calls"]
            assert all("odometer_m" in e for e in history)
            assert history[0]["source_distance_m"] == 0
    events = [
        json.loads(line)
        for p in tmp_path.glob("recordings/segments/*/events.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert any(e["kind"] == "segment_end" for e in events)
    assert all("odometer_m" in e for e in events)


def test_steering_holds_while_lateral_inactive_then_slews_and_holds_on_stop():
    clock = [0.0]
    dynamics = parameters()
    dynamics.launch_delay_s = 1.3
    native = VirtualNative(100, dynamics, clock=lambda: clock[0])
    assert motion(native, direction="left", steering_percent=20)["status"] == "accepted"
    clock[0] = 1
    status = native.status()
    assert status["standstill"] and not status["lateral_active"]
    assert status["steering_deg"] == 0
    clock[0] = 1.8
    status = native.status()
    assert status["lateral_active"] and status["speed_mps"] > 0
    assert 0 < status["steering_deg"] <= dynamics.steering_rate_deg_s * 0.5 + 1e-6
    angle = status["steering_deg"]
    native.call(
        dict(
            protocol=PROTOCOL,
            operation="stop",
            request_id="halt",
            boot_id=native.controller.boot_id,
            command_epoch=native.controller.command_epoch,
        )
    )
    clock[0] = 3
    status = native.status()
    assert status["held"]
    assert status["steering_deg"] == angle


def test_hazard_boundary_records_positive_speed_before_forced_halt():
    clock = [0.0]
    native = VirtualNative(10, parameters(), clock=lambda: clock[0], hazard_boundary_m=2)
    evidence = []
    native.on_evidence = lambda kind, **row: evidence.append({"kind": kind, **row})
    motion(native)
    clock[0] = 4
    native.status()
    assert native.end_reason == "hazard_boundary"
    assert native.odometer_m == 2
    assert native.speed_mps == 0
    assert len(evidence) == 1
    assert evidence[0]["kind"] == "hazard_boundary_crossing"
    assert evidence[0]["speed_mps"] > 0
    clock[0] = 6
    native.status()
    assert native.odometer_m == 2 and len(evidence) == 1


def test_expiry_cannot_be_relabelled_as_explicit_braking():
    clock = [0.0]
    native = VirtualNative(100, parameters(), clock=lambda: clock[0])
    assert motion(native, duration_s=5)["status"] == "accepted"
    clock[0] = 5.1
    native.status()
    assert native.speed_mps > 0
    assert native.last_stop_cause == "command_expiry"
    native.call(
        dict(
            protocol=PROTOCOL,
            operation="stop",
            request_id="late-stop",
            boot_id=native.controller.boot_id,
            command_epoch=native.controller.command_epoch,
        )
    )
    assert native.last_stop_cause == "command_expiry"
    clock[0] = 7
    native.status()
    assert native.speed_mps == 0 and native.last_stop_cause == "command_expiry"
    native.call(
        dict(
            protocol=PROTOCOL,
            operation="stop",
            request_id="held-stop",
            boot_id=native.controller.boot_id,
            command_epoch=native.controller.command_epoch,
        )
    )
    assert native.last_stop_cause == "command_expiry"
