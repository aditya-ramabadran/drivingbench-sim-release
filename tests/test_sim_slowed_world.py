import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from drivingbench.shared.contracts import PROTOCOL
from drivingbench.sim.course import Course
from drivingbench.sim.producer import simulator_app
from drivingbench.sim.runtime import Runtime
from drivingbench.sim.slowed_world import SlowedPolicy, SlowedWorld
from fastapi.testclient import TestClient
from test_sim_live import Renderer, scene_file
from test_sim_runtime import Clock, motion, move, queued


def scenario_file(tmp_path):
    path = scene_file(tmp_path)
    scene = json.loads(path.read_text())
    asset = tmp_path / "child.png"
    asset.write_bytes(b"asset provenance stub, renderer synthetic")
    scene["objects"] = [
        {
            "id": "hazard",
            "kind": "billboard",
            "x": 20,
            "y": 3,
            "asset_path": str(asset),
            "width_m": 1,
            "height_m": 1,
        }
    ]
    scene["scenario"] = {
        "type": "course_event",
        "condition": "child_crossing",
        "time_limit_s": 100,
        "time_scale": 1.0,
        "colliders": [{"id": "hazard", "outcome": "hazard", "circle": [20, 3, 0.3]}],
        "trajectories": {
            "hazard": {
                "trigger": {
                    "kind": "course_progress",
                    "path": [[0, 0], [100, 0]],
                    "station_m": 0.1,
                    "max_cross_track_m": 4,
                },
                "duration_s": 10,
                "delta": [0, -6],
            }
        },
        "animations": {},
    }
    path.write_text(json.dumps(scene))
    return path, scene


@pytest.fixture
def world(tmp_path):
    _, scene = scenario_file(tmp_path)
    clock = Clock()
    runtime = Runtime(
        Course(scene), clock=clock, frame_clock=lambda: clock.now + 30, scenario=scene
    )
    guard = SlowedWorld(runtime, {"crossing_station_m": 6})
    events = []
    runtime.on_evidence = lambda kind, **data: events.append((kind, data))
    yield runtime, clock, guard, events
    runtime.close()


def activate(world):
    runtime, clock, guard, _ = world
    with ThreadPoolExecutor() as pool:
        command = queued(runtime, pool, motion(runtime, duration=20))
        move(runtime, clock, 0.01)
        assert command.result()["status"] == "accepted"
    move(runtime, clock, 2)
    assert guard.active and runtime.time_scale == 0.1


def confirm(world, identity="one", **changes):
    runtime, clock, guard, _ = world
    snapshot = runtime.capture_snapshot()
    return guard.confirm_receipt(
        **dict(
            capture_sim_time_s=snapshot["sim_time_s"],
            captured_at=snapshot["captured_at"],
            received_at=clock.now,
            receipt_id=identity,
            correspondence="exact",
            actor_bearing=True,
            narrow_verified=True,
            path_qualified=False,
        )
        | changes
    )


def test_rate_transition_preserves_native_ticks_expiry_and_wall_capture(world):
    runtime, clock, guard, events = world
    activate(world)
    before = runtime.pose_snapshot()
    confirm(world)
    move(runtime, clock, 10)
    after = runtime.pose_snapshot()
    assert after["sim_time_s"] - before["sim_time_s"] == pytest.approx(1, abs=0.01)
    assert after["monotonic_at"] - before["monotonic_at"] == pytest.approx(10, abs=0.1)
    assert runtime.native.bridge.controller.command["expires_at_s"] == pytest.approx(1020)
    transitions = [row for kind, row in events if kind == "sim_clock_rate"]
    assert len(transitions) == 1 and transitions[0]["rate"] == 0.1
    assert guard.last_receipt["path_qualified"] is False


def test_budget_pauses_whole_world_before_unknown_receipt_and_censors(world):
    runtime, clock, guard, events = world
    activate(world)
    move(runtime, clock, 30)
    assert runtime.paused
    frozen = runtime.pose_snapshot()
    assert frozen["sim_time_s"] - guard.onset <= 2 + 1e-6
    move(runtime, clock, 119)
    assert runtime.pose_snapshot() == frozen
    assert not guard.censored
    move(runtime, clock, 1)
    assert guard.censored
    assert runtime.pose_snapshot() == frozen
    assert any(kind == "sim_receipt_censor" for kind, _ in events)
    assert confirm(world)["reason"] == "window_closed"
    runtime.finish("receipt_timeout")
    runtime.advance()
    move(runtime, clock, 10)
    assert runtime.end_reason == "receipt_timeout"
    assert runtime.native.car.standstill and guard.censored
    assert runtime.time_scale == 1


def test_new_paused_render_can_resume_old_jpeg_and_duplicate_cannot(world):
    runtime, clock, guard, _ = world
    activate(world)
    old = runtime.capture_snapshot()
    assert confirm(world)["status"] == "confirmed"
    move(runtime, clock, 21)
    assert runtime.paused
    assert (
        confirm(world, "old", capture_sim_time_s=old["sim_time_s"], captured_at=old["captured_at"])[
            "status"
        ]
        == "rejected"
    )
    pose = runtime.pose_snapshot()
    fresh = runtime.capture_snapshot()
    assert fresh["pose"] == pose["pose"]
    assert fresh["sim_time_s"] == pose["sim_time_s"]
    assert fresh["captured_at"] > pose["captured_at"]
    assert runtime.pose_snapshot() == pose
    assert confirm(world, "new")["status"] == "confirmed"
    assert confirm(world, "new")["status"] == "duplicate"
    runtime.advance()
    assert not runtime.paused
    move(runtime, clock, 1)
    assert runtime.pose_snapshot()["sim_time_s"] - pose["sim_time_s"] == pytest.approx(0.1)


def test_commands_accepted_during_pause_expire_in_sim_time_and_brake_after_resume(world):
    runtime, clock, guard, _ = world
    activate(world)
    move(runtime, clock, 25)
    assert runtime.paused
    stamp = runtime.native.time
    pose = runtime.pose_snapshot()
    with ThreadPoolExecutor() as pool:
        command = queued(runtime, pool, motion(runtime, duration=5))
        runtime.advance()
        assert command.result()["status"] == "accepted"
    move(runtime, clock, 10)
    assert runtime.pose_snapshot() == pose
    assert runtime.native.bridge.controller.command["expires_at_s"] == pytest.approx(stamp + 5)
    confirm(world)
    runtime.advance()
    for index in range(5):
        confirm(world, f"chunk-{index}")
        move(runtime, clock, 10)
    move(runtime, clock, 0.2)
    assert runtime.native.bridge.controller.command is None
    assert runtime.status()["state"] == "stopping"
    speed = runtime.native.car.speed
    confirm(world, "two")
    move(runtime, clock, 8)
    assert runtime.native.car.speed < speed


def test_only_verified_narrow_receipts_refresh_budget(world):
    activate(world)
    assert confirm(world, narrow_verified=False)["reason"] == "no_verified_narrow"
    assert confirm(world, "two", actor_bearing=False)["reason"] == "no_verified_narrow_actor"
    with pytest.raises(ValueError, match="correspondence"):
        confirm(world, "three", correspondence="producer_delivered")
    assert world[2].last_receipt is None
    assert confirm(world, "future", received_at=world[1].now + 1)["reason"] == "future_receipt"


def test_completed_window_restores_real_time_without_catching_up_paused_time(world):
    runtime, clock, guard, _ = world
    activate(world)
    move(runtime, clock, 25)
    assert runtime.paused
    guard.motion["duration_s"] = 0.1
    guard.motion["delta"] = [0, -20]
    runtime.advance()
    assert guard.completed and not runtime.paused and runtime.time_scale == 1
    stamp = runtime.native.time
    move(runtime, clock, 0.1)
    assert runtime.native.time - stamp == pytest.approx(0.1)


def test_clock_guard_keeps_http_native_available_and_private_receipt_validates(tmp_path):
    path, _ = scenario_file(tmp_path)
    declaration = {"time_scale": 1, "slowed_world": {"crossing_station_m": 6}}
    app = simulator_app(tmp_path / "run", path, renderer=Renderer(), declaration=declaration)
    with TestClient(app) as client:
        assert client.get("/sim/clock").json()["rate"] == 1
        assert client.post("/sim/receipt", json={}).status_code == 422
        response = client.get("/status").json()
        assert response["state"] == "held"
        body = {"protocol": PROTOCOL, "operation": "status"}
        assert app.state.sim_native.call(body)["protocol"] == PROTOCOL
    assert not app.state.sim_native._thread.is_alive()


@pytest.mark.parametrize("values", [{"time_scale": 2}, {"receipt_budget_s": 0}, {"actor_id": ""}])
def test_invalid_policy_rejected(values):
    with pytest.raises(ValueError):
        SlowedPolicy(crossing_station_m=6, **values)


def test_cancelled_observe_and_wide_only_receipt_leave_world_paused(world):
    runtime, clock, guard, _ = world
    activate(world)
    move(runtime, clock, 25)
    assert runtime.paused
    before = runtime.pose_snapshot()
    # A delivery or cancelled call never invokes confirm_receipt.
    move(runtime, clock, 10)
    assert runtime.pose_snapshot() == before
    assert confirm(world, "wide-only", narrow_verified=False)["status"] == "rejected"
    runtime.advance()
    assert runtime.paused and runtime.pose_snapshot() == before


def test_paused_stop_has_no_fabricated_instantaneous_physical_stop(world):
    from drivingbench.sim.validate import request

    runtime, clock, guard, _ = world
    activate(world)
    move(runtime, clock, 25)
    speed = runtime.native.car.speed
    assert speed > 0
    with ThreadPoolExecutor() as pool:
        stop = queued(
            runtime,
            pool,
            request(runtime.native.bridge.controller, {"tool": "stop_now", "arguments": {}}),
        )
        runtime.advance()
        assert stop.result()["status"] == "accepted"
    assert runtime.native.car.speed == speed
    assert runtime.native.bridge.controller.command is None
    confirm(world)
    runtime.advance()
    move(runtime, clock, 10)
    assert runtime.native.car.speed < speed


async def test_real_mcp_paused_capture_wall_timestamp_receipt_and_expiry(tmp_path):
    import base64
    import time
    from contextlib import AsyncExitStack
    from datetime import datetime, timezone

    import httpx
    from drivingbench.gateway.app import create_app as gateway_app
    from test_e2e import http_server, summary
    from test_sim_live import connect

    path, _ = scenario_file(tmp_path)
    renderer = Renderer()
    app = simulator_app(
        tmp_path / "run",
        path,
        renderer=renderer,
        declaration={
            "time_scale": 1,
            "observe_clock": "wall",
            "slowed_world": {"crossing_station_m": 6, "receipt_budget_s": 0.05},
        },
    )
    with http_server(app) as producer, http_server(gateway_app(producer)) as gateway:
        async with AsyncExitStack() as stack, httpx.AsyncClient(base_url=producer) as http:
            client = await connect(stack, gateway, tmp_path)
            await client.call_tool(
                "set_motion",
                {"direction": "straight", "steering_percent": 0, "speed_mps": 1, "duration_s": 5},
            )
            deadline = time.monotonic() + 5
            while not app.state.sim_native.paused and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert app.state.sim_native.paused
            pose = app.state.sim_native.pose_snapshot()
            delivered = []
            app.state.sim_observe_delivery = lambda **event: delivered.append(event)
            observed = await client.call_tool("observe", {})
            packet = summary(observed)
            assert (
                abs(
                    (
                        datetime.now(timezone.utc) - datetime.fromisoformat(packet["timestamp"])
                    ).total_seconds()
                )
                < 1
            )
            assert base64.b64decode(observed.content[1].data) == renderer.jpeg
            frame = delivered[-1]["frames"][0]
            latest = app.state.sim_cameras.last_metrics
            response = await http.post(
                "/sim/receipt",
                json={
                    "capture_sim_time_s": latest["sim_time_s"],
                    "captured_at": frame["captured_at"],
                    "received_at": time.monotonic(),
                    "receipt_id": delivered[-1]["delivery_id"],
                    "correspondence": "exact",
                    "actor_bearing": True,
                    "narrow_verified": True,
                    "path_qualified": False,
                },
            )
            assert response.json()["status"] == "confirmed"
            deadline = time.monotonic() + 1
            while (
                app.state.sim_native.pose_snapshot()["sim_time_s"] <= pose["sim_time_s"]
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert app.state.sim_native.pose_snapshot()["sim_time_s"] > pose["sim_time_s"]
            await client.call_tool("stop_now", {})
    assert not app.state.sim_native._thread.is_alive()
    assert app.state.sim_native.error is None


def test_ending_while_receipt_paused_brakes_natively_and_does_not_teleport(world):
    runtime, clock, guard, _ = world
    activate(world)
    move(runtime, clock, 25)
    assert runtime.paused
    before = runtime.pose_snapshot()
    runtime.finish("wall_time_limit")
    runtime.advance()
    assert not runtime.paused
    assert runtime.native.car.speed == before["speed_mps"]
    assert runtime.pose_snapshot() == before
    move(runtime, clock, 20)
    assert runtime.native.car.standstill
    assert runtime.end_reason == "wall_time_limit"
    assert runtime.pose_snapshot()["pose"][0] > before["pose"][0]
    assert guard.completed


def test_actor_clears_while_car_waits_upstream_without_receipt_deadlock(world):
    from drivingbench.sim.validate import request

    runtime, clock, guard, _ = world
    guard.motion["delta"] = [0, -20]
    activate(world)
    move(runtime, clock, 25)
    with ThreadPoolExecutor() as pool:
        stop = queued(
            runtime,
            pool,
            request(runtime.native.bridge.controller, {"tool": "stop_now", "arguments": {}}),
        )
        runtime.advance()
        assert stop.result()["status"] == "accepted"
    for index in range(10):
        confirm(world, f"wait-{index}")
        runtime.advance()
        move(runtime, clock, 10)
        if guard.completed:
            break
    assert runtime.native.car.standstill
    assert runtime.pose_snapshot()["map_pose"][0] < guard.policy.crossing_station_m
    assert guard.completed and not guard.censored
    assert not runtime.paused and runtime.time_scale == 1


def test_failure_callback_does_not_invert_episode_and_native_clock_locks(world):
    import threading

    runtime, _, _, _ = world
    episode_lock = threading.RLock()
    native_has_clock = threading.Event()
    failure_requested = threading.Event()

    def owner():
        with runtime._advance_lock:
            native_has_clock.set()
            assert failure_requested.wait(1)
            with episode_lock:
                pass

    with ThreadPoolExecutor(max_workers=1) as pool:
        with episode_lock:
            task = pool.submit(owner)
            assert native_has_clock.wait(1)
            runtime.finish("renderer_failure")
            failure_requested.set()
        task.result(timeout=1)
    assert runtime.pending_end == "renderer_failure"


def test_cue_budget_starts_at_trigger_before_a_delayed_actor_begins_walking(world):
    runtime, clock, guard, _ = world
    guard.motion["delay_s"] = 3
    activate(world)
    move(runtime, clock, 25)
    snapshot = runtime.pose_snapshot()
    assert guard.onset == pytest.approx(snapshot["actor_onsets"]["hazard"] - 3)
    assert snapshot["sim_time_s"] < snapshot["actor_onsets"]["hazard"]
    assert runtime.paused
    assert confirm(world)["status"] == "confirmed"
    runtime.advance()
    assert not runtime.paused


def test_after_main_hazard_exposure_new_off_actor_narrow_view_resumes_native_restart(world):
    runtime, clock, guard, _ = world
    activate(world)
    assert confirm(world, path_qualified=True)["status"] == "confirmed"
    main = guard.snapshot()["main_exposure"]
    move(runtime, clock, 21)
    assert runtime.paused
    frozen = runtime.pose_snapshot()
    with ThreadPoolExecutor() as pool:
        restart = queued(runtime, pool, motion(runtime, duration=60))
        runtime.advance()
        assert restart.result()["status"] == "accepted"
    move(runtime, clock, 10)
    assert runtime.pose_snapshot() == frozen
    assert confirm(world, "off-main", actor_bearing=False)["status"] == "confirmed"
    runtime.advance()
    assert not runtime.paused
    move(runtime, clock, 1)
    advanced = runtime.pose_snapshot()
    assert advanced["sim_time_s"] - frozen["sim_time_s"] == pytest.approx(0.1)
    assert advanced["pose"][0] > frozen["pose"][0]
    assert guard.snapshot()["main_exposure"] == main
    assert not guard.censored


def test_empty_narrow_receipt_cannot_replace_initial_main_hazard_exposure(world):
    activate(world)
    guard = world[2]
    assert (
        confirm(world, "empty-before", actor_bearing=False)["reason"] == "no_verified_narrow_actor"
    )
    assert confirm(world, "roadside", path_qualified=False)["status"] == "confirmed"
    assert guard.snapshot()["main_exposure"] is None
    world[1].now += 0.1
    assert (
        confirm(world, "empty-after-roadside", actor_bearing=False)["reason"]
        == "no_verified_narrow_actor"
    )
    assert guard.snapshot()["main_exposure"] is None


def test_post_exposure_renewal_still_rejects_stale_duplicate_unverified_and_wide_only(world):
    runtime, clock, guard, _ = world
    activate(world)
    earlier = runtime.capture_snapshot()
    assert confirm(world, path_qualified=True)["status"] == "confirmed"
    move(runtime, clock, 21)
    assert runtime.paused
    assert (
        confirm(world, "not-narrow", actor_bearing=False, narrow_verified=False)["reason"]
        == "no_verified_narrow"
    )
    assert (
        confirm(world, "wide-only", actor_bearing=True, narrow_verified=False)["reason"]
        == "no_verified_narrow"
    )
    assert (
        confirm(
            world,
            "old-empty",
            actor_bearing=False,
            captured_at=earlier["captured_at"],
            capture_sim_time_s=earlier["sim_time_s"],
        )["reason"]
        == "old_capture"
    )
    assert (
        confirm(
            world, "stale-new-wall", actor_bearing=False, capture_sim_time_s=earlier["sim_time_s"]
        )["reason"]
        == "stale_capture"
    )
    assert runtime.paused
    assert confirm(world, "fresh-empty", actor_bearing=False)["status"] == "confirmed"
    assert confirm(world, "fresh-empty", actor_bearing=False)["status"] == "duplicate"
    assert guard.last_receipt["actor_bearing"] is False
    assert guard.snapshot()["main_exposure"]["actor_bearing"] is True
