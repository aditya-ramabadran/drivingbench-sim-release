import json

import pytest
from drivingbench.sim.episode import EpisodePolicy
from drivingbench.sim.producer import simulator_app
from fastapi.testclient import TestClient
from test_sim_live import Renderer, scene_file


def test_camera_shutdown_failure_closes_all_other_workers(tmp_path):
    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=Renderer())
    close = app.state.sim_cameras.close

    def failing_close():
        close()
        raise RuntimeError("injected close failure")

    app.state.sim_cameras.close = failing_close
    with pytest.raises(RuntimeError, match="injected close failure"):
        with TestClient(app) as client:
            assert client.get("/status").status_code == 200
    assert not app.state.sim_native._thread.is_alive()
    assert not app.state.recording.thread.is_alive()
    assert not app.state.sim_episode._worker.is_alive()


def test_run_metadata_copies_declaration_and_calibration(tmp_path):
    declaration = {"time_scale": 1.0, "restrictions": {}}
    app = simulator_app(
        tmp_path / "run", scene_file(tmp_path), renderer=Renderer(), declaration=declaration
    )
    declaration["restrictions"]["brakes_failed"] = True
    with TestClient(app):
        metadata = json.loads((tmp_path / "run/episode/metadata.json").read_text())["metadata"]
        assert metadata["run_declaration"]["restrictions"] == {}
        assert len(metadata["calibration_sha256"]) == 64
        assert {"car.py", "params.json", "visual_composite.py"} <= metadata["renderer"].keys()


@pytest.mark.parametrize("tolerance", [0.0, 0.5, 0.8])
def test_declared_boundary_tolerance_reaches_physics_and_metadata(tmp_path, tolerance):
    app = simulator_app(
        tmp_path / "run",
        scene_file(tmp_path),
        renderer=Renderer(),
        policy=EpisodePolicy(boundary_tolerance_m=tolerance),
    )
    with TestClient(app):
        metadata = json.loads((tmp_path / "run/episode/metadata.json").read_text())
        assert app.state.sim_native.course.boundary_tolerance_m == tolerance
        assert metadata["metadata"]["boundary_geometry"]["tolerance_m"] == tolerance


def test_wide_run_provenance_hashes_each_camera_asset(tmp_path):
    import hashlib

    path = scene_file(tmp_path)
    scene = json.loads(path.read_text())
    scene["wide_camera"] = {}
    for name in ("checkpoint", "calibration", "vehicle_layer"):
        asset = tmp_path / name
        asset.write_bytes(name.encode())
        scene["wide_camera"][name] = str(asset)
    path.write_text(json.dumps(scene))
    app = simulator_app(tmp_path / "run", path, renderer=Renderer())
    with TestClient(app):
        metadata = json.loads((tmp_path / "run/episode/metadata.json").read_text())["metadata"]
        for name, value in metadata["wide_camera"].items():
            assert value["path"] == str(tmp_path / name)
            assert value["sha256"] == hashlib.sha256(name.encode()).hexdigest()
        assert {"visual_wide.py", "wide_renderer.py"} <= metadata["renderer"].keys()


def test_undeclared_implementation_rejects_scenario_restrictions(tmp_path):
    with pytest.raises(ValueError, match="explicit runtime"):
        simulator_app(
            tmp_path / "run",
            scene_file(tmp_path),
            renderer=Renderer(),
            declaration={"time_scale": 1, "restrictions": {"brakes_failed_speed_mps": 5.5}},
        )


def test_terminal_course_cause_stays_out_of_observe(tmp_path):
    import time

    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=Renderer())
    with TestClient(app) as client:
        app.state.sim_native.finish("marker_hit")
        deadline = time.monotonic() + 2
        while app.state.sim_native.end_reason is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        packet = client.get("/observe").json()
        assert packet["reason"] == "operator_brake"
        assert not {"end_reason", "collision_ids", "pose", "outside_course"} & packet.keys()
        assert app.state.sim_native.end_reason == "marker_hit"


def test_invalid_declarations_fail_before_native_construction(tmp_path):
    scene = scene_file(tmp_path)
    called = []

    def factory(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("Native must not be constructed")

    for change in (
        {"restrictions": {"unknown": True}},
        {"restrictions": {"command_latency_s": -1}},
        {"observe_clock": "imaginary"},
        {"time_scale": 0},
        {"paused": "yes"},
    ):
        declaration = {"time_scale": 1, "restrictions": {}}
        declaration.update(change)
        with pytest.raises(ValueError):
            simulator_app(
                tmp_path / "run",
                scene,
                renderer=Renderer(),
                runtime_factory=factory,
                declaration=declaration,
            )
    assert called == []


def test_disabled_observe_keeps_status_and_stop_endpoints_available(tmp_path):
    from drivingbench.sim.restrictions import runtime_for

    declaration = {"time_scale": 1, "restrictions": {"accepted_commands": ["stop_now"]}}
    app = simulator_app(
        tmp_path / "run",
        scene_file(tmp_path),
        renderer=Renderer(),
        declaration=declaration,
        runtime_factory=runtime_for(declaration),
    )
    with TestClient(app) as client:
        assert client.get("/observe").json() == {
            "status": "rejected",
            "reason": "scenario_restriction",
        }
        assert client.get("/status").status_code == 200
        assert client.post("/stop", json={}).json()["status"] == "stopping"
    events = [
        json.loads(line)
        for line in (tmp_path / "run/episode/events.jsonl").read_text().splitlines()
    ]
    assert any(event["kind"] == "sim_observe_rejected" for event in events)


def test_observe_latency_keeps_physics_and_status_running(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from drivingbench.sim.restrictions import runtime_for

    declaration = {"time_scale": 1, "restrictions": {"command_latency_s": 0.3}}
    app = simulator_app(
        tmp_path / "run",
        scene_file(tmp_path),
        renderer=Renderer(),
        declaration=declaration,
        runtime_factory=runtime_for(declaration),
    )
    with TestClient(app) as client, ThreadPoolExecutor() as pool:
        before = app.state.sim_native.pose_snapshot()["sim_time_s"]
        start = time.monotonic()
        observation = pool.submit(client.get, "/observe")
        assert client.get("/status").status_code == 200
        assert not observation.done()
        assert observation.result().status_code == 200
        assert time.monotonic() - start >= 0.3
        assert app.state.sim_native.pose_snapshot()["sim_time_s"] - before >= 0.25
    events = [
        json.loads(line)
        for line in (tmp_path / "run/episode/events.jsonl").read_text().splitlines()
    ]
    delivery = next(event for event in events if event["kind"] == "sim_observe_delivery")
    assert delivery["delivered_at"] - delivery["arrived_at"] >= 0.3


def test_simulated_observe_timestamp_does_not_change_wall_frame_age_or_recording(tmp_path):
    import time
    from datetime import datetime

    from drivingbench.sim.restrictions import runtime_for

    renderer = Renderer()
    declaration = {"time_scale": 0.25, "observe_clock": "simulated", "restrictions": {}}
    app = simulator_app(
        tmp_path / "run",
        scene_file(tmp_path),
        renderer=renderer,
        declaration=declaration,
        runtime_factory=runtime_for(declaration),
    )
    with TestClient(app) as client:
        deadline = time.monotonic() + 2
        while client.get("/camera/narrow").status_code != 200:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        renderer.block.set()
        assert renderer.entered.wait(2)
        first = client.get("/observe").json()
        time.sleep(0.24)
        second = client.get("/observe").json()
        elapsed = (
            datetime.fromisoformat(second["timestamp"]) - datetime.fromisoformat(first["timestamp"])
        ).total_seconds()
        assert 0.04 <= elapsed <= 0.12
        assert second["image_age_s"] - first["image_age_s"] >= 0.22
        assert second["images"][0]["data"] == first["images"][0]["data"]
        history = client.get("/history").json()["calls"]
        observed = [call for call in history if call.get("tool") == "observe"]
        assert observed[-1]["outcome"]["timestamp"] != second["timestamp"]
        renderer.release.set()
    metadata = json.loads((tmp_path / "run/episode/metadata.json").read_text())["metadata"]
    epoch = datetime.fromisoformat(metadata["observe_timestamp_epoch"])
    events = [
        json.loads(line)
        for line in (tmp_path / "run/episode/events.jsonl").read_text().splitlines()
    ]
    delivered = [event for event in events if event["kind"] == "sim_observe_delivery"][-1]
    assert delivered["delivered_timestamp"] == second["timestamp"]
    assert delivered["original_wall_timestamp"] == observed[-1]["outcome"]["timestamp"]
    assert (datetime.fromisoformat(second["timestamp"]) - epoch).total_seconds() == pytest.approx(
        delivered["sim_time_s"]
    )
    assert {"restrictions.py", "observe_policy.py"} <= metadata["renderer"].keys()


def test_default_runtime_honors_initial_pause_without_changing_default_observe_clock(tmp_path):
    from datetime import datetime, timezone

    declaration = {"time_scale": 1, "paused": True, "restrictions": {}}
    app = simulator_app(
        tmp_path / "run", scene_file(tmp_path), renderer=Renderer(), declaration=declaration
    )
    with TestClient(app) as client:
        native = app.state.sim_native
        assert native.paused
        before = native.pose_snapshot()
        packet = client.get("/observe").json()
        assert native.pose_snapshot() == before
        assert (
            abs(
                (
                    datetime.now(timezone.utc) - datetime.fromisoformat(packet["timestamp"])
                ).total_seconds()
            )
            < 1
        )


def test_very_slow_simulator_keeps_producer_available_with_new_render_exposures(tmp_path):
    import time

    from drivingbench.shared.contracts import NATIVE_WATCHDOG_S

    app = simulator_app(
        tmp_path / "run",
        scene_file(tmp_path),
        renderer=Renderer(),
        declaration={"time_scale": 0.001, "restrictions": {}},
    )
    with TestClient(app) as client:
        before = app.state.sim_native.pose_snapshot()
        time.sleep(NATIVE_WATCHDOG_S + 0.15)
        state = client.get("/status").json()
        assert state["state"] != "unavailable"
        assert state.get("reason") != "producer_unavailable"
        packet = client.get("/observe").json()
        assert packet["state"] != "unavailable"
        assert packet["image_age_s"] < 0.2
        assert app.state.sim_native.pose_snapshot() == before
        assert time.monotonic() - app.state.sim_native.status()["received_at"] < 0.2


def test_private_observe_hook_tracks_exact_capture_ids_for_identical_images(tmp_path):
    import hashlib
    import time
    from concurrent.futures import ThreadPoolExecutor

    from drivingbench.device.cameras import Frame

    renderer = Renderer()
    app = simulator_app(tmp_path / "run", scene_file(tmp_path), renderer=renderer)
    deliveries = []
    app.state.sim_observe_delivery = lambda **event: deliveries.append(event)
    with TestClient(app) as client:
        deadline = time.monotonic() + 2
        while client.get("/camera/narrow").status_code != 200:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not deliveries
        renderer.block.set()
        assert renderer.entered.wait(2)
        cameras = app.state.sim_cameras
        with cameras.lock:
            first = cameras.frames["narrow"]
        packet = client.get("/observe").json()
        assert "delivery_id" not in packet and "frames" not in packet
        suffix = hashlib.sha256(first.jpeg).hexdigest()[:12]
        assert deliveries[0]["frames"][0]["frame_id"] == first.key + "-" + suffix
        assert deliveries[0]["frames"][0]["captured_at"] == first.captured_at
        second = Frame("second-capture", first.captured_at + 1, first.jpeg)
        cameras.put("narrow", second)
        with ThreadPoolExecutor() as pool:
            futures = [
                pool.submit(client.get, path) for path in ("/observe", "/camera/narrow", "/observe")
            ]
            assert all(future.result().status_code == 200 for future in futures)
        assert len(deliveries) == 3
        assert len({event["delivery_id"] for event in deliveries}) == 3
        assert all(len(event["frames"]) == 1 for event in deliveries)
        assert all(
            event["frames"][0]["frame_id"] == second.key + "-" + suffix for event in deliveries[1:]
        )
        assert deliveries[0]["frames"][0]["frame_id"] != deliveries[1]["frames"][0]["frame_id"]
        assert len({event["frames"][0]["sha256"] for event in deliveries}) == 1
        renderer.release.set()
