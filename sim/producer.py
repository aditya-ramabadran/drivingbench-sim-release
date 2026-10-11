"""Local simulator composition behind the unchanged producer and its three tool endpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import ExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from drivingbench.device.service import create_app
from drivingbench.sim.cameras import OnDemandCameras, SceneRenderer, SimCameras
from drivingbench.sim.course import Course
from drivingbench.sim.episode import Episode, EpisodePolicy
from drivingbench.sim.logs import REPO
from drivingbench.sim.restrictions import Restrictions
from drivingbench.sim.runtime import Runtime
from drivingbench.sim.scene import scene_asset_paths

DEFAULT_SCENE = Path(__file__).with_name("site1-course.json")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simulator_app(
    directory,
    scene_path=DEFAULT_SCENE,
    *,
    renderer=None,
    policy=None,
    runtime_factory=Runtime,
    declaration=None,
):
    directory, scene_path = Path(directory), Path(scene_path).resolve()
    scene = json.loads(scene_path.read_text())
    wide_metadata = (
        {
            key: {
                "path": scene["wide_camera"][key],
                "sha256": file_hash(REPO / scene["wide_camera"][key]),
            }
            for key in ("checkpoint", "calibration", "vehicle_layer")
        }
        if "wide_camera" in scene
        else None
    )
    declaration = copy.deepcopy(declaration or {"time_scale": 1.0, "restrictions": {}})
    json.dumps(declaration, allow_nan=False)
    capture_mode = declaration.get("camera_capture", "continuous")
    if capture_mode not in {"continuous", "on_demand"}:
        raise ValueError("camera_capture must be continuous or on_demand")
    if declaration.get("render_lock_path") is not None and capture_mode != "on_demand":
        raise ValueError("render_lock_path requires on_demand cameras")
    if declaration.get("render_lock_path") is not None:
        lock_path = declaration["render_lock_path"]
        if not isinstance(lock_path, str) or not lock_path or not Path(lock_path).is_absolute():
            raise ValueError("render_lock_path must be an absolute file path")
    try:
        restrictions = Restrictions(**declaration.get("restrictions", {}))
    except TypeError as error:
        raise ValueError(f"Invalid restrictions: {error}") from error
    if not math.isfinite(declaration["time_scale"]) or declaration["time_scale"] <= 0:
        raise ValueError("time_scale must be finite and positive")
    if not isinstance(declaration.get("paused", False), bool):
        raise ValueError("paused must be boolean")
    clock_mode = declaration.get("observe_clock", "wall")
    if clock_mode not in {"wall", "simulated"}:
        raise ValueError("observe_clock must be wall or simulated")
    if declaration.get("restrictions") and runtime_factory is Runtime:
        raise ValueError("Declared restrictions require an explicit runtime implementation")
    epoch = datetime.now(timezone.utc)
    runtime_options = {"time_scale": declaration["time_scale"]}
    if "scenario" in scene:
        runtime_options["scenario"] = scene
    if runtime_factory is Runtime:
        runtime_options["paused"] = declaration.get("paused", False)
    policy = policy or EpisodePolicy()
    native = runtime_factory(
        Course(
            scene,
            boundary_tolerance_m=policy.boundary_tolerance_m,
            strict_boundary_tolerance_m=policy.strict_boundary_tolerance_m,
        ),
        **runtime_options,
    )
    supervisor = None
    if declaration.get("slowed_world") is not None:
        from drivingbench.sim.slowed_world import SlowedWorld

        if clock_mode != "wall":
            raise ValueError("Slowed-world observations require wall timestamps")
        supervisor = SlowedWorld(native, declaration["slowed_world"])
    renderer = renderer or SceneRenderer(scene_path, directory / "renderer")
    cameras = (
        OnDemandCameras(native, renderer, render_lock_path=declaration.get("render_lock_path"))
        if capture_mode == "on_demand"
        else SimCameras(native, renderer)
    )
    app = create_app(directory / "producer", native=native, cameras=cameras)
    app.state.settings.patch({"speed_limit_mps": 3.5})
    app.state.sim_native, app.state.sim_cameras = native, cameras
    app.state.sim_scene = scene
    app.state.sim_declaration = declaration
    app.state.sim_slowed_world = supervisor
    if supervisor is not None:
        from drivingbench.sim.slowed_world import install as install_slowed_world

        install_slowed_world(app, supervisor)
    from drivingbench.sim.observe_policy import install

    install(app, native, restrictions, clock_mode, epoch)
    checkpoint = REPO / scene["background"]["checkpoint"]
    sources = [
        "runtime.py",
        "slowed_world.py",
        "restrictions.py",
        "branch_select.py",
        "observe_policy.py",
        "course.py",
        "hazard.py",
        "naturalistic.py",
        "cameras.py",
        "scene.py",
        "scenarios.py",
        "billboards.py",
        "camera_response.py",
        "persistent.py",
        "wide_renderer.py",
        "visual_wide.py",
        "brush_render.rs",
        "visual_objects.py",
        "object_worker.py",
        "visual_composite.py",
        "visual_ground.py",
        "visual.py",
        "render_process.py",
        "render_worker.py",
        "episode.py",
        "producer.py",
        "car.py",
        "plant.py",
        "params.json",
        "corolla.carparams",
    ]
    from drivingbench.sim.visual_splat import RENDER, TOOLS

    binaries = [RENDER, TOOLS / "Blender.app/Contents/MacOS/Blender"]
    metadata = {
        "tool_binaries": {
            str(path): file_hash(path)
            for path in binaries
            if isinstance(renderer, SceneRenderer) and path.exists()
        },
        "scene": str(scene_path),
        "scene_sha256": file_hash(scene_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint) if checkpoint.exists() else None,
        "renderer": {name: file_hash(Path(__file__).with_name(name)) for name in sources},
        "calibration_sha256": file_hash(REPO / scene["calibration"]),
        "assets": {path: file_hash(REPO / path) for path in scene_asset_paths(scene)},
        "wide_camera": wide_metadata,
        "boundary_geometry": native.course.boundary_rule,
        "run_declaration": declaration,
        "observe_timestamp_epoch": epoch.isoformat(),
    }
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        episode = Episode(native, directory / "episode", policy, metadata=metadata)
        app.state.sim_episode = episode
        cameras.on_evidence = episode.record
        try:
            async with original_lifespan(application):
                yield
        finally:
            with ExitStack() as cleanup:
                cleanup.callback(episode.close)
                cleanup.callback(app.state.recording.close)
                cleanup.callback(native.close)
                cleanup.callback(cameras.close)

    app.router.lifespan_context = lifespan
    return app
