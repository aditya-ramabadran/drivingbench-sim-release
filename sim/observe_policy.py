"""Simulator-only observation admission, latency and optional simulated timestamps."""

from __future__ import annotations

import asyncio
import hashlib
import time
from contextvars import ContextVar
from datetime import timedelta
from uuid import uuid4

from fastapi import Request
from fastapi.routing import APIRoute, request_response
from starlette.concurrency import run_in_threadpool


def install(app, native, restrictions, clock_mode, epoch):
    route = next(
        route for route in app.routes if isinstance(route, APIRoute) and route.path == "/observe"
    )
    original = route.endpoint
    selected = ContextVar("sim_observe_frames", default=None)
    cameras = app.state.sim_cameras
    camera_frame = cameras.frame

    def frame(camera, adjustments):
        context = selected.get()
        result = (context[0].frame if context is not None else camera_frame)(camera, adjustments)
        if context is not None and result is not None:
            context[1].append(
                {
                    "camera": camera,
                    "frame_id": result.key,
                    "captured_at": result.captured_at,
                    "sha256": hashlib.sha256(result.jpeg).hexdigest(),
                }
            )
        return result

    cameras.frame = frame

    def invoke(request):
        frames = []
        snapshot = cameras.observation_snapshot()
        token = selected.set((snapshot, frames))
        try:
            return original(request), frames, snapshot.errors
        finally:
            selected.reset(token)

    def record(kind, **data):
        app.state.sim_episode.record(kind, **data)

    async def observe(request: Request):
        arrived = time.monotonic()
        if "observe" not in restrictions.accepted_commands:
            record("sim_observe_rejected", arrived_at=arrived, reason="scenario_restriction")
            return {"status": "rejected", "reason": "scenario_restriction"}
        delay = restrictions.command_latency_s
        if delay:
            record("sim_observe_latency", arrived_at=arrived, latency_s=delay)
            await asyncio.sleep(delay)
        result, frames, errors = await run_in_threadpool(invoke, request)
        ages = {image["camera"]: image["age_s"] for image in result.get("images", [])}
        for frame in frames:
            frame["age_s"] = ages.get(frame["camera"])
        simulated = native.pose_snapshot()["sim_time_s"]
        wall_timestamp = result["timestamp"]
        if clock_mode == "simulated":
            result = dict(result, timestamp=(epoch + timedelta(seconds=simulated)).isoformat())
        event = dict(
            delivery_id=uuid4().hex,
            frames=frames,
            camera_errors=errors,
            arrived_at=arrived,
            delivered_at=time.monotonic(),
            original_wall_timestamp=wall_timestamp,
            delivered_timestamp=result["timestamp"],
            sim_time_s=simulated,
            observe_clock=clock_mode,
            image_age_s=result.get("image_age_s"),
        )
        record("sim_observe_delivery", **event)
        callback = getattr(app.state, "sim_observe_delivery", None)
        if callback is not None:
            callback(**event)
        return result

    route.endpoint = route.dependant.call = observe
    route.app = request_response(route.get_route_handler())
