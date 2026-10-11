"""Operator-only simulator views, independent of the producer's tool contract."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

from drivingbench.shared.contracts import DURATION_RANGE_S, PROTOCOL, SPEED_RANGE_MPS
from drivingbench.sim.course import LENGTH_M, WIDTH_M
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict

UI_VERSION = 1


class BrowsePose(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float
    y: float
    heading_deg: float


def attach_operator(app):
    """Attach read-only telemetry and queued free-camera rendering to a simulator app."""

    @app.get("/sim")
    def page():
        return FileResponse(
            Path(__file__).with_name("operator.html"), headers={"Cache-Control": "no-store"}
        )

    @app.get("/sim/state")
    def state():
        native, cameras = app.state.sim_native, app.state.sim_cameras
        episode = getattr(app.state, "sim_episode", None)
        settings, revision = app.state.settings.snapshot()
        return {
            "ui_version": UI_VERSION,
            "protocol": PROTOCOL,
            "pose": native.pose_snapshot(),
            "scene": app.state.sim_scene,
            "vehicle": {"length_m": LENGTH_M, "width_m": WIDTH_M},
            "episode": episode.summary() if episode else None,
            "episode_policy": asdict(episode.policy) if episode else None,
            "settings": settings.model_dump(),
            "settings_revision": revision,
            "controls": {"speed_range_mps": SPEED_RANGE_MPS, "duration_range_s": DURATION_RANGE_S},
            "camera": {
                "ready": cameras.ready,
                "errors": dict(cameras.errors),
                "last_metrics": dict(cameras.last_metrics),
            },
            "declaration": copy.deepcopy(
                getattr(app.state, "sim_declaration", {"time_scale": 1.0, "restrictions": {}})
            ),
        }

    @app.post("/sim/browse")
    def browse(body: BrowsePose, request: Request):
        if request.headers.get("X-Sim-Operator-Version") != str(UI_VERSION):
            raise HTTPException(409, "operator_version_mismatch")
        try:
            jpeg = app.state.sim_cameras.browse((body.x, body.y, body.heading_deg))
        except ValueError as error:
            raise HTTPException(422, f"invalid_browse_pose: {error}") from error
        except (TimeoutError, OSError, RuntimeError) as error:
            raise HTTPException(503, f"browse_unavailable: {error}") from error
        return Response(
            jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Sim-View": "browse"},
        )

    return app
