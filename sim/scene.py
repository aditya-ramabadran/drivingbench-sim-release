"""An editable ground-map course connected to the calibrated camera and object renderer."""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path

from drivingbench.sim.course import CAMERA_OFFSET_M, Course
from drivingbench.sim.logs import REPO
from drivingbench.sim.visual import Renderer, splat_backend
from drivingbench.sim.visual_ground import GroundCalibration
from drivingbench.sim.visual_objects import PlacedObject, object_renderer_settings

DEFAULT_SCENE = Path(__file__).with_name("site1-course.json")


def scene_asset_paths(data):
    """Include every animation frame, not just the frame visible at startup."""
    paths = {
        row[key]
        for row in data.get("objects", []) + data.get("markers", [])
        for key in ("asset_path", "texture_path")
        if row.get(key)
    }
    lighting = data.get("lighting") or {}
    if lighting.get("environment_path"):
        paths.add(lighting["environment_path"])
    for animation in data.get("scenario", {}).get("animations", {}).values():
        paths.update(animation["frame_paths"])
        for key in ("idle", "hold_clip"):
            paths.update(animation.get(key, {}).get("frame_paths", []))
    return sorted(paths)


class Scene:
    """Scene positions use the calibrated flat map; render objects use reconstruction ENU."""

    def __init__(self, path=DEFAULT_SCENE, *, backend=None):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        self.course = Course(self.data)
        if "scenario" in self.data:
            from drivingbench.sim.scenarios import validate_scenario

            validate_scenario(self.data)
        self.ground = GroundCalibration.load(REPO / self.data["calibration"])
        self.lighting = dict(self.data.get("lighting") or {}) or None
        if self.lighting and self.lighting.get("environment_path"):
            self.lighting["environment_path"] = str(
                (REPO / self.lighting["environment_path"]).resolve()
            )
        self.object_renderer = object_renderer_settings(self.data.get("object_renderer"))
        self.objects = self.objects_at(0)
        self.renderer = Renderer(
            backend
            or splat_backend(
                REPO / self.data["background"]["checkpoint"],
                ground=self.ground,
                ground_polygon=self.course.corridor,
                lighting=self.lighting,
                object_renderer=self.object_renderer,
            ),
            origin=(*self.course.start[:2], math.radians(self.course.start[2])),
            rpy=(0.0, 0.0, 0.0),
            ground=self.ground,
            camera_offset=CAMERA_OFFSET_M,
        )

    def objects_at(self, sim_time_s=0, actor_onsets=None, actor_states=None):
        from drivingbench.sim.scenarios import object_rows_at

        rows = (
            object_rows_at(
                self.data, sim_time_s, actor_onsets=actor_onsets, actor_states=actor_states
            )
            if "scenario" in self.data
            else self.data["objects"]
        )
        base = {row["id"]: row for row in self.data["objects"] if "id" in row}
        current = {row["id"]: row for row in rows if "id" in row}
        native_shadows = set()
        for row in self.data["markers"] + rows:
            if row["kind"] == "ground_decal" and row.get("z", 0) != 0:
                raise ValueError("Ground decals must lie at calibrated ground z=0")
            if "shadow_of" not in row:
                continue
            target = row["shadow_of"]
            if (
                row["kind"] != "ground_decal"
                or not isinstance(target, str)
                or target not in base
                or base[target]["kind"] != "billboard"
            ):
                raise ValueError("shadow_of must link a ground decal to a scene billboard")
            native_shadows.add(target)
        transform = self.ground.map_from_ground
        allowed = {field.name for field in fields(PlacedObject)}
        objects = []
        for row in self.data["markers"] + rows:
            extra = row.keys() - allowed - {"id", "collision_radius_m", "shadow_of"}
            if extra:
                raise ValueError(f"Unknown placed-object fields: {sorted(extra)}")
            spec = {key: value for key, value in row.items() if key in allowed}
            if row["kind"] == "billboard" and row.get("id") in native_shadows:
                spec["shadow_strength"] = 0.0
            if "shadow_of" in row:
                target = row["shadow_of"]
                for axis in ("x", "y", "z"):
                    spec[axis] = (
                        spec.get(axis, 0) + current[target].get(axis, 0) - base[target].get(axis, 0)
                    )
            obj = PlacedObject(**spec)
            point = transform[:3, :3] @ [obj.x, obj.y, obj.z] + transform[:3, 3]
            spec.update(zip(("x", "y", "z"), map(float, point)))
            for key in ("asset_path", "texture_path"):
                if getattr(obj, key):
                    spec[key] = str((REPO / getattr(obj, key)).resolve())
            objects.append(PlacedObject(**spec))
        return objects

    def render(self, local_pose, sim_time_s=0):
        return self.renderer.render(*local_pose, objects=self.objects_at(sim_time_s))

    def camera_matrix(self, local_pose):
        return self.renderer.camera_matrix(*local_pose)
