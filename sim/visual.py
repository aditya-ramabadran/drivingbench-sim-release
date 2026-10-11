"""Pose-to-JPEG renderer interface; deliberately independent of the producer lifecycle."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
from drivingbench.sim.logs import REPO
from drivingbench.sim.visual_pose import camera_to_world, register_pose
from PIL import Image

CAMERA = {"w": 1344, "h": 760, "fl_x": 1141.5, "fl_y": 1141.5, "cx": 672, "cy": 380}


def novel_queries(reference):
    """Four synthetic stress cameras, with explicit labels that prevent paired scoring."""
    base = np.asarray(reference["transform_matrix"])
    rows = []
    for key, values in (("side_m", (-2.0, 2.0)), ("yaw_deg", (-15.0, 15.0))):
        for value in values:
            pose = base.copy()
            heading = reference["heading"]
            if key == "side_m":
                right = pose[:2, 0] / np.linalg.norm(pose[:2, 0])
                pose[:2, 3] += value * right
                label = f"side{value:+g}"
            else:
                angle = np.radians(value)
                c, s = np.cos(angle), np.sin(angle)
                pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ pose[:3, :3]
                heading += angle
                label = f"yaw{value:+g}"
            rows.append(
                dict(
                    reference,
                    name=f"{Path(reference['name']).stem}-{label}.jpg",
                    split="novel",
                    x=float(pose[0, 3]),
                    y=float(pose[1, 3]),
                    heading=float(heading),
                    transform_matrix=pose.tolist(),
                    perturbation={key: value},
                )
            )
    return rows


@dataclass
class Renderer:
    """SimNative.pose (metres/degrees) → JPEG; origin heading and mount rpy are radians."""

    render_camera: Callable[[np.ndarray, tuple], bytes]
    origin: tuple[float, float, float]
    rpy: tuple[float, float, float]
    height: float = 1.21
    ground: object | None = None
    camera_offset: tuple[float, float] = (0.0, 0.0)

    def camera_matrix(self, x, y, heading_deg):
        if not np.isfinite(
            [x, y, heading_deg, *self.origin, *self.rpy, self.height, *self.camera_offset]
        ).all():
            raise ValueError("Camera pose must be finite")
        pose = register_pose((x, y, np.radians(heading_deg)), self.origin)
        cx, cy, heading = register_pose((*self.camera_offset, 0), pose)
        if self.ground is not None:
            return self.ground.camera_from_ground(cx, cy, heading)
        return camera_to_world(cx, cy, heading, self.rpy, self.height)

    def render(self, x, y, heading_deg, camera="narrow", *, objects=()):
        if camera != "narrow":
            raise ValueError("Only the reconstructed narrow road camera is available")
        jpeg = self.render_camera(self.camera_matrix(x, y, heading_deg), tuple(objects))
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.format != "JPEG" or image.size != (CAMERA["w"], CAMERA["h"]):
                raise ValueError("Renderer must return a native-resolution JPEG")
        return jpeg


def ground_region(points, polygon, *, margin=0.0):
    """Ground-ray polygon with an optional metric collar; invalid rays remain outside."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Ground margin must be finite and nonnegative")
    points, polygon = np.asarray(points), np.asarray(polygon)
    valid = np.isfinite(points[..., :2]).all(axis=-1)
    x, y = np.where(valid, points[..., 0], 0), np.where(valid, points[..., 1], 0)
    inside = np.zeros_like(valid)
    collar = np.zeros_like(valid)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if a[1] != b[1]:
            inside ^= ((a[1] > y) != (b[1] > y)) & (
                x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            )
        if margin > 0:
            delta = b - a
            length2 = float(delta @ delta)
            fraction = np.clip(((x - a[0]) * delta[0] + (y - a[1]) * delta[1]) / length2, 0, 1)
            collar |= (x - a[0] - fraction * delta[0]) ** 2 + (
                y - a[1] - fraction * delta[1]
            ) ** 2 <= margin**2
    return (inside | collar) & valid


def splat_backend(
    checkpoint,
    *,
    blender=None,
    repair_markers=False,
    ground=None,
    ground_polygon=(),
    lighting=None,
    object_renderer=None,
):
    """Photoreal background with placed objects occluded by expected splat depth."""
    from drivingbench.sim.billboards import BillboardCache
    from drivingbench.sim.visual_splat import TOOLS, render_frames

    billboards = BillboardCache()

    def render(matrix, objects=()):
        root = REPO / "runs/sim/visual"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="render-", dir=root) as temporary:
            cameras = dict(
                CAMERA,
                frames=[
                    {
                        "name": "query",
                        "transform_matrix": (matrix @ np.diag([1, -1, -1, 1])).tolist(),
                    }
                ],
            )
            path = render_frames(checkpoint, cameras, Path(temporary), depth=bool(objects))[0]
            if not objects:
                return path.read_bytes()
            from drivingbench.sim.visual_objects import (
                ObjectLayer,
                PlacedObject,
                composite_grounded,
                composite_objects,
                marker_repair_mask,
                render_objects,
                repair_background,
            )

            layer = render_objects(
                objects,
                matrix,
                Path(temporary) / "objects",
                blender or TOOLS / "Blender.app/Contents/MacOS/Blender",
                **({"ground_normal": ground.normal} if ground is not None else {}),
                lighting=lighting,
                object_renderer=object_renderer,
            )
            with Image.open(path.with_name("query_color.png")) as background:
                pixels = np.asarray(background.convert("RGB"))
                depth = np.load(path.with_name("query_depth.npy"))
                alpha = np.load(path.with_name("query_depth_alpha.npy"))
                if repair_markers:
                    marker_ids = [
                        i
                        for i, obj in enumerate(objects, 1)
                        if (obj.kind if isinstance(obj, PlacedObject) else obj["kind"])
                        == "dome_marker"
                    ]
                    if marker_ids:
                        if layer.instance is None:
                            raise ValueError("Marker repair requires object instance IDs")
                        rgba = layer.rgba.copy()
                        rgba[~np.isin(layer.instance, marker_ids), 3] = 0
                        mask = marker_repair_mask(ObjectLayer(rgba, layer.depth), depth, alpha)
                        pixels, _ = repair_background(pixels, mask)
                placed = [
                    obj if isinstance(obj, PlacedObject) else PlacedObject(**obj) for obj in objects
                ]
                layers = [
                    layer,
                    *billboards.layers(
                        placed, matrix, CAMERA, ground.normal if ground is not None else (0, 0, 1)
                    ),
                ]
                if ground is None:
                    pixels, _ = composite_objects(pixels, depth, alpha, layers)
                else:
                    points, plane_depth = ground.ray_intersections(matrix, CAMERA)
                    transform = ground.map_from_ground
                    local = (points - transform[:3, 3]) @ transform[:3, :3]
                    region = ground_region(local, ground_polygon, margin=0.4)
                    pixels, _ = composite_grounded(
                        pixels,
                        depth,
                        alpha,
                        layers,
                        objects,
                        points,
                        plane_depth,
                        region,
                        ground_normal=ground.normal,
                    )
            output = io.BytesIO()
            Image.fromarray(pixels).save(output, "JPEG", quality=92)
            return output.getvalue()

    return render


def engine_backend(manifest, blender, *, hybrid=False):
    from drivingbench.sim.visual_engine import render_batch

    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]

    def render(matrix, objects=()):
        if objects:
            raise ValueError("Placed objects require the splat background renderer")
        root = REPO / "runs/sim/visual"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="render-", dir=root) as temporary:
            out = Path(temporary)
            query = {"name": "query.jpg", "transform_matrix": matrix.tolist()}
            render_batch(rows, [query], out / "engine", blender, hybrid)
            return (out / ("hybrid" if hybrid else "engine") / "query.jpg").read_bytes()

    return render
