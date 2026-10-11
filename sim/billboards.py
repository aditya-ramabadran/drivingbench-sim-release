"""Cached upright RGBA cutouts and ground decals with explicit per-pixel camera depth."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np
from drivingbench.sim.visual_ground import road_rotation
from drivingbench.sim.visual_objects import (
    _linear,
    _srgb,
    empty_layer,
    projected_bounds,
)
from PIL import Image


class BillboardCache:
    """Bound decoded clips in RAM; interpolate premultiplied RGB to avoid dark fringes."""

    def __init__(self, objects=(), *, max_bytes=64 * 1024**2):
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("Billboard cache budget must be a positive byte count")
        self.images = OrderedDict()
        self.max_bytes, self.cached_bytes = max_bytes, 0
        self.lock = RLock()
        for obj in objects:
            if obj.kind in ("billboard", "ground_decal"):
                self.load(obj.asset_path)

    def load(self, path):
        path = str(Path(path).resolve())
        with self.lock:
            if path in self.images:
                self.images.move_to_end(path)
                return self.images[path]
            with Image.open(path) as image:
                if "A" not in image.getbands():
                    raise ValueError("Billboard PNG must contain an alpha channel")
                rgba = np.asarray(image.convert("RGBA"))
            alpha = rgba[..., 3:4].astype(float) / 255
            if not np.any(alpha) or np.all(alpha == 1):
                raise ValueError("Billboard PNG needs visible content and transparent background")
            decoded = np.concatenate((_linear(rgba[..., :3]) * alpha, alpha), axis=-1).astype(
                np.float32
            )
            while self.images and self.cached_bytes + decoded.nbytes > self.max_bytes:
                _, evicted = self.images.popitem(last=False)
                self.cached_bytes -= evicted.nbytes
            if decoded.nbytes <= self.max_bytes:
                self.images[path] = decoded
                self.cached_bytes += decoded.nbytes
            return decoded

    def layers(self, objects, matrix, camera, ground_normal=(0, 0, 1)):
        """Keep surfaces separate until scene occlusion has been applied to each depth."""
        layers = []
        for index, obj in enumerate(objects):
            if obj.kind in ("billboard", "ground_decal"):
                projected = self.project(obj, matrix, camera, ground_normal)
                projected.instance[projected.rgba[..., 3] > 0] = -(index + 1)
                layers.append(projected)
        return layers

    def projection_geometry(self, obj, matrix, camera, ground_normal=(0, 0, 1)):
        """Return the same source-UV homography and corners used by the compositor."""
        matrix = np.asarray(matrix, dtype=float)
        basis = road_rotation(ground_normal, 0)
        yaw = obj.yaw
        if obj.face_camera:
            toward_camera = basis.T @ (matrix[:3, 3] - [obj.x, obj.y, obj.z])
            if np.linalg.norm(toward_camera[:2]) > 1e-9:
                yaw = math.atan2(toward_camera[1], toward_camera[0]) + math.pi / 2
        rotation = basis @ np.array(
            [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        )
        right = rotation[:, 0] * obj.width_m * obj.scale
        up = (
            rotation[:, 1] * obj.depth_m * obj.scale
            if obj.kind == "ground_decal"
            else rotation[:, 2] * obj.height_m * obj.scale
        )
        bottom = np.array([obj.x, obj.y, obj.z])
        anchor_x, anchor_y = obj.anchor_uv or (
            (0.5, 0.5) if obj.kind == "ground_decal" else (0.5, 1.0)
        )
        top_left = bottom - right * anchor_x + up * anchor_y
        corners = np.array([top_left, top_left + right, top_left - up, top_left + right - up])
        intrinsic = np.array(
            [[camera["fl_x"], 0, camera["cx"]], [0, camera["fl_y"], camera["cy"]], [0, 0, 1]]
        )
        homography = (
            intrinsic @ matrix[:3, :3].T @ np.column_stack((right, -up, top_left - matrix[:3, 3]))
        )
        return homography, corners

    def project(self, obj, matrix, camera, ground_normal=(0, 0, 1)):
        image = self.load(obj.asset_path)
        matrix = np.asarray(matrix, dtype=float)
        homography, corners = self.projection_geometry(obj, matrix, camera, ground_normal)
        layer = empty_layer(camera)
        bounds = projected_bounds([corners], matrix, camera, padding=1)
        if bounds is None:
            return layer
        if abs(np.linalg.det(homography)) < 1e-10:
            return layer
        x0, y0, x1, y1 = bounds
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rays = np.stack((xx + 0.5, yy + 0.5, np.ones(xx.shape)), axis=-1)
        coordinates = rays @ np.linalg.inv(homography).T
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = (
                coordinates[..., 0] / coordinates[..., 2],
                coordinates[..., 1] / coordinates[..., 2],
            )
            z = 1 / coordinates[..., 2]
        inside = (z > 0.01) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        if not inside.any():
            return layer
        ih, iw = image.shape[:2]
        sx = np.clip(u[inside] * iw - 0.5, 0, iw - 1)
        sy = np.clip(v[inside] * ih - 0.5, 0, ih - 1)
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        dx, dy = (sx - ix)[:, None], (sy - iy)[:, None]
        jx, jy = np.minimum(ix + 1, iw - 1), np.minimum(iy + 1, ih - 1)
        values = (1 - dy) * ((1 - dx) * image[iy, ix] + dx * image[iy, jx]) + dy * (
            (1 - dx) * image[jy, ix] + dx * image[jy, jx]
        )
        alpha = values[:, 3]
        rgba = np.zeros((len(values), 4), dtype=np.uint8)
        covered = alpha > 0
        color = values[covered, :3] / alpha[covered, None]
        if obj.material_saturation is not None and obj.material_saturation != 1:
            luminance = color @ np.array([0.2126, 0.7152, 0.0722])
            color = luminance[:, None] + obj.material_saturation * (color - luminance[:, None])
        rgba[covered, :3] = _srgb(color)
        rgba[:, 3] = np.rint(alpha * 255).astype(np.uint8)
        layer.rgba[y0:y1, x0:x1][inside] = rgba
        layer.depth[y0:y1, x0:x1][inside] = np.where(rgba[:, 3] > 0, z[inside], np.nan)
        return layer
