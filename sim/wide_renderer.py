"""Warm calibrated fisheye rendering from composed pinhole tiles and an opaque body layer."""

from __future__ import annotations

import copy
import io
import json
import time
from pathlib import Path

import numpy as np
from drivingbench.sim.logs import REPO
from drivingbench.sim.persistent import PersistentRenderer
from drivingbench.sim.visual_objects import pinhole_camera
from drivingbench.sim.visual_wide import WideCalibration, compose_native, native_maps
from PIL import Image


def visible_billboards(tile_masks, maps, weights, body_alpha, object_ids):
    """Count each contributing native pixel once, after the image warp and opaque body."""
    import cv2

    counts = {}
    for identity in sorted(object_ids):
        contribution = np.zeros(body_alpha.shape, np.float32)
        for masks, mapping, weight in zip(tile_masks, maps, weights, strict=True):
            mask = masks.get(identity)
            if mask is not None:
                contribution += (
                    cv2.remap(
                        mask.astype(np.float32), mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR
                    )
                    * weight
                )
        counts[identity] = {
            "visible_pixels": int(np.count_nonzero((contribution > 0) & (body_alpha < 1)))
        }
    return counts


class WideRenderer:
    def __init__(self, scene_path, *, output, new_session=True):
        scene = json.loads(Path(scene_path).read_text())
        config = scene["wide_camera"]
        self.calibration = WideCalibration.load(REPO / config["calibration"])
        k = self.calibration.intrinsics
        if (k["w"], k["h"]) != (1344, 760):
            raise ValueError("Wide camera must output 1344x760")
        with Image.open(REPO / config["vehicle_layer"]) as image:
            if image.mode != "RGBA" or image.size != (1344, 760):
                raise ValueError("Vehicle layer must be 1344x760 RGBA")
            body = np.array(image)
        self.body_rgb, self.body_alpha = body[..., :3], body[..., 3] / 255.0
        if np.any((body[..., 3] != 0) & (body[..., 3] != 255)) or np.all(body[..., 3] == 255):
            raise ValueError("Vehicle layer needs binary alpha and visible exterior")
        self.tiles = copy.deepcopy(config["tiles"])
        if not self.tiles or len({tile["name"] for tile in self.tiles}) != len(self.tiles):
            raise ValueError("Wide tiles must have unique names")
        self.cameras = [pinhole_camera(tile) for tile in self.tiles]
        for tile in self.tiles:
            rotation = np.asarray(tile["rotation_wide_from_tile"], dtype=float)
            if (
                rotation.shape != (3, 3)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1)
            ):
                raise ValueError("Tile rotation must be proper and orthonormal")
        self.maps, self.weights = native_maps(self.calibration, self.tiles)
        coverage = np.sum(self.weights, axis=0)
        uncovered = np.count_nonzero((self.body_alpha == 0) & (coverage <= 0))
        if uncovered:
            raise ValueError(f"Wide tiles leave {uncovered} exterior pixels uncovered")
        self.renderer = PersistentRenderer(
            scene_path,
            output=output,
            new_session=new_session,
            camera=self.cameras[0],
            checkpoint_override=REPO / config["checkpoint"],
        )
        self.last_metrics = {}

    def start(self):
        started = time.perf_counter()
        try:
            ready = self.renderer.start()
            self.render((0, 0, 0))
        except Exception:
            self.close()
            raise
        return dict(ready, warmup_s=time.perf_counter() - started)

    def render(self, pose, sim_time_s=0, actor_onsets=None, actor_states=None):
        started = time.perf_counter()
        wide = self.calibration.camera_pose(self.renderer.scene.camera_matrix(pose))
        views = []
        for tile, camera in zip(self.tiles, self.cameras, strict=True):
            matrix = wide.copy()
            matrix[:3, :3] = wide[:3, :3] @ np.asarray(tile["rotation_wide_from_tile"])
            views.append((matrix, camera))
        results = self.renderer.render_matrices_rgb(
            views,
            sim_time_s=sim_time_s,
            actor_onsets=actor_onsets,
            visibility_masks=True,
            **({"actor_states": actor_states} if actor_states is not None else {}),
        )
        images, metrics, masks = [], {}, []
        for tile, camera, (image, timing, visibility) in zip(
            self.tiles, self.cameras, results, strict=True
        ):
            if image.shape != (camera["h"], camera["w"], 3) or image.dtype != np.uint8:
                raise ValueError("Tile renderer must return camera-sized uint8 RGB")
            images.append(image)
            metrics[tile["name"]] = timing
            masks.append(visibility)
        rendering = time.perf_counter()
        image = compose_native(images, self.maps, self.weights, self.body_rgb, self.body_alpha)
        billboards = visible_billboards(
            masks,
            self.maps,
            self.weights,
            self.body_alpha,
            {identity for timing in metrics.values() for identity in timing.get("billboards", {})},
        )
        output = io.BytesIO()
        Image.fromarray(image).save(output, "JPEG", quality=92)
        self.last_metrics = dict(
            elapsed_s=time.perf_counter() - started,
            tiles_s=rendering - started,
            warp_encode_s=time.perf_counter() - rendering,
            tiles=metrics,
            billboards=billboards,
            camera=self.calibration.intrinsics,
            body_pixels=int(np.count_nonzero(self.body_alpha)),
        )
        return output.getvalue()

    def close(self):
        self.renderer.close()
