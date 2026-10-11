"""Warm Gaussian and object workers with bounded crash recovery and one JPEG encode."""

from __future__ import annotations

import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from drivingbench.sim.actor_visibility import whole_actor_visibility
from drivingbench.sim.billboards import BillboardCache
from drivingbench.sim.camera_response import response_profile
from drivingbench.sim.logs import REPO
from drivingbench.sim.render_process import RenderProcess
from drivingbench.sim.scene import DEFAULT_SCENE, Scene
from drivingbench.sim.visual import CAMERA
from drivingbench.sim.visual_composite import compose
from drivingbench.sim.visual_objects import ObjectLayer, build_object_job, pinhole_camera
from drivingbench.sim.visual_splat import RENDER, TOOLS
from PIL import Image


class PersistentRenderer:
    def __init__(
        self,
        scene_path=DEFAULT_SCENE,
        *,
        output,
        timeout=60,
        device="METAL",
        new_session=True,
        camera=None,
        checkpoint_override=None,
    ):
        self.scene = Scene(scene_path)
        self.camera = pinhole_camera(CAMERA if camera is None else camera)
        self.checkpoint = REPO / (
            checkpoint_override
            if checkpoint_override is not None
            else self.scene.data["background"]["checkpoint"]
        )
        self.response = response_profile(self.scene.data.get("camera_response"))
        self.billboards = BillboardCache(self.scene.objects)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.timeout, self.device = timeout, device
        self.new_session = new_session
        self.splat = self.objects = None
        self.restarts = 0
        self.last_metrics = {}

    def _splat_start(self):
        self.splat = RenderProcess(
            [RENDER, self.checkpoint, "--serve"],
            self.output / "splat.log",
            timeout=self.timeout,
            new_session=self.new_session,
            cwd=TOOLS,
        )
        return self.splat.start()

    def _objects_start(self, matrix):
        job = build_object_job(
            [obj for obj in self.scene.objects if obj.kind not in ("billboard", "ground_decal")],
            self.scene.ground.normal,
            camera=self.camera,
            lighting=self.scene.lighting,
            object_renderer=self.scene.object_renderer,
        )
        job.update(
            transform_matrix=matrix.tolist(),
            output=str(self.output / "warmup"),
            device=self.device,
        )
        path = self.output / "object-job.json"
        path.write_text(json.dumps(job))
        self.objects = RenderProcess(
            [
                TOOLS / "Blender.app/Contents/MacOS/Blender",
                "--background",
                "--python",
                Path(__file__).with_name("object_worker.py"),
                "--",
                path,
            ],
            self.output / "objects.log",
            timeout=self.timeout,
            new_session=self.new_session,
        )
        return self.objects.start()

    def start(self):
        started = time.perf_counter()
        matrix = self.scene.camera_matrix((0, 0, 0))
        try:
            self._splat_start()
            for attempt in range(3):
                try:
                    ready = self._objects_start(matrix)
                    self.device = ready["device"]
                    break
                except (OSError, RuntimeError, TimeoutError):
                    self.restarts += 1
                    if attempt == 2:
                        raise
                    if attempt == 1:
                        self.device = "CPU"
            self.render((0, 0, 0))
        except Exception:
            self.close()
            raise
        return {
            "ready": True,
            "device": self.device,
            "warmup_s": time.perf_counter() - started,
            "lighting": ready.get("lighting"),
        }

    def _request(self, kind, matrix, output, *, camera=None):
        camera = pinhole_camera(CAMERA if camera is None else camera)
        for attempt in range(3):
            try:
                if attempt:
                    self.restarts += 1
                    if kind == "objects":
                        if attempt == 2:
                            self.device = "CPU"
                        self._objects_start(matrix)
                    else:
                        self._splat_start()
                payload = {
                    "transform_matrix": matrix.tolist(),
                    "output": str(output),
                    "camera": camera,
                }
                return getattr(self, kind).request(**payload)
            except (OSError, RuntimeError, TimeoutError):
                getattr(self, kind).close()
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def render(self, pose, sim_time_s=0, actor_onsets=None, actor_states=None):
        return self.render_matrix(
            self.scene.camera_matrix(pose),
            sim_time_s=sim_time_s,
            actor_onsets=actor_onsets,
            actor_states=actor_states,
        )

    def render_matrix(self, matrix, *, camera=None, **kwargs):
        started = time.perf_counter()
        image = self.render_matrix_rgb(matrix, camera=camera, **kwargs)
        encoding = time.perf_counter()
        output = io.BytesIO()
        Image.fromarray(image).save(output, "JPEG", quality=92)
        jpeg_s = time.perf_counter() - encoding
        self.last_metrics.update(
            elapsed_s=time.perf_counter() - started,
            composite_s=self.last_metrics["composite_s"] + jpeg_s,
            jpeg_s=jpeg_s,
        )
        return output.getvalue()

    def render_matrix_rgb(
        self,
        matrix,
        *,
        camera=None,
        sim_time_s=0,
        actor_onsets=None,
        actor_states=None,
        support_output=None,
    ):
        frame = self._draw_matrix(matrix, camera=camera)
        image, self.last_metrics = self._compose_frame(
            frame,
            sim_time_s=sim_time_s,
            actor_onsets=actor_onsets,
            actor_states=actor_states,
            support_output=support_output,
        )
        return image

    def render_matrices_rgb(
        self, views, *, sim_time_s=0, actor_onsets=None, actor_states=None, visibility_masks=False
    ):
        """Overlap immutable tile composition with the next draw on the same GPU workers."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            pending = []
            for matrix, camera in views:
                frame = self._draw_matrix(matrix, camera=camera)
                masks = {} if visibility_masks else None
                future = executor.submit(
                    self._compose_frame,
                    frame,
                    sim_time_s=sim_time_s,
                    actor_onsets=actor_onsets,
                    **({"actor_states": actor_states} if actor_states is not None else {}),
                    **({"visibility_masks": masks} if masks is not None else {}),
                )
                pending.append((future, masks))
            return [
                (*future.result(), masks) if visibility_masks else future.result()
                for future, masks in pending
            ]

    def _draw_matrix(self, matrix, *, camera=None):
        camera = pinhole_camera(self.camera if camera is None else camera)
        matrix = np.asarray(matrix, dtype=float)
        started = time.perf_counter()
        with TemporaryDirectory(prefix="frame-", dir=self.output) as temporary:
            root = Path(temporary)
            with ThreadPoolExecutor(max_workers=2) as executor:
                pending = {
                    kind: executor.submit(self._request, kind, matrix, root / kind, camera=camera)
                    for kind in ("splat", "objects")
                }
                splat, objects = (pending[kind].result() for kind in ("splat", "objects"))
            rendered = time.perf_counter()
            rgb = np.load(root / "splat/rgb.npy")
            depth = np.load(root / "splat/depth.npy")
            alpha = np.load(root / "splat/alpha.npy")
            layer = ObjectLayer(**{key: np.load(path) for key, path in objects["paths"].items()})
        return dict(
            matrix=matrix,
            camera=camera,
            rgb=rgb,
            depth=depth,
            alpha=alpha,
            layer=layer,
            started=started,
            rendered=rendered,
            splat=splat,
            objects=objects,
            restarts=self.restarts,
            device=self.device,
        )

    def _compose_frame(
        self,
        frame,
        *,
        sim_time_s=0,
        actor_onsets=None,
        actor_states=None,
        support_output=None,
        visibility_masks=None,
    ):
        matrix, camera = frame["matrix"], frame["camera"]
        rgb, depth, alpha, layer = (frame[key] for key in ("rgb", "depth", "alpha", "layer"))
        started, rendered = frame["started"], time.perf_counter()

        placed = self.scene.objects_at(
            sim_time_s,
            actor_onsets=actor_onsets,
            **({"actor_states": actor_states} if actor_states is not None else {}),
        )
        layer = [
            layer,
            *self.billboards.layers(placed, matrix, camera, self.scene.ground.normal),
        ]
        annotations = self.scene.data.get("scenario", {}).get("visibility_regions", {})
        instance_masks = {} if visibility_masks is not None or annotations else None
        composed = compose(
            self.scene,
            matrix,
            rgb,
            depth,
            alpha,
            layer,
            objects=placed,
            diagnostics=True,
            instance_masks=instance_masks,
            return_support=support_output is not None,
            response=self.response,
            camera=camera,
            seed=int.from_bytes(
                hashlib.sha256(
                    np.asarray(matrix, dtype="<f8").tobytes()
                    + np.asarray(sim_time_s, dtype="<f8").tobytes()
                ).digest()[:8],
                "little",
            ),
        )
        image, visibility = composed[:2]
        if support_output is not None:
            support_path = Path(support_output)
            support_path.parent.mkdir(parents=True, exist_ok=True)
            with support_path.open("xb") as target:
                Image.fromarray(composed[2].astype(np.uint8) * 255).save(target, "PNG")
        rows = self.scene.data["markers"] + self.scene.data["objects"]
        billboards = {
            rows[index].get("id", str(index)): visibility.get(
                index,
                {
                    "projected_pixels": 0,
                    "visible_pixels": 0,
                    "occluded_pixels": 0,
                    "unknown_pixels": 0,
                },
            )
            for index, obj in enumerate(placed)
            if obj.kind == "billboard"
        }
        if annotations and visibility_masks is None:
            for index, obj in enumerate(placed):
                identity = rows[index].get("id", str(index))
                if obj.kind == "billboard" and identity in annotations:
                    boxes = annotations[identity]
                    if "frames" in boxes or "default" in boxes:
                        boxes = boxes.get("frames", {}).get(obj.asset_path, boxes.get("default"))
                    billboards[identity]["whole_actor_visibility"] = whole_actor_visibility(
                        self.billboards,
                        obj,
                        matrix,
                        camera,
                        instance_masks.get(index, np.zeros((camera["h"], camera["w"]), bool)),
                        boxes,
                        self.scene.ground.normal,
                    )
        if visibility_masks is not None:
            visibility_masks.update(
                {
                    rows[index].get("id", str(index)): instance_masks[index]
                    for index, obj in enumerate(placed)
                    if obj.kind == "billboard" and index in instance_masks
                }
            )
        metrics = {
            "elapsed_s": time.perf_counter() - started,
            "composite_s": time.perf_counter() - rendered,
            "splat": frame["splat"],
            "objects": frame["objects"],
            "restarts": frame["restarts"],
            "device": frame["device"],
            "billboards": billboards,
            "camera_response": self.scene.data.get("camera_response"),
            "camera": camera,
        }
        return image, metrics

    def close(self):
        for worker in (self.objects, self.splat):
            if worker:
                worker.close()
