"""Persistent Blender object scene; JSONL replies use a prefix distinct from render logs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PREFIX = "DRIVINGBENCH "


def emit(message):
    print(PREFIX + json.dumps(message, allow_nan=False), flush=True)


def serve_requests(lines, render, respond=emit):
    """Keep request errors local; a crashed native renderer is recovered by its supervisor."""
    for line in lines:
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not isinstance(request.get("id"), (str, int)):
                raise ValueError("Object requests require an id")
            if request.get("shutdown") is True:
                respond({"id": request["id"], "ok": True, "shutdown": True})
                return
            result = render(request)
            respond({"id": request["id"], "ok": True, **result})
        except Exception as error:
            respond(
                {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )


class BlenderObjects:
    def __init__(self, job):
        if __package__:
            from . import visual_objects
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import visual_objects
        self.module = visual_objects
        self.job = {"crop": True} | job
        self.device = visual_objects._blender_scene(job)
        self.lighting = visual_objects.blender_lighting_status()

    def __call__(self, request):
        matrix = np.asarray(request["transform_matrix"], dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Object camera must be a finite 4x4 OpenCV matrix")
        output = Path(request["output"]).resolve()
        camera = self.module.pinhole_camera(request.get("camera", self.job["camera"]))
        start = time.perf_counter()
        region = self.module._blender_frame(
            self.job
            | {"transform_matrix": matrix.tolist(), "output": str(output), "camera": camera},
            arrays=True,
        )
        rendered = time.perf_counter()
        if region["empty"]:
            layer = self.module.empty_layer(camera)
        else:
            rgba = np.rint(
                np.clip(self.module._blender_pixels(output / "objects.png", noncolor=True), 0, 1)
                * 255
            ).astype(np.uint8)
            layer = self.module.repair_edge_depth(
                rgba,
                region["depth"],
                region["instance"],
            )
        arrays = {
            "rgba": layer.rgba,
            "depth": layer.depth,
            "instance": layer.instance,
            "repaired": layer.repaired,
        }
        paths = {}
        for name, array in arrays.items():
            path = output / f"{name}.npy"
            np.save(path, array)
            paths[name] = str(path)
        end = time.perf_counter()
        return {
            "paths": paths,
            "lighting": self.lighting,
            "render_bounds": region["bounds"],
            "render_s": rendered - start,
            "export_s": end - rendered,
            "elapsed_s": end - start,
        }


def main(job_path):
    job = json.loads(Path(job_path).read_text())
    started = time.perf_counter()
    renderer = BlenderObjects(job)
    warm = renderer(job)
    emit(
        {
            "ready": True,
            "device": renderer.device,
            "lighting": renderer.lighting,
            "warmup_s": time.perf_counter() - started,
            "warmup": warm,
        }
    )
    serve_requests(sys.stdin, renderer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    main(args.job)
