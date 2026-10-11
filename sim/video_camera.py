"""Progress-indexed generated video; physical outcomes belong to the runtime."""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import math
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from drivingbench.sim.car import DT
from drivingbench.sim.logs import REPO
from PIL import Image, ImageOps

CLIPS = ("approach", "keep", "switch")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_calibration(path, *, verify_hashes=True):
    """Validate anchors and source identity without decoding videos."""
    data = json.loads(Path(path).read_text())
    if data.get("version") != 1 or set(data.get("clips", {})) != set(CLIPS):
        raise ValueError("Expected version 1 and approach/keep/switch clips")
    end, tip = data.get("approach_end_m"), data.get("fork_tip_m")
    if not (_finite(end) and _finite(tip) and 0 < end < tip):
        raise ValueError("The approach endpoint must be positive and before the physical tip")
    for name, clip in data["clips"].items():
        for key in ("width", "height", "frame_count"):
            if type(clip.get(key)) is not int or clip[key] < 1:
                raise ValueError(f"Invalid {name} {key}")
        if not _finite(clip.get("fps")) or clip["fps"] <= 0:
            raise ValueError(f"Invalid {name} fps")
        distances, indices = clip.get("progress_m", []), clip.get("frame_indices", [])
        if (
            len(distances) < 2
            or len(distances) != len(indices)
            or not all(_finite(value) for value in distances)
            or any(b <= a for a, b in zip(distances, distances[1:]))
            or any(type(i) is not int for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))
            or indices[0] != 0
            or indices[-1] != clip["frame_count"] - 1
        ):
            raise ValueError(f"Invalid {name} distance/frame anchors")
        if distances[0] != (0 if name == "approach" else end):
            raise ValueError(f"Invalid {name} starting progress")
        if name == "approach" and distances[-1] != end:
            raise ValueError("Approach final anchor disagrees with approach_end_m")
        source = Path(clip["path"])
        source = source if source.is_absolute() else REPO / source
        clip["path"] = str(source.resolve())
        digest = clip.get("sha256", "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid {name} source hash")
        if verify_hashes and hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name} source hash mismatch")
    sizes = {(clip["width"], clip["height"]) for clip in data["clips"].values()}
    total = sum(c["width"] * c["height"] * c["frame_count"] * 3 for c in data["clips"].values())
    if len(sizes) != 1 or total > 1_200_000_000:
        raise ValueError("Video dimensions must agree and decoded cache must be at most 1.2 GB")
    return data


def _decode(clip):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        clip["path"],
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-fps_mode",
        "passthrough",
        "-threads",
        "1",
        "pipe:1",
    ]
    frame_bytes = clip["width"] * clip["height"] * 3
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            frames = []
            for _ in range(clip["frame_count"]):
                raw = process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise ValueError("Video ended before its declared frame count")
                frames.append(raw)
            if process.stdout.read(1):
                raise ValueError("Video exceeds its declared frame count")
            if process.wait(timeout=30):
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
            return frames
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


class VideoCamera:
    """Decode one pairing once, then select by longitudinal progress, never time."""

    def __init__(self, calibration):
        self.calibration = load_calibration(calibration)
        self.frames = {}
        self.last_metadata = {}

    def start(self):
        if not self.frames:
            decoded = {name: _decode(clip) for name, clip in self.calibration["clips"].items()}
            self.frames = decoded
        return {"camera": "narrow", "clips": {k: len(v) for k, v in self.frames.items()}}

    def render(self, progress_m, branch="keep", *, mirror=False, lateral_m=0, yaw_deg=0):
        if branch not in ("keep", "switch") or not _finite(progress_m):
            raise ValueError("Expected a finite progress and keep/switch branch")
        if (
            not (_finite(lateral_m) and _finite(yaw_deg))
            or abs(lateral_m) > 0.25
            or abs(yaw_deg) > 2
        ):
            raise ValueError(
                "Approximate display warp is limited to 0.25 m lateral and 2 degrees yaw"
            )
        if lateral_m or yaw_deg:
            if not all(
                _finite(self.calibration.get(k)) and self.calibration[k] > 0
                for k in ("focal_px", "warp_depth_m")
            ):
                raise ValueError("Display warp requires focal_px and warp_depth_m")
        name = "approach" if progress_m <= self.calibration["approach_end_m"] else branch
        clip = self.calibration["clips"][name]
        distances, indices = clip["progress_m"], clip["frame_indices"]
        if not distances[0] <= progress_m <= distances[-1]:
            raise ValueError(f"Progress {progress_m} is outside supported {name} views")
        segment = min(max(bisect.bisect_right(distances, progress_m) - 1, 0), len(distances) - 2)
        fraction = (progress_m - distances[segment]) / (distances[segment + 1] - distances[segment])
        index = round(indices[segment] + fraction * (indices[segment + 1] - indices[segment]))
        self.start()
        jpeg = self._jpeg(name, index, bool(mirror), lateral_m, yaw_deg)
        self.last_metadata = {
            "clip": name,
            "frame_index": index,
            "source_time_s": index / clip["fps"],
            "source_sha256": clip["sha256"],
            "progress_m": progress_m,
            "branch": branch,
            "before_tip": progress_m < self.calibration["fork_tip_m"],
            "mirror": bool(mirror),
            "lateral_warp_m": lateral_m,
            "yaw_warp_deg": yaw_deg,
            "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
        }
        return jpeg

    @lru_cache(maxsize=32)
    def _jpeg(self, name, index, mirror, lateral_m, yaw_deg):
        clip = self.calibration["clips"][name]
        image = Image.frombytes("RGB", (clip["width"], clip["height"]), self.frames[name][index])
        if mirror:
            image = ImageOps.mirror(image)
        if lateral_m or yaw_deg:
            f = self.calibration["focal_px"]
            angle = math.radians(yaw_deg)
            c, s = math.cos(angle), math.sin(angle)
            cx, cy = image.width / 2, image.height / 2
            dx = f * lateral_m / self.calibration["warp_depth_m"]
            # Inverse yaw homography with a single-depth lateral approximation.
            a, b = c - cx * s / f, f * s + cx * c
            denom = c + s * cx / f
            coefficients = (
                a / denom,
                0,
                (b - a * cx + dx) / denom,
                -cy * s / f / denom,
                1 / denom,
                (cy * c - cy + cy * s * cx / f) / denom,
                -s / f / denom,
                0,
            )
            image = image.transform(
                image.size, Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    def close(self):
        self.frames.clear()
        self._jpeg.cache_clear()


class VideoSnapshotRenderer:
    """One-image adapter; source-path residual warps stay disabled by default."""

    def __init__(self, scene_path, output=None):
        scene = json.loads(Path(scene_path).read_text())
        config = scene["video_camera"]
        path = Path(config["calibration"])
        self.camera = VideoCamera(path if path.is_absolute() else REPO / path)
        self.mirror = bool(config.get("mirror", False))
        self.default_branch = config.get("default_branch", "left" if self.mirror else "right")
        if self.default_branch not in ("left", "right"):
            raise ValueError("Expected a physical default branch left/right")
        self.metrics, self.errors = {}, {}

    def start(self):
        return self.camera.start()

    def render_snapshot(self, snapshot):
        state = snapshot["video_fork"]
        physical_branch = (
            state.get("locked_branch") or state.get("requested_branch") or state["actual_branch"]
        )
        if physical_branch not in ("left", "right"):
            raise ValueError("Expected physical requested/locked branch left/right")
        branch = "keep" if physical_branch == self.default_branch else "switch"
        actual_progress = state["progress_m"]
        progress, overshoot = actual_progress, 0.0
        if state.get("endpoint_reached"):
            endpoint = self.camera.calibration["clips"][branch]["progress_m"][-1]
            step = state.get("endpoint_step_m")
            overshoot = actual_progress - endpoint
            reported = state.get("endpoint_overshoot_m")
            if (
                state.get("phase") != "assigned_outcome"
                or state.get("locked_branch") != physical_branch
                or not _finite(state.get("endpoint_m"))
                or not math.isclose(state["endpoint_m"], endpoint, abs_tol=1e-9, rel_tol=0)
                or not _finite(step)
                or not 0 <= step <= 6 * DT
                or not -1e-9 <= overshoot <= step + 1e-9
                or not _finite(reported)
                or not math.isclose(reported, max(0.0, overshoot), abs_tol=1e-9, rel_tol=0)
            ):
                raise ValueError(
                    "Terminal video frame requires the selected endpoint and at most one physics tick of overshoot"
                )
            progress = endpoint
        jpeg = self.camera.render(progress, branch, mirror=self.mirror)
        self.metrics = dict(
            self.camera.last_metadata,
            physical_branch=physical_branch,
            actual_progress_m=actual_progress,
            endpoint_overshoot_m=max(0.0, overshoot),
            terminal_frame=bool(state.get("endpoint_reached")),
        )
        return jpeg

    def close(self):
        self.camera.close()
