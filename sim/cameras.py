"""Continuous pose-stamped camera capture, independent of physics and tool requests."""

from __future__ import annotations

import fcntl
import hashlib
import math
import queue
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from drivingbench.device.cameras import Cameras, Frame, camera_clock
from drivingbench.sim.logs import REPO
from drivingbench.sim.render_process import RenderProcess


@contextmanager
def render_slot(path=None, stop=None):
    """Share one cancelable GPU slot between renderer warmup and paired captures."""
    stop = stop or threading.Event()
    if stop.is_set():
        raise RuntimeError("Camera closed")
    if path is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        while not stop.is_set():
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                stop.wait(0.01)
        else:
            raise RuntimeError("Camera closed")
        try:
            if stop.is_set():
                raise RuntimeError("Camera closed")
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_renderer_priority(priority):
    """Validate the one explicit optional launch policy; never adjust a live process."""
    if priority is None:
        return None
    if (
        sys.platform != "darwin"
        or type(priority) is not dict
        or set(priority) != {"executable", "increment", "executable_sha256"}
        or priority["executable"] != "/usr/bin/nice"
        or type(priority["increment"]) is not int
        or priority["increment"] != 5
        or not isinstance(priority["executable_sha256"], str)
        or len(priority["executable_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in priority["executable_sha256"])
    ):
        raise ValueError("Renderer priority requires the exact Darwin nice5 executable record")
    executable = Path(priority["executable"])
    if (
        not executable.is_file()
        or hashlib.sha256(executable.read_bytes()).hexdigest() != priority["executable_sha256"]
    ):
        raise ValueError("Renderer priority executable differs from the prospective binding")
    return dict(priority)


class SceneRenderer:
    """Keep optional visual dependencies in their own Python environment."""

    def __init__(self, scene_path, output, *, priority=None):
        self.priority = validate_renderer_priority(priority)
        self.output = Path(output).resolve()
        self.worker = RenderProcess(
            [
                REPO / "runs/sim/visual/venv/bin/python",
                "-m",
                "drivingbench.sim.render_worker",
                "--scene",
                Path(scene_path).resolve(),
                "--output",
                self.output,
            ],
            self.output / "renderer.log",
            timeout=180,
            cwd=REPO,
        )
        if self.priority is not None:
            self.worker.command = [
                self.priority["executable"],
                "-n",
                str(self.priority["increment"]),
                *self.worker.command,
            ]
        self._launch_command = tuple(self.worker.command)
        self.metrics = {}
        self.errors = {}
        self.ready = None

    def start(self):
        if self.ready is None:
            validate_renderer_priority(self.priority)
            if self.priority is not None and tuple(self.worker.command) != self._launch_command:
                raise ValueError("Renderer priority launch command changed before startup")
            self.ready = self.worker.start()
        return self.ready

    def render(self, pose):
        return self.render_snapshot({"pose": pose, "sim_time_s": 0})

    def render_snapshot(self, snapshot):
        frames = self.render_frames(snapshot)
        if "narrow" not in frames:
            raise RuntimeError(self.errors.get("narrow", "narrow_camera_unavailable"))
        return frames["narrow"]

    def render_frames(self, snapshot):
        with TemporaryDirectory(prefix="camera-", dir=self.output) as output:
            result = self.worker.request(
                pose=list(snapshot["pose"]),
                sim_time_s=snapshot["sim_time_s"],
                actor_onsets=snapshot.get("actor_onsets", {}),
                output=output,
                **(
                    {"actor_states": snapshot["actor_states"]} if "actor_states" in snapshot else {}
                ),
            )
            self.metrics = result["metrics"]
            self.errors = dict(result.get("errors", {}))
            frames = {}
            for camera, path in result["paths"].items():
                if camera in self.errors:
                    continue
                try:
                    frames[camera] = Path(path).read_bytes()
                except OSError as error:
                    self.errors[camera] = f"{type(error).__name__}: {error}"
            return frames

    def close(self):
        self.worker.close()


class _ObservationCameras:
    """One publication view sharing the upstream adjustment cache, never its live frames."""

    def __init__(self, frames, errors, cache, lock):
        self.frames, self.errors, self.cache, self.lock = frames, errors, cache, lock
        self.workers = {}

    frame = Cameras.frame


class SimCameras(Cameras):
    def __init__(self, native, renderer, *, interval=0.05):
        super().__init__()
        self.native, self.renderer, self.interval = native, renderer, interval
        self.thread = None
        self.on_evidence = None
        self.ready = None
        self.browse_requests = queue.Queue(maxsize=1)
        self.last_metrics = {}
        self.errors["wide"] = "camera_unavailable"

    def observation_snapshot(self):
        """Pin both source frames and errors atomically; adjustments happen after unlock."""
        with self.lock:
            return _ObservationCameras(dict(self.frames), dict(self.errors), self.cache, self.lock)

    def _render_error(self, error, *, startup=False):
        stopping = self.stop.is_set()
        if not stopping:
            with self.lock:
                names = (
                    ("narrow", "wide") if hasattr(self.renderer, "render_frames") else ("narrow",)
                )
                for camera in names:
                    prefix = "startup: " if startup else ""
                    self.errors[camera] = f"{prefix}{type(error).__name__}: {error}"
        if self.on_evidence:
            self.on_evidence(
                "sim_render_shutdown" if stopping else "sim_render_error", error=str(error)
            )
        return stopping

    def _capture_once(self, *, requested_at=None, queue_wait_s=0.0):
        snapshot = getattr(self.native, "capture_snapshot", self.native.pose_snapshot)()
        started = time.perf_counter()
        render_frames = getattr(self.renderer, "render_frames", None)
        if render_frames:
            images = render_frames(snapshot)
            errors = dict(getattr(self.renderer, "errors", {}))
        else:
            render_snapshot = getattr(self.renderer, "render_snapshot", None)
            images = {
                "narrow": render_snapshot(snapshot)
                if render_snapshot
                else self.renderer.render(snapshot["pose"])
            }
            errors = {}
        if self.stop.is_set():
            return
        if set(images) - {"narrow", "wide"} or set(errors) - {"narrow", "wide"}:
            raise ValueError("Unknown rendered camera")
        frames = {}
        for camera, jpeg in images.items():
            if not isinstance(jpeg, bytes) or not jpeg:
                errors[camera] = "invalid_camera_jpeg"
            elif camera not in errors:
                frames[camera] = Frame(uuid4().hex, snapshot["captured_at"], jpeg)
        with self.lock:
            for camera in ("narrow", "wide"):
                if camera in frames:
                    self.frames[camera] = frames[camera]
                    self.errors.pop(camera, None)
                else:
                    if camera in self.frames or camera == "narrow":
                        errors.setdefault(camera, "missing_camera_frame")
                    self.errors[camera] = errors.get(camera, "camera_unavailable")
        self.last_metrics = {
            "captured_at": snapshot["captured_at"],
            "requested_at": requested_at,
            "queue_wait_s": queue_wait_s,
            "request_to_publication_s": (
                time.monotonic() - requested_at if requested_at is not None else None
            ),
            "pose_sample_age_at_capture_s": (
                max(0.0, snapshot["monotonic_at"] - snapshot["pose_sample_monotonic_at"])
                if "pose_sample_monotonic_at" in snapshot
                else None
            ),
            "pose_sample_monotonic_at": snapshot.get("pose_sample_monotonic_at"),
            "pose_sample_age_s": (
                max(0.0, time.monotonic() - snapshot["pose_sample_monotonic_at"])
                if "pose_sample_monotonic_at" in snapshot
                else None
            ),
            "render_s": time.perf_counter() - started,
            "image_age_s": camera_clock() - snapshot["captured_at"],
            "frame_ids": {camera: frame.key for camera, frame in frames.items()},
            "camera_errors": errors,
            "pose": snapshot["pose"],
            "sim_time_s": snapshot["sim_time_s"],
            "actor_onsets": snapshot.get("actor_onsets", {}),
            "worker": getattr(self.renderer, "metrics", {}),
            **({"actor_states": snapshot["actor_states"]} if "actor_states" in snapshot else {}),
        }
        if self.on_evidence:
            if "narrow" in frames:
                self.on_evidence("sim_frame", **self.last_metrics)
            for camera, error in errors.items():
                self.on_evidence("sim_render_error", camera=camera, error=error)

    def start(self):
        if self.thread:
            return

        def capture():
            try:
                self.ready = self.renderer.start()
                while not self.stop.is_set():
                    started = time.perf_counter()
                    try:
                        self._capture_once()
                    except Exception as error:
                        if self._render_error(error):
                            break
                    try:
                        pose, result, done = self.browse_requests.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        try:
                            result.append(self.renderer.render(pose))
                        except Exception as error:
                            result.append(error)
                        finally:
                            done.set()
                    self.stop.wait(max(0.0, self.interval - (time.perf_counter() - started)))
            except Exception as error:
                self._render_error(error, startup=True)
            finally:
                self.renderer.close()

        self.thread = threading.Thread(target=capture, name="sim-camera", daemon=True)
        self.thread.start()

    def _browse_pose(self, map_pose):
        x, y, heading = map(float, map_pose)
        if not all(math.isfinite(value) for value in (x, y, heading)):
            raise ValueError("Browse pose must be finite")
        sx, sy, sh = self.native.course.start
        angle = math.radians(sh)
        dx, dy = x - sx, y - sy
        pose = (
            dx * math.cos(angle) + dy * math.sin(angle),
            -dx * math.sin(angle) + dy * math.cos(angle),
            heading - sh,
        )
        return pose

    def browse(self, map_pose):
        if self.stop.is_set():
            raise RuntimeError("Camera closed")
        pose = self._browse_pose(map_pose)
        result, done = [], threading.Event()
        try:
            self.browse_requests.put_nowait((pose, result, done))
        except queue.Full as error:
            raise RuntimeError("Browse camera busy") from error
        if not done.wait(15):
            raise TimeoutError("Browse render timed out")
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.renderer.close()
                self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise TimeoutError("sim_camera_did_not_close")


class OnDemandCameras(SimCameras):
    """Render a new paired exposure for each observe, without holding the physics lock."""

    def __init__(self, native, renderer, *, render_lock_path=None):
        super().__init__(native, renderer)
        self.render_lock_path = (
            None if render_lock_path is None else Path(render_lock_path).resolve()
        )
        self._render_lock = threading.Lock()
        self._startup_done = threading.Event()

    @contextmanager
    def _render_slot(self):
        with self._render_lock, render_slot(self.render_lock_path, self.stop):
            yield

    def start(self):
        if self.thread:
            return

        def warmup():
            try:
                with self._render_slot():
                    if self.stop.is_set():
                        return
                    self.ready = self.renderer.start()
                    if not self.stop.is_set():
                        self._capture_once()
            except Exception as error:
                self._render_error(error, startup=True)
            finally:
                self._startup_done.set()

        self.thread = threading.Thread(target=warmup, name="sim-camera-warmup", daemon=True)
        self.thread.start()

    def observation_snapshot(self):
        requested = time.monotonic()
        if not self._startup_done.wait(180):
            raise TimeoutError("Camera startup timed out")
        with self._render_slot():
            if self.stop.is_set():
                raise RuntimeError("Camera closed")
            try:
                self._capture_once(
                    requested_at=requested, queue_wait_s=time.monotonic() - requested
                )
            except Exception as error:
                self._render_error(error)
            return super().observation_snapshot()

    def browse(self, map_pose):
        pose = self._browse_pose(map_pose)
        if not self._startup_done.wait(180):
            raise TimeoutError("Camera startup timed out")
        with self._render_slot():
            if self.stop.is_set():
                raise RuntimeError("Camera closed")
            return self.renderer.render(pose)

    def close(self):
        self.stop.set()
        self.renderer.close()
        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise TimeoutError("sim_camera_did_not_close")
