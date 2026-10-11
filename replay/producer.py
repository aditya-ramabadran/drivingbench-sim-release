"""Reuse the real producer with local virtual-native and recorded-camera adapters."""

import io
import math
import threading
import time

from drivingbench.controller.core import Controller, Sample
from drivingbench.device.cameras import Cameras, Frame, camera_clock
from drivingbench.device.service import create_app
from drivingbench.shared.contracts import PROTOCOL, Settings


class VirtualNative:
    """Controller acceptance/expiry is real; measured motion is an explicit approximation."""

    def __init__(
        self,
        length_m,
        dynamics,
        *,
        hard_steer_percent=30,
        clock=time.monotonic,
        hazard_boundary_m=None,
    ):
        self.controller = Controller()
        self.length_m, self.dynamics = length_m, dynamics
        if hazard_boundary_m is not None and not 0 < hazard_boundary_m < length_m:
            raise ValueError("hazard boundary must lie strictly inside the source window")
        self.hazard_boundary_m = hazard_boundary_m
        self.on_evidence = None
        self.last_stop_cause = "initial_hold"
        self.hard_steer_percent, self.clock = hard_steer_percent, clock
        self.odometer_m = self.speed_mps = self.steering_deg = 0.0
        self.launch_at = None
        self.last_time = clock()
        self.end_reason = None
        self.pending_end = None
        self.error = None
        self.on_status = None
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self._sample(self.last_time)

    def _sample(self, now):
        # Refresh heartbeat before physical state: the native controller remains the owner.
        self.controller.handle({"protocol": PROTOCOL, "operation": "heartbeat"}, now)
        noise = self.dynamics.speed_noise_mps * math.sin(now * 7.1) if self.speed_mps else 0
        self.controller.refresh(
            Sample(
                speed_mps=max(0, self.speed_mps + noise),
                steering_deg=self.steering_deg,
                gear="drive",
                enabled=self.end_reason is None,
                sensors_valid=True,
                car_state_valid=True,
                standstill=self.speed_mps <= 0.001,
                lateral_active=self.speed_mps > 0.001,
                longitudinal_active=True,
            ),
            now,
        )

    def advance(self, now=None):
        with self.lock:
            now = self.clock() if now is None else now
            if now < self.last_time:
                raise ValueError("clock moved backwards")
            # Substeps preserve expiry timing even when an HTTP call arrives between ticks.
            while self.last_time < now:
                dt = min(0.05, now - self.last_time)
                self.last_time += dt
                self._sample(self.last_time)
                previous_command = self.controller.command
                selection = self.controller.step(self.last_time)
                if previous_command is not None and self.controller.command is None:
                    self.last_stop_cause = "command_expiry"
                command = self.controller.command
                target = command["speed_mps"] if command and not selection.should_stop else 0
                if target and self.launch_at is not None and self.last_time < self.launch_at:
                    target = 0
                rate = (
                    self.dynamics.accel_mps2
                    if target > self.speed_mps
                    else self.dynamics.brake_mps2
                )
                previous = self.speed_mps
                self.speed_mps += max(-rate * dt, min(rate * dt, target - previous))
                self.odometer_m += (previous + self.speed_mps) * 0.5 * dt
                angle = selection.angle_deg
                delta = self.dynamics.steering_rate_deg_s * dt
                self.steering_deg += max(-delta, min(delta, angle - self.steering_deg))
                if (
                    self.end_reason is None
                    and self.hazard_boundary_m is not None
                    and self.odometer_m >= self.hazard_boundary_m
                ):
                    self.odometer_m = self.hazard_boundary_m
                    if self.on_evidence:
                        self.on_evidence(
                            "hazard_boundary_crossing",
                            odometer_m=self.odometer_m,
                            speed_mps=self.speed_mps,
                            stop_cause=self.last_stop_cause,
                        )
                    self.finish("hazard_boundary")
                if self.odometer_m >= self.length_m:
                    self.odometer_m = self.length_m
                    self.finish("window_end")
                self._sample(self.last_time)
            return self.snapshot()

    def snapshot(self):
        return {
            **self.controller.status(self.last_time),
            "received_at": self.last_time,
            "odometer_m": self.odometer_m,
        }

    def status(self):
        return self.advance()

    def call(self, body, **_):
        with self.lock:
            self.advance()
            before = self.controller.command
            reply = self.controller.handle(body, self.last_time)
            command = self.controller.command
            if body.get("operation") == "motion" and reply.get("status") == "accepted":
                self.last_stop_cause = None
                if before is None and self.speed_mps <= 0.001:
                    self.launch_at = self.last_time + self.dynamics.launch_delay_s
                scale = self.controller.settings.max_steering_angle_deg
                if abs(command["target_angle_deg"]) / scale * 100 >= self.hard_steer_percent:
                    self.pending_end = "hard_steer"
            if body.get("operation") == "stop" and reply.get("status") == "accepted":
                # A stop after expiry cannot retroactively cause the earlier braking.
                if before is not None:
                    self.last_stop_cause = "explicit_stop"
                elif self.speed_mps <= 0.001 and self.last_stop_cause == "initial_hold":
                    self.last_stop_cause = "explicit_stop_at_rest"
            if self.on_status:
                self.on_status(self.snapshot())
            return reply

    def finish(self, reason):
        with self.lock:
            if self.end_reason is not None:
                return
            self.end_reason = reason
            self.controller.cancel()
            self.speed_mps = 0.0
            self._sample(self.last_time)

    def start(self):
        def tick():
            try:
                while not self.stop.is_set():
                    with self.lock:
                        snapshot = self.advance()
                        if self.on_status:
                            self.on_status(snapshot)
                    self.stop.wait(0.05)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.finish("infrastructure_error")

        self.thread = threading.Thread(target=tick, name="offline-native", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(2)


class ReplayCameras(Cameras):
    """Same image path; on-demand composition cached 0.5 s, not independent capture."""

    def __init__(self, clip, native):
        super().__init__()
        self.clip, self.native = clip, native
        self.capture_at = None
        self.source_distance_m = 0.0

    def start(self):
        pass

    def frame(self, camera, adjustments):
        with self.lock:
            now = camera_clock()
            if self.capture_at is None or now - self.capture_at >= 0.5:
                self.source_distance_m = self.native.status()["odometer_m"]
                # Both cameras reference the same frozen odometer snapshot.
                for name in ("narrow", "wide"):
                    try:
                        image = self.clip.frame_at(self.source_distance_m, name)
                    except (KeyError, ValueError):
                        continue
                    output = io.BytesIO()
                    image.convert("RGB").save(output, "JPEG", quality=92)
                    self.frames[name] = Frame(f"{now}-{name}", now, output.getvalue())
                self.capture_at = now
        return super().frame(camera, adjustments)

    def close(self):
        pass


def replay_app(
    directory, clip, dynamics, *, settings=None, hard_steer_percent=30, hazard_boundary_m=None
):
    """No physical native socket, camera worker, or network device is created."""
    native = VirtualNative(
        clip.length_m,
        dynamics,
        hard_steer_percent=hard_steer_percent,
        hazard_boundary_m=hazard_boundary_m,
    )
    cameras = ReplayCameras(clip, native)
    app = create_app(directory, native=native, cameras=cameras)
    app.state.settings.patch((settings or Settings()).model_dump())
    recorder = app.state.recording
    original_record = recorder.record

    def record(kind, **data):
        # Evidence fields never enter observe/motion/stop tool responses.
        data.setdefault("odometer_m", native.odometer_m)
        data.setdefault("stop_cause", native.last_stop_cause)
        data.setdefault("actual_speed_mps", native.speed_mps)
        if kind == "tool" and data.get("tool") == "observe":
            data["source_distance_m"] = cameras.source_distance_m
        event = original_record(kind, **data)
        if kind == "tool" and data.get("tool") == "set_motion" and native.pending_end:
            native.finish(native.pending_end)
            native.pending_end = None
        return event

    recorder.record = record
    native.on_evidence = lambda kind, **data: recorder.record(kind, **data)
    app.state.virtual_native = native
    app.state.replay_cameras = cameras
    return app
