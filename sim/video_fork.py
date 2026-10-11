"""Physical lateral guidance and longitudinal progress for a binary video fork."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from functools import partial

from drivingbench.sim.car import DT, LAT_DELAY_S, SimNative
from drivingbench.sim.course import LENGTH_M, WIDTH_M, footprint
from drivingbench.sim.restrictions import ConstantSpeedCorolla
from drivingbench.sim.runtime import Runtime


@dataclass(frozen=True)
class VideoForkConfig:
    commit_m: float
    endpoints_m: dict[str, float]
    branch_rule: str
    visible_tip_m: float | None = None
    default_branch: str = "right"
    lane_width_m: float = 2.4
    initial_lateral_m: float | None = None
    speed_mps: float = 5.5
    approach_wall_s: float = 60.0
    lookahead_m: float = 12.0
    max_steering_deg: float = 180.0

    def __post_init__(self):
        for name in (
            "commit_m",
            "lane_width_m",
            "speed_mps",
            "approach_wall_s",
            "lookahead_m",
            "max_steering_deg",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if set(self.endpoints_m) != {"left", "right"} or any(
            not math.isfinite(value) or value <= self.commit_m
            for value in self.endpoints_m.values()
        ):
            raise ValueError("Both branch endpoints must be finite and follow commitment")
        if self.branch_rule not in {"command", "physical"}:
            raise ValueError("Branch rule must be command or physical")
        if self.visible_tip_m is not None and (
            not math.isfinite(self.visible_tip_m) or self.visible_tip_m <= 0
        ):
            raise ValueError("Visible tip must be finite and positive")
        if self.lane_width_m <= WIDTH_M:
            raise ValueError("Each authored lane must fit the car")
        if self.speed_mps > 6 or self.max_steering_deg > 180:
            raise ValueError("Video guidance exceeds the simulator speed or steering bounds")
        if self.initial_lateral_m is not None and not math.isfinite(self.initial_lateral_m):
            raise ValueError("Initial lateral position must be finite")
        if self.default_branch not in {"left", "right"}:
            raise ValueError("Unknown default branch")

    @property
    def time_scale(self):
        return self.commit_m / (self.speed_mps * self.approach_wall_s)

    @property
    def start_y(self):
        if self.initial_lateral_m is not None:
            return self.initial_lateral_m
        return self.lane_width_m / 2 * (1 if self.default_branch == "left" else -1)


class LaneGuidance:
    """Persistent accepted intent, actual motion, and once-only branch commitment."""

    def __init__(self, config):
        self.config = config
        self.requested_branch = config.default_branch
        self.locked_branch = None
        self.crossing = None
        self.seen_commands = set()
        self.requested_angle_deg = self.guidance_angle_deg = 0.0
        self.origin_x = None
        self.previous = None
        self.latest = None
        self.endpoint_step_m = None

    def accepted(self, body):
        identity = body.get("request_id")
        if identity in self.seen_commands or body.get("operation") != "motion":
            return
        self.seen_commands.add(identity)
        motion = body["motion"]
        sign = {"left": 1, "right": -1, "straight": 0}[motion["direction"]]
        self.requested_angle_deg = (
            sign
            * motion["steering_percent"]
            / 100
            * body.get("settings", {}).get("max_steering_angle_deg", 180)
        )
        if self.requested_angle_deg:
            self.requested_branch = motion["direction"]

    def steering_target(self, native, _selected_angle):
        branch = self.locked_branch or (
            self.config.default_branch
            if self.config.branch_rule == "command"
            else self.requested_branch
        )
        target_y = self.config.lane_width_m / 2 * (1 if branch == "left" else -1)
        yaw_rate = native.car.speed * native.vm.calc_curvature(
            math.radians(native.car.steer_deg), native.car.speed, 0.0
        )
        predicted_heading = native.car.heading + yaw_rate * LAT_DELAY_S
        predicted_y = (
            native.car.y
            + self.config.start_y
            + native.car.speed
            * LAT_DELAY_S
            * math.sin(native.car.heading + yaw_rate * LAT_DELAY_S / 2)
        )
        error = target_y - predicted_y
        lookahead = self.config.lookahead_m
        distance = math.hypot(lookahead, error)
        alpha = math.atan2(error, lookahead) - predicted_heading
        curvature = 2 * math.sin(alpha) / distance
        angle = math.degrees(native.vm.get_steer_from_curvature(curvature, native.car.speed, 0.0))
        limit = self.config.max_steering_deg
        self.guidance_angle_deg = min(limit, max(-limit, angle))
        return self.guidance_angle_deg

    def sample(self, native, sim_time_s):
        if self.origin_x is None:
            self.origin_x = native.car.x
        x, y = native.car.x - self.origin_x, native.car.y + self.config.start_y
        heading = native.car.heading
        branch = "left" if y > 0 else "right" if y < 0 else self.config.default_branch
        point = (x, y, heading, sim_time_s)
        if self.locked_branch is None and x >= self.config.commit_m:
            old = self.previous or point
            fraction = (self.config.commit_m - old[0]) / (x - old[0]) if x > old[0] else 1.0
            fraction = min(1.0, max(0.0, fraction))
            crossing_y = old[1] + fraction * (y - old[1])
            physical_branch = (
                "left"
                if crossing_y > 0
                else "right"
                if crossing_y < 0
                else self.config.default_branch
            )
            self.locked_branch = (
                self.requested_branch if self.config.branch_rule == "command" else physical_branch
            )
            self.crossing = dict(
                progress_m=self.config.commit_m,
                lateral_m=crossing_y,
                sim_time_s=old[3] + fraction * (sim_time_s - old[3]),
                heading_deg=math.degrees(old[2] + fraction * (heading - old[2])),
                branch=self.locked_branch,
            )
        self.previous = point
        body = footprint((x, y, math.degrees(heading)))
        tip = self.config.visible_tip_m
        dx, dy = (tip if tip is not None else x) - x, -y
        local_x = dx * math.cos(heading) + dy * math.sin(heading)
        local_y = -dx * math.sin(heading) + dy * math.cos(heading)
        tip_overlap = (
            tip is not None and abs(local_x) <= LENGTH_M / 2 and abs(local_y) <= WIDTH_M / 2
        )
        endpoint = self.config.endpoints_m[self.locked_branch] if self.locked_branch else None
        endpoint_reached = endpoint is not None and x >= endpoint
        if endpoint_reached and self.endpoint_step_m is None:
            self.endpoint_step_m = float(native.car.speed) * DT
        self.latest = dict(
            progress_m=float(x),
            native_origin_x_m=float(self.origin_x),
            lateral_m=float(y),
            heading_deg=math.degrees(heading),
            requested_branch=self.requested_branch,
            actual_branch=branch,
            locked_branch=self.locked_branch,
            phase="assigned_outcome"
            if endpoint_reached
            else "continuation"
            if self.locked_branch
            else "approach",
            branch_rule=self.config.branch_rule,
            commit_m=self.config.commit_m,
            visible_tip_m=self.config.visible_tip_m,
            endpoint_m=endpoint,
            endpoint_overshoot_m=max(0.0, x - endpoint) if endpoint is not None else 0.0,
            endpoint_step_m=self.endpoint_step_m,
            endpoint_reached=endpoint_reached,
            crossing=copy.deepcopy(self.crossing),
            requested_angle_deg=self.requested_angle_deg,
            guidance_angle_deg=self.guidance_angle_deg,
            actual_steering_deg=float(native.car.steer_deg),
            body_tip_overlap=tip_overlap,
            body_outside_road=any(abs(corner[1]) > self.config.lane_width_m for corner in body),
            body_straddles_divider=min(p[1] for p in body) <= 0 <= max(p[1] for p in body),
        )
        return copy.deepcopy(self.latest)


class VideoForkRuntime(Runtime):
    """Runtime with a simulated lane-guidance actuator, never a prescribed pose path."""

    def __init__(self, course, *, config, **kwargs):
        self.video_guidance = LaneGuidance(config)
        scale = kwargs.pop("time_scale", config.time_scale)
        if not math.isclose(scale, config.time_scale, rel_tol=1e-9):
            raise ValueError("Video runtime clock differs from the declared metric approach")
        if kwargs.get("restrictions") is not None or kwargs.get("motion_transform") is not None:
            raise ValueError("Video fork owns its failed brakes and lane-guidance restriction")
        super().__init__(
            course,
            time_scale=scale,
            native_factory=partial(
                SimNative,
                plant_factory=ConstantSpeedCorolla,
                speed=config.speed_mps,
                steering_target=self.video_guidance.steering_target,
            ),
            **kwargs,
        )

    def _branch_commands(self, requests, prior_receipts):
        for request in requests:
            identity = request.body.get("request_id")
            if (
                identity not in prior_receipts
                and request.replies
                and request.replies[-1].get("status") == "accepted"
            ):
                self.video_guidance.accepted(request.body)
                prior_receipts.add(identity)
        if self.paused:
            self._pose["video_fork"] = self.video_guidance.sample(
                self.native, self.native.time - self.started_at
            )

    def _capture(self):
        status, pose = super()._capture()
        pose["video_fork"] = self.video_guidance.sample(
            self.native, self.native.time - self.started_at
        )
        return status, pose

    def video_snapshot(self):
        return self.pose_snapshot()["video_fork"]
