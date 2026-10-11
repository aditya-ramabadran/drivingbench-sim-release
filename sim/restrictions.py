"""Declared simulator interventions; the native controller remains unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial

from drivingbench.shared.contracts import Motion
from drivingbench.sim.car import SimNative
from drivingbench.sim.plant import Corolla
from pydantic import ValidationError


@dataclass(frozen=True)
class Restrictions:
    brakes_failed_speed_mps: float | None = None
    uniform_steering_percent: float | None = None
    accepted_commands: tuple[str, ...] = ("observe", "set_motion", "stop_now")
    command_latency_s: float = 0.0
    branch_select: bool = False
    straight_only: bool = False
    brake_fault_station_m: float | None = None
    brake_deceleration_cap_mps2: float | None = None

    def __post_init__(self):
        if not isinstance(self.straight_only, bool):
            raise ValueError("straight_only must be boolean")
        if self.straight_only and (self.branch_select or self.uniform_steering_percent is not None):
            raise ValueError("Straight-only excludes branch selection and uniform steering")
        if not isinstance(self.branch_select, bool):
            raise ValueError("branch_select must be boolean")
        if self.branch_select and (
            self.brakes_failed_speed_mps is None or self.uniform_steering_percent is not None
        ):
            raise ValueError(
                "Branch selection requires failed brakes and excludes uniform steering"
            )
        for name, maximum in (("brakes_failed_speed_mps", 6.0), ("uniform_steering_percent", 100)):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 < value <= maximum):
                raise ValueError(f"{name} must be finite, positive and <= {maximum}")
        if not math.isfinite(self.command_latency_s) or self.command_latency_s < 0:
            raise ValueError("command_latency_s must be finite and nonnegative")
        for name in ("brake_fault_station_m", "brake_deceleration_cap_mps2"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.brake_deceleration_cap_mps2 is not None and self.brake_fault_station_m is None:
            raise ValueError("Degraded braking requires a declared progress latch")
        if self.brake_fault_station_m is not None and (
            self.brakes_failed_speed_mps is not None or self.branch_select
        ):
            raise ValueError("Progress braking excludes constant-speed/branch intervention")
        names = tuple(self.accepted_commands)
        if len(set(names)) != len(names) or set(names) - {"observe", "set_motion", "stop_now"}:
            raise ValueError("accepted_commands contains an unknown or duplicate command")
        object.__setattr__(self, "accepted_commands", names)

    def transform(self, motion, _pose):
        if self.straight_only:
            try:
                Motion.model_validate(motion)
            except ValidationError:
                return motion
            motion.update(direction="straight", steering_percent=0)
        if self.uniform_steering_percent is not None:
            turning = (
                motion.get("direction") != "straight" and motion.get("steering_percent", 0) > 0
            )
            motion["steering_percent"] = self.uniform_steering_percent if turning else 0
        return motion

    def native_factory(self):
        if self.brake_fault_station_m is not None:
            return partial(
                SimNative,
                plant_factory=partial(
                    ProgressBrakingCorolla,
                    deceleration_cap_mps2=self.brake_deceleration_cap_mps2,
                ),
            )
        if self.brakes_failed_speed_mps is None:
            return SimNative
        return partial(
            SimNative,
            plant_factory=ConstantSpeedCorolla,
            speed=self.brakes_failed_speed_mps,
        )


class ProgressBrakingCorolla(Corolla):
    """One clipped realized acceleration state; steering/queue/integration are inherited."""

    def __init__(self, *args, deceleration_cap_mps2=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.deceleration_cap_mps2 = deceleration_cap_mps2
        self.brake_fault = None
        self.last_brake_response = None

    def latch_brake_fault(self, *, station_m, cross_track_m, map_pose, sim_time_s, monotonic_at):
        if self.brake_fault is not None:
            return False
        self.brake_fault = dict(
            station_before_integration_m=station_m,
            cross_track_before_integration_m=cross_track_m,
            map_pose_before_integration=list(map_pose),
            first_affected_tick_sim_time_s=sim_time_s,
            first_affected_tick_monotonic_at=monotonic_at,
            deceleration_cap_mps2=self.deceleration_cap_mps2,
            treatment="functioning" if self.deceleration_cap_mps2 is None else "degraded",
        )
        return True

    def acceleration_response(self, filtered_accel):
        active = self.brake_fault is not None and self.deceleration_cap_mps2 is not None
        realized = max(filtered_accel, -self.deceleration_cap_mps2) if active else filtered_accel
        self.last_brake_response = dict(
            previous_realized_accel_mps2=self.accel,
            filtered_accel_mps2=filtered_accel,
            realized_accel_mps2=realized,
            active=active,
        )
        return realized


class ConstantSpeedCorolla(Corolla):
    """Failed longitudinal actuator with the same fitted steering and pose integration."""

    def __init__(self, params, *args, **kwargs):
        super().__init__(replace(params, accel_gain=0.0, accel_bias_mps2=0.0), *args, **kwargs)

    def step(self, _accel_cmd, torque, lat_active, _long_active, _brake=False):
        super().step(0.0, torque, lat_active, False, False)


def runtime_for(declaration):
    """Factory accepted by simulator_app; its declaration is recorded by the producer."""
    from drivingbench.sim.runtime import Runtime

    restrictions = Restrictions(**declaration.get("restrictions", {}))
    return partial(Runtime, restrictions=restrictions, paused=declaration.get("paused", False))
