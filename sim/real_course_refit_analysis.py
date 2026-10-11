"""Offline real-telemetry plant candidates; excluded from live-study source membership."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from drivingbench.sim.car import DT, SimNative
from drivingbench.sim.plant import Corolla, Params
from drivingbench.sim.validate import LONG_STATES


@dataclass(frozen=True)
class RefitConfig:
    params: Params = field(default_factory=Params.load)
    yaw_coefficients: tuple[float, float, float] = (1.0, 0.0, 0.0)
    launch_boost_mps2: float = 0.0
    launch_boost_tau_s: float = 0.7

    def __post_init__(self):
        if (
            len(self.yaw_coefficients) != 3
            or not np.isfinite(
                (*self.yaw_coefficients, self.launch_boost_mps2, self.launch_boost_tau_s)
            ).all()
            or self.launch_boost_mps2 < 0
            or self.launch_boost_tau_s <= 0
        ):
            raise ValueError(
                "Candidate coefficients must be finite with a nonnegative boost and positive decay"
            )

    def yaw_scale(self, wheel_deg):
        intercept, absolute, left = self.yaw_coefficients
        return np.clip(
            intercept + absolute * np.abs(wheel_deg) / 180 + left * np.maximum(wheel_deg, 0) / 180,
            0.65,
            1.15,
        )

    def launch_boost(self, age_s):
        return (
            self.launch_boost_mps2 * math.exp(-age_s / self.launch_boost_tau_s)
            if age_s < 3
            else 0.0
        )


class CandidateCorolla(Corolla):
    """Existing plant with bounded wheel-based yaw and a decaying launch residual."""

    def __init__(self, p, curvature, dt=DT, *, config, **state):
        def corrected(angle, speed):
            return float(config.yaw_scale(math.degrees(angle))) * curvature(angle, speed)

        super().__init__(p, corrected, dt, **state)
        self.config, self.launch_age_s = config, math.inf

    def step(self, accel_cmd, torque, lat_active, long_active, brake=False):
        was_stopped = self.standstill
        if not was_stopped:
            boost = self.config.launch_boost(self.launch_age_s) / self.p.accel_gain
            if self.accel_queue:
                self.accel_queue[0] += boost
            else:
                accel_cmd = (accel_cmd if long_active else 0.0) + boost
                long_active = True
        super().step(accel_cmd, torque, lat_active, long_active, brake)
        if not was_stopped:
            self.launch_age_s += self.dt
        elif not self.standstill:
            self.launch_age_s = 0.0


def make_plant(config):
    """Return the existing SimNative plant_factory hook, with no controller changes."""
    return partial(CandidateCorolla, config=config)


def measured_launch_age(speed, dt=DT):
    """Seconds since each measured standstill departure; unknown initially moving age is infinite."""
    ages = np.full(len(speed), math.inf)
    age = math.inf
    for index, value in enumerate(speed):
        if value <= 0.05:
            age = math.inf
        elif index and speed[index - 1] <= 0.05:
            age = 0.0
        elif math.isfinite(age):
            age += dt
        ages[index] = age
    return ages


class CandidateSim:
    """Logged-state native initialization, including logged actuator and launch history."""

    def __init__(self, cp, c, k, *, config, plant_factory=None):
        self.native = SimNative(
            cp, params=config.params, plant_factory=plant_factory or make_plant(config),
            steer_ratio=float(np.nanmedian(c["steer_ratio"])),
            stiffness=float(np.nanmedian(c["stiffness"])), time=float(c["t"][k]) - DT,
            speed=float(c["v"][k]), steer_deg=float(c["angle"][k]),
            hold=bool(c["cruise_standstill"][k]),
        )  # fmt: skip
        car = self.native.car
        if k < len(car.accel_queue):
            raise ValueError("Candidate needs complete logged acceleration history")
        car.accel = float(c["a"][k])
        history = np.nan_to_num(c["accel_cmd"][k - len(car.accel_queue) : k])
        car.accel_queue.clear()
        car.accel_queue.extend(history.tolist())
        if config.launch_boost_mps2:
            car.launch_age_s = float(measured_launch_age(c["v"][max(0, k - 400) : k + 1])[-1])
        self.native.desired_curvature = float(c["curvature_cmd"][k])
        self.native.torque = int(round(float(c["torque_out"][k]) * self.native.limits.STEER_MAX))
        self.native.long.long_control_state = LONG_STATES[int(c["long_state"][k])]
        self.controller = self.native.bridge.controller

    def call(self, body):
        return self.native.call(body)

    def advance(self, until):
        self.native.advance(until)

    def state(self):
        return self.native.car.speed, self.native.car.steer_deg
