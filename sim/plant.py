"""Step 2b: the Corolla below openpilot's controllers, fitted from our logs (docs/sim-dynamics.md).

Inputs are what openpilot sends the car: `actuators.accel` and the Toyota-limited steering
torque (`actuatorsOutput.torque`, −1…1). The step functions take arrays so the fit and the
simulator share them.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PARAMS = Path(__file__).with_name("params.json")


@dataclass(frozen=True)
class Params:
    accel_delay_s: float = 0.8
    accel_tau_s: float = 0.7
    accel_gain: float = 1.0  # steady acceleration per unit actuators.accel
    accel_bias_mps2: float = 0.0
    launch_s: float = 1.4  # positive accel request at standstill until the car moves
    launch_accel_mps2: float = 1.4  # the lurch as the brakes release
    hold_launch_s: float = (
        3.0  # the same after a stop held by the operator's brake or a disengagement
    )
    steer_gain: float = 24.0  # wheel deg/s per unit of torque beyond friction, turning out
    return_gain: float = 62.0  # the same, turning back toward center
    steer_gain_per_mps: float = 54.0
    steer_hold_deg: float = 244.0  # angle where full torque balances the tire's resistance
    steer_friction: float = 0.33  # torque that moves nothing
    eps_torque_scale: float = 2116.0  # steeringTorqueEps per unit applied torque

    @classmethod
    def load(cls, path: Path = PARAMS) -> Params:
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path = PARAMS):
        values = {name: round(value, 4) for name, value in asdict(self).items()}
        path.write_text(json.dumps(values, indent=2) + "\n")


def accel_step(p: Params, accel, command, dt):
    steady = p.accel_gain * command + p.accel_bias_mps2
    return accel + (steady - accel) * min(1.0, dt / p.accel_tau_s)


def steer_step(p: Params, angle, torque, speed, dt):
    excess = torque - angle / p.steer_hold_deg
    push = np.sign(excess) * np.maximum(np.abs(excess) - p.steer_friction, 0.0)
    gain = np.where(push * angle < 0, p.return_gain, p.steer_gain) + p.steer_gain_per_mps * speed
    return angle + gain * push * dt


class Corolla:
    """Speed, steering-wheel angle and planar pose, stepped at controlsd's 100 Hz."""

    def __init__(self, p: Params, curvature, dt=0.01, speed=0.0, steer_deg=0.0, hold=False):
        self.p, self.curvature, self.dt = p, curvature, dt
        self.speed, self.accel, self.steer_deg, self.steer_rate = speed, 0.0, steer_deg, 0.0
        self.x = self.y = self.heading = 0.0
        self.hold, self.waiting_s = hold, 0.0
        self.accel_queue = deque([0.0] * round(p.accel_delay_s / dt))

    @property
    def standstill(self):
        return self.speed == 0.0

    def acceleration_response(self, filtered_accel):
        """Identity unless an explicitly scoped simulator plant overrides it."""
        return filtered_accel

    def step(self, accel_cmd, torque, lat_active, long_active, brake=False):
        p, dt = self.p, self.dt
        self.accel_queue.append(accel_cmd if long_active else 0.0)
        delayed_accel = self.accel_queue.popleft()
        if self.standstill:
            self.accel = 0.0
            self.waiting_s = self.waiting_s + dt if long_active and accel_cmd > 0 else 0.0
            if self.waiting_s >= (p.hold_launch_s if self.hold else p.launch_s) - dt / 2:
                self.speed, self.accel, self.hold = 1e-3, p.launch_accel_mps2, False
            else:
                self.hold |= brake or not long_active
        else:
            self.accel = self.acceleration_response(
                float(accel_step(p, self.accel, delayed_accel, dt))
            )
            self.speed = max(0.0, self.speed + self.accel * dt)
            self.waiting_s = 0.0
        # Without openpilot's torque the wheel stays put: lateral control is off below 0.3 m/s.
        previous = self.steer_deg
        if lat_active:
            self.steer_deg = float(steer_step(p, self.steer_deg, torque, self.speed, dt))
        self.steer_rate = (self.steer_deg - previous) / dt
        yaw_rate = self.speed * self.curvature(math.radians(self.steer_deg), self.speed)
        self.heading += yaw_rate * dt
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
