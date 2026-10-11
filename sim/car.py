"""Step 2c: the Sandbox native stack driving the fitted Corolla, tick for tick.

DrivingBench's own `Bridge` and `Controller` and openpilot's `LongControl`, `LatControlTorque`
and Toyota torque limiter run as `controlsd.state_control` runs them on the comma; only the car
below them is a model (sim/plant.py). `SimNative` has the producer's native interface
(`call`, `status`), like replay's `VirtualNative`.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace as NS

from drivingbench.device.native import Bridge
from drivingbench.shared.contracts import PROTOCOL
from drivingbench.sim.logs import car_params, use_openpilot
from drivingbench.sim.plant import Corolla, Params

use_openpilot()
from opendbc.car.lateral import apply_meas_steer_torque_limits  # noqa: E402
from opendbc.car.toyota.values import CarControllerParams  # noqa: E402
from opendbc.car.vehicle_model import VehicleModel  # noqa: E402
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque  # noqa: E402
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState  # noqa: E402

DT = 0.01
COROLLA = Path(__file__).with_name(
    "corolla.carparams"
)  # logged CarParams, VIN and firmware removed
HEARTBEAT_S = 0.05  # NativeClient's poll interval
LAT_DELAY_S = 0.32  # lateralDelay in every logged drive; modeld's LAT_SMOOTH_SECONDS is 0
LINEAR_TORQUE = NS(
    torque_from_lateral_accel=lambda: lambda accel, tp: accel / float(tp.latAccelFactor),
    lateral_accel_from_torque=lambda: lambda torque, tp: torque * float(tp.latAccelFactor),
)  # Toyota keeps CarInterfaceBase's linear torque model


class Inbox:
    """Bridge's server: requests queued by `call`, answered on the next native tick."""

    def __init__(self):
        self.pending = []

    def poll(self, handler):
        for message, replies in self.pending:
            replies.append(handler(message))
        self.pending.clear()


class Bus(dict):
    """The SubMaster subset the Bridge reads; every simulated input is valid."""

    def __init__(self, **messages):
        super().__init__(messages)

    def all_checks(self, _names=None):
        return True


class SimNative:
    def __init__(
        self,
        cp=None,
        *,
        params=None,
        steer_ratio=14.8,
        stiffness=1.0,
        time=0.0,
        plant_factory=Corolla,
        steering_target=None,
        **state,
    ):
        cp = cp or car_params(COROLLA.read_bytes())
        self.time = time
        self.steering_target = steering_target
        self.inbox = Inbox()
        self.bridge = Bridge(self.inbox, clock=lambda: self.time)
        self.vm = VehicleModel(cp)
        self.vm.update_params(stiffness, steer_ratio)
        self.lp = NS(
            angleOffsetDeg=0.0, roll=0.0, steerRatio=steer_ratio, stiffnessFactor=stiffness
        )
        self.long = LongControl(cp)
        self.lat = LatControlTorque(cp, LINEAR_TORQUE, DT)
        self.limits = CarControllerParams(cp)
        self.car = plant_factory(
            params or Params.load(), lambda sa, u: self.vm.calc_curvature(sa, u, 0.0), DT, **state
        )
        self.enabled, self.brake = True, False
        self.desired_curvature, self.torque = 0.0, 0  # torque in Toyota CAN units
        self.snapshot, self.next_heartbeat = {}, time
        self.tick()  # the native loop is always running before the producer calls it

    def call(self, body, **_):
        replies = []
        self.inbox.pending.append((body, replies))
        self.tick()
        return replies[0]

    def status(self):
        return dict(self.snapshot)

    def advance(self, until):
        while self.time + DT / 2 < until:
            self.tick()

    def tick(self):
        if self.time >= self.next_heartbeat:
            self.next_heartbeat += HEARTBEAT_S
            replies = []
            self.inbox.pending.insert(
                0, ({"protocol": PROTOCOL, "operation": "heartbeat"}, replies)
            )
        else:
            replies = None
        car, limits = self.car, self.limits
        cs = NS(
            vEgo=car.speed, aEgo=car.accel, steeringAngleDeg=car.steer_deg,
            steeringRateDeg=car.steer_rate, standstill=car.standstill, gearShifter="drive",
            brakePressed=self.brake, gasPressed=False, steeringPressed=False, canValid=True,
            cruiseState=NS(standstill=car.hold and car.standstill),
        )  # fmt: skip
        sm = Bus(
            carState=cs,
            vehicleParameters=self.lp,
            selfdriveState=NS(enabled=self.enabled, state="enabled"),
        )

        # controlsd.state_control, with the stock planner and model replaced by the Bridge.
        lat_active = self.enabled and not (cs.vEgo <= 0.3 or cs.standstill)
        long_active = self.enabled
        selection = self.bridge.tick(sm, lateral_active=lat_active, longitudinal_active=long_active)
        lat_active = lat_active and self.bridge.controller.sample.sensors_valid
        if self.steering_target is not None and lat_active and self.bridge.failure is None:
            angle = float(self.steering_target(self, selection.angle_deg))
            if not math.isfinite(angle):
                raise ValueError("Simulator steering target must be finite")
            bound = self.bridge.controller.settings.max_steering_angle_deg
            selection.angle_deg = min(bound, max(-bound, angle))
        if not lat_active:
            self.lat.reset()
        if not long_active:
            self.long.reset()
        if selection.resume and long_active and not cs.brakePressed:
            self.long.long_control_state = LongCtrlState.pid
        if not selection.should_stop and cs.standstill:
            self.long.pid.reset()
        accel = float(
            self.long.update(
                long_active, cs, selection.acceleration_mps2, selection.should_stop,
                (limits.ACCEL_MIN, limits.ACCEL_MAX),
            )
        )  # fmt: skip
        self.desired_curvature, limited = self.bridge.apply_curvature(
            self.vm, cs.vEgo, 0.0, self.desired_curvature
        )
        torque, _, _ = self.lat.update(
            lat_active, cs, self.vm, self.lp, False, self.desired_curvature, limited, LAT_DELAY_S
        )
        # Toyota CarController: rate and motor-error limits on the applied torque.
        eps = car.p.eps_torque_scale * self.torque / limits.STEER_MAX
        new = int(round(torque * limits.STEER_MAX))
        self.torque = (
            apply_meas_steer_torque_limits(new, self.torque, eps, limits) if lat_active else 0
        )
        car.step(accel, self.torque / limits.STEER_MAX, lat_active, long_active, self.brake)
        self.time += DT
        if replies:
            self.snapshot = dict(replies[0], received_at=self.time)

    @property
    def pose(self):
        return self.car.x, self.car.y, math.degrees(self.car.heading)
