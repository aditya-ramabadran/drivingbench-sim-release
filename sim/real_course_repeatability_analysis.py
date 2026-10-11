"""Offline empirical response draws; not a validated physical-repeatability model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from drivingbench.sim.real_course_refit_analysis import CandidateCorolla


class ResponseDraw:
    """Keep run-level steering and conditional launch pairs correlated."""

    def __init__(self, config, seed, *, groups, launches, steering_center, launch_center):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.group = groups[int(self.rng.integers(len(groups)))]
        self.launch_log = []
        self.launches = launches
        self.launch_center = launch_center
        gain = self.group["steer_gain_multiplier"] / steering_center["gain_multiplier_median"]
        hold = self.group["steer_hold_multiplier"] / steering_center["hold_multiplier_median"]
        self.params = replace(
            config.params,
            steer_gain=config.params.steer_gain * gain,
            return_gain=config.params.return_gain * gain,
            steer_gain_per_mps=config.params.steer_gain_per_mps * gain,
            steer_hold_deg=config.params.steer_hold_deg * hold,
        )

    def launch(self, held):
        pool = [x for x in self.launches if x["held"] == held and x["group"] == self.group["id"]]
        scope = "response_run"
        if not pool:
            sessions = set(self.group["session_ids"])
            pool = [
                x
                for x in self.launches
                if x["held"] == held and sessions.intersection(x["session_ids"])
            ]
            scope = "same_session"
        if not pool:
            pool = [
                x for x in self.launches if x["held"] == held and x["route"] == self.group["route"]
            ]
            scope = "same_route"
        if not pool:
            pool = [x for x in self.launches if x["held"] == held]
            scope = "all_training_routes"
        if not pool:
            raise ValueError("No conditional launch stratum available")
        sample = pool[int(self.rng.integers(len(pool)))]
        center = self.launch_center[str(held)]
        raw = (self.config.params.hold_launch_s if held else self.config.params.launch_s) + (
            sample["delay_s"] - center["delay_median_s"]
        )
        delay = max(0.1, raw)
        accel = self.config.params.launch_accel_mps2 * (
            sample["lurch_mps2"] / center["lurch_median_mps2"]
        )
        self.launch_log.append(
            {
                "sample_id": sample["id"],
                "pool_scope": scope,
                "held": held,
                "delay_s": delay,
                "launch_accel_mps2": accel,
                "delay_floor_applied": delay != raw,
            }
        )
        return replace(
            self.params, **{"hold_launch_s" if held else "launch_s": delay}, launch_accel_mps2=accel
        )


class ResponseCorolla(CandidateCorolla):
    """Sample once at each standstill request, retaining the chosen steering response."""

    def __init__(self, p, curvature, dt=0.01, *, draw, **state):
        super().__init__(draw.params, curvature, dt, config=draw.config, **state)
        self.draw = draw
        self.draw_requested = False

    def step(self, accel_cmd, torque, lat_active, long_active, brake=False):
        requesting = self.standstill and long_active and accel_cmd > 0
        if requesting and not self.draw_requested:
            self.p = self.draw.launch(bool(self.hold))
            self.draw_requested = True
        if not requesting:
            self.draw_requested = False
        return super().step(accel_cmd, torque, lat_active, long_active, brake)
