"""Scoped fault preserves the fitted plant and native requests outside negative response."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from drivingbench.sim.course import Course
from drivingbench.sim.plant import Corolla, Params
from drivingbench.sim.restrictions import ProgressBrakingCorolla, Restrictions
from drivingbench.sim.runtime import DT, Runtime
from drivingbench.sim.scenarios import hazard_scene
from drivingbench.sim.validate import request


def test_functioning_progress_plant_has_exact_default_dynamics():
    def curvature(angle, speed):
        return angle / 100

    normal = Corolla(Params(), curvature)
    functioning = ProgressBrakingCorolla(Params(), curvature)
    functioning.latch_brake_fault(
        station_m=8, cross_track_m=0, map_pose=(8, 0, 0), sim_time_s=1, monotonic_at=2
    )
    for step in range(1800):
        acceleration = 1 if step < 600 else -2 if step < 1000 else 1
        arguments = (acceleration, 0.8, True, True, step >= 600)
        normal.step(*arguments)
        functioning.step(*arguments)
        assert (normal.speed, normal.accel, normal.steer_deg, normal.x, normal.y) == (
            functioning.speed,
            functioning.accel,
            functioning.steer_deg,
            functioning.x,
            functioning.y,
        )
        assert normal.accel_queue == functioning.accel_queue


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize("cap", [None, 0.2])
def test_native_stop_keeps_latched_treatment_and_uses_actual_preintegration_station(cap):
    scene = hazard_scene(
        {"frame": "pilot_ground", "calibration": "unused", "background": {"checkpoint": "unused"}},
        "child_entering_distance",
        assets={},
    )
    scene["scenario"]["type"] = "course_event"
    scene["design"] = {"path": [[0, 0], [100, 0]]}
    clock = Clock()
    runtime = Runtime(
        Course(scene),
        clock=clock,
        frame_clock=clock,
        scenario=scene,
        restrictions=Restrictions(brake_fault_station_m=8, brake_deceleration_cap_mps2=cap),
    )
    try:
        car = runtime.native.car
        car.x, car.speed = 7.999, 1.0
        clock.now += DT
        runtime.advance()
        assert car.x > 8
        assert car.brake_fault is None  # The previous tick began below the plane.
        preintegration = car.x
        clock.now += DT
        runtime.advance()
        latch = dict(car.brake_fault)
        assert latch["station_before_integration_m"] == pytest.approx(preintegration)
        assert latch["first_affected_tick_sim_time_s"] == pytest.approx(0.02)
        car.x = 7  # Reversal cannot unlatch or reattribute the fault.
        with ThreadPoolExecutor() as pool:
            body = request(runtime.native.bridge.controller, {"tool": "stop_now", "arguments": {}})
            future = pool.submit(runtime.call, body, timeout=1)
            with runtime._condition:
                assert runtime._condition.wait_for(lambda: runtime._pending, timeout=1)
            clock.now += DT
            runtime.advance()
            assert future.result()["status"] == "accepted"
        samples = []
        for _ in range(250):
            clock.now += DT
            runtime.advance()
            samples.append(car.speed)
            assert car.brake_fault == latch
            response = runtime.pose_snapshot()["brake_response"]["sample"]
            if response:
                assert response["active"] is (cap is not None)
                if cap is not None:
                    assert response["realized_accel_mps2"] >= -cap
        assert runtime.status()["command"] is None
        if cap is None:
            assert car.speed == 0
            assert runtime.pose_snapshot()["brake_response"]["sample"] is None
        else:
            assert car.speed > 0.4  # An accepted stop cannot restore functioning brakes.
            assert samples[-1] < samples[0]
    finally:
        runtime.close()
