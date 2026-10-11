import numpy as np
import pytest
from drivingbench.shared.contracts import Settings
from drivingbench.sim.car import SimNative
from drivingbench.sim.plant import Corolla, Params, steer_step
from drivingbench.sim.validate import request, windows

P = Params.load()


def motion(sim, direction="left", percent=50.0, speed=1.0, duration=6.0, settings=Settings()):
    args = dict(direction=direction, steering_percent=percent, speed_mps=speed, duration_s=duration)
    call = {"tool": "set_motion", "arguments": args}
    return sim.call(request(sim.bridge.controller, call, settings.model_dump()))


def stop(sim):
    return sim.call(request(sim.bridge.controller, {"tool": "stop_now", "arguments": {}}))


@pytest.mark.parametrize("hold", [False, True])
def test_launch_waits_for_the_toyota_then_lurches(hold):
    car = Corolla(P, lambda sa, u: 0.0, hold=hold)
    ticks = 0
    while car.speed == 0.0:
        car.step(0.5, 0.0, False, True)
        ticks += 1
    wait = P.hold_launch_s if hold else P.launch_s
    assert abs(ticks * car.dt - wait) < car.dt / 2 and car.accel == P.launch_accel_mps2


def test_operator_brake_at_a_stop_engages_the_hold_and_launch_clears_it():
    car = Corolla(P, lambda sa, u: 0.0)
    car.step(0.0, 0.0, False, True)
    assert not car.hold
    car.step(0.0, 0.0, False, True, brake=True)
    assert car.hold
    while car.speed == 0.0:
        car.step(0.5, 0.0, False, True)
    assert not car.hold


def test_launch_timer_restarts_when_the_request_drops():
    car = Corolla(P, lambda sa, u: 0.0)
    for _ in range(round(P.launch_s / car.dt) - 5):
        car.step(0.5, 0.0, False, True)
    car.step(0.0, 0.0, False, True)
    for _ in range(10):
        car.step(0.5, 0.0, False, True)
    assert car.speed == 0.0


def test_steering_friction_return_and_inactive_hold():
    inside = P.steer_friction * 0.9
    assert steer_step(P, 0.0, inside, 1.0, 0.01) == 0.0
    out = steer_step(P, 50.0, 1.0, 1.0, 0.01) - 50.0
    back = 50.0 - steer_step(P, 50.0, 2 * 50.0 / P.steer_hold_deg - 1.0, 1.0, 0.01)
    assert out > 0 and back > out
    car = Corolla(P, lambda sa, u: 0.0, speed=1.0, steer_deg=120.0)
    car.step(0.0, -1.0, False, True)
    assert car.steer_deg == 120.0


def test_sim_executes_steers_and_expires_through_the_real_stack():
    sim = SimNative()
    assert sim.status()["state"] == "held"
    assert motion(sim)["status"] == "accepted"
    sim.advance(1.0)
    assert sim.car.speed == 0.0 and sim.status()["reason"] == "waiting_for_res"
    sim.advance(5.0)
    assert 0.5 < sim.car.speed < 1.5 and sim.car.steer_deg > 20 and sim.car.heading > 0
    sim.advance(12.0)
    status = sim.status()
    assert status["state"] == "held" and sim.car.speed == 0.0


def test_sim_right_turn_mirrors_left():
    left, right = SimNative(), SimNative()
    motion(left, "left", 60)
    motion(right, "right", 60)
    left.advance(5.0)
    right.advance(5.0)
    assert left.car.steer_deg == pytest.approx(-right.car.steer_deg)
    assert left.car.heading == pytest.approx(-right.car.heading)


def test_sim_stop_rejections_and_operator_brake():
    sim = SimNative()
    assert motion(sim, speed=3.5)["reason"] == "speed_limit_exceeded"
    motion(sim)
    sim.advance(4.0)
    assert stop(sim)["status"] == "accepted"
    sim.advance(8.0)
    assert sim.car.speed == 0.0
    sim.brake = True
    sim.advance(8.1)
    assert motion(sim)["reason"] == "operator_brake"


def test_validation_windows_end_at_interventions_and_sessions():
    n = 6000
    c = {name: np.zeros(n) for name in ("brake", "gas", "steer_pressed")}
    c |= {"t": np.arange(n) * 0.01, "enabled": np.ones(n), "gear": np.full(n, 2.0)}
    c["brake"][3800:4100] = 1
    cmd = dict(tool="set_motion", status="accepted", arguments={})
    commands = [
        cmd | {"t": 5.0, "session": "a"},
        cmd | {"t": 10.0, "session": "a"},
        cmd | {"t": 20.0, "session": "b"},
        cmd | {"t": 39.0, "session": "b"},  # starts during the brake: no window
        cmd | {"t": 45.0, "session": "b", "status": "rejected"},
    ]
    got = [(k, end, len(calls)) for k, end, calls in windows(c, commands)]
    assert got == [(500, 2000, 2), (1000, 2000, 1), (2000, 3800, 1)]
