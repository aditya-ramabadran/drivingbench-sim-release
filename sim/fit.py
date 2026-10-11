"""Step 2b: fit the Corolla plant (sim/params.json) on the training routes.

    python -m drivingbench.sim.fit [--series runs/sim/series]

Speed and steering are fitted separately by multi-step simulation error over 3 s windows:
speed from `actuators.accel`, steering angle from the applied torque. Launch timing is measured
directly from standstill departures. The baselines are an ideal car: speed follows
`actuators.accel` at once, and the wheel sits at the angle `actuators.curvature` asks for.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from drivingbench.sim.logs import REPO, car_params, use_openpilot
from drivingbench.sim.plant import Params, accel_step, steer_step
from drivingbench.sim.series import load

TRAIN = ("0000003c--0388a7fd07", "0000003d--a44338c090", "00000041--fdf9cede4b")
HELD_OUT = ("00000042--544757d3f5",)
DRIVE, REVERSE = 2, 4  # car.capnp GearShifter
DT, HORIZON, STRIDE, HISTORY = 0.01, 300, 50, 100


def route(path: Path) -> dict[str, np.ndarray]:
    """Columns plus `desired_deg`, the wheel angle VehicleModel gives for actuators.curvature."""
    use_openpilot()
    from opendbc.car.vehicle_model import VehicleModel

    z = np.load(path)
    _, _, c = load(path)
    vm = VehicleModel(car_params(z["car_params"].tobytes()))
    vm.update_params(np.nanmedian(c["stiffness"]), np.nanmedian(c["steer_ratio"]))
    roll = np.nan_to_num(c["roll"])
    c["desired_deg"] = np.degrees(vm.get_steer_from_curvature(-c["curvature_cmd"], c["v"], roll))
    c["angle"] = c["steer_deg"] - c["angle_offset"]
    return c


def starts(mask: np.ndarray) -> np.ndarray:
    """Window starts whose history and horizon lie wholly inside `mask`."""
    run = np.convolve(mask.astype(int), np.ones(HISTORY + HORIZON, int), "valid")
    return np.where(run == HISTORY + HORIZON)[0][::STRIDE] + HISTORY


def longitudinal(c):
    ok = (c["long_active"] == 1) & (c["brake"] == 0) & (c["gas"] == 0) & (c["gear"] == DRIVE)
    return c, starts(ok & (c["v"] > 0.05))


def lateral(c):
    ok = (c["lat_active"] == 1) & (c["steer_pressed"] == 0) & (c["gear"] == DRIVE)
    return c, starts(ok & np.isfinite(c["desired_deg"]) & np.isfinite(c["angle"]))


def speed_error(p: Params, sets) -> float:
    errors = []
    delay = round(p.accel_delay_s / DT)
    for c, s in sets:
        idx = s[:, None] + np.arange(HORIZON)[None]
        command, truth = c["accel_cmd"][idx - delay], c["v"][idx]
        v, a = truth[:, 0].copy(), c["a"][s].copy()
        for k in range(1, HORIZON):
            a = accel_step(p, a, command[:, k - 1], DT)
            v = np.maximum(0.0, v + a * DT)
            errors.append(np.square(v - truth[:, k]))
    return float(np.sqrt(np.mean(np.concatenate(errors))))


def steer_error(p: Params, sets) -> float:
    errors = []
    for c, s in sets:
        idx = s[:, None] + np.arange(HORIZON)[None]
        torque, speed, truth = c["torque_out"][idx], c["v"][idx], c["angle"][idx]
        angle = truth[:, 0].copy()
        for k in range(1, HORIZON):
            angle = steer_step(p, angle, torque[:, k - 1], speed[:, k - 1], DT)
            errors.append(np.square(angle - truth[:, k]))
    return float(np.sqrt(np.mean(np.concatenate(errors))))


def ideal_steer_error(sets) -> float:
    errors = [
        np.square(c["desired_deg"][s[:, None] + k] - c["angle"][s[:, None] + k]).ravel()
        for c, s in sets
        for k in range(1, HORIZON)
    ]
    return float(np.sqrt(np.mean(np.concatenate(errors))))


def search(error, p: Params, ranges: dict[str, tuple[float, float]], rounds=4, points=9):
    """Coordinate descent over shrinking grids; returns (params, error)."""
    best = error(p)
    for r in range(rounds):
        for name, (lo, hi) in ranges.items():
            span = (hi - lo) / 2 ** (r + 1)
            here = getattr(p, name)
            for value in np.linspace(max(lo, here - span), min(hi, here + span), points):
                trial = replace(p, **{name: float(value)})
                if (e := error(trial)) < best:
                    p, best = trial, e
    return p, best


def launches(c) -> tuple[list[float], list[float], list[float]]:
    """Launch delay (positive accel request → moving) without and with the standstill hold,
    and the mean acceleration over the first 0.2 s."""
    free, held, lurch = [], [], []
    moving = c["v"] > 0.05
    for i in np.where(moving[1:] & ~moving[:-1])[0] + 1:
        j = i
        while j > 0 and c["accel_cmd"][j - 1] > 0 and not moving[j - 1]:
            j -= 1
        if (
            j < i
            and c["long_active"][j:i].all()
            and c["gear"][i] == DRIVE
            and not c["brake"][j:i].any()
        ):
            (held if c["cruise_standstill"][j] else free).append((i - j) * DT)
            lurch.append((c["v"][i + 20] - c["v"][i]) / 0.2)
    return free, held, lurch


def closed_loop(p: Params, paths, pool) -> tuple[Params, float]:
    """Refine the longitudinal plant on what the simulator is for: whole-stack replay."""
    from drivingbench.sim.validate import replay

    def error(q):
        rows = replay(paths, pool, q, ("sim",))
        return float(np.mean([r["sim"]["rms"]["speed_mps"] for r in rows]))

    ranges = {
        "accel_delay_s": (0.0, 1.2), "accel_tau_s": (0.05, 1.5), "accel_gain": (0.5, 4.0),
        "accel_bias_mps2": (-0.3, 0.3), "launch_s": (0.8, 2.0), "launch_accel_mps2": (0.0, 3.0),
    }  # fmt: skip
    return search(error, p, ranges, rounds=3, points=7)


IDEAL = Params(accel_delay_s=0.0, accel_tau_s=DT, accel_bias_mps2=0.0)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--series", type=Path, default=REPO / "runs/sim/series")
    args = a.parse_args()
    train = [route(args.series / f"{r}.npz") for r in TRAIN]
    held = [route(args.series / f"{r}.npz") for r in HELD_OUT]
    long_train, long_held = map(longitudinal, train), map(longitudinal, held)
    long_train, long_held = list(long_train), list(long_held)
    lat_train, lat_held = [lateral(c) for c in train], [lateral(c) for c in held]

    p = Params()
    p, _ = search(
        lambda q: speed_error(q, long_train),
        p,
        {"accel_delay_s": (0.0, 1.0), "accel_tau_s": (0.05, 2.0), "accel_bias_mps2": (-0.3, 0.3)},
    )
    p, _ = search(
        lambda q: steer_error(q, lat_train),
        p,
        {
            "steer_gain": (0.0, 400.0),
            "return_gain": (0.0, 800.0),
            "steer_gain_per_mps": (-200.0, 800.0),
            "steer_hold_deg": (60.0, 400.0),
            "steer_friction": (0.0, 0.6),
        },
        rounds=6,
    )
    free, held_launch, lurch = [], [], []
    for c in train:
        f, h, u = launches(c)
        free, held_launch, lurch = free + f, held_launch + h, lurch + u
    active = np.concatenate([c["lat_active"] == 1 for c in train])
    torque = np.concatenate([c["torque_out"] for c in train])[active]
    eps = np.concatenate([c["eps_torque"] for c in train])[active]
    p = replace(
        p,
        eps_torque_scale=float(np.dot(torque, eps) / np.dot(torque, torque)),
        launch_s=float(np.median(free)),
        launch_accel_mps2=float(np.median(lurch)),
        hold_launch_s=float(np.median(held_launch)),
    )
    with ProcessPoolExecutor() as pool:
        paths = [args.series / f"{r}.npz" for r in TRAIN]
        p, error = closed_loop(p, paths, pool)
    print(f"closed-loop mean speed RMS on training windows: {error:.3f} m/s")
    p.save()
    print(p)
    print(f"launches: {len(free)} free, {len(held_launch)} held")
    for name, error, ideal, sets in (
        ("speed m/s", speed_error, speed_error(IDEAL, long_held), (long_train, long_held)),
        ("steer deg", steer_error, ideal_steer_error(lat_held), (lat_train, lat_held)),
    ):
        windows = [sum(len(s) for _, s in x) for x in sets]
        print(
            f"{name}: 3 s RMS train {error(p, sets[0]):.3f} / held-out {error(p, sets[1]):.3f}"
            f" (ideal car {ideal:.3f}; windows {windows[0]} / {windows[1]})"
        )


if __name__ == "__main__":
    main()
