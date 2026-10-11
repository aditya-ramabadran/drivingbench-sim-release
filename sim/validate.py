"""Step 2d: replay recorded tool calls through the simulator and score it against the car.

    python -m drivingbench.sim.validate [--routes ...] [--out runs/sim/validate]

Every accepted `set_motion` opens a window. Each model starts from the logged state and gets
the same tool calls at their logged times; the window ends after 30 s, at the next session, or
at the first operator input, disengagement or gear change in the log. Truth is the logged speed
and wheel angle, gyro yaw, and the logged speed dead-reckoned along it. The baseline is replay's
`VirtualNative` (fixed launch delay, acceleration and steering rate).
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
from drivingbench.replay.dynamics import Dynamics
from drivingbench.replay.producer import VirtualNative
from drivingbench.shared.contracts import PROTOCOL, Settings
from drivingbench.sim.car import DT, LongCtrlState, SimNative
from drivingbench.sim.fit import DRIVE, HELD_OUT, route
from drivingbench.sim.logs import REPO, car_params

HORIZON_S, CHECKPOINTS = 30.0, (5.0, 10.0, 30.0)
SETTINGS = Settings(speed_limit_mps=3.5).model_dump()  # every logged session's settings
LONG_STATES = (LongCtrlState.off, LongCtrlState.pid, LongCtrlState.stopping)


def request(controller, call, settings=SETTINGS):
    """The envelope device/service.py sends for a tool call."""
    motion = call["tool"] == "set_motion"
    return {
        "protocol": PROTOCOL, "request_id": uuid.uuid4().hex,
        "operation": "motion" if motion else "stop", "boot_id": controller.boot_id,
        "command_epoch": controller.command_epoch,
        "motion": dict(call["arguments"], reason="") if motion else None,
        "settings": settings if motion else None,
    }  # fmt: skip


class Sim:
    """SimNative started from the logged state at row k, actuator history included."""

    def __init__(self, cp, c, k, params=None):
        self.native = SimNative(
            cp, params=params, steer_ratio=float(np.nanmedian(c["steer_ratio"])),
            stiffness=float(np.nanmedian(c["stiffness"])), time=float(c["t"][k]) - DT,
            speed=float(c["v"][k]), steer_deg=float(c["angle"][k]),
            hold=bool(c["cruise_standstill"][k]),
        )  # fmt: skip
        car = self.native.car
        car.accel = float(c["a"][k])
        history = np.nan_to_num(c["accel_cmd"][k - len(car.accel_queue) : k])
        car.accel_queue.clear()
        car.accel_queue.extend(history.tolist())
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


class Replay:
    """replay.producer.VirtualNative on a clock this loop owns."""

    def __init__(self, cp, c, k, params=None):
        self.now = float(c["t"][k])
        self.native = VirtualNative(1e9, Dynamics(), hard_steer_percent=101, clock=lambda: self.now)
        self.native.speed_mps, self.native.steering_deg = float(c["v"][k]), float(c["angle"][k])
        self.controller = self.native.controller

    def call(self, body):
        return self.native.call(body)

    def advance(self, until):
        self.now = until
        self.native.advance(until)

    def state(self):
        return self.native.speed_mps, self.native.steering_deg


MODELS = {"sim": Sim, "replay": Replay}


def windows(c, commands):
    """(start row, end row, the window's tool calls) per accepted set_motion."""
    t = c["t"]
    stop = (
        (c["brake"] == 1) | (c["gas"] == 1) | (c["steer_pressed"] == 1)
        | (c["enabled"] != 1) | (c["gear"] != DRIVE)
    )  # fmt: skip
    for i, cmd in enumerate(commands):
        k = int(np.searchsorted(t, cmd["t"]))
        if cmd["tool"] != "set_motion" or cmd["status"] != "accepted" or k < 200 or stop[k]:
            continue
        end = min(int(np.searchsorted(t, cmd["t"] + HORIZON_S)) + 1, len(t) - 1)
        if hits := np.flatnonzero(stop[k:end]).tolist():
            end = k + hits[0]
        later = []
        for other in commands[i + 1 :]:
            if other["t"] >= t[end]:
                break
            if other["session"] != cmd["session"]:
                end = int(np.searchsorted(t, other["t"]))
                break
            later.append(other)
        yield k, end, [cmd, *later]


def pose(speed, heading, dt):
    x = np.cumsum(speed * np.cos(heading) * dt)
    y = np.cumsum(speed * np.sin(heading) * dt)
    return x, y


def simulate(model, c, k, end, calls, vm):
    """Model speed, wheel angle, heading and position at each log row of the window."""
    t, pending = c["t"], list(calls)
    speed, angle = np.zeros(end - k), np.zeros(end - k)
    for n, j in enumerate(range(k, end)):
        while pending and pending[0]["t"] <= t[j]:
            model.call(request(model.controller, pending.pop(0)))
        model.advance(t[j])
        speed[n], angle[n] = model.state()
    dt = np.diff(t[k:end], prepend=t[k])
    curvature = np.array([vm.calc_curvature(math.radians(a), s, 0.0) for s, a in zip(speed, angle)])
    heading = np.cumsum(speed * curvature * dt)
    return speed, angle, heading, *pose(speed, heading, dt)


def observed(c, k, end, bias):
    dt = np.diff(c["t"][k:end], prepend=c["t"][k])
    heading = np.cumsum(np.nan_to_num(c["gyro_x"][k:end] - bias) * dt)
    speed = c["v"][k:end]
    return speed, c["angle"][k:end], heading, *pose(speed, heading, dt)


def score(model, real, t):
    """Errors at each checkpoint (by log time, t from the window start) and over the window."""
    speed, angle, heading, x, y = model
    rs, ra, rh, rx, ry = real
    errors = {
        "speed_mps": np.abs(speed - rs),
        "angle_deg": np.abs(angle - ra),
        "heading_deg": np.degrees(np.abs(heading - rh)),
        "position_m": np.hypot(x - rx, y - ry),
    }
    row = {"length_s": float(t[-1])}
    for h in CHECKPOINTS:
        n = int(np.searchsorted(t, h - DT))
        if n < len(t):
            row[f"{h:g}s"] = {name: float(e[n]) for name, e in errors.items()}
    row["rms"] = {name: float(np.sqrt(np.mean(e**2))) for name, e in errors.items()}
    return row


@cache
def loaded(path: Path):
    """Columns, CarParams, VehicleModel, gyro rest bias and commands for one route."""
    from opendbc.car.vehicle_model import VehicleModel

    z, c = np.load(path), route(path)
    cp = car_params(z["car_params"].tobytes())
    vm = VehicleModel(cp)
    vm.update_params(float(np.nanmedian(c["stiffness"])), float(np.nanmedian(c["steer_ratio"])))
    rest = (c["v"] < 0.01) & np.isfinite(c["gyro_x"])
    return c, cp, vm, float(np.median(c["gyro_x"][rest])), json.loads(str(z["meta"]))["commands"]


def evaluate(job):
    """Rows for every `parts`-th window of one route, starting at `part`."""
    path, params, models, part, parts = job
    c, cp, vm, bias, commands = loaded(path)
    rows = []
    for n, (k, end, calls) in enumerate(windows(c, commands)):
        if n % parts != part:
            continue
        real, t = observed(c, k, end, bias), c["t"][k:end] - c["t"][k]
        row = {"route": path.stem, "t": float(c["t"][k]), "session": calls[0]["session"]}
        row["calls"] = len(calls)
        for model in models:
            made = MODELS[model](cp, c, k, params)
            row[model] = score(simulate(made, c, k, end, calls, vm), real, t)
        rows.append(row)
    return rows


def replay(paths, pool, params=None, models=tuple(MODELS)):
    parts = pool._max_workers
    jobs = [(path, params, models, part, parts) for path in paths for part in range(parts)]
    return sorted((r for rows in pool.map(evaluate, jobs) for r in rows), key=lambda r: r["t"])


def summarize(rows):
    out = {}
    for model in MODELS:
        out[model] = {}
        for key in [f"{h:g}s" for h in CHECKPOINTS] + ["rms"]:
            values = [r[model][key] for r in rows if key in r[model]]
            if values:
                out[model][key] = {
                    name: round(float(np.median([v[name] for v in values])), 3)
                    for name in values[0]
                } | {"windows": len(values)}
    return out


def figure(path: Path, starts: list[float], out: Path):
    """Car vs models for the windows starting at `starts` (speed, wheel angle, path)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, cp, vm, bias, commands = loaded(path)
    chosen = [w for w in windows(c, commands) if float(c["t"][w[0]]) in starts]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(13, 3.6 * len(chosen)), squeeze=False)
    styles = {"car": ("k", "-"), "sim": ("tab:blue", "-"), "replay": ("tab:orange", "--")}
    for row, (k, end, calls) in zip(axes, chosen):
        series = {"car": observed(c, k, end, bias)}
        for model in MODELS:
            series[model] = simulate(MODELS[model](cp, c, k), c, k, end, calls, vm)
        t = c["t"][k:end] - c["t"][k]
        for name, (speed, angle, _, x, y) in series.items():
            color, line = styles[name]
            row[0].plot(t, speed, color=color, ls=line, label=name)
            row[1].plot(t, angle, color=color, ls=line)
            row[2].plot(x, y, color=color, ls=line)
        for call in calls:
            for ax in row[:2]:
                ax.axvline(call["t"] - c["t"][k], color="0.8", lw=0.8, zorder=0)
        row[0].set(xlabel="s", ylabel="speed m/s")
        row[1].set(xlabel="s", ylabel="wheel angle deg")
        row[2].set(xlabel="m", ylabel="m", aspect="equal")
    axes[0][0].legend(loc="lower right")
    fig.suptitle("Held-out replay: same tool calls (grey) from the same start state")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--routes", nargs="*", default=list(HELD_OUT))
    a.add_argument("--series", type=Path, default=REPO / "runs/sim/series")
    a.add_argument("--out", type=Path, default=REPO / "runs/sim/validate")
    a.add_argument("--plot", type=int, default=2, help="figure of the N longest windows")
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        rows = replay([args.series / f"{name}.npz" for name in args.routes], pool)
    summary = summarize(rows)
    (args.out / "windows.json").write_text(json.dumps(rows, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.plot:
        first = args.series / f"{args.routes[0]}.npz"
        mine = [r for r in rows if r["route"] == first.stem and r["calls"] > 1]
        longest = sorted(mine, key=lambda r: -r["sim"]["length_s"])[: args.plot]
        figure(first, [r["t"] for r in longest], args.out / "windows.png")
    print(f"{len(rows)} windows from {', '.join(args.routes)} (median absolute error)")
    for key in [f"{h:g}s" for h in CHECKPOINTS] + ["rms"]:
        for model in MODELS:
            if s := summary[model].get(key):
                fields = "  ".join(f"{n} {v}" for n, v in s.items())
                print(f"{key:>4} {model:6s} {fields}")


if __name__ == "__main__":
    main()
