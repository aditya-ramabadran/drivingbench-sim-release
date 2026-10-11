"""Step 2a: one continuous 100 Hz series per openpilot route, plus the Sandbox commands.

    python -m drivingbench.sim.series [--root runs/artifacts/routes] [--out runs/sim/series]

Rows are carState ticks; every other signal is the latest value at that tick. Commands come
from traces/segments/*/events.jsonl (tool calls, wall time) and are placed on the route's
monotonic clock through rlog `clocks` messages.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
from drivingbench.sim.logs import REPO, events

TYPES = {
    "carState", "carControl", "carOutput", "selfdriveState", "vehicleParameters", "gyroscope",
    "gpsLocationExternal", "clocks", "carParams", "lateralTorqueParameters", "lateralDelay",
}  # fmt: skip
COLUMNS = (
    "t", "v", "a", "gear", "steer_deg", "steer_rate", "standstill", "cruise_standstill", "brake", "gas",
    "steer_pressed", "enabled", "lat_active", "long_active", "long_state", "accel_cmd", "accel_out", "curvature_cmd",
    "torque_cmd", "torque_out", "eps_torque", "lat_accel_factor", "lat_accel_offset", "friction",
    "torque_use_params", "lat_delay", "angle_offset", "steer_ratio", "stiffness", "roll", "gyro_x",
    "gyro_y", "gyro_z", "lat", "lon", "gps_accuracy",
)  # fmt: skip


def route_segments(root: Path) -> dict[str, list[Path]]:
    """Route id → its segment rlogs in order (folders are named <route>--<n>)."""
    routes = defaultdict(list)
    for rlog in root.glob("*/rlog.zst"):
        route, _, n = rlog.parent.name.rpartition("--")
        routes[route].append((int(n), rlog))
    return {r: [p for _, p in sorted(s)] for r, s in routes.items()}


def extract(item) -> tuple[str, dict]:
    route, rlogs = item
    latest = dict.fromkeys(COLUMNS, np.nan)
    rows, offsets, car_params = [], [], None
    for rlog in rlogs:
        for mono, kind, msg in events(rlog, TYPES):
            if kind == "carState":
                latest.update(
                    t=mono / 1e9, v=msg.vEgo, a=msg.aEgo, gear=msg.gearShifter.raw, steer_deg=msg.steeringAngleDeg,
                    steer_rate=msg.steeringRateDeg, standstill=msg.standstill,
                    cruise_standstill=msg.cruiseState.standstill, brake=msg.brakePressed, gas=msg.gasPressed, steer_pressed=msg.steeringPressed,
                    eps_torque=msg.steeringTorqueEps,
                )  # fmt: skip
                rows.append([float(latest[c]) for c in COLUMNS])
            elif kind == "selfdriveState":
                latest["enabled"] = msg.enabled
            elif kind == "carControl":
                act = msg.actuators
                latest.update(
                    lat_active=msg.latActive, long_active=msg.longActive,
                    long_state=act.longControlState.raw, accel_cmd=act.accel,
                    curvature_cmd=act.curvature, torque_cmd=act.torque,
                )  # fmt: skip
            elif kind == "carOutput":
                latest.update(
                    accel_out=msg.actuatorsOutput.accel, torque_out=msg.actuatorsOutput.torque
                )
            elif kind == "vehicleParameters":
                latest.update(
                    angle_offset=msg.angleOffsetDeg, steer_ratio=msg.steerRatio,
                    stiffness=msg.stiffnessFactor, roll=msg.roll,
                )  # fmt: skip
            elif kind == "lateralTorqueParameters":
                latest.update(
                    lat_accel_factor=msg.latAccelFactorFiltered,
                    lat_accel_offset=msg.latAccelOffsetFiltered,
                    friction=msg.frictionCoefficientFiltered, torque_use_params=msg.useParams,
                )  # fmt: skip
            elif kind == "lateralDelay":
                latest["lat_delay"] = msg.lateralDelay
            elif kind == "gyroscope":
                x, y, z = msg.gyroUncalibrated.v
                latest.update(gyro_x=x, gyro_y=y, gyro_z=z)
            elif kind == "gpsLocationExternal":
                if msg.hasFix:
                    latest.update(
                        lat=msg.latitude, lon=msg.longitude, gps_accuracy=msg.horizontalAccuracy
                    )
            elif kind == "clocks":
                offsets.append(msg.wallTimeNanos / 1e9 - mono / 1e9)
            elif kind == "carParams" and car_params is None:
                car_params = msg.as_builder().to_bytes()
    data = np.array(rows, dtype=np.float64) if rows else np.zeros((0, len(COLUMNS)))
    meta = {
        "route": route,
        "segments": [p.parent.name for p in rlogs],
        "wall_minus_mono_s": median(offsets) if offsets else None,
    }
    return route, {"data": data, "meta": meta, "car_params": car_params}


def commands(traces: Path) -> list[dict]:
    """Every Sandbox tool call that changes motion, with its wall time (s since epoch)."""
    out = []
    for log in sorted(traces.glob("segments/*/events.jsonl")):
        for line in log.read_text().splitlines():
            e = json.loads(line)
            if e.get("kind") == "tool" and e.get("tool") in ("set_motion", "stop_now"):
                out.append(
                    {
                        "wall": datetime.fromisoformat(e["timestamp"]).timestamp(),
                        "tool": e["tool"],
                        "arguments": {k: v for k, v in e["arguments"].items() if k != "reason"},
                        "status": (e.get("outcome") or {}).get("status"),
                        "session": e.get("session"),
                        "trace_segment": e.get("segment"),
                    }
                )
    return sorted(out, key=lambda c: c["wall"])


def load(path: Path) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """(data, meta, columns by name) for one saved route."""
    z = np.load(path, allow_pickle=False)
    data, meta = z["data"], json.loads(str(z["meta"]))
    return data, meta, {c: data[:, i] for i, c in enumerate(COLUMNS)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=REPO / "runs/artifacts/routes")
    p.add_argument("--traces", type=Path, default=REPO / "traces")
    p.add_argument("--out", type=Path, default=REPO / "runs/sim/series")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cmds = commands(a.traces)
    with ProcessPoolExecutor(a.workers) as pool:
        for route, r in pool.map(extract, route_segments(a.root).items()):
            t = r["data"][:, 0]
            if len(t) and r["meta"]["wall_minus_mono_s"] is not None:
                lo, hi = (
                    t[0] + r["meta"]["wall_minus_mono_s"],
                    t[-1] + r["meta"]["wall_minus_mono_s"],
                )
                mine = [
                    c | {"t": c["wall"] - r["meta"]["wall_minus_mono_s"]}
                    for c in cmds
                    if lo <= c["wall"] <= hi
                ]
            else:
                mine = []
            r["meta"]["commands"] = mine
            np.savez_compressed(
                a.out / f"{route}.npz",
                data=r["data"],
                meta=json.dumps(r["meta"]),
                car_params=np.frombuffer(r["car_params"] or b"", dtype=np.uint8),
            )
            hours = len(t) / 100 / 3600
            print(
                f"{route}: {len(r['meta']['segments'])} segments, {hours:.2f} h, {len(mine)} commands"
            )


if __name__ == "__main__":
    main()
