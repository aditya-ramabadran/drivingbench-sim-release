"""Offline accepted-command replay from logged states; never imported by live studies.

    python -m drivingbench.sim.real_course_analysis MANIFEST --out NEW_DIRECTORY

Manifest attempts declare segment, series_path, start_t and exclusive intervention end_t.
Map registration and measured scene assessment are deliberately separate analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from drivingbench.sim.logs import REPO
from drivingbench.sim.validate import Sim, loaded, request

FIELDS = ("t", "x", "y", "heading", "speed", "wheel_deg", "accel", "native_time")
SUCCESS = {"accepted", "stopping", "stopped"}


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_commands(commands, segment, first_row_t, end_t):
    """Successful motion/stop calls within one intervention-censored engagement."""
    return sorted(
        (
            row
            for row in commands
            if row["trace_segment"] == segment
            and first_row_t - 0.011 <= row["t"] < end_t
            and (
                (row["tool"] == "set_motion" and row["status"] == "accepted")
                or (row["tool"] == "stop_now" and row["status"] in SUCCESS)
            )
        ),
        key=lambda row: row["t"],
    )


def _window(columns, cp, start, end, calls, model_factory):
    model = model_factory(cp, columns, start)
    model.native.car.x = model.native.car.y = model.native.car.heading = 0.0
    pending = iter(calls)
    command = next(pending, None)
    rows, receipts = [], []
    for index in range(start, end):
        while command is not None and command["t"] <= columns["t"][index]:
            before = model.native.time
            response = model.call(request(model.controller, command))
            receipts.append(
                {
                    "command": command,
                    "scheduled_t": command["t"],
                    "before_tick_t": before,
                    "receipt_t": model.native.time,
                    "response": response,
                }
            )
            command = next(pending, None)
        model.advance(columns["t"][index])
        car = model.native.car
        rows.append(
            (
                columns["t"][index], car.x, car.y, car.heading, car.speed,
                car.steer_deg, car.accel, model.native.time,
            )
        )  # fmt: skip
    return np.asarray(rows), receipts


def replay_attempt(columns, cp, commands, attempt, *, model_factory=Sim):
    """Continuous prediction plus independent logged-state command-window predictions."""
    times = columns["t"]
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Logged times must be finite and strictly increasing")
    start_t, end_t = float(attempt["start_t"]), float(attempt["end_t"])
    if not np.isfinite([start_t, end_t]).all() or not times[0] <= start_t < end_t <= times[-1]:
        raise ValueError("Attempt must have a nonempty interval inside the logged series")
    start, end = (int(np.searchsorted(times, value)) for value in (start_t, end_t))
    if start == end:
        raise ValueError("Attempt has no logged samples before intervention")
    if model_factory is Sim and start < 200:
        raise ValueError("Native replay needs 200 preceding logged rows, matching sim.validate")
    calls = accepted_commands(commands, attempt["segment"], times[start], end_t)
    if not calls or calls[0]["tool"] != "set_motion":
        raise ValueError("Attempt needs an accepted initial motion")
    continuous, receipts = _window(columns, cp, start, end, calls, model_factory)
    pieces, windows = [], []
    for index, command in enumerate(calls):
        first = start if index == 0 else int(np.searchsorted(times, command["t"]))
        last = int(np.searchsorted(times, calls[index + 1]["t"])) if index + 1 < len(calls) else end
        if last <= first:
            # The continuous replay retains these calls; no independent observed interval exists.
            continue
        piece, responses = _window(columns, cp, first, last, [command], model_factory)
        pieces.append(piece)
        windows.append(
            {
                "start_row": first - start,
                "end_row": last - start,
                "command": command,
                "responses": responses,
                "initial_state": {
                    "speed": float(columns["v"][first]),
                    "wheel_deg": float(columns["angle"][first]),
                    "accel": float(columns["a"][first]),
                    "hold": bool(columns["cruise_standstill"][first]),
                    "torque": float(columns["torque_out"][first]),
                    "desired_curvature": float(columns["curvature_cmd"][first]),
                    "long_state": int(columns["long_state"][first]),
                },
            }
        )
    arrays = {
        "continuous": continuous,
        "reanchored_local": np.concatenate(pieces),
        "observed_speed": columns["v"][start:end],
        "observed_wheel": columns["angle"][start:end],
        "fields": np.asarray(FIELDS),
    }
    summary = {
        "segment": attempt["segment"], "start_t": float(times[start]),
        "end_t": float(times[end]), "samples": end - start,
        "accepted_calls": len(calls),
        "zero_duration_command_windows": len(calls) - len(windows),
        "accepted_replays": sum(row["response"].get("status") in SUCCESS for row in receipts),
        "max_schedule_lateness_s": max(abs(row["receipt_t"] - row["scheduled_t"]) for row in receipts),
        "windows": windows, "continuous_receipts": receipts,
    }  # fmt: skip
    return arrays, summary


def run(manifest_path, out, *, root=REPO):
    """Save CPU replay artifacts in a fresh directory without altering source evidence."""
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    began, summaries = time.monotonic(), []
    seen = set()
    for attempt in manifest["attempts"]:
        segment = attempt["segment"]
        if Path(segment).name != segment or segment in {"", ".", ".."} or segment in seen:
            raise ValueError("Manifest needs unique segment directory names")
        seen.add(segment)
        series = root / attempt["series_path"]
        columns, cp, _, _, commands = loaded(series)
        arrays, summary = replay_attempt(columns, cp, commands, attempt)
        destination = out / segment
        destination.mkdir()
        np.savez_compressed(destination / "local-replay.npz", **arrays)
        summary.update(
            route_sha256=_hash(series),
            source_metadata=attempt,
            timing="Existing sim.validate logged-row scheduling; native request advances one 10 ms tick. Intervention row excluded.",
            initialization="Logged speed/wheel/acceleration history/curvature/Toyota torque/LongControl state. Hidden PID integrals and launch elapsed time not restored; preceding lease reconstructible but not restored here. Pose origin only, no path conditioning.",
            reference_convention="Nominal body center; independent real-only map registration applied after dynamics. Command resets are separate predictions, not a continuous drive.",
        )
        (destination / "local-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(
            {
                key: summary[key]
                for key in (
                    "segment", "samples", "accepted_calls", "accepted_replays",
                    "max_schedule_lateness_s",
                )
            }
        )  # fmt: skip
        print(segment, summary["samples"], summary["accepted_calls"], flush=True)
    report = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _hash(manifest_path),
        "script_sha256": _hash(Path(__file__)),
        "wall_seconds": time.monotonic() - began,
        "attempts": summaries,
        "sources": {
            name: _hash(REPO / "sim" / name)
            for name in ("car.py", "plant.py", "validate.py", "params.json", "corolla.carparams")
        },
    }
    (out / "native-local-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO, help="Base for relative series paths")
    args = parser.parse_args()
    report = run(args.manifest, args.out, root=args.root)
    print(json.dumps({"attempts": len(report["attempts"]), "seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
