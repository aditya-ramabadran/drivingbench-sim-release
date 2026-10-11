"""Fit a deliberately simple replay model to recorded, uninterrupted maneuvers.

This estimates effective low-speed behavior, not Toyota actuator physics. No device
connection is made. Operator interventions are excluded; resume latency cannot be
separated from actuator latency in these recordings.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class Dynamics:
    launch_delay_s: float = 1.3
    accel_mps2: float = 0.55
    brake_mps2: float = 0.4
    steering_rate_deg_s: float = 7.1459
    speed_noise_mps: float = 0.006749

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid dynamics parameter {name}: {value}")
        if min(self.accel_mps2, self.brake_mps2, self.steering_rate_deg_s) <= 0:
            raise ValueError("Acceleration, braking and steering rate must be positive")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(**data.get("parameters", data))


def clean(row):
    return (
        row.get("enabled", False)
        and row.get("sensors_valid", False)
        and not any(row.get(k) for k in ("brake_pressed", "gas_pressed", "steering_pressed"))
        and row.get("reason") in (None, "waiting_for_res", "steering_unavailable")
    )


def collect(paths):
    launches, stops, steering, noise = [], [], [], []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in rows if r["kind"] == "telemetry" and "received_at" in r]
        segment = path.parent.name
        # Contiguous command runs avoid conflating successive target updates.
        i = 0
        while i < len(rows):
            row = rows[i]
            command = row.get("command")
            if command is None:
                i += 1
                continue
            j = i + 1
            while (
                j < len(rows)
                and (rows[j].get("command") or {}).get("request_id") == command["request_id"]
            ):
                j += 1
            episode = rows[i:j]
            origin = command["accepted_at_s"]
            if (
                episode
                and episode[0]["speed_mps"] < 0.05
                and command["speed_mps"] > 0.2
                and all(clean(r) for r in episode)
                and max(r["speed_mps"] for r in episode) > 0.2
                and episode[0]["received_at"] - origin < 0.5
            ):
                launches.append(
                    {
                        "segment": segment,
                        "target": command["speed_mps"],
                        "origin": "ready_after_res"
                        if origin != command["accepted_at_s"]
                        else "command_accepted",
                        "samples": [
                            (r["received_at"] - origin, max(0, r["speed_mps"])) for r in episode
                        ],
                    }
                )
            # Cancellation without intervention is ordinary expiry or explicit stop.
            if j < len(rows) and not rows[j].get("command") and clean(rows[j - 1]):
                end = j
                while end < len(rows) and not rows[end].get("command") and clean(rows[end]):
                    if rows[end]["speed_mps"] <= 0.05:
                        end += 1
                        break
                    end += 1
                episode = rows[j:end]
                if (
                    len(episode) >= 3
                    and episode[-1]["speed_mps"] <= 0.05
                    and rows[j - 1]["speed_mps"] > 0.2
                ):
                    start = rows[j - 1]
                    stops.append(
                        {
                            "segment": segment,
                            "initial": start["speed_mps"],
                            "samples": [
                                (r["received_at"] - start["received_at"], max(0, r["speed_mps"]))
                                for r in episode
                            ],
                        }
                    )
            i = j
        for a, b in zip(rows, rows[1:]):
            dt = b["received_at"] - a["received_at"]
            command = a.get("command")
            if not (
                0.03 <= dt <= 0.3
                and clean(a)
                and clean(b)
                and command
                and command == b.get("command")
            ):
                continue
            angle_error = command["target_angle_deg"] - a["steering_deg"]
            rate = (b["steering_deg"] - a["steering_deg"]) / dt
            if (
                a.get("lateral_active")
                and a["speed_mps"] > 0.2
                and abs(angle_error) > 15
                and rate * angle_error > 0
            ):
                steering.append((segment, abs(rate)))
            if (
                abs(a["speed_mps"] - command["speed_mps"]) < 0.1
                and abs(b["speed_mps"] - command["speed_mps"]) < 0.1
            ):
                noise.append((segment, (b["speed_mps"] - a["speed_mps"]) / math.sqrt(2)))
    return launches, stops, steering, noise


def fit(traces):
    import numpy as np

    paths = sorted(Path(traces).glob("segments/2026-09-17*/events.jsonl"))
    launches, stops, steering, noise = collect(paths)
    if not launches or not stops or not steering:
        raise ValueError("Insufficient clean launch, ordinary stop, or steering samples")
    # Whole-segment holdout prevents adjacent telemetry leaking between sets.
    segments = sorted({e["segment"] for e in launches + stops})
    holdout = set(segments[::5]) if len(segments) >= 5 else set()
    train_launches = [e for e in launches if e["segment"] not in holdout]
    train_stops = [e for e in stops if e["segment"] not in holdout]
    if not train_launches or not train_stops:
        raise ValueError("Insufficient training episodes after segment holdout")

    def loss(episodes, delay, rate, braking=False):
        # Equal episode weighting prevents long plateau recordings dominating.
        errors = []
        for e in episodes:
            t, v = np.asarray(e["samples"]).T
            predicted = (
                np.maximum(0, e["initial"] - rate * t)
                if braking
                else np.minimum(e["target"], rate * np.maximum(0, t - delay))
            )
            errors.append(float(np.mean((predicted - v) ** 2)))
        return float(np.mean(errors)) if errors else None

    _, delay, accel = min(
        (loss(train_launches, d, a), d, a)
        for d in np.arange(0, 3.01, 0.1)
        for a in np.arange(0.1, 2.01, 0.05)
    )
    _, brake = min((loss(train_stops, 0, b, True), b) for b in np.arange(0.1, 3.01, 0.025))
    rates = [v for s, v in steering if s not in holdout]
    jitter = [v for s, v in noise if s not in holdout]
    noise_scale = 1.4826 * median(abs(v - median(jitter)) for v in jitter) if jitter else 0.0
    parameters = Dynamics(
        round(float(delay), 4),
        round(float(accel), 4),
        round(float(brake), 4),
        round(median(rates), 4),
        round(noise_scale, 6),
    )
    metrics = {}
    for label, predicate in (
        ("train", lambda s: s not in holdout),
        ("holdout", lambda s: s in holdout),
    ):
        for kind, episodes, rate in (("launch", launches, accel), ("stop", stops, brake)):
            subset = [e for e in episodes if predicate(e["segment"])]
            mse = loss(subset, delay, rate, kind == "stop")
            metrics[f"{label}_{kind}"] = {
                "episodes": len(subset),
                "samples": sum(len(e["samples"]) for e in subset),
                "speed_rmse_mps": math.sqrt(mse) if mse is not None else None,
            }
    for label, predicate in (
        ("train", lambda s: s not in holdout),
        ("holdout", lambda s: s in holdout),
    ):
        subset = [v for s, v in steering if predicate(s)]
        metrics[f"{label}_steering_rate"] = {
            "samples": len(subset),
            "rate_rmse_deg_s": math.sqrt(
                sum((v - parameters.steering_rate_deg_s) ** 2 for v in subset) / len(subset)
            )
            if subset
            else None,
            "p10_median_p90_deg_s": np.quantile(subset, [0.1, 0.5, 0.9]).tolist()
            if subset
            else None,
        }
    return {
        "parameters": asdict(parameters),
        "source_glob": "segments/2026-09-17*/events.jsonl",
        "source_segments": len(paths),
        "holdout_segments": sorted(holdout),
        "metrics": metrics,
        "steering_samples": len(rates),
        "noise_samples": len(jitter),
        "method": "Grid least squares with equal episode weights; whole-segment holdout. Steering rate is median toward-target rate at >15 degree error; noise is robust first-difference scale near target.",
        "limitations": [
            "Effective low-speed approximation, not physical steering or collision verification.",
            "Excludes operator intervention and invalid sensors. Recorded waiting_for_res is a standstill status, not a measurement of manual RES; launch delay includes any unobserved resume latency.",
            "Telemetry command disappearance brackets braking onset by one sample; stop and expiry pooled.",
            "Constant acceleration cannot reproduce overshoot, creep, actuator latency or controller oscillation.",
            "Steering rate depends on vehicle speed and load; one median cannot reproduce all conditions.",
            "Noise estimate includes vehicle dynamics and filtering; it is not independent sensor noise.",
            "Holdout is within one day's sessions, not an independent vehicle/day validation.",
        ],
        "episodes": {
            kind: [
                {
                    "segment": e["segment"],
                    "samples": len(e["samples"]),
                    "duration_s": e["samples"][-1][0],
                    "target_mps" if kind == "launch" else "initial_mps": e.get(
                        "target", e.get("initial")
                    ),
                    "speed_rmse_mps": math.sqrt(
                        loss([e], delay, accel if kind == "launch" else brake, kind == "stop")
                    ),
                }
                for e in episodes
            ]
            for kind, episodes in (("launch", launches), ("stop", stops))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = fit(args.traces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
