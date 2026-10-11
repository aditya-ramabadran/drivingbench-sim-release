"""Summarize a completed simulator batch without reclassifying model intentions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MODEL_CLIENTS = {"MCP", "codex", "claude", "cursor"}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def events(path):
    with path.open() as stream:
        for line, text in enumerate(stream, 1):
            row = json.loads(text)
            row["source"] = dict(path=str(path), line=line, id=row.get("id", row.get("sequence")))
            yield row


def evidence(row):
    keys = "source sim_time_s map_pose pose speed_mps collision_ids outside_course finish_reached valid_finish reason terminal_reason"
    return {key: row[key] for key in keys.split() if key in row}


def trajectory(path):
    points, private = [], []
    first = dict(contact=None, outside=None, finish_region=None, cause=None)
    distance, previous = 0.0, None
    for row in events(path) if path.exists() else ():
        if row["kind"] == "sim_tick":
            point = [row["sim_time_s"], *row["map_pose"][:2]]
            if not all(math.isfinite(v) for v in point):
                raise ValueError(f"Nonfinite trajectory at {row['source']}")
            if previous:
                if point[0] < previous[0]:
                    raise ValueError(f"Nonmonotonic trajectory at {row['source']}")
                distance += math.hypot(point[1] - previous[1], point[2] - previous[2])
            previous = point
            points.append(point)
            for key, condition in (
                ("contact", row.get("collision_ids")),
                ("outside", row.get("outside_course")),
                ("finish_region", row.get("finish_reached")),
            ):
                if condition and first[key] is None:
                    first[key] = evidence(row)
        elif row["kind"] in {"episode_trigger", "sim_terminal_impact", "infrastructure_failure"}:
            value = dict(kind=row["kind"], **evidence(row))
            if "evidence" in row:
                value["evidence"] = evidence(row["evidence"])
            private.append(value)
            if row["kind"] == "episode_trigger" and first["cause"] is None:
                first["cause"] = value
    return points, distance if points else None, first, private


def tool_summary(paths):
    counts, clients, outcomes = Counter(), defaultdict(Counter), Counter()
    stops, native_ends = [], []
    previous = None
    rows = sorted((row for path in paths for row in events(path)), key=lambda row: row["timestamp"])
    for row in rows:
        if row["kind"] == "telemetry":
            previous = row
        elif row["kind"] == "segment_end":
            native_ends.append({key: row.get(key) for key in ("source", "timestamp", "reason")})
        elif row["kind"] == "tool":
            tool, client = row["tool"], row.get("client", "unknown")
            counts[tool] += 1
            clients[client][tool] += 1
            outcome = row.get("outcome", {})
            outcomes[f"{tool}:{outcome.get('status', outcome.get('error', 'unspecified'))}"] += 1
            if tool == "stop_now":
                preceding = (
                    None
                    if previous is None or previous.get("segment") != row.get("segment")
                    else {
                        key: previous.get(key)
                        for key in "source timestamp speed_mps standstill held reason".split()
                    }
                )
                stops.append(
                    dict(
                        source=row["source"],
                        timestamp=row["timestamp"],
                        client=client,
                        actor="model_transport"
                        if client in MODEL_CLIENTS
                        else "operator"
                        if client == "operator"
                        else "unknown",
                        arguments=row.get("arguments", {}),
                        outcome=outcome,
                        preceding_telemetry=preceding,
                    )
                )
    return dict(
        recording_files=[str(path) for path in paths],
        available=bool(paths),
        counts=dict(counts),
        by_client={k: dict(v) for k, v in clients.items()},
        outcomes=dict(outcomes),
        stop_calls=stops,
        native_segment_ends=native_ends,
    )


def maximum_progress(points, centerline):
    """Nearest-polyline arc coordinate; only defined for an explicit map-frame centerline."""
    if not points or centerline is None:
        return None
    import numpy as np

    line = np.asarray(centerline, dtype=float)
    if line.ndim != 2 or line.shape[1] != 2 or len(line) < 2 or not np.isfinite(line).all():
        raise ValueError("centerline must contain finite map-frame [x,y] points")
    vectors = np.diff(line, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths == 0):
        raise ValueError("centerline contains a zero-length segment")
    offsets = np.r_[0, lengths.cumsum()[:-1]]
    maximum = 0.0
    for start in range(0, len(points), 512):
        xy = np.asarray(points[start : start + 512])[:, 1:3]
        fractions = np.clip(((xy[:, None] - line[:-1]) * vectors).sum(2) / lengths**2, 0, 1)
        distances = ((xy[:, None] - (line[:-1] + fractions[..., None] * vectors)) ** 2).sum(2)
        closest = distances.argmin(1)
        maximum = max(
            maximum,
            float(
                (offsets[closest] + fractions[np.arange(len(xy)), closest] * lengths[closest]).max()
            ),
        )
    return maximum


def summarize(root, scene_path, real_reference=None):
    root, scene_path = Path(root).resolve(), Path(scene_path).resolve()
    if not (root / "batch.json").exists():
        raise ValueError("Completed batch.json is required")
    batch, scene = read_json(root / "batch.json"), read_json(scene_path)
    if not scene:
        raise ValueError("Scene JSON is required")
    sources = {root / "batch.json", scene_path}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        sources.add(protocol_path)
        expected = read_json(protocol_path).get("scene_sha256")
        if expected and expected != hashlib.sha256(scene_path.read_bytes()).hexdigest():
            raise ValueError("Scene differs from frozen batch protocol")
    result = []
    for assignment in batch["assignments"]:
        model, number = assignment["model"], assignment["attempt"]
        folder = root / model / f"attempt-{number}"
        paths = [
            folder / "attempt.json",
            folder / "episode/summary.json",
            folder / "episode/events.jsonl",
            folder / "supervisor-error.json",
            root / model / "skipped.json",
            root / model / "remaining-skipped.json",
        ]
        sources.update(p for p in paths if p.exists())
        attempt = read_json(paths[0])
        fallback = next(
            (
                r
                for r in batch.get("attempts", [])
                if r.get("model") == model and r.get("attempt") == number
            ),
            {},
        )
        attempt = attempt or fallback
        summary = read_json(paths[1]) or attempt.get("episode", {})
        points, distance, first, private = trajectory(paths[2])
        recordings = sorted((folder / "producer/recordings").glob("**/events.jsonl"))
        sources.update(recordings)
        skipped = read_json(paths[4])
        remaining = read_json(paths[5])
        if remaining and number > remaining["after_attempt"]:
            skipped = remaining
        incomplete = assignment["status"] == "attempted" and not summary
        row = dict(
            **assignment,
            outcome=summary.get("outcome", "incomplete_evidence" if incomplete else "not_run"),
            skip=skipped or None,
            batch_error=batch.get("batch_error"),
            infrastructure_failure=summary.get("infrastructure_failure"),
            supervisor_error=read_json(paths[3]) or attempt.get("infrastructure_error"),
            client_errors=attempt.get("client_errors", []),
            failure_kind=attempt.get("failure_kind"),
            wall_s=attempt.get("wall_s"),
            elapsed_sim_s=summary.get("elapsed_s"),
            distance_m=distance,
            max_progress_m=maximum_progress(points, scene.get("centerline")),
            max_progress_method="nearest explicit centerline arc coordinate"
            if scene.get("centerline")
            else None,
            first_cause=summary.get("first_cause", first["cause"]),
            first_contact=first["contact"] or summary.get("first_contact"),
            first_outside=first["outside"] or summary.get("first_exit"),
            first_finish_region=first["finish_region"] or summary.get("first_finish"),
            stopped=summary.get("stopped"),
            native_end_reason=summary.get("native_end_reason"),
            private_termination_events=private,
            tools=tool_summary(recordings),
            path_samples=points[:: max(1, math.ceil(len(points) / 999))],
            tick_count=len(points),
            missing_sources=[str(p) for p in paths[:3] if not p.exists()],
        )
        if points and row["path_samples"][-1] != points[-1]:
            row["path_samples"].append(points[-1])
        result.append(row)
    reference = None
    if real_reference:
        real_reference = Path(real_reference).resolve()
        if not real_reference.exists():
            raise ValueError("Real reference does not exist")
        sources.add(real_reference)
        real = read_json(real_reference)
        reference = dict(
            path=str(real_reference),
            models=real["primary_models"],
            caveat="Curated real outcomes; 11 attempts and dual-camera observations, not matched simulator geometry or finish rules.",
        )
    hashes = {}
    for path in sorted(sources):
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(
        schema_version=1,
        source_sha256=hashes,
        attempts=result,
        real_reference=reference,
        caveats=[
            "Path travel is not finish progress; first finish-region entry is not finish dwell success.",
            "A stop call is not braking onset; generic operator_brake native endings do not identify a human.",
            "Transport client labels are recorded attribution; no reasoning or intention is inferred.",
        ],
    )


def plot_paths(report, scene, output):
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(row["model"] for row in report["attempts"]))
    count = max(row["attempt"] for row in report["attempts"])
    fig, axes = plt.subplots(
        count, len(models), figsize=(4 * len(models), 3.5 * count), squeeze=False
    )
    boundaries = scene.get("boundaries", {})
    corridor = boundaries.get("left", []) + boundaries.get("right", [])[::-1]
    for row in report["attempts"]:
        ax = axes[row["attempt"] - 1, models.index(row["model"])]
        for shape, color in (
            (scene.get("corridor", corridor), "gray"),
            (scene.get("finish", {}).get("polygon", []), "green"),
        ):
            if shape:
                xy = [*shape, shape[0]]
                ax.plot(*zip(*xy), color=color, linewidth=0.8)
        markers = scene.get("markers", [])
        if markers:
            ax.scatter([m["x"] for m in markers], [m["y"] for m in markers], s=5, c="orange")
        points = row["path_samples"]
        if points:
            ax.plot([p[1] for p in points], [p[2] for p in points], color="#1764a0")
            ax.scatter(points[-1][1], points[-1][2], c="red", s=16)
        else:
            ax.text(0.5, 0.5, row["outcome"].replace("_", " "), transform=ax.transAxes, ha="center")
        ax.set_title(f"{row['model']} #{row['attempt']}: {row['outcome']}", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("map x (m)")
        ax.set_ylabel("map y (m)")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-reference", type=Path)
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    report = summarize(args.root, args.scene, args.real_reference)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.figure:
        plot_paths(report, read_json(args.scene), args.output / "paths.jpg")


if __name__ == "__main__":
    main()
