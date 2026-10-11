"""Heading-aware training support, measured held-out quality, and frozen spatial test splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

SITE_RANGES = (
    ("00000041--fdf9cede4b", 2040, 2160),
    ("00000041--fdf9cede4b", 2880, 3000),
    ("00000042--544757d3f5", 720, 840),
    ("00000042--544757d3f5", 960, 1080),
    ("00000042--544757d3f5", 2760, 2880),
    ("00000042--544757d3f5", 3840, 3960),
    ("00000042--544757d3f5", 340, 480),
)


def split_rows(rows, ranges=SITE_RANGES, guard_s=10):
    """Freeze whole time intervals and guards without consulting appearance or refined poses."""
    if guard_s < 0 or any(end <= start for _, start, end in ranges):
        raise ValueError("Split ranges must increase and guard must be nonnegative")
    result = []
    for row in rows:
        intervals = [(start, end) for route, start, end in ranges if route == row["route"]]
        split = "train"
        if any(start - guard_s <= row["t"] < end + guard_s for start, end in intervals):
            split = "guard"
        if any(start <= row["t"] < end for start, end in intervals):
            split = "test"
        result.append(dict(row, split=split))
    return result


def prepare_validation(manifest, metadata_path, out):
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if any("seed_pose" in row for row in rows):
        raise ValueError("Validation must start from original pose seeds, not a fitted map")
    rows = split_rows(rows)
    metadata = json.loads(metadata_path.read_text())
    for key in ("test_route", "test_time_s"):
        metadata.pop(key, None)
    metadata["validation_split"] = {
        "source_manifest": str(manifest.resolve()),
        "purpose": "six spatially distributed time ranges; original development interval retained",
        "selection": "camera pose coverage only, before scoring this split",
        "guard_s": 10,
        "ranges": [dict(route=r, start_s=s, end_s=e) for r, s, e in SITE_RANGES],
        "counts": dict(Counter(row["split"] for row in rows)),
        "requires": "fresh training-only geometry and appearance; original image bytes reused",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata["validation_split"]


def support(poses, training):
    """Classify ENU x/y metres and heading radians by 2 m/15° then 5 m/30° support."""
    if not training or any(row["split"] != "train" for row in training):
        raise ValueError("Support requires training cameras only")
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    if poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise ValueError("Expected finite x/y/heading poses")
    reference = np.array([[r["x"], r["y"], r["heading"]] for r in training])
    result = []
    for pose in poses:
        delta = reference - pose
        distance = np.linalg.norm(delta[:, :2], axis=1)
        angle = np.abs(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2])))
        near = (distance <= 2) & (angle <= np.radians(15))
        marginal = (distance <= 5) & (angle <= np.radians(30))
        state = "supported" if near.any() else "marginal" if marginal.any() else "unsupported"
        aligned = distance[angle <= np.radians(15)]
        result.append(
            dict(
                state=state,
                nearest_m=float(distance.min()),
                nearest_aligned_m=float(aligned.min()) if len(aligned) else None,
                nearby_views=int(near.sum()),
            )
        )
    return result


def support_grid(training, spacing=1.0, directions=16):
    """Sample location and heading separately; a supported location need not support a turn."""
    if not training or any(r["split"] != "train" for r in training):
        raise ValueError("Grid requires training cameras only")
    xy = np.array([[r["x"], r["y"]] for r in training])
    heading = np.array([r["heading"] for r in training])
    if spacing <= 0 or directions < 4 or not len(xy):
        raise ValueError(
            "Grid needs training cameras, positive spacing, and at least four directions"
        )
    lower, upper = np.floor(xy.min(0) - 10), np.ceil(xy.max(0) + 10)
    x, y = [np.arange(a, b + spacing, spacing) for a, b in zip(lower, upper)]
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    angles = np.arange(directions) * (2 * np.pi / directions)
    states = np.zeros((directions, len(points)), dtype=np.uint8)
    for k, angle in enumerate(angles):
        difference = np.abs(np.arctan2(np.sin(heading - angle), np.cos(heading - angle)))
        for state, radius, tolerance in ((1, 5, 30), (2, 2, 15)):
            reference = xy[difference <= np.radians(tolerance)]
            if not len(reference):
                continue
            for start in range(0, len(points), 512):
                query = points[start : start + 512]
                distance = ((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
                valid = distance.min(axis=1) <= radius**2
                states[k, start : start + len(query)][valid] = state
    return x, y, angles, states.reshape(directions, len(y), len(x))


def measured_frames(rows, metrics, model="Gaussian"):
    """Join measured scores only to genuinely held-out observations, never to a grid cell."""
    observed = {r["name"]: r for r in rows if r["split"] == "test"}
    training = [r for r in rows if r["split"] == "train"]
    result, names = [], set()
    for score in metrics["frames"]:
        name = score["name"]
        if name not in observed or name in names:
            raise ValueError("Measured scores need unique held-out frame names")
        names.add(name)
        if "error" in score["scores"][model]:
            continue
        row = observed[name]
        evidence = support([[row[k] for k in ("x", "y", "heading")]], training)[0]
        result.append(dict(row, score=score["scores"][model], support=evidence))
    return result


def diagnostic_queries(training, grid, count=8):
    """Spatially spread synthetic poses in each support class; no paired target images."""
    from drivingbench.sim.visual_pose import camera_to_world

    x, y, angles, states = grid
    xy = np.array([[r["x"], r["y"]] for r in training])
    queries = []
    for state, label in enumerate(("unsupported", "marginal", "supported")):
        index = np.argwhere(states == state)
        poses = np.column_stack((x[index[:, 2]], y[index[:, 1]], angles[index[:, 0]]))
        embedding = np.column_stack(
            (poses[:, :2], 3 * np.cos(poses[:, 2]), 3 * np.sin(poses[:, 2]))
        )
        selected = []
        distances = np.linalg.norm(embedding - embedding.mean(0), axis=1)
        for _ in range(min(count, len(poses))):
            k = int(np.argmax(distances))
            selected.append(k)
            distances = np.minimum(distances, np.linalg.norm(embedding - embedding[k], axis=1))
            distances[selected] = -1
        for k in selected:
            east, north, heading = poses[k]
            nearest = training[int(np.argmin(np.linalg.norm(xy - poses[k, :2], axis=1)))]
            name = f"coverage-{len(queries) + 1:02d}.jpg"
            pose = camera_to_world(east, north, heading, nearest["rpy"])
            queries.append(
                dict(
                    name=name,
                    split="novel",
                    x=float(east),
                    y=float(north),
                    heading=float(heading),
                    support=label,
                    rpy=nearest["rpy"],
                    transform_matrix=pose.tolist(),
                )
            )
    return queries


def coverage_figures(training, measured, grid, out, prefix="visual-site1-coverage"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    x, y, angles, states = grid
    dx, dy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    extent = [x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy]
    xy = np.array([[r["x"], r["y"]] for r in training])
    observed = np.array([[r["x"], r["y"]] for r in measured])
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    cover = axes[0].imshow(
        (states == 2).sum(0),
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0,
        vmax=len(angles),
        interpolation="nearest",
    )
    axes[0].scatter(*xy.T, s=1, c="#222222", alpha=0.25)
    axes[0].set_title("Training support by viewing direction")
    fig.colorbar(
        cover, ax=axes[0], label=f"Headings within 2 m / 15° (of {len(angles)})", shrink=0.78
    )
    axes[1].scatter(*xy.T, s=2, c="#cccccc")
    scored = axes[1].scatter(
        *observed.T,
        c=[r["score"]["psnr_db"] for r in measured],
        cmap="RdYlGn",
        s=24,
        edgecolors="#333333",
        linewidths=0.3,
    )
    axes[1].set_title(f"Measured quality: {len(measured)} held-out cameras")
    fig.colorbar(scored, ax=axes[1], label="PSNR (dB), observed cameras only", shrink=0.78)
    for ax in axes:
        ax.set(
            xlabel="East (m)", ylabel="North (m)", aspect="equal", xlim=extent[:2], ylim=extent[2:]
        )
    fig.savefig(out / f"{prefix}.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    colors = ListedColormap(["#eeeeee", "#e9bd75", "#387da5"])
    for ax, k, title in zip(
        axes.ravel(), range(0, len(angles), len(angles) // 4), ("East", "North", "West", "South")
    ):
        ax.imshow(states[k], origin="lower", extent=extent, cmap=colors, vmin=0, vmax=2)
        ax.scatter(*xy.T, s=1, c="#222222", alpha=0.3)
        ax.set(
            title=f"Looking {title.lower()}", xlabel="East (m)", ylabel="North (m)", aspect="equal"
        )
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in zip(
                colors.colors, ("Unsupported", "Marginal: ≤5 m / 30°", "Supported: ≤2 m / 15°")
            )
        ],
        loc="outside lower center",
        ncol=3,
    )
    fig.savefig(out / f"{prefix}-directions.png", dpi=150)
    plt.close(fig)


def build_coverage(manifest, metrics_path, out, figures, prefix="visual-site1-coverage"):
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    training = [r for r in rows if r["split"] == "train"]
    metrics = json.loads(metrics_path.read_text())
    measured = measured_frames(rows, metrics)
    grid = support_grid(training)
    queries = diagnostic_queries(training, grid)
    out.mkdir(parents=True, exist_ok=True)
    x, y, headings, states = grid
    np.savez_compressed(out / "grid.npz", x=x, y=y, heading=headings, state=states)
    (out / "diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in queries))
    camera = json.loads((manifest.parent / "transforms_train.json").read_text())
    camera = {key: camera[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
    frames = [
        dict(
            r,
            transform_matrix=(np.asarray(r["transform_matrix"]) @ np.diag([1, -1, -1, 1])).tolist(),
        )
        for r in queries
    ]
    (out / "cameras.json").write_text(json.dumps(dict(camera, frames=frames), indent=2))
    report = dict(
        training=len(training),
        measured=len(measured),
        measured_support=dict(Counter(r["support"]["state"] for r in measured)),
        grid_spacing_m=1,
        grid_heading_count=len(headings),
        any_heading_supported_area_m2=int(np.any(states == 2, axis=0).sum()),
        all_headings_supported_area_m2=int(np.all(states == 2, axis=0).sum()),
        supported_pose_fraction=float(np.mean(states == 2)),
        metrics=metrics["summary"],
        measured_bounds_xy=[
            np.min([[r["x"], r["y"]] for r in measured], axis=0).tolist(),
            np.max([[r["x"], r["y"]] for r in measured], axis=0).tolist(),
        ],
        synthetic_diagnostics=len(queries),
        interpretation="Support is a distance/heading heuristic. Only observed held-out cameras have measured quality; no operationally trusted area is established.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    coverage_figures(training, measured, grid, figures, prefix)
    return report


def diagnostic_figure(queries_path, renders, out):
    """Show actual novel renders and opacity holes; opacity does not measure fidelity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    queries = [json.loads(line) for line in queries_path.read_text().splitlines()]
    if any(row["split"] != "novel" for row in queries):
        raise ValueError("Diagnostic montage requires synthetic queries")
    fig, axes = plt.subplots((len(queries) + 3) // 4, 4, figsize=(16, 13), squeeze=False)
    diagnostics = []
    for ax, row in zip(axes.ravel(), queries):
        with Image.open(renders / row["name"]) as im:
            ax.imshow(im)
        with Image.open(renders / f"{Path(row['name']).stem}_alpha.png") as im:
            alpha = np.asarray(im.convert("L"), dtype=float) / 255
        low = float(np.mean(alpha < 0.9))
        diagnostics.append(
            dict(
                name=row["name"],
                support=row["support"],
                mean_alpha=float(alpha.mean()),
                fraction_alpha_lt_0_9=low,
            )
        )
        ax.set_title(
            f"{row['support']} · ({row['x']:.0f}, {row['y']:.0f}) m · "
            f"{np.degrees(row['heading']):.0f}°\nOpacity <0.9: {low:.0%}",
            fontsize=10,
        )
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Synthetic camera poses: visual diagnostics, no paired ground truth", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    (queries_path.parent / "opacity.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return diagnostics


def summarize_intervals(source_rows, metrics, localization, ranges=SITE_RANGES):
    """Keep original/new interval scores and missing-pose denominators explicit."""
    from drivingbench.sim.visual_eval import summarize

    expected = {row["name"]: row for row in source_rows if row["split"] == "test"}
    scored = {row["name"]: row for row in metrics["frames"]}
    if len(scored) != len(metrics["frames"]) or not set(scored) <= set(expected):
        raise ValueError("Interval scores need unique names from the frozen test split")
    rows = {
        name: scored.get(
            name,
            dict(name=name, scores={"Gaussian": {"error": "not localized or missing evaluation"}}),
        )
        for name in expected
    }

    def aggregate(names):
        return summarize([rows[name] for name in sorted(names)], ["Gaussian"])["Gaussian"]

    intervals = []
    for route, start, end in ranges:
        names = [
            name
            for name, row in expected.items()
            if row["route"] == route and start <= row["t"] < end
        ]
        intervals.append(dict(route=route, start_s=start, end_s=end, **aggregate(names)))
    development = {
        name
        for name, row in expected.items()
        if row["route"] == "00000042--544757d3f5" and 340 <= row["t"] < 480
    }
    low_inliers = [
        dict(row, route=expected[row["name"]]["route"], t=expected[row["name"]]["t"])
        for row in localization
        if row["name"] in expected and row["localized"] and row["inliers"] < 100
    ]
    return dict(
        device=metrics.get("device"),
        all_test=aggregate(expected),
        development=aggregate(development),
        new_intervals=aggregate(set(expected) - development),
        intervals=intervals,
        low_inlier_test_poses=low_inliers,
        interpretation="Report new and development intervals separately; verify model-training provenance before claiming independence. Low-inlier poses are retained.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("split")
    p.add_argument("manifest", type=Path)
    p.add_argument("metadata", type=Path)
    p.add_argument("out", type=Path)
    p = sub.add_parser("map")
    p.add_argument("manifest", type=Path)
    p.add_argument("metrics", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--figures", type=Path, default=Path("docs/sim"))
    p.add_argument("--prefix", default="visual-site1-coverage")
    p = sub.add_parser("diagnostics")
    p.add_argument("queries", type=Path)
    p.add_argument("renders", type=Path)
    p.add_argument("out", type=Path)
    p = sub.add_parser("summarize")
    p.add_argument("source", type=Path)
    p.add_argument("metrics", type=Path)
    p.add_argument("localization", type=Path)
    p.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.action == "split":
        result = prepare_validation(args.manifest, args.metadata, args.out)
    elif args.action == "map":
        result = build_coverage(args.manifest, args.metrics, args.out, args.figures, args.prefix)
    elif args.action == "diagnostics":
        result = diagnostic_figure(args.queries, args.renders, args.out)
    else:
        result = summarize_intervals(
            [json.loads(line) for line in args.source.read_text().splitlines()],
            json.loads(args.metrics.read_text()),
            json.loads(args.localization.read_text())["localization"],
        )
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
