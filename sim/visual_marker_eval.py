"""Calibrated appearance study on the frozen marker crops; not an independent holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from drivingbench.sim.visual_cones import K, project
from PIL import Image

ANNOTATION_SHA = "fd7fc6732d59fe5a724c6d62f2e818881e8bfd5c87d4d077412bbafcc31ef9c8"
SEMANTIC_SHA = "da3fdf3f45a58ceb49bad6cdbe6981a8dbeb1ed24b60fd58615fa3c50ffd8c0e"
ROOT = Path("runs/sim/visual")


def annotation_digest(annotations):
    content = [{k: v for k, v in row.items() if k != "file_path"} for row in annotations]
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def material_colors(observed):
    rgb = np.asarray(observed)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    albedo = np.clip(linear / np.array([0.56, 0.576, 0.608]), 0, 1)
    return np.where(
        albedo <= 0.0031308, albedo * 12.92, 1.055 * albedo ** (1 / 2.4) - 0.055
    ).tolist()


def matched_markers(annotations, frames, landmarks, max_error=15):
    """One-to-one same-color projected-base assignments, retaining unmatched denominators."""
    from scipy.optimize import linear_sum_assignment

    matches = []
    for name in sorted({a["name"] for a in annotations}):
        frame = frames[name]
        for color in ("orange_red", "yellow_green"):
            observed = [
                a
                for a in annotations
                if a["name"] == name and a["color"] == color and a["kind"] == "dome_marker"
            ]
            candidates = []
            for marker in landmarks:
                if marker["kind"] != "dome_marker" or marker["color"] != color:
                    continue
                uv, depth = project(frame, marker["position_enu_m"])
                if depth > 0 and -50 < uv[0] < 1394 and -50 < uv[1] < 810:
                    candidates.append((marker, uv, depth))
            if not observed or not candidates:
                continue
            pixels = np.array(
                [[(a["bbox"][0] + a["bbox"][2] - 1) / 2, a["bbox"][3] - 1] for a in observed]
            )
            cost = np.linalg.norm(
                pixels[:, None] - np.array([c[1] for c in candidates])[None], axis=2
            )
            penalty = (min(cost.shape) + 1) * max_error + 1
            rows, columns = linear_sum_assignment(np.where(cost <= max_error, cost, penalty))
            for row, column in zip(rows, columns):
                if cost[row, column] <= max_error:
                    marker, uv, depth = candidates[column]
                    matches.append(
                        {
                            "annotation_id": observed[row]["id"],
                            "name": name,
                            "marker_id": marker["id"],
                            "position_enu_m": marker["position_enu_m"],
                            "depth_m": float(depth),
                            "reprojection_px": float(cost[row, column]),
                            "projected_base": uv.tolist(),
                        }
                    )
    return matches


def apparent_size(bbox, point, camera):
    """Small-object ellipsoid silhouette approximation in the registered map's metric scale."""
    world_to_camera = np.asarray(camera)[:3, :3].T
    local = world_to_camera @ (np.asarray(point) - np.asarray(camera)[:3, 3])
    depth = local[2]
    if not np.isfinite(local).all() or depth <= 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("Apparent size requires finite visible geometry and a positive rectangle")
    diameter = (bbox[2] - bbox[0]) * depth / K[0, 0]
    jacobian = K[1, 1] * (world_to_camera[1] * depth - local[1] * world_to_camera[2]) / depth**2
    if abs(jacobian[2]) < 1e-8:
        raise ValueError("Height is unobservable along this camera axis")
    ground_half = diameter / 2 * np.linalg.norm(jacobian[:2])
    height_pixels = bbox[3] - bbox[1]
    height = np.sqrt(max(0, height_pixels**2 - 2 * height_pixels * ground_half)) / abs(jacobian[2])
    return float(diameter), float(height)


def calibrate(annotation_path, manifest_path, landmark_path, out):
    from skimage.color import rgb2hsv

    digest = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    annotations = json.loads(annotation_path.read_text())
    semantic = annotation_digest(annotations)
    if semantic != SEMANTIC_SHA:
        raise ValueError("Rectangles/labels differ from the frozen 97-crop sample")
    frames = {f["name"]: f for f in map(json.loads, manifest_path.read_text().splitlines())}
    landmarks = json.loads(landmark_path.read_text())["landmarks"]
    matches = matched_markers(annotations, frames, landmarks)
    by_id = {m["annotation_id"]: m for m in matches}
    rows, colors = [], {"orange_red": [], "yellow_green": []}
    images = {}
    for annotation in annotations:
        if annotation["kind"] != "dome_marker":
            continue
        name, color = annotation["name"], annotation["color"]
        if name not in images:
            images[name] = np.asarray(Image.open(annotation["file_path"]).convert("RGB")) / 255
        x0, y0, x1, y1 = annotation["bbox"]
        rgb = images[name][y0:y1, x0:x1]
        hsv = rgb2hsv(rgb)
        hue = (
            hsv[..., 0] < 0.15
            if color == "orange_red"
            else ((hsv[..., 0] >= 0.15) & (hsv[..., 0] < 0.28))
        )
        keep = hue & (hsv[..., 1] > 0.28) & (hsv[..., 2] > 0.24)
        if keep.any():
            colors[color].append(np.median(rgb[keep], axis=0).tolist())
        row = {
            "annotation_id": annotation["id"],
            "name": name,
            "color": color,
            "color_pixels": int(keep.sum()),
            "observed_color_srgb": np.median(rgb[keep], axis=0).tolist() if keep.any() else None,
        }
        if match := by_id.get(annotation["id"]):
            diameter, height = apparent_size(
                annotation["bbox"], match["position_enu_m"], frames[name]["transform_matrix"]
            )
            row.update(match, apparent_diameter_m=diameter, apparent_height_m=height)
        rows.append(row)
    sizes = np.array(
        [
            [r["apparent_diameter_m"], r["apparent_height_m"]]
            for r in rows
            if "apparent_diameter_m" in r
        ]
    )
    if not len(sizes) or any(not values for values in colors.values()):
        raise ValueError("Calibration needs matched dome geometry and both observed colors")
    measured = {color: np.median(values, axis=0).tolist() for color, values in colors.items()}
    report = {
        "evaluation": "calibrated/in-sample appearance evaluation; same 97 crops calibrate and score",
        "annotations_sha256": digest,
        "annotations_semantic_sha256": semantic,
        "original_frozen_sha256": ANNOTATION_SHA,
        "annotations": len(annotations),
        "dome_annotations": len(rows),
        "matched_domes": len(sizes),
        "unmatched_dome_ids": [r["annotation_id"] for r in rows if "marker_id" not in r],
        "diameter_m": float(np.median(sizes[:, 0])),
        "height_m": float(np.median(sizes[:, 1])),
        "diameter_quantiles_m": np.quantile(sizes[:, 0], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        "height_quantiles_m": np.quantile(sizes[:, 1], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        "observed_color_srgb": measured,
        "material_color_srgb": {k: material_colors(v) for k, v in measured.items()},
        "colors_observations": {k: len(v) for k, v in colors.items()},
        "position_source": str(landmark_path),
        "rows": rows,
        "size_semantics": "apparent saturated-color bounds plus registered depth; ellipsoid projection approximation, not surveyed physical dimensions",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def fit_marker_assets(calibration, annotations, frames):
    """Fit one persistent asset per observed marker, with fixed map positions/cameras."""
    from drivingbench.sim.visual_objects import dome_mesh
    from scipy.optimize import least_squares

    boxes = {a["id"]: a["bbox"] for a in annotations}
    assets = {}
    marker_ids = sorted({r["marker_id"] for r in calibration["rows"] if "marker_id" in r})
    for marker_id in marker_ids:
        rows = [r for r in calibration["rows"] if r.get("marker_id") == marker_id]

        def residual(size):
            vertices = np.asarray(dome_mesh(*size)[0])
            errors = []
            for row in rows:
                camera = np.asarray(frames[row["name"]]["transform_matrix"])
                local = (vertices + row["position_enu_m"] - camera[:3, 3]) @ camera[:3, :3]
                pixels = local @ K.T
                extent = np.ptp(pixels[:, :2] / pixels[:, 2:], axis=0)
                box = boxes[row["annotation_id"]]
                errors.extend(extent - [box[2] - box[0], box[3] - box[1]])
            return errors

        fit = least_squares(
            residual,
            [calibration["diameter_m"], calibration["height_m"]],
            bounds=([0.07, 0.025], [0.30, 0.15]),
            loss="soft_l1",
            f_scale=1,
            max_nfev=100,
        )
        observed = np.median(
            [r["observed_color_srgb"] for r in rows if r["observed_color_srgb"] is not None], axis=0
        )
        assets[marker_id] = {
            "diameter_m": float(fit.x[0]),
            "height_m": float(fit.x[1]),
            "observed_color_srgb": observed.tolist(),
            "material_color_srgb": material_colors(observed),
            "observations": len(rows),
            "extent_rmse_px": float(np.sqrt(np.mean(np.square(residual(fit.x))))),
        }
    return assets


def render_study(annotations, frames, landmarks, calibration, out, blender):
    from drivingbench.sim.visual_objects import (
        PlacedObject,
        composite_objects,
        marker_repair_mask,
        render_objects,
        repair_background,
    )

    markers = [m for m in landmarks if m["kind"] == "dome_marker"]
    assets = calibration.get("marker_assets", {})
    objects = [
        PlacedObject(
            "dome_marker",
            *marker["position_enu_m"],
            diameter_m=assets.get(marker["id"], calibration)["diameter_m"],
            height_m=assets.get(marker["id"], calibration)["height_m"],
            color_srgb=tuple(
                assets[marker["id"]]["material_color_srgb"]
                if marker["id"] in assets
                else calibration["material_color_srgb"][marker["color"]]
            ),
        )
        for marker in markers
    ]
    rows = []
    for name in sorted({a["name"] for a in annotations}):
        stem = Path(name).stem
        background = np.asarray(Image.open(out / "background" / f"{stem}_color.png").convert("RGB"))
        depth = np.load(out / "background" / f"{stem}_depth.npy")
        alpha = np.load(out / "background" / f"{stem}_depth_alpha.npy")
        layer = render_objects(
            objects, frames[name]["transform_matrix"], out / "objects" / stem, blender
        )
        inserted, masks = composite_objects(background, depth, alpha, layer)
        mask = marker_repair_mask(layer, depth, alpha, radius=3)
        clean, repair = repair_background(background, mask)
        repaired, second_masks = composite_objects(clean, depth, alpha, layer)
        assert all(np.array_equal(masks[k], second_masks[k]) for k in masks)
        assert np.array_equal(clean[~repair["repaired"]], background[~repair["repaired"]])
        for variant, rgb in (
            ("background", background),
            ("inserted", inserted),
            ("repaired", repaired),
        ):
            folder = out / "renders" / variant
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(folder / name, quality=92)
        diagnostic = out / "objects" / stem
        for key, value in {**masks, **{f"repair_{k}": v for k, v in repair.items()}}.items():
            Image.fromarray(np.uint8(value) * 255).save(diagnostic / f"{key}.png")
        instances = []
        for index, marker in enumerate(markers, 1):
            present = layer.instance == index
            counts = {key: int((value & present).sum()) for key, value in masks.items()}
            instances.append({"marker_id": marker["id"], "pixels": int(present.sum()), **counts})
        row = {
            "name": name,
            "instances": instances,
            "pixels": {key: int(value.sum()) for key, value in masks.items()},
            "unsupported_instance_pixels": int(
                ((layer.rgba[..., 3] > 0) & (layer.instance == 0)).sum()
            ),
            "repair_pixels": {key: int(value.sum()) for key, value in repair.items()},
        }
        rows.append(row)
        (out / "visibility.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(name, row["pixels"], row["repair_pixels"], flush=True)
    return rows


def evaluate_study(annotations, manifest, out):
    from drivingbench.sim.visual_cones import evaluate as evaluate_crops
    from drivingbench.sim.visual_eval import evaluate as evaluate_frames
    from drivingbench.sim.visual_eval import summarize

    renders = {name: out / "renders" / name for name in ("background", "inserted", "repaired")}
    crops = evaluate_crops(annotations, renders, out / "crop-metrics.json")
    queries = [f for f in manifest if f["name"] in {a["name"] for a in annotations}]
    full = evaluate_frames(manifest, queries, renders, out / "frame-metrics.json")
    kinds = {a["id"]: a["kind"] for a in annotations}
    report = {
        "evaluation": "calibrated/in-sample appearance; same frozen crops calibrate asset dimensions/colors and score; see calibration.json for shared or persistent per-marker fits",
        "annotations_semantic_sha256": annotation_digest(annotations),
        "single_roi_sensitivity": {
            "96_single_markers": summarize([r for r in crops["rows"] if r["id"] != 173], renders),
            "91_single_domes": summarize(
                [r for r in crops["rows"] if r["id"] != 173 and kinds[r["id"]] == "dome_marker"],
                renders,
            ),
        },
        "full_frames": full["summary"],
        "all_97_crops": crops["summary"],
        "crop_groups": {
            kind: summarize([r for r in crops["rows"] if kinds[r["id"]] == kind], renders)
            for kind in sorted(set(kinds.values()))
        },
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def comparison_figure(annotations, out, target):
    import matplotlib.pyplot as plt
    from drivingbench.sim.visual_cones import crop_bounds

    chosen = []
    for name in sorted({a["name"] for a in annotations}):
        domes = [
            a
            for a in annotations
            if a["name"] == name
            and a["kind"] == "dome_marker"
            and a["bbox"][2] - a["bbox"][0] <= 100
        ]
        chosen.append(
            max(domes, key=lambda a: (a["bbox"][2] - a["bbox"][0]) * (a["bbox"][3] - a["bbox"][1]))
        )
    fig, axes = plt.subplots(len(chosen), 4, figsize=(10, 15), squeeze=False)
    for index, annotation in enumerate(chosen):
        x0, y0, x1, y1 = crop_bounds(annotation["bbox"], padding=12)
        paths = [Path(annotation["file_path"])] + [
            out / "renders" / key / annotation["name"]
            for key in ("background", "inserted", "repaired")
        ]
        for axis, path in zip(axes[index], paths):
            axis.imshow(np.asarray(Image.open(path))[y0:y1, x0:x1], interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[index, 0].set_ylabel(f"{annotation['t']:.1f}s / #{annotation['id']}")
    for axis, title in zip(axes[0], ("Real", "Gaussian", "+ measured domes", "Repair + domes")):
        axis.set_title(title)
    fig.suptitle(
        "Calibrated appearance: largest single-marker box (width ≤100 px) in each frozen frame"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("calibrate", "render", "evaluate", "figure"))
    parser.add_argument("--annotations", type=Path, default=ROOT / "cones/annotations.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "pilot/sfm/frames.jsonl")
    parser.add_argument("--landmarks", type=Path, default=ROOT / "cones/landmarks.json")
    parser.add_argument("--out", type=Path, default=ROOT / "marker-eval")
    parser.add_argument(
        "--blender", type=Path, default=ROOT / "tools/Blender.app/Contents/MacOS/Blender"
    )
    parser.add_argument("--personalized", action="store_true")
    parser.add_argument("--figure", type=Path, default=Path("docs/sim/marker-eval.png"))
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text())
    if annotation_digest(annotations) != SEMANTIC_SHA:
        raise ValueError("Rectangles/labels differ from the frozen 97-crop sample")
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if args.operation == "calibrate":
        result = calibrate(args.annotations, args.manifest, args.landmarks, args.out)
        if args.personalized:
            result["marker_assets"] = fit_marker_assets(
                result, annotations, {f["name"]: f for f in manifest}
            )
            (args.out / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("rows", "marker_assets")}, indent=2
            )
        )
    elif args.operation == "render":
        render_study(
            annotations,
            {f["name"]: f for f in manifest},
            json.loads(args.landmarks.read_text())["landmarks"],
            json.loads((args.out / "calibration.json").read_text()),
            args.out,
            args.blender,
        )
    elif args.operation == "evaluate":
        print(json.dumps(evaluate_study(annotations, manifest, args.out), indent=2))
    else:
        comparison_figure(annotations, args.out, args.figure)


if __name__ == "__main__":
    main()
