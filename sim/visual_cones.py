"""Auditable marker crops and multi-view landmarks; no inferred course boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from drivingbench.sim.visual_eval import image, pixel_scores, summarize
from PIL import Image, ImageDraw

K = np.array([[1141.5, 0, 672], [0, 1141.5, 380], [0, 0, 1]])


def proposals(path):
    """Color components are proposals, not semantic detections or ground truth."""
    from scipy.ndimage import find_objects, label
    from skimage.color import rgb2hsv

    rgb = image(path)
    hsv = rgb2hsv(rgb)
    result = []
    for color, hue in (
        ("orange", (hsv[:, :, 0] < 0.15)),
        ("yellow_green", (hsv[:, :, 0] >= 0.15) & (hsv[:, :, 0] < 0.28)),
    ):
        mask = hue & (hsv[:, :, 1] > 0.28) & (hsv[:, :, 2] > 0.24)
        mask[:440] = False
        components, count = label(mask)
        for k, extent in enumerate(find_objects(components), 1):
            if extent is None:
                continue
            yy, xx = np.where(components[extent] == k)
            yy, xx = yy + extent[0].start, xx + extent[1].start
            if not 8 <= len(xx) <= 2500:
                continue
            x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
            width, height = x1 - x0, y1 - y0
            if not 4 <= width <= 100 or not 2 <= height <= 100 or width > height * 7:
                continue
            if x0 < 4 or x1 > 1340 or y1 > 755:
                continue
            if len(xx) / (width * height) < 0.28:
                continue
            neighborhood = hsv[max(0, y0 - 5) : y1 + 5, max(0, x0 - 5) : x1 + 5, 1].copy()
            if np.median(neighborhood) > 0.23:
                continue
            result.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "color": color,
                    "kind": "dome" if width > height * 1.4 else "tall_or_unknown",
                    "pixel": [(x0 + x1 - 1) / 2, y1 - 1],
                }
            )
    return sorted(result, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def crop_bounds(bbox, padding=4):
    x0, y0, x1, y1 = bbox
    px = max(padding, (11 - (x1 - x0) + 1) // 2)
    py = max(padding, (11 - (y1 - y0) + 1) // 2)
    return max(0, x0 - px), max(0, y0 - py), min(1344, x1 + px), min(760, y1 + py)


def proposal_sheet(rows, out):
    tiles = []
    annotations = []
    for frame in rows:
        source = Image.open(frame["file_path"]).convert("RGB")
        for observation in proposals(frame["file_path"]):
            record = {
                "id": len(annotations),
                "name": frame["name"],
                "t": frame["t"],
                "file_path": frame["file_path"],
                **observation,
            }
            annotations.append(record)
            crop = source.crop(crop_bounds(observation["bbox"], 8))
            crop.thumbnail((100, 74))
            crop = crop.resize(
                (
                    max(1, crop.width * min(3, 100 // crop.width)),
                    max(1, crop.height * min(3, 100 // crop.width)),
                ),
                Image.Resampling.NEAREST,
            )
            tile = Image.new("RGB", (160, 120), "#eeeeee")
            tile.paste(crop, (0, 22))
            ImageDraw.Draw(tile).text(
                (3, 3), f"{record['id']} t={frame['t']:.1f} {observation['color']}", fill="black"
            )
            ImageDraw.Draw(tile).text((3, 103), str(observation["bbox"]), fill="black")
            tiles.append(tile)
    sheet = Image.new("RGB", (160 * 8, 120 * ((len(tiles) + 7) // 8)), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 8) * 160, (i // 8) * 120))
    sheet.save(out)
    return annotations


def rays(frame, pixels):
    camera = np.asarray(frame["transform_matrix"])
    pixels = np.atleast_2d(pixels)
    directions = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    directions = directions @ camera[:3, :3].T
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.broadcast_to(camera[:3, 3], directions.shape), directions


def triangulate(centers, directions):
    """Least-squares intersection; condition number exposes weak baselines."""
    eye = np.eye(3)
    projectors = eye - directions[:, :, None] * directions[:, None, :]
    lhs = np.sum(projectors, axis=0)
    point = np.linalg.solve(lhs, np.einsum("nij,nj->i", projectors, centers))
    error = np.linalg.norm(np.einsum("nij,nj->ni", projectors, point - centers), axis=1)
    return point, error, float(np.linalg.cond(lhs))


def project(frame, point):
    camera = np.asarray(frame["transform_matrix"])
    p = camera[:3, :3].T @ (np.asarray(point) - camera[:3, 3])
    uv = K @ p
    return uv[:2] / uv[2], p[2]


def fit_landmark(observations, rng=None):
    """RANSAC base-point rays; reject near-parallel and fewer than five-view solutions."""
    if len(observations) < 5:
        return None
    centers = np.array([o["center"] for o in observations])
    directions = np.array([o["direction"] for o in observations])
    best = np.array([], dtype=int)
    rng = np.random.default_rng(1) if rng is None else rng
    for _ in range(100):
        pair = rng.choice(len(observations), 2, replace=False)
        try:
            point, _, condition = triangulate(centers[pair], directions[pair])
        except np.linalg.LinAlgError:
            continue
        if condition > 1e5 or not -1 < point[2] < 1:
            continue
        delta = point - centers
        depth = np.sum(delta * directions, axis=1)
        error = np.linalg.norm(delta - depth[:, None] * directions, axis=1)
        use = np.flatnonzero((error / np.maximum(depth, 1e-6) * K[0, 0] < 4) & (depth > 0))
        if len(use) > len(best):
            best = use
    if len(best) < 5:
        return None
    point, error, condition = triangulate(centers[best], directions[best])
    baseline = np.degrees(np.arccos(np.clip(np.min(directions[best] @ directions[best].T), -1, 1)))
    if baseline < 4 or condition > 1e5:
        return None
    depth = np.sum((point - centers[best]) * directions[best], axis=1)
    return {
        "position": point.tolist(),
        "count": len(best),
        "baseline_deg": float(baseline),
        "condition": condition,
        "reprojection_median_px": float(np.median(error / depth * K[0, 0])),
        "observations": [observations[i] for i in best],
    }


def landmark_candidates(frames, out):
    """Training-only candidate landmarks, requiring a separate visual identity review."""
    observations = []
    for frame in frames:
        if frame["split"] != "train":
            raise ValueError("Landmark reconstruction must use training frames only")
        for detection in proposals(frame["file_path"]):
            centers, directions = rays(frame, detection["pixel"])
            center, direction = centers[0], directions[0]
            distance = -center[2] / direction[2]
            if 2 < distance < 35:
                observations.append(
                    {
                        "name": frame["name"],
                        "file_path": frame["file_path"],
                        "t": frame["t"],
                        **detection,
                        "center": center.tolist(),
                        "direction": direction.tolist(),
                        "ground": (center + distance * direction).tolist(),
                    }
                )
    (out / "training-proposals.json").write_text(json.dumps(observations))
    clusters, locations, members = [], [], []
    for observation in sorted(observations, key=lambda o: -(o["bbox"][2] - o["bbox"][0])):
        point = np.array(observation["ground"])[:2]
        distances = np.linalg.norm(point - np.array(locations), axis=1) if locations else []
        possible = [
            int(i)
            for i in np.argsort(distances)
            if distances[i] < 0.65 and observation["name"] not in members[i]
        ]
        if possible:
            i = possible[0]
            clusters[i].append(observation)
            members[i].add(observation["name"])
            locations[i] = np.median([o["ground"][:2] for o in clusters[i]], axis=0)
        else:
            clusters.append([observation])
            locations.append(point)
            members.append({observation["name"]})
    results = []
    rng = np.random.default_rng(1)
    for group in clusters:
        if fit := fit_landmark(group, rng):
            results.append({"id": len(results), **fit})
    (out / "landmark-candidates.json").write_text(json.dumps(results, indent=2))
    return results


def evaluate(annotations, renders, out):
    """Native tight crops for PSNR/SSIM; separate resized-64 LPIPS for tiny objects."""
    import os

    os.environ.setdefault("TORCH_HOME", str(Path("runs/sim/visual/pilot/torch-cache").resolve()))
    import lpips
    import torch

    torch.set_num_threads(6)
    metric = lpips.LPIPS(net="alex", version="0.1").eval()
    result = []
    for annotation in annotations:
        bounds = crop_bounds(annotation["bbox"])
        x0, y0, x1, y1 = bounds
        reference = image(annotation["file_path"])[y0:y1, x0:x1]

        def tensor(array):
            resized = Image.fromarray(np.uint8(np.round(array * 255))).resize(
                (64, 64), Image.Resampling.BILINEAR
            )
            return (
                torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)[None].float() / 127.5
                - 1
            )

        real_tensor = tensor(reference)
        row = {
            "id": annotation["id"],
            "name": annotation["name"],
            "bbox_padded": bounds,
            "scores": {},
        }
        for name, folder in renders.items():
            path = folder / annotation["name"]
            if not path.exists():
                row["scores"][name] = {"error": "missing render"}
                continue
            rendered = image(path)[y0:y1, x0:x1]
            scores = pixel_scores(reference, rendered)
            with torch.no_grad():
                scores["lpips_alex"] = float(metric(real_tensor, tensor(rendered)).item())
            row["scores"][name] = scores
        result.append(row)
    report = {
        "protocol": "PSNR/SSIM native bbox+4 px (minimum 11px extent); LPIPS bilinear resize of same crop to 64x64",
        "annotations": len(annotations),
        "frames": len({a["name"] for a in annotations}),
        "summary": summarize(result, renders),
        "rows": result,
    }
    out.write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("propose", "evaluate", "landmarks"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("runs/sim/visual/pilot/sfm/frames.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("runs/sim/visual/cones"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--render", action="append", default=[], help="name=folder")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.operation == "propose":
        test = [
            json.loads(s)
            for s in args.manifest.read_text().splitlines()
            if json.loads(s)["split"] == "test"
        ]
        rows = [test[i] for i in np.linspace(0, len(test) - 1, 10, dtype=int)]
        annotations = proposal_sheet(rows, args.out / "proposals.jpg")
        (args.out / "proposals.json").write_text(json.dumps(annotations, indent=2))
    elif args.operation == "landmarks":
        frames = [json.loads(line) for line in args.manifest.read_text().splitlines()]
        landmark_candidates([r for r in frames if r["split"] == "train"], args.out)
    else:
        annotations = json.loads(args.annotations.read_text())
        renders = {k: Path(v) for k, v in (r.split("=", 1) for r in args.render)}
        print(
            json.dumps(
                evaluate(annotations, renders, args.out / "crop-metrics.json")["summary"], indent=2
            )
        )


if __name__ == "__main__":
    main()
