"""Bounded Gaussian background cleanup with explicit spatial and color selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_ply(path):
    """Read Brush's binary float vertex table without changing property order."""
    with Path(path).open("rb") as stream:
        header = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("Incomplete PLY header")
            line = raw.decode("ascii").rstrip("\n")
            header.append(line)
            if line == "end_header":
                break
        if header[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("Expected binary little-endian PLY")
        elements = [line for line in header if line.startswith("element ")]
        if len(elements) != 1 or not elements[0].startswith("element vertex "):
            raise ValueError("Expected one vertex element")
        properties = [line.split() for line in header if line.startswith("property ")]
        if any(len(p) != 3 or p[1] != "float" for p in properties):
            raise ValueError("Expected scalar float properties")
        count = int(elements[0].split()[-1])
        vertices = np.fromfile(stream, dtype=[(p[2], "<f4") for p in properties], count=count)
        if len(vertices) != count or stream.read(1):
            raise ValueError("PLY vertex payload size mismatch")
    return header, vertices


def write_ply(path, header, vertices):
    header = [
        f"element vertex {len(vertices)}" if line.startswith("element vertex ") else line
        for line in header
    ]
    with Path(path).open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        vertices.tofile(stream)


def in_course(points, polygon, margin=1.0):
    """Inside a convex polygon, including a bounded outward half-plane margin."""
    points, polygon = np.asarray(points), np.asarray(polygon)
    edge = np.roll(polygon, -1, axis=0) - polygon
    if (
        len(polygon) < 3
        or not np.isfinite(polygon).all()
        or np.any(np.linalg.norm(edge, axis=1) == 0)
    ):
        raise ValueError("Course requires a finite convex polygon")
    signed_area = np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    if signed_area == 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Course polygon must have area and margin must be nonnegative")
    turns = edge[:, 0] * np.roll(edge[:, 1], -1) - edge[:, 1] * np.roll(edge[:, 0], -1)
    if np.any(np.sign(signed_area) * turns < -1e-10):
        raise ValueError("Course polygon must be convex and consistently ordered")
    delta = points[:, None, :2] - polygon[None]
    cross = edge[None, :, 0] * delta[..., 1] - edge[None, :, 1] * delta[..., 0]
    return np.all(np.sign(signed_area) * cross >= -margin * np.linalg.norm(edge, axis=1), axis=1)


def in_region(points, region, margin=1.0):
    inside = in_course(points, region["polygon_xy"], margin)
    if "corridor_xy" in region:
        path = np.asarray(region["corridor_xy"], dtype=float)
        radius = float(region["corridor_radius_m"])
        if (
            path.ndim != 2
            or path.shape[1] != 2
            or len(path) < 2
            or not np.isfinite(path).all()
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("Corridor requires finite points and positive radius")
        for start, end in zip(path[:-1], path[1:]):
            delta = end - start
            length2 = delta @ delta
            if length2 == 0:
                raise ValueError("Corridor segments must have length")
            relative = np.asarray(points)[:, :2] - start
            along = np.clip(relative @ delta / length2, 0, 1)
            inside |= np.linalg.norm(relative - along[:, None] * delta, axis=1) <= radius
    return inside


def selection(
    vertices,
    polygon,
    plane,
    *,
    height=0.8,
    saturation=0.20,
    margin=1.0,
    corridor=None,
    max_scale=None,
    halo=0.0,
):
    """Select orange/yellow near-ground Gaussian centres only within the course."""
    from skimage.color import rgb2hsv

    plane = np.asarray(plane, dtype=float)
    if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) == 0:
        raise ValueError("Plane must be finite [nx, ny, nz, d]")
    if not np.isfinite([height, saturation]).all() or height <= 0 or not 0 <= saturation <= 1:
        raise ValueError("Invalid height or saturation threshold")
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    rgb = np.clip(
        np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) * 0.28209479177387814 + 0.5, 0, 1
    )
    hsv = rgb2hsv(rgb)
    distance = (xyz @ plane[:3] + plane[3]) / np.linalg.norm(plane[:3])
    course = in_region(xyz, {"polygon_xy": polygon, **(corridor or {})}, margin)
    near = np.abs(distance) <= height
    colored = (
        ((hsv[:, 0] < 0.30) | (hsv[:, 0] > 0.96)) & (hsv[:, 1] >= saturation) & (hsv[:, 2] >= 0.12)
    )
    selected = course & near & colored
    seed_count = int(selected.sum())
    scales = (
        np.exp(np.column_stack([vertices[f"scale_{i}"] for i in range(3)])).max(axis=1)
        if max_scale is not None or halo > 0
        else None
    )
    if max_scale is not None:
        if not np.isfinite(max_scale) or max_scale <= 0:
            raise ValueError("Maximum splat scale must be positive")
        selected &= scales <= max_scale
    if not np.isfinite(halo) or not 0 <= halo <= 0.2:
        raise ValueError("Halo must be between zero and 0.2 metres")
    if halo and selected.any():
        from scipy.spatial import cKDTree

        distance_to_seed = cKDTree(xyz[selected]).query(xyz, distance_upper_bound=halo)[0]
        selected |= course & near & (scales <= 0.15) & (distance_to_seed <= halo)
    return selected, {
        "course_centres": int(course.sum()),
        "near_ground_centres": int((course & near).sum()),
        "colored_candidates": seed_count,
        "selected": int(selected.sum()),
        "total": len(vertices),
    }


def clean(
    source, output, region, *, height=0.8, saturation=0.20, margin=1.0, max_scale=None, halo=0.0
):
    header, vertices = read_ply(source)
    mask, report = selection(
        vertices,
        region["polygon_xy"],
        region["plane"],
        height=height,
        saturation=saturation,
        margin=margin,
        corridor={k: region[k] for k in ("corridor_xy", "corridor_radius_m") if k in region},
        max_scale=max_scale,
        halo=halo,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output, header, vertices[~mask])
    report.update(
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output=str(output),
        region=region,
        height_m=height,
        saturation=saturation,
        margin_m=margin,
        max_scale_m=max_scale,
        halo_m=halo,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare(before, after, cameras, region, out):
    """Measure image changes and surviving warm-colored pixels in a projected ground region."""
    from PIL import Image
    from skimage.color import rgb2hsv

    data = json.loads(Path(cameras).read_text())
    y, x = np.mgrid[: data["h"], : data["w"]]
    rays = np.stack(
        [(x - data["cx"]) / data["fl_x"], (y - data["cy"]) / data["fl_y"], np.ones_like(x)], axis=-1
    )
    plane = np.asarray(region["plane"])
    rows = []
    for frame in data["frames"]:
        name = Path(frame.get("name", frame.get("file_path", ""))).stem
        pose = np.asarray(frame["transform_matrix"]) @ np.diag([1, -1, -1, 1])
        direction = rays @ pose[:3, :3].T
        denominator = direction @ plane[:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = -(pose[:3, 3] @ plane[:3] + plane[3]) / denominator
            point = pose[:3, 3] + direction * distance[..., None]
        mask = in_region(point.reshape(-1, 3), region, 1.0).reshape(x.shape) & (distance > 0)
        original = np.asarray(Image.open(before / f"{name}_color.png").convert("RGB"))
        cleaned = np.asarray(Image.open(after / f"{name}_color.png").convert("RGB"))
        change = np.max(np.abs(cleaned.astype(float) - original), axis=-1)
        counts = []
        for rgb in (original, cleaned):
            hsv = rgb2hsv(rgb.astype(float) / 255)
            counts.append(
                int(
                    (
                        mask
                        & ((hsv[..., 0] < 0.30) | (hsv[..., 0] > 0.96))
                        & (hsv[..., 1] >= 0.20)
                        & (hsv[..., 2] >= 0.12)
                    ).sum()
                )
            )
        alpha_before = np.asarray(Image.open(before / f"{name}_alpha.png"))
        alpha_after = np.asarray(Image.open(after / f"{name}_alpha.png"))
        rows.append(
            {
                "name": name,
                "inside_pixels": int(mask.sum()),
                "outside_pixels": int((~mask).sum()),
                "inside_changed_gt3": int(((change > 3) & mask).sum()),
                "outside_changed_gt3": int(((change > 3) & ~mask).sum()),
                "outside_mean_max_channel_difference": float(change[~mask].mean()),
                "warm_color_before": counts[0],
                "warm_color_after": counts[1],
                "new_low_alpha_pixels": int(((alpha_before >= 128) & (alpha_after < 128)).sum()),
            }
        )
    out.write_text(
        json.dumps(
            {
                "interpretation": "Changes relative to original splat, not real-image accuracy; warm pixels include non-marker colors; course is projected fitted ground polygon plus1m",
                "frames": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("region", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=float, default=0.8)
    parser.add_argument("--saturation", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--max-scale", type=float)
    parser.add_argument("--halo", type=float, default=0.0)
    args = parser.parse_args()
    print(
        json.dumps(
            clean(
                args.source,
                args.output,
                json.loads(args.region.read_text()),
                height=args.height,
                saturation=args.saturation,
                margin=args.margin,
                max_scale=args.max_scale,
                halo=args.halo,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
