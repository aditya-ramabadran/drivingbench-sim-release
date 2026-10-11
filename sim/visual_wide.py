"""Wide-camera indexing and calibrated equidistant rays; optional vision tools stay lazy."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from drivingbench.sim.logs import events
from drivingbench.sim.visual_data import decode, validate_index, write_json, write_rows

INTRINSICS = dict(w=1344, h=760, fl_x=425.25, fl_y=425.25, cx=672.0, cy=380.0)


def wide_index(path):
    """Read the authoritative presentation index for the wide HEVC stream."""
    rows = []
    for _, _, message in events(path, {"wideRoadEncodeIdx"}):
        if str(message.type) == "fullHEVC":
            rows.append(
                dict(
                    segment=int(message.segmentNum),
                    index=int(message.segmentId),
                    frame_id=int(message.frameId),
                    sof_ns=int(message.timestampSof),
                    eof_ns=int(message.timestampEof),
                )
            )
    return validate_index(rows)


def pair_frame(narrow, index, output):
    matches = [r for r in index if r["frame_id"] == narrow["frame_id"]]
    if len(matches) != 1:
        raise ValueError("wide frame ID must match exactly once")
    wide = matches[0]
    delta = (wide["eof_ns"] - narrow["eof_ns"]) * 1e-9
    if abs(delta) > 0.005 or wide["segment"] != narrow["segment"]:
        raise ValueError("wide/narrow exposure or segment mismatch")
    return dict(
        narrow,
        **{key: value for key, value in wide.items() if key not in narrow},
        wide_index=wide["index"],
        wide_sof_ns=wide["sof_ns"],
        wide_eof_ns=wide["eof_ns"],
        pair_delta_s=delta,
        narrow_file_path=narrow["file_path"],
        file_path=str(Path(output).resolve() / narrow["name"]),
    )


def extract_pairs(rows, routes, output, *, workers=4):
    """Preserve the narrow split and pose while decoding paired wide images."""
    output = Path(output)
    if (output / "frames.jsonl").exists():
        raise ValueError("wide extraction output already contains a manifest")
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    groups = {}
    for row in rows:
        groups.setdefault((row["route"], row["segment"]), []).append(row)
    paths = [Path(routes) / f"{route}--{segment}" for route, segment in groups]
    result, jobs = [], []
    with ProcessPoolExecutor(workers) as pool:
        indexes = pool.map(wide_index, [p / "rlog.zst" for p in paths])
        for path, source, index in zip(paths, groups.values(), indexes, strict=True):
            paired = [pair_frame(row, index, images) for row in source]
            result.extend(paired)
            ordered = sorted(paired, key=lambda r: r["wide_index"])
            jobs.append((path / "ecamera.hevc", [dict(r, index=r["wide_index"]) for r in ordered]))
        list(pool.map(decode, jobs))
    write_rows(output / "frames.jsonl", result)
    write_json(
        output / "metadata.json",
        dict(
            camera=INTRINSICS,
            projection="nominal intrinsics only; distortion not supplied by vendor",
            count=len(result),
            max_pair_delta_s=max(abs(r["pair_delta_s"]) for r in result),
            camera_to_world="transform_matrix remains narrow OpenCV; rig is applied separately",
        ),
    )
    return result


def fisheye_project(points, intrinsics, distortion):
    """Project forward rays with the OpenCV four-coefficient equidistant model."""
    points = np.asarray(points, dtype=float)
    coefficients = np.asarray(distortion, dtype=float)
    if points.shape[-1:] != (3,) or coefficients.shape != (4,):
        raise ValueError("expected 3D points and four fisheye coefficients")
    if not np.isfinite(points).all() or not np.isfinite(coefficients).all():
        raise ValueError("fisheye inputs must be finite")
    if np.any(np.linalg.norm(points, axis=-1) < 1e-12):
        raise ValueError("zero ray has no projection")
    radial = np.linalg.norm(points[..., :2], axis=-1)
    theta = np.arctan2(radial, points[..., 2])
    theta2 = theta * theta
    polynomial = 1 + sum(k * theta2 ** (i + 1) for i, k in enumerate(coefficients))
    radius = theta * polynomial
    scale = np.divide(radius, radial, out=np.zeros_like(radial), where=radial > 1e-12)
    xy = points[..., :2] * scale[..., None]
    pixels = xy * [intrinsics["fl_x"], intrinsics["fl_y"]] + [intrinsics["cx"], intrinsics["cy"]]
    undefined = (radial <= 1e-12) & (points[..., 2] < 0)
    return np.where(undefined[..., None], np.nan, pixels)


@dataclass
class WideCalibration:
    """A fixed narrow-to-wide rig and native equidistant projection."""

    intrinsics: dict
    distortion: list
    rotation_wide_from_narrow: list
    translation_wide_from_narrow: list

    def __post_init__(self):
        values = np.array([self.intrinsics[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")])
        self.rotation = np.array(self.rotation_wide_from_narrow, dtype=float)
        self.translation = np.array(self.translation_wide_from_narrow, dtype=float)
        self.coefficients = np.array(self.distortion, dtype=float)
        if not np.isfinite(values).all() or np.any(values[:4] <= 0):
            raise ValueError("invalid wide intrinsics")
        if (
            self.rotation.shape != (3, 3)
            or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(self.rotation), 1)
        ):
            raise ValueError("wide rig rotation must be proper and orthonormal")
        if self.translation.shape != (3,) or not np.isfinite(self.translation).all():
            raise ValueError("invalid rig translation")
        if self.coefficients.shape != (4,) or not np.isfinite(self.coefficients).all():
            raise ValueError("expected four finite fisheye coefficients")
        theta = np.linspace(0, 2.2, 1001)
        if np.any(self._derivative(theta) <= 0):
            raise ValueError("fisheye polynomial must be monotonic through 126 degrees")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})

    def _radius(self, theta):
        return theta * (1 + sum(k * theta ** (2 * i + 2) for i, k in enumerate(self.coefficients)))

    def _derivative(self, theta):
        return 1 + sum(
            (2 * i + 3) * k * theta ** (2 * i + 2) for i, k in enumerate(self.coefficients)
        )

    def project(self, points):
        return fisheye_project(points, self.intrinsics, self.coefficients)

    def rays(self, pixels):
        """Return unit native rays through 126 degrees; larger angles become NaN."""
        pixels = np.asarray(pixels, dtype=float)
        if pixels.shape[-1:] != (2,) or not np.isfinite(pixels).all():
            raise ValueError("pixels must be finite 2D coordinates")
        k = self.intrinsics
        xy = (pixels - [k["cx"], k["cy"]]) / [k["fl_x"], k["fl_y"]]
        radius = np.linalg.norm(xy, axis=-1)
        valid = radius <= self._radius(2.2)
        theta = np.minimum(radius, 2.2)
        for _ in range(12):
            theta = np.clip(
                theta - (self._radius(theta) - radius) / self._derivative(theta), 0, 2.2
            )
        scale = np.divide(np.sin(theta), radius, out=np.ones_like(radius), where=radius > 1e-12)
        rays = np.concatenate((xy * scale[..., None], np.cos(theta)[..., None]), axis=-1)
        return np.where(valid[..., None], rays, np.nan)

    def camera_pose(self, narrow_c2w):
        narrow = np.asarray(narrow_c2w, dtype=float)
        if narrow.shape != (4, 4) or not np.isfinite(narrow).all():
            raise ValueError("expected a finite 4x4 narrow pose")
        result = narrow.copy()
        result[:3, :3] = narrow[:3, :3] @ self.rotation.T
        result[:3, 3] -= result[:3, :3] @ self.translation
        return result


def perspective_tiles(size=896, focal=448.0):
    """Three overlapping upward-tilted pinhole cameras in the wide optical frame."""
    tiles = []
    pitch = np.radians(10)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )
    for name, yaw in (("left", -55), ("center", 0), ("right", 55)):
        angle = np.radians(yaw)
        ry = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
        tiles.append(
            dict(
                name=name,
                w=size,
                h=size,
                fl_x=focal,
                fl_y=focal,
                cx=size / 2,
                cy=size / 2,
                rotation_wide_from_tile=(ry @ rx).tolist(),
            )
        )
    return tiles


def tile_map(calibration, tile):
    """Map rectified tile pixels to native wide pixels for image and mask sampling."""
    yy, xx = np.indices((tile["h"], tile["w"]), dtype=float)
    rays = np.stack(
        ((xx - tile["cx"]) / tile["fl_x"], (yy - tile["cy"]) / tile["fl_y"], np.ones_like(xx)),
        axis=-1,
    )
    return calibration.project(rays @ np.asarray(tile["rotation_wide_from_tile"]).T).astype(
        np.float32
    )


def native_maps(calibration, tiles):
    """Return tile sampling coordinates and blend weights for native wide output."""
    k = calibration.intrinsics
    yy, xx = np.indices((k["h"], k["w"]), dtype=float)
    rays = calibration.rays(np.stack((xx, yy), axis=-1))
    maps, weights = [], []
    for tile in tiles:
        local = rays @ np.asarray(tile["rotation_wide_from_tile"])
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = local[..., :2] / local[..., 2:3]
        uv = uv * [tile["fl_x"], tile["fl_y"]] + [tile["cx"], tile["cy"]]
        margin = np.minimum.reduce(
            (uv[..., 0], uv[..., 1], tile["w"] - 1 - uv[..., 0], tile["h"] - 1 - uv[..., 1])
        )
        weight = np.where((local[..., 2] > 0) & np.isfinite(margin), np.clip(margin / 64, 0, 1), 0)
        maps.append(np.nan_to_num(uv, nan=-1).astype(np.float32))
        weights.append(weight)
    total = np.sum(weights, axis=0)
    return maps, [np.divide(w, total, out=np.zeros_like(w), where=total > 0) for w in weights]


def compose_native(images, maps, weights, body_rgb=None, body_alpha=None):
    """Blend rendered RGB tiles into native fisheye pixels, then overlay opaque car body."""
    import cv2

    if not (len(images) == len(maps) == len(weights)) or not images:
        raise ValueError("one image, map, and weight required per tile")
    result = np.zeros((*weights[0].shape, 3), dtype=float)
    for image, mapping, weight in zip(images, maps, weights, strict=True):
        sampled = cv2.remap(np.asarray(image), mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR)
        result += sampled * weight[..., None]
    if (body_rgb is None) != (body_alpha is None):
        raise ValueError("body RGB and alpha must be supplied together")
    if body_rgb is not None:
        alpha = np.asarray(body_alpha, dtype=float)
        rgb = np.asarray(body_rgb)
        if (
            alpha.shape != result.shape[:2]
            or rgb.shape != result.shape
            or not np.isfinite(alpha).all()
            or np.any((alpha < 0) | (alpha > 1))
        ):
            raise ValueError("body layer must match native dimensions with alpha in [0,1]")
        result = result * (1 - alpha[..., None]) + rgb * alpha[..., None]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]
