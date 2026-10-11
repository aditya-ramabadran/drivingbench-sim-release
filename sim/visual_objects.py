"""Transparent placed objects over a fixed Gaussian scene, with explicit depth validity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CONE_SIZE = (0.35, 0.35, 0.55)


@dataclass(frozen=True)
class PlacedObject:
    """Metres in ENU, origin at the object's ground contact; yaw is radians."""

    kind: str
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    scale: float = 1.0
    asset_path: str | None = None
    texture_path: str | None = None
    diameter_m: float | None = None
    height_m: float | None = None
    color_srgb: tuple[float, float, float] | None = None
    width_m: float | None = None
    depth_m: float | None = None
    anchor_uv: tuple[float, float] | None = None
    shadow_strength: float | None = None
    material_saturation: float | None = None
    face_camera: bool = False

    def __post_init__(self):
        kinds = (
            "cone",
            "dome_marker",
            "asset",
            "billboard",
            "ground_decal",
            "barrier",
            "paint",
            "jersey_barrier",
            "traffic_barrel",
        )
        if self.kind not in kinds:
            raise ValueError(f"Object kind must be one of {kinds}")
        if not isinstance(self.face_camera, bool) or (
            self.face_camera and self.kind != "billboard"
        ):
            raise ValueError("face_camera requires a boolean and is only enabled for billboards")
        if self.material_saturation is not None:
            value = self.material_saturation
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("material_saturation must be finite between zero and one")
            if self.kind != "billboard" and (
                self.kind not in ("asset", "jersey_barrier", "traffic_barrel")
                or not (self.asset_path and Path(self.asset_path).suffix.lower() == ".glb")
            ):
                raise ValueError("material_saturation requires a billboard or static imported GLB")
        if not np.isfinite([self.x, self.y, self.z, self.yaw, self.scale]).all() or self.scale <= 0:
            raise ValueError("Object placement must be finite and scale positive")
        if self.shadow_strength is not None and (
            not np.isfinite(self.shadow_strength) or not 0 <= self.shadow_strength <= 1
        ):
            raise ValueError("shadow_strength must be between zero and one")
        if self.anchor_uv is not None:
            anchor = np.asarray(self.anchor_uv)
            if (
                self.kind not in ("billboard", "ground_decal")
                or anchor.shape != (2,)
                or not np.isfinite(anchor).all()
                or not np.all((anchor >= 0) & (anchor <= 1))
            ):
                raise ValueError(
                    "anchor_uv requires two normalized billboard/decal pixel coordinates"
                )
        if self.texture_path is not None and (
            self.kind != "jersey_barrier" or Path(self.texture_path).suffix.lower() != ".png"
        ):
            raise ValueError("texture_path requires a local .png Jersey barrier material")
        suffix = {"asset": ".glb", "billboard": ".png", "ground_decal": ".png"}.get(self.kind)
        if suffix and (not self.asset_path or Path(self.asset_path).suffix.lower() != suffix):
            raise ValueError(f"{self.kind} requires a local {suffix} asset")
        if not suffix and self.asset_path is not None:
            if (
                self.kind not in ("jersey_barrier", "traffic_barrel")
                or Path(self.asset_path).suffix.lower() != ".glb"
            ):
                raise ValueError("Construction assets require a local .glb")
            if self.texture_path is not None:
                raise ValueError("Choose asset_path PBR materials or texture_path, not both")
        required = {
            "dome_marker": ("diameter_m", "height_m"),
            "billboard": ("width_m", "height_m"),
            "ground_decal": ("width_m", "depth_m"),
            "barrier": ("width_m", "depth_m", "height_m"),
            "paint": ("width_m", "depth_m"),
            "jersey_barrier": ("width_m", "depth_m", "height_m"),
            "traffic_barrel": ("diameter_m", "height_m"),
        }.get(self.kind, ())
        for name in ("diameter_m", "height_m", "width_m", "depth_m"):
            value = getattr(self, name)
            if name in required:
                if value is None or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{self.kind} requires finite positive {name}")
            elif value is not None:
                raise ValueError(f"{name} is not used by {self.kind}")
        if self.kind == "dome_marker" and self.color_srgb is None:
            raise ValueError("Dome markers require color_srgb")
        if self.color_srgb is not None:
            color = np.asarray(self.color_srgb)
            if self.kind not in (
                "dome_marker",
                "barrier",
                "paint",
                "jersey_barrier",
                "traffic_barrel",
            ) or (
                color.shape != (3,)
                or not np.isfinite(color).all()
                or not np.all((color >= 0) & (color <= 1))
            ):
                raise ValueError("color_srgb requires three values between zero and one")


@dataclass
class ObjectLayer:
    rgba: np.ndarray
    depth: np.ndarray
    repaired: np.ndarray | None = None
    instance: np.ndarray | None = None


def repair_edge_depth(rgba, depth, instance=None):
    """Fill covered edge samples from one original 8-neighbor ring; never propagate."""
    covered = np.asarray(rgba)[..., 3] > 0
    result = np.asarray(depth, dtype=np.float32).copy()
    valid = covered & np.isfinite(result) & (result > 0)
    source = np.pad(np.where(valid, result, np.nan), 1, constant_values=np.nan)
    missing = covered & ~valid
    repaired = np.zeros_like(covered)
    ids = None if instance is None else np.asarray(instance, dtype=np.int32).copy()
    source_ids = None if ids is None else np.pad(ids, 1)
    height, width = covered.shape
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    unambiguous = np.ones_like(covered)
    if ids is not None:
        candidates = np.stack(
            [
                np.where(
                    np.isfinite(source[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]),
                    source_ids[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width],
                    0,
                )
                for dy, dx in offsets
            ]
        )
        low = np.where(candidates > 0, candidates, np.iinfo(np.int32).max).min(axis=0)
        unambiguous = low == candidates.max(axis=0)
    for dy, dx in offsets:
        neighbor = source[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        fill = missing & ~repaired & np.isfinite(neighbor)
        if ids is not None:
            neighbor_ids = source_ids[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            fill &= (neighbor_ids > 0) & ((ids == neighbor_ids) | ((ids == 0) & unambiguous))
            ids[fill] = neighbor_ids[fill]
        result[fill] = neighbor[fill]
        repaired |= fill
    return ObjectLayer(np.asarray(rgba), result, repaired, ids)


def _linear(rgb):
    value = np.asarray(rgb, dtype=float) / 255
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _srgb(linear):
    value = np.clip(linear, 0, 1)
    srgb = np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)
    return np.rint(srgb * 255).astype(np.uint8)


def repair_background(background, mask, *, max_component_pixels=4096, max_component_span=128):
    """Bounded appearance-only harmonic fill; preserves every pixel outside accepted masks."""
    rgb, requested = np.asarray(background), np.asarray(mask, dtype=bool)
    if rgb.shape != (*requested.shape, 3) or requested.ndim != 2:
        raise ValueError("RGB background and two-dimensional mask must match")
    output, remaining, repaired = rgb.copy(), requested.copy(), np.zeros_like(requested)
    height, width = requested.shape
    for sy, sx in zip(*np.where(requested)):
        if not remaining[sy, sx]:
            continue
        pending, points = [(int(sy), int(sx))], []
        remaining[sy, sx] = False
        while pending:
            y, x = pending.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width and remaining[yy, xx]:
                        remaining[yy, xx] = False
                        pending.append((yy, xx))
        coordinates = np.asarray(points)
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0) + 1
        if (
            len(points) > max_component_pixels
            or max(y1 - y0, x1 - x0) > max_component_span
            or y0 == 0
            or x0 == 0
            or y1 == height
            or x1 == width
        ):
            continue
        local = np.zeros((y1 - y0 + 2, x1 - x0 + 2), dtype=bool)
        local[coordinates[:, 0] - y0 + 1, coordinates[:, 1] - x0 + 1] = True
        if np.any(requested[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] & ~local):
            continue
        work = _linear(rgb[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1])
        work[local] = work[~local].mean(axis=0)
        inside = local[1:-1, 1:-1]
        for _ in range(512):
            average = (work[:-2, 1:-1] + work[2:, 1:-1] + work[1:-1, :-2] + work[1:-1, 2:]) / 4
            change = float(np.max(np.abs(work[1:-1, 1:-1][inside] - average[inside])))
            work[1:-1, 1:-1][inside] = average[inside]
            if change < 1e-5:
                break
        fill = _srgb(work[1:-1, 1:-1])
        output[y0:y1, x0:x1][inside] = fill[inside]
        repaired[y0:y1, x0:x1] |= inside
    return output, {"repaired": repaired, "unrepaired": requested & ~repaired}


def marker_repair_mask(layer, scene_depth, scene_alpha, radius=3, margin_m=0.05):
    """Bound silhouette dilation to four pixels and apply the original scene-depth test."""
    if not isinstance(radius, int) or not 0 <= radius <= 4:
        raise ValueError("Marker repair radius must be an integer from zero to four pixels")
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("Depth margin must be finite and nonnegative")
    depth, alpha = np.asarray(scene_depth), np.asarray(scene_alpha)
    if depth.shape != layer.depth.shape or alpha.shape != depth.shape:
        raise ValueError("Object and scene depth/alpha dimensions must match")
    support = (layer.rgba[..., 3] > 0) & np.isfinite(layer.depth) & (layer.depth > 0)
    projected = np.where(support, layer.depth, np.nan)
    height, width = projected.shape
    for _ in range(radius):
        source = np.pad(projected, 1, constant_values=np.nan)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            neighbor = source[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            fill = ~np.isfinite(projected) & np.isfinite(neighbor)
            projected[fill] = neighbor[fill]
    return (
        np.isfinite(projected)
        & np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(alpha)
        & (alpha >= 0.5)
        & (projected < depth - margin_m)
    )


def shadow_radius(obj):
    """Radial contact support; the long barrier footprint uses separate local axes."""
    if obj.shadow_strength == 0:
        return 0.0
    if obj.kind in ("dome_marker", "traffic_barrel"):
        return obj.diameter_m * obj.scale / 2
    if obj.kind == "cone" and obj.shadow_strength is not None:
        return max(CONE_SIZE[:2]) * obj.scale / 2
    if obj.kind in ("billboard", "barrier", "jersey_barrier"):
        return 0.8 * math.hypot(
            obj.width_m * obj.scale / 2 + 0.12,
            (obj.depth_m or 0.35) * obj.scale / 2 + 0.12,
        )
    return 0.0


def contact_shadows(
    background,
    objects,
    ground_points,
    ground_visible,
    *,
    strength=0.30,
    ground_normal=(0.0, 0.0, 1.0),
):
    """Multiply linear RGB by bounded soft dome shadows on explicitly visible ground."""
    rgb = np.asarray(background)
    points, allowed = np.asarray(ground_points), np.asarray(ground_visible, dtype=bool)
    if points.shape != rgb.shape or rgb.shape != (*allowed.shape, 3):
        raise ValueError("RGB, map ground points, and ground visibility dimensions must match")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Contact shadow strength must be between zero and one")
    supported = allowed & np.isfinite(points).all(axis=-1)
    if not supported.all():
        result, opacity = rgb.copy(), np.zeros(allowed.shape, dtype=np.float32)
        if supported.any():
            pixels, shadow = contact_shadows(
                rgb[supported][:, None],
                objects,
                points[supported][:, None],
                np.ones((int(supported.sum()), 1), bool),
                strength=strength,
                ground_normal=ground_normal,
            )
            result[supported], opacity[supported] = pixels[:, 0], shadow[:, 0]
        return result, opacity
    projected_points = None
    transmission = np.ones(allowed.shape)
    for value in objects:
        obj = value if isinstance(value, PlacedObject) else PlacedObject(**value)
        radius = shadow_radius(obj)
        if radius == 0:
            continue
        if obj.kind in ("dome_marker", "traffic_barrel", "cone"):
            distance_squared = np.sum((points - [obj.x, obj.y, obj.z]) ** 2, axis=-1)
            u = distance_squared / (2 * radius) ** 2
        else:
            from drivingbench.sim.visual_ground import road_rotation

            if projected_points is None:
                basis = road_rotation(ground_normal, 0)
                projected_points = points @ basis
            delta = projected_points - np.asarray([obj.x, obj.y, obj.z]) @ basis
            c, s = math.cos(obj.yaw), math.sin(obj.yaw)
            along = delta[..., 0] * c + delta[..., 1] * s
            across = -delta[..., 0] * s + delta[..., 1] * c
            half_width = obj.width_m * obj.scale / 2 + 0.12
            half_depth = (obj.depth_m or 0.35) * obj.scale / 2 + 0.12
            u = (along / (half_width * 1.6)) ** 2 + (across / (half_depth * 1.6)) ** 2
        inside = supported & (u < 1)
        kernel = np.zeros(u.shape)
        kernel[inside] = np.expm1(4 * (1 - u[inside])) / np.expm1(4)
        opacity = strength if obj.shadow_strength is None else obj.shadow_strength
        transmission *= 1 - opacity * kernel
    opacity = 1 - transmission
    return _srgb(_linear(rgb) * transmission[..., None]), opacity.astype(np.float32)


def composite_objects(
    background, scene_depth, scene_alpha, layer, margin_m=0.05, *, instance_masks=None
):
    """Test each object's depth before back-to-front linear-light alpha compositing."""
    background = np.asarray(background)
    depth, alpha = np.asarray(scene_depth), np.asarray(scene_alpha)
    layers = list(layer) if isinstance(layer, (list, tuple)) else [layer]
    shape = background.shape[:2]
    if (
        background.shape != (*shape, 3)
        or depth.shape != shape
        or alpha.shape != shape
        or not layers
        or any(value.rgba.shape != (*shape, 4) or value.depth.shape != shape for value in layers)
    ):
        raise ValueError("Background, object layer, depth, and opacity dimensions must match")
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("Depth margin must be finite and nonnegative")
    selected = np.logical_or.reduce([value.rgba[..., 3] > 0 for value in layers])
    if not selected.all():
        result = background.copy()
        masks = {name: np.zeros(shape, bool) for name in ("visible", "occluded", "unknown")}
        if selected.any():
            subset = [
                ObjectLayer(
                    value.rgba[selected][:, None],
                    value.depth[selected][:, None],
                    instance=None if value.instance is None else value.instance[selected][:, None],
                )
                for value in layers
            ]
            selected_masks = {} if instance_masks is not None else None
            pixels, chosen = _composite_layers(
                background[selected][:, None],
                depth[selected][:, None],
                alpha[selected][:, None],
                subset,
                margin_m,
                instance_masks=selected_masks,
            )
            if instance_masks is not None:
                for identity, mask in selected_masks.items():
                    full = np.zeros(shape, bool)
                    full[selected] = mask[:, 0]
                    instance_masks[identity] = full
            result[selected] = pixels[:, 0]
            for name, mask in masks.items():
                mask[selected] = chosen[name][:, 0]
            if "instances" in chosen:
                masks["instances"] = chosen["instances"]
        return result, masks
    return _composite_layers(
        background, depth, alpha, layers, margin_m, instance_masks=instance_masks
    )


def _composite_layers(background, depth, alpha, layers, margin_m, *, instance_masks=None):
    shape = background.shape[:2]
    present = np.stack([value.rgba[..., 3] > 0 for value in layers])
    z = np.stack([value.depth for value in layers])
    known = (
        np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(alpha)
        & (alpha >= 0.5)
        & np.isfinite(z)
        & (z > 0)
    )
    visible = present & known & (z < depth - margin_m)
    rgba = np.stack([value.rgba for value in layers])
    weights = rgba[..., 3].astype(float) / 255 * visible
    order = np.argsort(np.where(visible, z, np.inf), axis=0, kind="stable")
    linear = _linear(background)
    for rank in order[::-1]:
        color = np.take_along_axis(rgba[..., :3], rank[None, ..., None], axis=0)[0]
        weight = np.take_along_axis(weights, rank[None], axis=0)[0][..., None]
        linear = _linear(color) * weight + linear * (1 - weight)
    masks = {
        "visible": visible.any(axis=0),
        "occluded": (present & known & ~visible).any(axis=0),
        "unknown": (present & ~known).any(axis=0),
    }
    instances = {}
    if any(value.instance is not None and np.any(value.instance < 0) for value in layers):
        transmission = np.ones(shape)
        contributes = np.zeros(visible.shape, dtype=bool)
        for rank in order:
            weight = np.take_along_axis(weights, rank[None], axis=0)[0]
            np.put_along_axis(contributes, rank[None], (weight * transmission > 0)[None], axis=0)
            transmission *= 1 - weight
        for index, value in enumerate(layers):
            if value.instance is None:
                continue
            for identity in np.unique(value.instance[value.instance < 0]):
                selected = value.instance == identity
                if instance_masks is not None:
                    instance_masks[int(-identity - 1)] = selected & contributes[index]
                foreground_hidden = visible[index] & ~contributes[index]
                instances[int(-identity - 1)] = {
                    "projected_pixels": int(selected.sum()),
                    "visible_pixels": int((selected & contributes[index]).sum()),
                    "occluded_pixels": int(
                        (selected & ((known[index] & ~visible[index]) | foreground_hidden)).sum()
                    ),
                    "unknown_pixels": int((selected & ~known[index]).sum()),
                }
    if instances:
        masks["instances"] = instances
    return _srgb(linear), masks


def composite_grounded(
    background,
    scene_depth,
    scene_alpha,
    layer,
    objects,
    ground_points,
    ground_depth,
    ground_region,
    *,
    foreground_margin_m=0.5,
    contact_tolerance_m=0.05,
    shadow_strength=0.30,
    ground_normal=(0.0, 0.0, 1.0),
    instance_masks=None,
):
    """Use calibrated asphalt in its declared region while retaining nearer scene geometry."""
    raw, alpha = np.asarray(scene_depth), np.asarray(scene_alpha)
    plane, region = np.asarray(ground_depth), np.asarray(ground_region, dtype=bool)
    if raw.shape != plane.shape or region.shape != plane.shape or alpha.shape != raw.shape:
        raise ValueError("Scene/ground depth, opacity, and ground region dimensions must match")
    if (
        not np.isfinite([foreground_margin_m, contact_tolerance_m]).all()
        or min(foreground_margin_m, contact_tolerance_m) < 0
    ):
        raise ValueError("Ground depth margins must be finite and nonnegative")
    supported = (
        region
        & np.isfinite(plane)
        & (plane > 0)
        & np.isfinite(raw)
        & (raw > 0)
        & np.isfinite(alpha)
        & (alpha >= 0.5)
    )
    foreground = supported & (raw < plane - foreground_margin_m)
    ground_visible = supported & ~foreground
    shaded, shadow = contact_shadows(
        background,
        objects,
        ground_points,
        ground_visible,
        strength=shadow_strength,
        ground_normal=ground_normal,
    )
    effective = np.where(ground_visible, plane + contact_tolerance_m, raw - 0.05)
    result, masks = composite_objects(
        shaded,
        effective,
        alpha,
        layer,
        margin_m=0,
        **({"instance_masks": instance_masks} if instance_masks is not None else {}),
    )
    return result, masks | {
        "ground_visible": ground_visible,
        "foreground": foreground,
        "ground_unknown": region & ~supported,
        "shadow_opacity": shadow,
    }


def blender_depth(pixels):
    """Cycles Z is camera-forward metres; Blender image storage starts at the bottom."""
    depth = np.asarray(pixels, dtype=np.float32)[::-1].copy()
    depth[(depth <= 0) | (depth > 1000)] = np.nan
    return depth


def instance_moments(mean, second, count):
    """Reject mixed labels, including mixtures whose mean is another valid integer ID."""
    mean, second = np.asarray(mean, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if mean.shape != second.shape:
        raise ValueError("Instance moments must have matching dimensions")
    nearest = np.rint(mean)
    known = (
        np.isfinite(mean)
        & np.isfinite(second)
        & (nearest > 0)
        & (nearest <= count)
        & (np.abs(mean - nearest) < 1e-5)
        & (np.abs(second - mean**2) < 1e-3)
    )
    return np.where(known, nearest, 0).astype(np.int32)


def asset_normalization(lower, upper):
    """Map an asset's combined bounds to centered unit XY and ground-contact Z=0."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite([lower, upper]).all()
        or np.any(upper <= lower)
    ):
        raise ValueError("Asset bounds must have finite positive extents")
    matrix = np.diag([*(1 / (upper - lower)), 1.0])
    matrix[:3, 3] = -np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]]) / (
        upper - lower
    )
    return matrix


def pinhole_camera(camera):
    """Normalize a finite pinhole camera without assuming centered or square pixels."""
    values = {key: camera[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Camera intrinsics must be finite")
    if (
        any(
            isinstance(values[key], bool) or values[key] <= 0 or int(values[key]) != values[key]
            for key in ("w", "h")
        )
        or min(values["fl_x"], values["fl_y"]) <= 0
    ):
        raise ValueError("Camera dimensions must be positive integers and focal lengths positive")
    return {key: int(value) if key in ("w", "h") else float(value) for key, value in values.items()}


def blender_camera_settings(camera):
    """Translate OpenCV intrinsics to Blender's horizontal sensor-fit convention."""
    camera = pinhole_camera(camera)
    width, height = camera["w"], camera["h"]
    ratio = camera["fl_x"] / camera["fl_y"]
    return {
        "resolution_x": width,
        "resolution_y": height,
        "pixel_aspect_x": max(1.0, 1 / ratio),
        "pixel_aspect_y": max(1.0, ratio),
        "lens": camera["fl_x"] * 36 / width,
        "shift_x": (width / 2 - camera["cx"]) / width,
        "shift_y": (camera["cy"] - height / 2) * ratio / width,
    }


def _blender_camera(scene, camera):
    settings = blender_camera_settings(camera)
    for key in ("resolution_x", "resolution_y", "pixel_aspect_x", "pixel_aspect_y"):
        setattr(scene.render, key, settings[key])
    scene.render.resolution_percentage = 100
    data = scene.camera.data
    data.type, data.sensor_fit, data.sensor_width = "PERSP", "HORIZONTAL", 36
    for key in ("lens", "shift_x", "shift_y"):
        setattr(data, key, settings[key])


def _lathe(profile, segments=64, cap=False):
    vertices = [
        (r * math.cos(a * 2 * math.pi / segments), r * math.sin(a * 2 * math.pi / segments), z)
        for r, z in profile
        for a in range(segments)
    ]
    faces = [
        (
            ring * segments + a,
            ring * segments + (a + 1) % segments,
            (ring + 1) * segments + (a + 1) % segments,
            (ring + 1) * segments + a,
        )
        for ring in range(len(profile) - 1)
        for a in range(segments)
    ]
    if cap:
        faces.append(tuple(range((len(profile) - 1) * segments, len(vertices))))
    return vertices, faces


def dome_mesh(diameter_m, height_m):
    """Rounded disc-cone profile with a thin outer lip and small central opening."""
    profile = (
        (0.50, 0),
        (0.50, 0.04),
        (0.44, 0.12),
        (0.36, 0.38),
        (0.25, 0.75),
        (0.14, 0.96),
        (0.075, 1),
    )
    return _lathe([(radius * diameter_m, z * height_m) for radius, z in profile])


def jersey_mesh(width_m, depth_m, height_m):
    """Closed tapered barrier with its measured rectangular base on z=0."""
    if not np.isfinite([width_m, depth_m, height_m]).all() or min(width_m, depth_m, height_m) <= 0:
        raise ValueError("Jersey dimensions must be finite and positive")
    section = [
        (-0.5, 0),
        (0.5, 0),
        (0.5, 0.11),
        (0.29, 0.40),
        (0.12, 1),
        (-0.12, 1),
        (-0.29, 0.40),
        (-0.5, 0.11),
    ]
    vertices = [
        (x, y * depth_m, z * height_m) for x in (-width_m / 2, width_m / 2) for y, z in section
    ]
    count = len(section)
    faces = [tuple(reversed(range(count))), tuple(range(count, count * 2))]
    faces += [(i, (i + 1) % count, (i + 1) % count + count, i + count) for i in range(count)]
    return vertices, faces


def barrel_mesh(diameter_m, height_m, segments=64):
    """Closed molded traffic drum; its broad rubber foot defines the collision diameter."""
    if not np.isfinite([diameter_m, height_m]).all() or min(diameter_m, height_m) <= 0:
        raise ValueError("Barrel dimensions must be finite and positive")
    profile = [
        (0.50, 0),
        (0.50, 0.045),
        (0.45, 0.06),
        (0.44, 0.14),
        (0.46, 0.15),
        (0.46, 0.18),
        (0.435, 0.20),
        (0.425, 0.30),
        (0.415, 0.40),
        (0.405, 0.50),
        (0.395, 0.60),
        (0.385, 0.70),
        (0.375, 0.78),
        (0.395, 0.79),
        (0.395, 0.82),
        (0.35, 0.85),
        (0.30, 0.875),
    ]
    vertices, faces = _lathe(
        [(r * diameter_m, z * height_m) for r, z in profile], segments=segments, cap=True
    )
    faces.append(tuple(reversed(range(segments))))
    return vertices, faces


def environment_lighting(lighting, ground_normal):
    """Declare an LDR panorama in calibrated-ground axes, without reading external files."""
    from drivingbench.sim.visual_ground import road_rotation

    if lighting is None:
        return None
    allowed = {"environment_path", "strength", "ao_distance_m", "bake_irradiance"}
    if not isinstance(lighting, dict) or lighting.keys() - allowed:
        raise ValueError("Unknown environment lighting fields")
    path = lighting.get("environment_path")
    if (
        not isinstance(path, str)
        or not path
        or "://" in path
        or Path(path).suffix.lower() != ".png"
    ):
        raise ValueError("Environment lighting requires a local LDR PNG environment_path")
    strength, distance = lighting.get("strength", 1.0), lighting.get("ao_distance_m", 0.0)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in (strength, distance)
    ):
        raise ValueError("Environment strength and AO distance must be finite and nonnegative")
    bake = lighting.get("bake_irradiance", False)
    if not isinstance(bake, bool):
        raise ValueError("bake_irradiance must be boolean")
    lookup = np.diag([1, -1, 1]) @ road_rotation(ground_normal, 0).T
    return dict(
        environment_path=path,
        strength=float(strength),
        ao_distance_m=float(distance),
        bake_irradiance=bake,
        environment_lookup_matrix=lookup.tolist(),
    )


def _blender_lighting(job):
    import bpy
    from mathutils import Matrix

    lighting = job.get("lighting")
    scene = bpy.context.scene
    if lighting is None:
        scene["drivingbench_lighting"] = "null"
        return
    nodes, links = scene.world.node_tree.nodes, scene.world.node_tree.links
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(lighting["environment_path"], check_existing=True)
    environment.image.colorspace_settings.name = "sRGB"
    environment.projection = "EQUIRECTANGULAR"
    coordinates = nodes.new("ShaderNodeNewGeometry")
    vector = nodes.new("ShaderNodeCombineXYZ")
    for row, axis in zip(lighting["environment_lookup_matrix"], "XYZ"):
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = row
        links.new(coordinates.outputs["Position"], dot.inputs[0])
        links.new(dot.outputs["Value"], vector.inputs[axis])
    links.new(vector.outputs[0], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], nodes["Background"].inputs["Color"])
    nodes["Background"].inputs["Strength"].default_value = lighting["strength"]
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    materials = {slot.material for obj in meshes for slot in obj.material_slots if slot.material}
    ao_count = 0
    if lighting["ao_distance_m"] > 0:
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = lighting["ao_distance_m"]
        for material in materials:
            if not material.use_nodes:
                continue
            graph = material.node_tree
            for bsdf in list(graph.nodes):
                if bsdf.bl_idname != "ShaderNodeBsdfPrincipled":
                    continue
                ao_count += 1
                base = bsdf.inputs["Base Color"]
                ao = graph.nodes.new("ShaderNodeAmbientOcclusion")
                ao.samples = 16
                ao.inputs["Distance"].default_value = lighting["ao_distance_m"]
                ao.inputs["Color"].default_value = base.default_value
                if base.links:
                    graph.links.new(base.links[0].from_socket, ao.inputs["Color"])
                graph.links.new(ao.outputs["Color"], base)
    status = dict(
        lighting,
        engine=scene.render.engine,
        irradiance_status="not_requested",
        bake_cache="none",
        material_ao_count=ao_count,
    )
    if lighting["bake_irradiance"]:
        if not meshes:
            status["irradiance_status"] = "not_needed_no_static_mesh"
        elif scene.render.engine != "BLENDER_EEVEE_NEXT":
            status["irradiance_status"] = "cycles_direct_environment"
        else:
            bpy.context.view_layer.update()
            basis = np.diag([1, -1, 1]) @ np.asarray(lighting["environment_lookup_matrix"])
            points = []
            for obj in meshes:
                matrix = np.asarray(obj.matrix_world)
                points.extend(np.asarray(obj.bound_box) @ matrix[:3, :3].T + matrix[:3, 3])
            points = np.asarray(points) @ basis.T
            lower, upper = points.min(axis=0) - 1, points.max(axis=0) + 1
            data = bpy.data.lightprobes.new("DrivingBench environment irradiance", "VOLUME")
            resolution = np.clip(np.ceil((upper - lower) / 8).astype(int) + 1, 2, 8)
            data.resolution_x, data.resolution_y, data.resolution_z = map(int, resolution)
            data.bake_samples = 128
            data.capture_world = True
            probe = bpy.data.objects.new("DrivingBench environment irradiance", data)
            scene.collection.objects.link(probe)
            probe.location = basis.T @ ((lower + upper) / 2)
            probe.rotation_euler = Matrix(basis.T).to_euler()
            probe.scale = (upper - lower) / 2
            bpy.ops.object.select_all(action="DESELECT")
            probe.select_set(True)
            bpy.context.view_layer.objects.active = probe
            bpy.context.view_layer.update()
            if "FINISHED" not in bpy.ops.object.lightprobe_cache_bake(subset="SELECTED"):
                raise RuntimeError("Requested environment irradiance bake did not finish")
            status.update(
                irradiance_status="baked",
                bake_cache="in_memory_worker_lifetime",
                bake_resolution=resolution.tolist(),
                bake_samples=128,
            )
    scene["drivingbench_lighting"] = json.dumps(status)


def blender_lighting_status():
    import bpy

    return json.loads(bpy.context.scene.get("drivingbench_lighting", "null"))


def object_renderer_settings(renderer=None):
    """Validate the optional static-object engine before launching a worker."""
    if renderer is None:
        renderer = {}
    if not isinstance(renderer, dict) or renderer.keys() - {"engine", "samples"}:
        raise ValueError("Object renderer accepts only engine and samples")
    engine = renderer.get("engine", "CYCLES")
    if engine not in ("CYCLES", "BLENDER_EEVEE_NEXT"):
        raise ValueError("Object engine must be CYCLES or BLENDER_EEVEE_NEXT")
    samples = renderer.get("samples", 16 if engine == "BLENDER_EEVEE_NEXT" else 128)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 256:
        raise ValueError("Object samples must be an integer between one and 256")
    return {"engine": engine, "samples": samples}


def _validate_object_ids(objects, engine):
    if engine == "BLENDER_EEVEE_NEXT" and any(
        index > 45 and obj["kind"] not in ("billboard", "ground_decal")
        for index, obj in enumerate(objects, 1)
    ):
        raise ValueError(
            "EEVEE depth IDs support static instance IDs 1–45: squared IDs exceed the "
            "guaranteed exact float16 range beyond this limit. Use CYCLES for this job."
        )


def _blender_material_saturation(objects, saturation):
    """Mix only imported base color with its linear luminance on private material copies."""
    if saturation is None or saturation == 1:
        return
    copies = {}
    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            source = slot.material
            if source not in copies:
                if source is None or not source.use_nodes:
                    raise ValueError("Material saturation requires a Principled node material")
                material = source.copy()
                graph = material.node_tree
                shaders = [n for n in graph.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"]
                if not shaders:
                    raise ValueError("Material saturation requires a Principled node material")
                for shader in shaders:
                    base = shader.inputs["Base Color"]
                    gray = graph.nodes.new("ShaderNodeRGBToBW")
                    blend = graph.nodes.new("ShaderNodeMixRGB")
                    blend.blend_type = "MIX"
                    blend.inputs[0].default_value = saturation
                    gray.inputs[0].default_value = base.default_value
                    blend.inputs[2].default_value = base.default_value
                    if base.links:
                        original = base.links[0].from_socket
                        graph.links.new(original, gray.inputs[0])
                        graph.links.new(original, blend.inputs[2])
                    graph.links.new(gray.outputs[0], blend.inputs[1])
                    graph.links.new(blend.outputs[0], base)
                copies[source] = material
            slot.link, slot.material = "OBJECT", copies[source]


def _blender_scene(job):
    settings = object_renderer_settings({k: job[k] for k in ("engine", "samples") if k in job})
    _validate_object_ids(job["objects"], settings["engine"])
    import bpy
    from mathutils import Matrix

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    engine, samples = settings["engine"], settings["samples"]
    requested_device = job.get("device", "METAL")
    if requested_device not in ("METAL", "CPU"):
        raise ValueError("Object device must be METAL or CPU")
    if requested_device == "CPU" and engine == "BLENDER_EEVEE_NEXT":
        engine, samples = "CYCLES", 128
    scene.render.engine = engine
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.device = "CPU"
    scene.render.use_persistent_data = engine == "CYCLES"
    if engine == "BLENDER_EEVEE_NEXT":
        scene.eevee.taa_render_samples = samples
    elif requested_device == "METAL":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == "METAL"
        if any(d.type == "METAL" for d in prefs.devices):
            scene.cycles.device = "GPU"
    camera = job["camera"]
    width, height = camera["w"], camera["h"]
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.compression = 0
    scene.view_settings.view_transform = "Standard"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.7, 0.72, 0.76, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.8

    def material(name, color, roughness):
        result = bpy.data.materials.new(name)
        result.use_nodes = True
        nodes, links = result.node_tree.nodes, result.node_tree.links
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        bsdf.inputs["Roughness"].default_value = roughness
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 180
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.12
        bump.inputs["Distance"].default_value = 0.0005
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        return result

    orange = material("weathered orange PVC", (0.95, 0.105, 0.012), 0.4)
    white = material("reflective white collar", (0.8, 0.82, 0.79), 0.3)
    rubber = material("rubber foot", (0.014, 0.017, 0.017), 0.86)

    def cone():
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.02))
        base = bpy.context.object
        base.name = "beveled rubber base"
        base.dimensions = (*CONE_SIZE[:2], 0.04)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        base.data.materials.append(rubber)
        bevel = base.modifiers.new("rounded base corners", "BEVEL")
        bevel.width, bevel.segments = 0.012, 3
        heights = (0.04, 0.27, 0.38, 0.55)
        segments = 64
        vertices, faces = _lathe(
            [(0.145 + (0.022 - 0.145) * (z - 0.04) / 0.51, z) for z in heights], cap=True
        )
        mesh = bpy.data.meshes.new("cone shell")
        mesh.from_pydata(vertices, [], faces)
        shell = bpy.data.objects.new("tapered cone", mesh)
        bpy.context.collection.objects.link(shell)
        mesh.materials.append(orange)
        mesh.materials.append(white)
        for polygon in mesh.polygons:
            polygon.material_index = int(segments <= polygon.index < 2 * segments)
            polygon.use_smooth = True
        return [base, shell]

    def dome(spec):
        vertices, faces = dome_mesh(spec["diameter_m"], spec["height_m"])
        mesh = bpy.data.meshes.new("dome marker shell")
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new("dome marker", mesh)
        bpy.context.collection.objects.link(obj)
        mesh.materials.append(
            material("marker plastic", _linear(np.asarray(spec["color_srgb"]) * 255), 0.55)
        )
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        shell = obj.modifiers.new("plastic thickness", "SOLIDIFY")
        shell.thickness = spec["diameter_m"] * 0.012
        shell.offset = -1
        return [obj]

    def block_surface(color, *, paint=False, segmented=False, texture_path=None):
        surface = material(
            "worn lane paint" if paint else "weathered concrete",
            _linear(np.asarray(color) * 255),
            0.88,
        )
        nodes, links = surface.node_tree.nodes, surface.node_tree.links
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 5
        ramp = nodes.new("ShaderNodeValToRGB")
        base = _linear(np.asarray(color) * 255)
        ramp.color_ramp.elements[0].color = (*(base * 0.65), 1)
        ramp.color_ramp.elements[1].color = (*base, 1)
        links.new(noise.outputs["Fac"], ramp.inputs[0])
        links.new(ramp.outputs[0], nodes["Principled BSDF"].inputs["Base Color"])
        if not paint:
            coordinates = nodes.new("ShaderNodeTexCoord")
            links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
            noise.inputs["Scale"].default_value = 2.5
            ramp.color_ramp.elements[0].color = (*(base * 0.7), 1)
            ramp.color_ramp.elements[1].color = (*(base * 1.05), 1)
            grain = nodes.new("ShaderNodeTexNoise")
            grain.inputs["Scale"].default_value = 55
            grain.inputs["Detail"].default_value = 3
            links.new(coordinates.outputs["Object"], grain.inputs["Vector"])
            roughness = nodes.new("ShaderNodeMapRange")
            roughness.inputs["To Min"].default_value = 0.78
            roughness.inputs["To Max"].default_value = 0.94
            links.new(grain.outputs["Fac"], roughness.inputs["Value"])
            links.new(roughness.outputs["Result"], nodes["Principled BSDF"].inputs["Roughness"])
            axis = nodes.new("ShaderNodeSeparateXYZ")
            links.new(coordinates.outputs["Object"], axis.inputs[0])
            repeat = nodes.new("ShaderNodeMath")
            repeat.operation = "MULTIPLY"
            repeat.inputs[1].default_value = 1 / 3
            links.new(axis.outputs["X"], repeat.inputs[0])
            fraction = nodes.new("ShaderNodeMath")
            fraction.operation = "FRACT"
            links.new(repeat.outputs[0], fraction.inputs[0])
            joint = nodes.new("ShaderNodeMath")
            joint.operation = "LESS_THAN"
            joint.inputs[1].default_value = 0 if segmented else 0.008
            links.new(fraction.outputs[0], joint.inputs[0])
            stain = nodes.new("ShaderNodeMixRGB")
            stain.blend_type = "MULTIPLY"
            stain.inputs[2].default_value = (0.65, 0.65, 0.65, 1)
            links.new(joint.outputs[0], stain.inputs[0])
            links.new(ramp.outputs[0], stain.inputs[1])
            links.new(stain.outputs[0], nodes["Principled BSDF"].inputs["Base Color"])
            recess = nodes.new("ShaderNodeMath")
            recess.operation = "MULTIPLY_ADD"
            recess.inputs[1].default_value = -0.35
            links.new(joint.outputs[0], recess.inputs[0])
            links.new(grain.outputs["Fac"], recess.inputs[2])
            bump = nodes["Bump"]
            bump.inputs["Strength"].default_value = 0.35
            bump.inputs["Distance"].default_value = 0.003
            links.new(recess.outputs[0], bump.inputs["Height"])
            if segmented:
                ramp.color_ramp.elements[0].position = 0.30
                ramp.color_ramp.elements[1].position = 0.70
                ramp.color_ramp.elements[0].color = (*(base * 0.50), 1)
                ramp.color_ramp.elements[1].color = (*(base * 1.25), 1)
                foot_dirt = nodes.new("ShaderNodeMapRange")
                foot_dirt.inputs["From Min"].default_value = 0
                foot_dirt.inputs["From Max"].default_value = 0.22
                foot_dirt.inputs["To Min"].default_value = 0.62
                foot_dirt.inputs["To Max"].default_value = 1
                links.new(axis.outputs["Z"], foot_dirt.inputs["Value"])
                weathered = nodes.new("ShaderNodeMixRGB")
                weathered.blend_type = "MULTIPLY"
                weathered.inputs[0].default_value = 1
                links.new(stain.outputs[0], weathered.inputs[1])
                links.new(foot_dirt.outputs["Result"], weathered.inputs[2])
                links.new(weathered.outputs[0], nodes["Principled BSDF"].inputs["Base Color"])
                if texture_path:
                    texture = nodes.new("ShaderNodeTexImage")
                    texture.image = bpy.data.images.load(texture_path, check_existing=True)
                    texture.image.colorspace_settings.name = "sRGB"
                    texture.projection = "BOX"
                    texture.projection_blend = 0.2
                    texture.extension = "REPEAT"
                    links.new(coordinates.outputs["Object"], texture.inputs["Vector"])
                    links.new(texture.outputs["Color"], weathered.inputs[1])
                    foot_dirt.inputs["To Min"].default_value = 0.9
                random = nodes.new("ShaderNodeObjectInfo")
                for texture in (noise, grain):
                    texture.noise_dimensions = "4D"
                    links.new(random.outputs["Random"], texture.inputs["W"])
        return surface

    def block(spec):
        paint = spec["kind"] == "paint"
        height = 0.005 if paint else spec["height_m"]
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, height / 2))
        obj = bpy.context.object
        obj.name = "worn lane paint" if paint else "weathered concrete barrier"
        obj.dimensions = (spec["width_m"], spec["depth_m"], height)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        color = spec.get("color_srgb") or ((0.83, 0.81, 0.7) if paint else (0.55, 0.54, 0.51))
        obj.data.materials.append(block_surface(color, paint=paint))
        if not paint:
            bevel = obj.modifiers.new("worn concrete edges", "BEVEL")
            bevel.width, bevel.segments = min(0.06, height / 8), 3
        return [obj]

    def jersey(spec):
        vertices, faces = jersey_mesh(spec["width_m"], spec["depth_m"], spec["height_m"])
        mesh = bpy.data.meshes.new("tapered jersey concrete")
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new("segmented jersey barrier", mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(
            block_surface(
                spec.get("color_srgb") or (0.58, 0.57, 0.54),
                segmented=True,
                texture_path=spec.get("texture_path"),
            )
        )
        bevel = obj.modifiers.new("cast concrete edge wear", "BEVEL")
        bevel.width, bevel.segments = min(0.012, spec["depth_m"] / 20), 2
        return [obj]

    def barrel(spec):
        vertices, faces = barrel_mesh(spec["diameter_m"], spec["height_m"])
        mesh = bpy.data.meshes.new("molded traffic drum")
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new("orange traffic barrel", mesh)
        bpy.context.collection.objects.link(obj)
        body = material(
            "faded orange drum plastic",
            _linear(np.array(spec.get("color_srgb") or (0.84, 0.27, 0.055)) * 255),
            0.67,
        )
        reflector = material(
            "scuffed reflective drum bands", _linear(np.array([0.76, 0.75, 0.69]) * 255), 0.53
        )
        for surface in (body, reflector):
            nodes, links = surface.node_tree.nodes, surface.node_tree.links
            coordinates = nodes.new("ShaderNodeTexCoord")
            dirt = nodes.new("ShaderNodeTexNoise")
            dirt.inputs["Scale"].default_value = 8
            dirt.inputs["Detail"].default_value = 3
            links.new(coordinates.outputs["Object"], dirt.inputs["Vector"])
            wear = nodes.new("ShaderNodeValToRGB")
            color = tuple(nodes["Principled BSDF"].inputs["Base Color"].default_value)
            wear.color_ramp.elements[0].color = tuple(c * 0.74 for c in color[:3]) + (1,)
            wear.color_ramp.elements[1].color = color
            links.new(dirt.outputs["Fac"], wear.inputs[0])
            links.new(wear.outputs[0], nodes["Principled BSDF"].inputs["Base Color"])

        mesh.materials.append(body)
        mesh.materials.append(reflector)
        mesh.materials.append(rubber)
        for polygon in mesh.polygons:
            z = (
                sum(mesh.vertices[i].co.z for i in polygon.vertices)
                / len(polygon.vertices)
                / spec["height_m"]
            )
            polygon.material_index = 2 if z < 0.06 else int(0.30 <= z < 0.40 or 0.60 <= z < 0.70)
            polygon.use_smooth = polygon.material_index != 2
        height = spec["height_m"]
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.05 * height,
            minor_radius=0.0125 * height,
            location=(0, 0, 0.9375 * height),
            rotation=(math.pi / 2, 0, 0),
            major_segments=32,
            minor_segments=8,
        )
        handle = bpy.context.object
        handle.name = "traffic drum carry handle"
        handle.data.materials.append(body)
        return [obj, handle]

    asset_cache = {}

    def construction_asset(spec):
        path = spec["asset_path"]
        if path not in asset_cache:
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=path)
            imported = set(bpy.data.objects) - before
            bpy.context.view_layer.update()
            graph = bpy.context.evaluated_depsgraph_get()
            meshes, positions = [], []
            for obj in imported:
                if obj.type != "MESH":
                    continue
                mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(graph))
                mesh.transform(obj.matrix_world)
                vertices = np.empty(len(mesh.vertices) * 3)
                mesh.vertices.foreach_get("co", vertices)
                positions.append(vertices.reshape(-1, 3))
                meshes.append((obj.name, mesh))
            if not positions:
                raise ValueError("Construction GLB contains no mesh")
            vertices = np.concatenate(positions)
            normalization = Matrix(
                asset_normalization(vertices.min(axis=0), vertices.max(axis=0)).tolist()
            )
            for _, mesh in meshes:
                mesh.transform(normalization)
            for obj in imported:
                bpy.data.objects.remove(obj, do_unlink=True)
            asset_cache[path] = meshes
        dimensions = (
            (spec["width_m"], spec["depth_m"], spec["height_m"])
            if spec["kind"] == "jersey_barrier"
            else (spec["diameter_m"], spec["diameter_m"], spec["height_m"])
        )
        objects = []
        for name, mesh in asset_cache[path]:
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.collection.objects.link(obj)
            obj.scale = dimensions
            objects.append(obj)
        return objects

    for instance_id, (spec, rotation) in enumerate(zip(job["objects"], job["rotations"]), 1):
        if spec["kind"] in ("jersey_barrier", "traffic_barrel") and spec.get("asset_path"):
            objects = construction_asset(spec)
        elif spec["kind"] == "cone":
            objects = cone()
        elif spec["kind"] == "dome_marker":
            objects = dome(spec)
        elif spec["kind"] in ("barrier", "paint"):
            objects = block(spec)
        elif spec["kind"] == "jersey_barrier":
            objects = jersey(spec)
        elif spec["kind"] == "traffic_barrel":
            objects = barrel(spec)
        elif spec["kind"] in ("billboard", "ground_decal"):
            continue
        else:
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=spec["asset_path"])
            objects = [o for o in set(bpy.data.objects) - before if o.parent is None]
        root = bpy.data.objects.new("placed object", None)
        bpy.context.collection.objects.link(root)
        for obj in objects:
            obj.parent = root
        for obj in root.children_recursive:
            obj.pass_index = instance_id
        _blender_material_saturation(root.children_recursive, spec.get("material_saturation"))
        root.location = (spec["x"], spec["y"], spec["z"])
        root.rotation_euler = Matrix(rotation).to_euler()
        root.scale = (spec["scale"],) * 3

    if engine == "BLENDER_EEVEE_NEXT":
        for name in ("ObjectID", "ObjectID2"):
            aov = scene.view_layers[0].aovs.add()
            aov.name, aov.type = name, "VALUE"
        materials = {}
        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            for slot in obj.material_slots:
                key = (slot.material.name, obj.pass_index)
                if key not in materials:
                    material = slot.material.copy()
                    for name, value in (
                        ("ObjectID", obj.pass_index),
                        ("ObjectID2", obj.pass_index**2),
                    ):
                        node = material.node_tree.nodes.new("ShaderNodeOutputAOV")
                        node.aov_name = name
                        node.inputs["Value"].default_value = value
                    materials[key] = material
                slot.link, slot.material = "OBJECT", materials[key]

    bpy.ops.object.camera_add()
    obj = bpy.context.object
    obj.data.sensor_fit, obj.data.sensor_width = "HORIZONTAL", 36
    obj.data.lens = camera["fl_x"] * 36 / width
    obj.data.clip_start, obj.data.clip_end = 0.01, 1000
    scene.camera = obj
    _blender_camera(scene, camera)
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    if engine == "BLENDER_EEVEE_NEXT":
        packed = nodes.new("CompositorNodeCombRGBA")
        for source, channel in (("Depth", "R"), ("ObjectID", "G"), ("ObjectID2", "B")):
            scene.node_tree.links.new(layers.outputs[source], packed.inputs[channel])
        packed.inputs["A"].default_value = 1
        outputs = [(packed.outputs["Image"], "auxiliary-")]
    else:
        outputs = [(layers.outputs["Depth"], "range-"), (layers.outputs["IndexOB"], "instance-")]
    for socket, prefix in outputs:
        target = nodes.new("CompositorNodeOutputFile")
        target.base_path = ""
        target.file_slots[0].path = prefix
        target.format.file_format = "OPEN_EXR"
        target.format.color_depth = "32"
        scene.node_tree.links.new(socket, target.inputs[0])
    _blender_lighting(job)
    return "METAL" if engine == "BLENDER_EEVEE_NEXT" or scene.cycles.device == "GPU" else "CPU"


def empty_layer(camera):
    shape = (camera["h"], camera["w"])
    return ObjectLayer(
        np.zeros((*shape, 4), dtype=np.uint8),
        np.full(shape, np.nan, dtype=np.float32),
        np.zeros(shape, dtype=bool),
        np.zeros(shape, dtype=np.int32),
    )


def _blender_pixels(path, *, noncolor=False):
    import bpy

    image = bpy.data.images.load(str(path), check_existing=False)
    if noncolor:
        image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)[::-1].copy()


def projected_bounds(boxes, matrix, camera, padding=16):
    """Conservative pixel rectangle for world-space bounding boxes; crossing near plane is full."""
    width, height = camera["w"], camera["h"]
    matrix = np.asarray(matrix)
    rectangles = []
    for corners in boxes:
        points = (np.asarray(corners) - matrix[:3, 3]) @ matrix[:3, :3]
        if np.max(points[:, 2]) <= 0.01:
            continue
        x, y, z = points.T
        planes = (
            camera["fl_x"] * x + camera["cx"] * z,
            (width - camera["cx"]) * z - camera["fl_x"] * x,
            camera["fl_y"] * y + camera["cy"] * z,
            (height - camera["cy"]) * z - camera["fl_y"] * y,
        )
        if any(np.max(plane) < 0 for plane in planes):
            continue
        if np.min(points[:, 2]) <= 0.01:
            return 0, 0, width, height
        x = camera["fl_x"] * points[:, 0] / points[:, 2] + camera["cx"]
        y = camera["fl_y"] * points[:, 1] / points[:, 2] + camera["cy"]
        if x.max() < 0 or x.min() >= width or y.max() < 0 or y.min() >= height:
            continue
        rectangles.append((x.min(), y.min(), x.max(), y.max()))
    if not rectangles:
        return None
    rectangles = np.asarray(rectangles)
    return (
        max(0, int(np.floor(rectangles[:, 0].min())) - padding),
        max(0, int(np.floor(rectangles[:, 1].min())) - padding),
        min(width, int(np.ceil(rectangles[:, 2].max())) + padding + 1),
        min(height, int(np.ceil(rectangles[:, 3].max())) + padding + 1),
    )


def _blender_frame(job, *, arrays=False):
    import bpy
    from mathutils import Matrix

    matrix = np.asarray(job["transform_matrix"], dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Object camera must be a finite 4x4 OpenCV matrix")
    scene = bpy.context.scene
    _blender_camera(scene, job["camera"])
    scene.camera.matrix_world = Matrix((matrix @ np.diag([1, -1, -1, 1])).tolist())
    scene.render.use_border = False
    bounds = (0, 0, job["camera"]["w"], job["camera"]["h"])
    if job.get("crop", False):
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        boxes = []
        for obj in scene.objects:
            if obj.type == "MESH":
                evaluated = obj.evaluated_get(graph)
                transform = np.asarray(evaluated.matrix_world)
                boxes.append(
                    np.asarray(evaluated.bound_box) @ transform[:3, :3].T + transform[:3, 3]
                )
        bounds = projected_bounds(boxes, matrix, job["camera"])
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            width, height = job["camera"]["w"], job["camera"]["h"]
            scene.render.use_border = True
            scene.render.use_crop_to_border = False
            scene.render.border_min_x, scene.render.border_max_x = x0 / width, x1 / width
            scene.render.border_min_y, scene.render.border_max_y = 1 - y1 / height, 1 - y0 / height
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    for node in scene.node_tree.nodes:
        if node.bl_idname == "CompositorNodeOutputFile":
            node.base_path = str(output)
    scene.render.filepath = str(output / "objects.png")
    if bounds is None:
        layer = empty_layer(job["camera"])
        depth, indices = layer.depth, layer.instance
    else:
        if "FINISHED" not in bpy.ops.render.render(write_still=True):
            raise RuntimeError("Object rendering did not finish")
        if scene.render.engine == "BLENDER_EEVEE_NEXT":
            packed = _blender_pixels(output / "auxiliary-0001.exr", noncolor=True)
            depth = packed[..., 0].copy()
            indices = instance_moments(packed[..., 1], packed[..., 2], len(job["objects"]))
            depth[indices == 0] = np.nan
        else:
            depth = _blender_pixels(output / "range-0001.exr")[:, :, 0]
            indices = np.rint(_blender_pixels(output / "instance-0001.exr")[:, :, 0]).astype(
                np.int32
            )
        depth[(depth <= 0) | (depth > 1000)] = np.nan
    if arrays:
        return {"empty": bounds is None, "bounds": bounds, "depth": depth, "instance": indices}
    np.save(output / "objects-depth-raw.npy", depth)
    np.save(output / "objects-instance-raw.npy", indices)
    (output / "complete.json").write_text(
        json.dumps(
            {
                "run_id": job.get("run_id"),
                "depth": "camera-forward z in metres; NaN outside objects",
                "rgba": "straight alpha, sRGB PNG",
                "shadow": "no ground shadow catcher",
                "objects": job["objects"],
            },
            indent=2,
        )
        + "\n"
    )

    return {"empty": bounds is None, "bounds": bounds}


def _blender_render(job):
    _blender_scene(job)
    _blender_frame(job)


def build_object_job(
    objects, ground_normal=(0.0, 0.0, 1.0), *, camera=None, lighting=None, object_renderer=None
):
    """Serialize static assets once; camera poses and output paths belong to requests."""
    from drivingbench.sim.visual import CAMERA
    from drivingbench.sim.visual_ground import road_rotation

    basis = road_rotation(ground_normal, 0)
    specs = [obj if isinstance(obj, PlacedObject) else PlacedObject(**obj) for obj in objects]
    settings = object_renderer_settings(object_renderer)
    rows = [asdict(obj) for obj in specs]
    _validate_object_ids(rows, settings["engine"])
    return {
        **settings,
        "objects": rows,
        "ground_normal": np.asarray(ground_normal).tolist(),
        "rotations": [
            (
                basis
                @ np.array(
                    [
                        [math.cos(obj.yaw), -math.sin(obj.yaw), 0],
                        [math.sin(obj.yaw), math.cos(obj.yaw), 0],
                        [0, 0, 1],
                    ]
                )
            ).tolist()
            for obj in specs
        ],
        "camera": pinhole_camera(CAMERA if camera is None else camera),
        **(
            {"lighting": environment_lighting(lighting, ground_normal)}
            if lighting is not None
            else {}
        ),
    }


def render_objects(
    objects,
    camera_matrix,
    out,
    blender,
    *,
    ground_normal=(0.0, 0.0, 1.0),
    lighting=None,
    object_renderer=None,
):
    from PIL import Image

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    job = build_object_job(
        objects, ground_normal, lighting=lighting, object_renderer=object_renderer
    ) | {
        "run_id": uuid.uuid4().hex,
        "transform_matrix": np.asarray(camera_matrix).tolist(),
        "output": str(out),
    }
    job_path = out / "job.json"
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    subprocess.run(
        [
            str(Path(blender).resolve()),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            str(job_path),
        ],
        check=True,
    )
    if json.loads((out / "complete.json").read_text()).get("run_id") != job["run_id"]:
        raise RuntimeError("Object render completion does not match this job")
    with Image.open(out / "objects.png") as image:
        rgba = np.asarray(image.convert("RGBA"))
    layer = repair_edge_depth(
        rgba, np.load(out / "objects-depth-raw.npy"), np.load(out / "objects-instance-raw.npy")
    )
    np.save(out / "objects-depth.npy", layer.depth)
    np.save(out / "objects-instance.npy", layer.instance)
    Image.fromarray(layer.repaired.astype(np.uint8) * 255).save(out / "objects-depth-repaired.png")
    metadata = json.loads((out / "complete.json").read_text())
    metadata["edge_depth_repair"] = {
        "neighbors": "one original 8-neighbor ring; orthogonal before diagonal",
        "filled_pixels": int(layer.repaired.sum()),
    }
    (out / "complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    _blender_render(json.loads(args.job.read_text()))
