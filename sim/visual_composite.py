"""Exact grounded compositing restricted to object and contact-shadow support."""

from __future__ import annotations

import numpy as np
from drivingbench.sim.camera_response import apply_response
from drivingbench.sim.visual_objects import (
    ObjectLayer,
    composite_grounded,
    pinhole_camera,
    shadow_radius,
)
from PIL import Image, ImageDraw


def _clip_polygon(points, width, height):
    for plane in ((0, 0, 1), (1, 0, 0), (-1, 0, width), (0, 1, 0), (0, -1, height)):
        if not len(points):
            break
        distances = points @ plane
        if plane == (0, 0, 1):
            distances -= 1e-8
        clipped = []
        for previous, current, a, b in zip(
            np.roll(points, 1, axis=0), points, np.roll(distances, 1), distances, strict=True
        ):
            if (a >= 0) != (b >= 0):
                clipped.append(previous + (current - previous) * (a / (a - b)))
            if b >= 0:
                clipped.append(current)
        points = np.asarray(clipped)
    return points


def _expand(mask, radius):
    padded = np.pad(mask, ((0, 0), (radius, radius)))
    horizontal = np.logical_or.reduce(
        [padded[:, shift : shift + mask.shape[1]] for shift in range(2 * radius + 1)]
    )
    padded = np.pad(horizontal, ((radius, radius), (0, 0)))
    return np.logical_or.reduce(
        [padded[shift : shift + mask.shape[0]] for shift in range(2 * radius + 1)]
    )


def _support(matrix, objects, rgba, camera, ground):
    selected = rgba[..., 3] > 0
    height, width = selected.shape
    shadow = Image.new("L", (width, height))
    draw = ImageDraw.Draw(shadow)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    basis = ground.road_rotation(0)[:, :2]
    for obj in objects:
        radius = 2 * shadow_radius(obj)
        if radius == 0:
            continue
        center = np.array([obj.x, obj.y, obj.z], dtype=float)
        center -= (center @ ground.normal + ground.offset) * ground.normal
        axes = basis * radius
        if obj.kind in {"billboard", "barrier", "jersey_barrier"}:
            c, s = np.cos(obj.yaw), np.sin(obj.yaw)
            axes = basis @ np.array([[c, -s], [s, c]])
            axes *= [
                1.6 * (obj.width_m * obj.scale / 2 + 0.12),
                1.6 * ((obj.depth_m or 0.35) * obj.scale / 2 + 0.12),
            ]
        world = center + corners @ axes.T
        local = (world - matrix[:3, 3]) @ matrix[:3, :3]
        projected = local.copy()
        projected[:, 0] = camera["fl_x"] * local[:, 0] + camera["cx"] * local[:, 2]
        projected[:, 1] = camera["fl_y"] * local[:, 1] + camera["cy"] * local[:, 2]
        polygon = _clip_polygon(projected, width, height)
        if len(polygon) >= 3:
            pixels = polygon[:, :2] / polygon[:, 2:3] - 0.5
            draw.polygon([tuple(point) for point in pixels], fill=255)
    selected |= _expand(np.asarray(shadow) > 0, 1)
    return np.nonzero(selected)


def _intersections(ground, matrix, y, x, camera):
    rays = np.column_stack(
        (
            (x + 0.5 - camera["cx"]) / camera["fl_x"],
            (y + 0.5 - camera["cy"]) / camera["fl_y"],
            np.ones(len(x)),
        )
    )
    directions = rays @ matrix[:3, :3].T
    denominator = directions @ ground.normal
    numerator = -(matrix[:3, 3] @ ground.normal + ground.offset)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = numerator / denominator
    depth[(np.abs(denominator) < 1e-8) | (depth <= 0)] = np.nan
    return matrix[:3, 3] + depth[:, None] * directions, depth


def compose(
    scene,
    matrix,
    rgb,
    depth,
    alpha,
    layer,
    *,
    objects=None,
    diagnostics=False,
    response=None,
    seed=0,
    return_support=False,
    camera=None,
    instance_masks=None,
):
    """Return native RGB with grounded depth/shadows, optionally visible support for offline QA.

    return_support yields (RGB, instance diagnostics, bool mask). Camera-response halos
    outside exact visible-object/contact-shadow support are deliberately excluded.
    instance_masks optionally receives per-object contributing pixels before camera response.
    """
    from drivingbench.sim.visual import CAMERA, ground_region

    camera = pinhole_camera(CAMERA if camera is None else camera)
    matrix, rgb, depth, alpha = map(np.asarray, (matrix, rgb, depth, alpha))
    shape = (camera["h"], camera["w"])
    layers = list(layer) if isinstance(layer, (list, tuple)) else [layer]
    if (
        rgb.shape != (*shape, 3)
        or rgb.dtype != np.uint8
        or depth.shape != shape
        or alpha.shape != shape
        or not layers
        or any(value.rgba.shape != (*shape, 4) or value.depth.shape != shape for value in layers)
    ):
        raise ValueError("Expected native uint8 RGB/RGBA and matching depth/opacity dimensions")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Camera transform must be finite and 4×4")
    objects = scene.objects if objects is None else objects
    output = rgb.copy()
    coverage = layers[0].rgba.copy()
    for value in layers[1:]:
        coverage[..., 3] |= value.rgba[..., 3]
    y, x = _support(matrix, objects, coverage, camera, scene.ground)
    if not len(x):
        if return_support:
            return output, {}, np.zeros(shape, dtype=bool)
        return (output, {}) if diagnostics else output
    if response is not None:
        selected = np.zeros(shape, bool)
        selected[y, x] = True
        y, x = np.nonzero(_expand(selected, response.radius))
    points, plane = _intersections(scene.ground, matrix, y, x, camera)
    transform = scene.ground.map_from_ground
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    region = ground_region(local, scene.course.corridor, margin=0.4)
    selected_masks = {} if instance_masks is not None else None
    pixels, masks = composite_grounded(
        rgb[y, x][:, None],
        depth[y, x][:, None],
        alpha[y, x][:, None],
        [
            ObjectLayer(
                value.rgba[y, x][:, None],
                value.depth[y, x][:, None],
                instance=None if value.instance is None else value.instance[y, x][:, None],
            )
            for value in layers
        ],
        objects,
        points[:, None],
        plane[:, None],
        region[:, None],
        ground_normal=scene.ground.normal,
        **({"instance_masks": selected_masks} if selected_masks is not None else {}),
    )
    if instance_masks is not None:
        for identity, mask in selected_masks.items():
            full = np.zeros(shape, dtype=bool)
            full[y, x] = mask[:, 0]
            instance_masks[identity] = full
    output[y, x] = pixels[:, 0]
    if response is not None:
        effective = np.where(masks["ground_visible"][:, 0], plane + 0.05, depth[y, x] - 0.05)
        surface = np.zeros(len(x))
        known = np.isfinite(effective) & np.isfinite(alpha[y, x]) & (alpha[y, x] >= 0.5)
        depths, weights = [], []
        for value in layers:
            z = value.depth[y, x]
            visible = known & np.isfinite(z) & (z > 0) & (z < effective)
            depths.append(z)
            weights.append(value.rgba[y, x, 3].astype(np.float32) / 255 * visible)
        depths, weights = np.asarray(depths), np.asarray(weights)
        transmission = np.ones(len(x), np.float32)
        order = np.argsort(np.where(weights > 0, depths, np.inf), axis=0, kind="stable")
        for rank in order:
            z = np.take_along_axis(depths, rank[None], axis=0)[0]
            amount = np.take_along_axis(weights, rank[None], axis=0)[0]
            surface = np.maximum(surface, np.where(amount * transmission > 0, z, 0))
            transmission *= 1 - amount
        weight = 1 - transmission
        shadow_visible = (masks["shadow_opacity"][:, 0] > 0) & (transmission > 0)
        surface = np.maximum(surface, np.where(shadow_visible, plane, 0))
        scene_depth, surface_depth, object_weight = [np.zeros(shape) for _ in range(3)]
        scene_depth[y, x], surface_depth[y, x], object_weight[y, x] = effective, surface, weight
        output = apply_response(
            rgb, output, scene_depth, alpha, surface_depth, object_weight, response, seed=seed
        )
    if return_support:
        support = np.zeros(shape, dtype=bool)
        support[y, x] = masks["visible"][:, 0] | (masks["shadow_opacity"][:, 0] > 0)
        return output, masks.get("instances", {}), support
    if not diagnostics:
        return output
    return output, masks.get("instances", {})
