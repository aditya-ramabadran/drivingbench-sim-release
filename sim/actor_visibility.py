"""Whole-actor projection denominators and regional survival for prospective exposure audits."""

from __future__ import annotations

import numpy as np

REGIONS = ("head", "body", "feet")


def region_rectangles(regions):
    """Require prospective source-image annotations, never infer regions from a cropped view."""
    if not isinstance(regions, dict) or set(regions) != set(REGIONS):
        raise ValueError("Actor visibility needs head, body and feet source rectangles")
    result = {}
    for key, value in regions.items():
        box = np.asarray(value, dtype=float)
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or np.any(box < 0)
            or np.any(box > 1)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise ValueError("Source region must be a finite normalized rectangle")
        result[key] = box
    return result


def whole_actor_visibility(cache, obj, matrix, camera, visible, regions, normal=(0, 0, 1)):
    """Count unoccluded projected alpha off-image too; compare exact post-depth instance pixels."""
    boxes = region_rectangles(regions)
    visible = np.asarray(visible, dtype=bool)
    if visible.shape != (camera["h"], camera["w"]):
        raise ValueError("Surviving actor mask must match the native camera")
    homography, _ = cache.projection_geometry(obj, matrix, camera, normal)
    corners = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]) @ homography.T
    if not np.isfinite(corners).all() or np.any(corners[:, 2] <= 0.01):
        return dict(valid=False, reason="projection_crosses_camera_plane")
    projected = corners[:, :2] / corners[:, 2:3]
    x0, y0 = np.floor(projected.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(projected.max(axis=0)).astype(int) + 1
    width, height = int(x1 - x0), int(y1 - y0)
    if width <= 0 or height <= 0 or width * height > 4_000_000:
        return dict(valid=False, reason="projection_outside_audit_budget")
    expanded = dict(camera, w=width, h=height, cx=camera["cx"] - x0, cy=camera["cy"] - y0)
    alpha = cache.project(obj, matrix, expanded, normal).rgba[..., 3] > 0
    total = int(alpha.sum())
    if not total:
        return dict(valid=False, reason="no_projected_actor_alpha")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    in_frame = (xx >= 0) & (xx < camera["w"]) & (yy >= 0) & (yy < camera["h"])
    survived = np.zeros(alpha.shape, bool)
    survived[in_frame] = visible[yy[in_frame], xx[in_frame]]
    survived &= alpha
    source_uv = np.stack((xx + 0.5, yy + 0.5, np.ones(xx.shape)), axis=-1)
    source_uv = source_uv @ np.linalg.inv(homography).T
    source_uv = source_uv[..., :2] / source_uv[..., 2:3]
    fractions = {}
    for name, (left, top, right, bottom) in boxes.items():
        selected = (
            alpha
            & (source_uv[..., 0] >= left)
            & (source_uv[..., 0] <= right)
            & (source_uv[..., 1] >= top)
            & (source_uv[..., 1] <= bottom)
        )
        count = int(selected.sum())
        if not count:
            return dict(valid=False, reason="no_projected_region", region=name)
        fractions[name] = int((selected & survived).sum()) / count
    inside = int((alpha & in_frame).sum())
    surviving = int(survived.sum())
    full_x, full_y = xx[alpha], yy[alpha]
    bounds = [int(full_x.min()), int(full_y.min()), int(full_x.max()), int(full_y.max())]
    if surviving:
        sx, sy = xx[survived], yy[survived]
        centroid, visible_height = (
            [float(sx.mean()), float(sy.mean())],
            int(sy.max() - sy.min() + 1),
        )
    else:
        centroid, visible_height = None, 0
    return dict(
        valid=True,
        projected_total_pixels=total,
        in_frame_pixels=inside,
        visible_pixels=surviving,
        visible_height_px=visible_height,
        visible_centroid_px=centroid,
        in_frame_fraction=inside / total,
        survival_fraction=surviving / inside if inside else 0.0,
        edge_margin_px=min(
            bounds[0], bounds[1], camera["w"] - 1 - bounds[2], camera["h"] - 1 - bounds[3]
        ),
        full_projected_bounds_px=bounds,
        region_survival=fractions,
        alpha_definition="nonzero compositor source alpha; before JPEG, after instance depth visibility",
    )
