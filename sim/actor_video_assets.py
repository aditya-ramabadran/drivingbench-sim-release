"""Normalize tracked video mattes without replacing their source motion."""

import numpy as np


def height_scale_factors(heights, *, fps, reference_height_px, window_s=1.5):
    """Remove slow apparent-size drift while retaining faster gait variation."""
    heights = np.asarray(heights, dtype=float)
    if (
        heights.ndim != 1
        or not len(heights)
        or not np.isfinite(heights).all()
        or np.any(heights <= 0)
        or not np.isfinite((fps, reference_height_px, window_s)).all()
        or min(fps, reference_height_px, window_s) <= 0
    ):
        raise ValueError("Finite positive heights, FPS, reference and window are required")
    window = max(1, round(fps * window_s)) | 1
    pad = window // 2
    samples = np.lib.stride_tricks.sliding_window_view(np.pad(heights, pad, mode="edge"), window)
    median = np.median(samples, axis=1)
    trend = np.convolve(np.pad(median, pad, mode="edge"), np.ones(window) / window, mode="valid")
    return trend, reference_height_px / trend


def nearby_components(boxes, *, primary, max_gap=12):
    """Retain adjacent matte islands such as a head separated at the neck."""
    boxes = np.asarray(boxes, dtype=float)
    if (
        boxes.ndim != 2
        or boxes.shape[1] != 4
        or not np.isfinite(boxes).all()
        or np.any(boxes[:, 2:] <= boxes[:, :2])
        or not 0 <= primary < len(boxes)
        or not np.isfinite(max_gap)
        or max_gap < 0
    ):
        raise ValueError("Valid boxes, primary component and nonnegative gap are required")
    selected = {primary}
    pending = [primary]
    while pending:
        current = pending.pop()
        gap = np.maximum(
            np.maximum(
                boxes[:, :2] - boxes[current, 2:],
                boxes[current, :2] - boxes[:, 2:],
            ),
            0,
        )
        for index in np.flatnonzero(np.linalg.norm(gap, axis=1) <= max_gap):
            if index not in selected:
                selected.add(int(index))
                pending.append(int(index))
    return tuple(sorted(selected))


def monotone_progress(anchors, *, direction):
    """Return source-derived progress and isotonic tracking correction in pixels."""
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != 2 or len(anchors) < 2:
        raise ValueError("At least two finite XY foot anchors are required")
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("A finite nonzero XY direction is required")
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(anchors).all():
        raise ValueError("A finite nonzero XY direction is required")
    coordinate = anchors @ (direction / norm)
    blocks = []
    for index, value in enumerate(coordinate):
        blocks.append([index, index + 1, float(value), 1])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            count = left[3] + right[3]
            blocks.append(
                [left[0], right[1], (left[2] * left[3] + right[2] * right[3]) / count, count]
            )
    fitted = np.empty_like(coordinate)
    for start, end, value, _ in blocks:
        fitted[start:end] = value
    distance = fitted[-1] - fitted[0]
    if distance <= 1:
        raise ValueError("Tracked translation must exceed one pixel")
    return (fitted - fitted[0]) / distance, fitted - coordinate


def matte_anchor(alpha, *, threshold=128, foot_fraction=0.12):
    """Estimate feet from the opaque silhouette's lower band."""
    alpha = np.asarray(alpha)
    if alpha.ndim != 2 or not 0 < foot_fraction <= 1:
        raise ValueError("A two-dimensional matte and valid foot fraction are required")
    ys, xs = np.nonzero(alpha >= threshold)
    if len(xs) == 0:
        raise ValueError("The matte has no opaque subject")
    top, bottom = int(ys.min()), int(ys.max())
    lower = ys >= bottom - max(1, (bottom - top) * foot_fraction)
    return (float(np.median(xs[lower])), float(bottom)), (
        int(xs.min()),
        top,
        int(xs.max()) + 1,
        bottom + 1,
    )


def normalized_extent(boxes, anchors, *, padding=12):
    """Choose one anchor-relative canvas that retains every tracked silhouette."""
    boxes = np.asarray(boxes, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or anchors.shape != (len(boxes), 2):
        raise ValueError("Each box needs one XY anchor")
    if not len(boxes) or not np.isfinite(boxes).all() or not np.isfinite(anchors).all():
        raise ValueError("Finite nonempty boxes and anchors are required")
    if padding < 0 or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Positive boxes and nonnegative padding are required")
    low = np.floor(np.min(boxes[:, :2] - anchors, axis=0)).astype(int) - padding
    high = np.ceil(np.max(boxes[:, 2:] - anchors, axis=0)).astype(int) + padding
    return tuple((high - low).tolist()), tuple((-low).tolist())


def metric_track(anchors, *, nominal_height_m, silhouette_height_px, direction):
    """Scale source translation by a declared actor height, retaining tracking error."""
    if (
        not np.isfinite((nominal_height_m, silhouette_height_px)).all()
        or nominal_height_m <= 0
        or silhouette_height_px <= 0
    ):
        raise ValueError("Actor and source silhouette heights must be positive")
    progress, correction = monotone_progress(anchors, direction=direction)
    scale = nominal_height_m / silhouette_height_px
    anchors = np.asarray(anchors, dtype=float)
    displacement = (anchors - anchors[0]) * scale
    distance = np.linalg.norm(displacement[-1])
    return {
        "metres_per_pixel": float(scale),
        "displacement_m": displacement.tolist(),
        "source_distance_m": float(distance),
        "progress_fraction": progress.tolist(),
        "tracking_correction_max_px": float(np.max(np.abs(correction))),
        "metric_scale_note": "Authored actor-height scale; monocular ground projection is approximate",
    }
