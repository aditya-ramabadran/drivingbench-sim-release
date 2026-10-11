"""A bounded camera response for already-visible inserted surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraResponse:
    sigma_px: float
    noise_sigma: float
    rgb_gain: tuple[float, float, float]

    @property
    def radius(self):
        return int(np.ceil(3 * self.sigma_px))


PROFILES = {
    "site1-overcast-v1": CameraResponse(0.55, 0.53, (1.025, 1.0, 0.975)),
    "site1-splat-softness-v2": CameraResponse(1.65, 0.53, (1.025, 1.0, 0.975)),
}


def response_profile(name):
    if name is None:
        return None
    if name not in PROFILES:
        raise ValueError(f"Unknown inserted-object camera response: {name}")
    return PROFILES[name]


def _filter(values, kernel, *, maximum=False):
    result = values
    radius = len(kernel) // 2
    for axis in (0, 1):
        padding = [(0, 0)] * values.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="constant")
        result = np.zeros_like(result)
        for index, weight in enumerate(kernel):
            selection = [slice(None)] * values.ndim
            selection[axis] = slice(index, index + result.shape[axis])
            shifted = padded[tuple(selection)]
            if maximum:
                np.maximum(result, shifted, out=result)
            else:
                result += shifted * weight
    return result


def apply_response(
    background, composite, depth, opacity, surface_depth, coverage, profile, *, seed=0
):
    """Soften visible contributions while preserving unknown and nearer real surfaces."""
    background, composite = np.asarray(background), np.asarray(composite)
    shape = background.shape[:2]
    arrays = tuple(map(np.asarray, (depth, opacity, surface_depth, coverage)))
    if (
        background.shape != (*shape, 3)
        or composite.shape != background.shape
        or background.dtype != np.uint8
        or composite.dtype != np.uint8
        or any(value.shape != shape for value in arrays)
    ):
        raise ValueError("Camera response needs RGB bytes and matching depth/coverage arrays")
    depth, opacity, surface_depth, coverage = arrays
    if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("Camera-response coverage must lie between zero and one")
    changed = np.any(background != composite, axis=-1)
    support = (changed | (coverage > 0)) & np.isfinite(surface_depth) & (surface_depth > 0)
    if not support.any():
        return composite.copy()
    y, x = np.nonzero(support)
    radius = profile.radius
    box = np.s_[
        max(0, y.min() - radius) : min(shape[0], y.max() + radius + 1),
        max(0, x.min() - radius) : min(shape[1], x.max() + radius + 1),
    ]
    base, composed = background[box].astype(np.float32), composite[box].astype(np.float32)
    weight = coverage[box].astype(np.float32)
    foreground = composed - base * (1 - weight[..., None])
    foreground += (
        foreground * (np.asarray(profile.rgb_gain, np.float32) - 1) * (weight > 0)[..., None]
    )
    foreground *= support[box][..., None]
    axis = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (axis / profile.sigma_px) ** 2).astype(np.float32)
    kernel /= kernel.sum()
    filtered_weight = _filter(weight, kernel)
    filtered = _filter(foreground, kernel) - base * filtered_weight[..., None]
    farthest = _filter(np.where(support[box], surface_depth[box], 0), kernel, maximum=True)
    known = np.isfinite(depth[box]) & np.isfinite(opacity[box]) & (opacity[box] >= 0.5)
    allowed = known & (farthest > 0) & (farthest < depth[box])
    noise = np.random.default_rng(seed).normal(0, profile.noise_sigma, (*weight.shape, 3))
    filtered += noise * filtered_weight[..., None]
    result = composite.copy()
    pixels = np.rint(np.clip(base + filtered, 0, 255)).astype(np.uint8)
    result[box] = np.where(allowed[..., None], pixels, composite[box])
    return result
