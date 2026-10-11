"""Offline generated-frame finishing inside an explicitly supplied visible support mask."""

from __future__ import annotations

import hashlib
import math

import numpy as np

NATIVE_SHAPE = (760, 1344)
MAX_FEATHER_PX = 8


def _rgb(value, label):
    value = np.asarray(value)
    if value.shape != (*NATIVE_SHAPE, 3) or value.dtype != np.uint8:
        raise ValueError(f"{label} must be native 1344×760 uint8 RGB")
    return value


def _mask(value):
    value = np.asarray(value)
    if value.shape != NATIVE_SHAPE:
        raise ValueError("Support mask must be native 1344×760")
    if value.dtype == bool:
        return value
    if value.dtype == np.uint8 and np.all((value == 0) | (value == 255)):
        return value == 255
    raise ValueError("Support mask must be bool or binary uint8 0/255")


def _inward_weight(mask, radius):
    inside = mask.copy()
    weight = inside.astype(np.float32)
    height, width = mask.shape
    for _ in range(radius):
        padded = np.pad(inside, 1, constant_values=False)
        inside = np.logical_and.reduce(
            [padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)]
        )
        weight += inside
    return weight / (radius + 1)


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def finish_frame(
    original_rgb, generated_rgb, support_mask, *, pose, sim_time_s, captured_at, feather_px=2
):
    """Return finished RGB and provenance; never expand the mask or advance capture time.

    Hashes cover native row-major RGB bytes and canonical mask bytes (0/255), not encoded
    files. Feathering is an inward Chebyshev-distance ramp, at most eight pixels wide.
    Save losslessly if the outside-mask byte-identity guarantee must survive encoding.
    Explicit captured_at=None preserves unknown capture time for an offline source.
    """
    original = _rgb(original_rgb, "Original frame")
    generated = _rgb(generated_rgb, "Generated frame")
    mask = _mask(support_mask)
    if (
        isinstance(feather_px, bool)
        or not isinstance(feather_px, int)
        or not 0 <= feather_px <= MAX_FEATHER_PX
    ):
        raise ValueError("Feather width must be an integer from zero to eight pixels")
    try:
        pose = tuple(float(value) for value in pose)
        stamp = None if captured_at is None else float(captured_at)
        sim_time = float(sim_time_s)
    except (TypeError, ValueError) as error:
        raise ValueError("Pose and capture/simulation times must be finite") from error
    if (
        len(pose) != 3
        or not all(math.isfinite(v) for v in (*pose, sim_time))
        or sim_time < 0
        or (stamp is not None and (not math.isfinite(stamp) or stamp < 0))
    ):
        raise ValueError("Pose needs three finite values and times must be finite/nonnegative")
    output = original.copy()
    weights = _inward_weight(mask, feather_px)[mask, None]
    output[mask] = np.rint(original[mask] * (1 - weights) + generated[mask] * weights).astype(
        np.uint8
    )
    changed = np.any(generated != original, axis=-1)
    record = {
        "schema_version": 1,
        "mode": "offline_masked_generated_finish",
        "dimensions": {"width": NATIVE_SHAPE[1], "height": NATIVE_SHAPE[0]},
        "hash_format": "sha256 row-major uint8 RGB; mask canonical uint8 0/255",
        "original_sha256": _hash(original),
        "generated_sha256": _hash(generated),
        "mask_sha256": _hash(mask.astype(np.uint8) * 255),
        "output_sha256": _hash(output),
        "pose": list(pose),
        "sim_time_s": sim_time,
        "captured_at": stamp,
        "capture_time_policy": "unchanged from original frame"
        if stamp is not None
        else "unknown; offline source has no capture timestamp",
        "feather_px": feather_px,
        "feather_policy": "inward only; sRGB-byte interpolation",
        "mask_pixels": int(mask.sum()),
        "changed_pixels": int(np.any(output != original, axis=-1).sum()),
        "generated_outside_mask_changes_ignored": int((changed & ~mask).sum()),
        "outside_mask_identical": bool(np.array_equal(output[~mask], original[~mask])),
        "live_integration": False,
        "visual_qa": "required; geometric pixel preservation does not validate generated content",
    }
    return output, record
