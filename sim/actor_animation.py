"""Time-keyed cutout samples and measured root progress, shared by images and collisions."""

from __future__ import annotations

import math
from pathlib import Path

FIELDS = (
    "frame_paths",
    "fps",
    "anchor_uv",
    "anchors_uv",
    "width_m",
    "height_m",
    "playback",
    "progress_fraction",
    "idle",
    "hold_clip",
)


def validate(animation):
    paths, fps = animation["frame_paths"], animation["fps"]
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
        or not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or Path(path).suffix.lower() != ".png" for path in paths)
    ):
        raise ValueError("Animation needs positive fps and nonempty PNG frame paths")
    if animation.get("playback", "loop") not in {"loop", "once"}:
        raise ValueError("Animation playback must be loop or once")
    anchors = animation.get("anchors_uv", [])
    if "anchors_uv" in animation and len(anchors) != len(paths):
        raise ValueError("Per-frame anchors must match the frame count")
    for anchor in [animation["anchor_uv"]] if "anchor_uv" in animation else []:
        anchors = [*anchors, anchor]
    for anchor in anchors:
        if len(anchor) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in anchor):
            raise ValueError("Animation anchor must be two normalized coordinates")
    for key in ("width_m", "height_m"):
        if key in animation and (not math.isfinite(animation[key]) or animation[key] <= 0):
            raise ValueError("Animation dimensions must be finite and positive")
    progress = animation.get("progress_fraction")
    if progress is not None:
        if (
            animation.get("playback") != "once"
            or len(progress) != len(paths)
            or len(progress) < 2
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in progress)
            or progress[0] != 0
            or progress[-1] != 1
            or any(a > b for a, b in zip(progress, progress[1:]))
        ):
            raise ValueError(
                "Measured progress needs one monotonic 0..1 sample per once-play frame"
            )
    for key in ("idle", "hold_clip"):
        if key not in animation:
            continue
        clip = animation[key]
        if any(field in clip for field in ("idle", "hold_clip", "progress_fraction")):
            raise ValueError("Stationary animation cannot translate or contain nested clips")
        validate(clip)
        if key == "hold_clip" and clip.get("playback") != "once":
            raise ValueError("Authored hold clip must play once at its native rate")


def phase(animation, elapsed, duration):
    """Once clips cover the authored motion duration; historical pose cycles retain their rate."""
    count = len(animation["frame_paths"])
    if animation.get("playback", "loop") == "once":
        return min(count - 1.0, max(0.0, elapsed / duration * (count - 1)))
    return max(0.0, elapsed) * animation["fps"] % count


def root_fraction(animation, elapsed, duration):
    fraction = min(1.0, max(0.0, elapsed / duration))
    progress = (animation or {}).get("progress_fraction")
    if progress is None:
        return fraction
    source = fraction * (len(progress) - 1)
    index = min(len(progress) - 2, int(source))
    weight = source - index
    return progress[index] * (1 - weight) + progress[index + 1] * weight


def frame_fields(animation, elapsed, duration, *, idle_time=0, paused=False, hold_time=None):
    """Select source frames without dissolves that would double moving limbs."""
    moving = not paused and 0 <= elapsed < duration
    idle = animation.get("idle")
    hold = animation.get("hold_clip") if hold_time is not None else None
    selected = hold or (idle if idle and not moving else animation)
    if hold:
        source = phase(hold, hold_time, max(1, len(hold["frame_paths"]) - 1) / hold["fps"])
    elif selected is idle:
        idle_duration = max(1, len(selected["frame_paths"]) - 1) / selected["fps"]
        source = phase(
            selected, idle_time if paused or elapsed < 0 else elapsed - duration, idle_duration
        )
    elif selected.get("playback", "loop") == "once":
        source = phase(selected, elapsed, duration)
    else:
        source = phase(selected, elapsed, duration) if moving else 0
    index = min(
        len(selected["frame_paths"]) - 1,
        int(source + (0.5 if selected.get("playback") == "once" else 0)),
    )
    fields = {key: selected[key] for key in ("anchor_uv", "width_m", "height_m") if key in selected}
    if "anchors_uv" in selected:
        fields["anchor_uv"] = selected["anchors_uv"][index]
    return fields | {"asset_path": selected["frame_paths"][index]}
