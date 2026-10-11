import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from drivingbench.sim.actor_animation import frame_fields, root_fraction, validate
from drivingbench.sim.billboards import BillboardCache
from drivingbench.sim.scenarios import (
    _offset,
    assess_step,
    hazard_scene,
    object_rows_at,
    validate_scenario,
)
from drivingbench.sim.scene import scene_asset_paths
from drivingbench.sim.visual_objects import PlacedObject
from PIL import Image

BASE = {
    "frame": "pilot_ground",
    "calibration": "ground.json",
    "background": {"checkpoint": "model.ply"},
    "start": {"x": 0, "y": 0, "heading_deg": 0},
    "boundaries": {"left": [[-5, 3], [40, 3]], "right": [[-5, -3], [40, -3]]},
    "finish": {"polygon": [[30, -3], [35, -3], [35, 3], [30, 3]]},
    "markers": [],
    "objects": [],
}


def animation():
    return {
        "frame_paths": ["frame0.png", "frame1.png", "frame2.png", "frame3.png"],
        "fps": 30,
        "playback": "once",
        "progress_fraction": [0, 0.1, 0.8, 1],
        "anchors_uv": [[0.45, 0.9], [0.48, 0.91], [0.5, 0.9], [0.52, 0.91]],
        "width_m": 1,
        "height_m": 1.3,
    }


def test_once_clip_clamps_and_preserves_native_pose_without_crossfade():
    clip = animation()
    validate(clip)
    assert frame_fields(clip, -5, 3)["asset_path"] == "frame0.png"
    assert frame_fields(clip, 1.49, 3)["asset_path"] == "frame1.png"
    at = frame_fields(clip, 1.51, 3)
    assert at["asset_path"] == "frame2.png"
    assert at["anchor_uv"] == [0.5, 0.9]
    assert frame_fields(clip, 100, 3)["asset_path"] == "frame3.png"
    assert root_fraction(clip, 1.5, 3) == pytest.approx(0.45)
    assert root_fraction(clip, -1, 3) == 0
    assert root_fraction(clip, 9, 3) == 1


def test_measured_root_is_shared_by_rendering_and_contact_geometry():
    clip = animation()
    scene = hazard_scene(BASE, "child_entering", assets={}, animations={"child": clip})
    validate_scenario(scene)
    movement = scene["scenario"]["trajectories"]["hazard"]
    movement.update(start_s=2, duration_s=3, delta=[1, -3])
    offset = _offset(scene, "hazard", 3.5)
    assert offset == pytest.approx((0.45, -1.35))
    row = next(row for row in object_rows_at(scene, 3.5) if row["id"] == "hazard")
    source = next(row for row in scene["objects"] if row["id"] == "hazard")
    assert (row["x"] - source["x"], row["y"] - source["y"]) == pytest.approx(offset)
    assert row["asset_path"] == "frame2.png"
    assert _offset(scene, "hazard", 20) == (1, -3)
    assert (
        next(row for row in object_rows_at(scene, 20) if row["id"] == "hazard")["asset_path"]
        == "frame3.png"
    )


def test_idle_plays_before_and_after_motion_without_changing_root():
    clip = animation() | {"idle": {"frame_paths": ["idle0.png", "idle1.png"], "fps": 2}}
    validate(clip)
    assert frame_fields(clip, -1, 3, idle_time=0.75)["asset_path"] == "idle1.png"
    assert frame_fields(clip, 3.75, 3)["asset_path"] == "idle1.png"
    scene = hazard_scene(BASE, "child_entering", assets={}, animations={"child": clip})
    assert _offset(scene, "hazard", 1) == (0, 0)
    assert set(scene_asset_paths(scene)) >= {"idle0.png", "idle1.png", "frame3.png"}


def test_standing_actor_idle_needs_no_physics_trajectory():
    scene = hazard_scene(BASE, "child_static", assets={})
    scene["scenario"]["animations"] = {
        "hazard": {"frame_paths": ["idle0.png", "idle1.png"], "fps": 2}
    }
    validate_scenario(scene)
    row = object_rows_at(scene, 10.75)[0]
    assert row["asset_path"] == "idle1.png"
    assert (row["x"], row["y"]) == (20, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"progress_fraction": [0, 0.8, 0.7, 1]},
        {"progress_fraction": [0, 0.2, 1]},
        {"progress_fraction": [0.1, 0.2, 0.8, 1]},
        {"progress_fraction": [0, 0.2, float("nan"), 1]},
        {"playback": "repeat"},
        {"anchors_uv": [[0.5, 0.9]]},
        {"anchors_uv": [[0.5, 1.1]] * 4},
        {"idle": {"frame_paths": ["idle.png"], "fps": 2, "idle": {}}},
    ],
)
def test_invalid_clip_is_rejected_at_scene_load(change):
    with pytest.raises(ValueError):
        validate(animation() | change)


def test_cache_bounds_long_clips_and_preserves_reused_pixels(tmp_path):
    objects = []
    for index in range(6):
        pixels = np.zeros((8, 8, 4), np.uint8)
        pixels[1:7, 1:7] = [30 * index, 50, 100, 255]
        path = tmp_path / f"{index}.png"
        Image.fromarray(pixels).save(path)
        objects.append(PlacedObject("billboard", 0, 5, asset_path=str(path), width_m=1, height_m=1))
    cache = BillboardCache(max_bytes=2 * 8 * 8 * 4 * 4)
    expected = copy.deepcopy(cache.load(objects[0].asset_path))
    for obj in objects:
        cache.load(obj.asset_path)
        assert cache.cached_bytes <= cache.max_bytes
        assert len(cache.images) <= 2
    np.testing.assert_array_equal(cache.load(objects[0].asset_path), expected)
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(cache.load, [objects[0].asset_path] * 8))
    assert all(value is values[0] for value in values)
    assert cache.cached_bytes == sum(image.nbytes for image in cache.images.values())
    assert replace(objects[0], x=1).asset_path == objects[0].asset_path


def test_video_root_contact_interval_contains_true_entry_between_samples():
    scene = hazard_scene(BASE, "child_entering", assets={}, animations={"child": animation()})
    scene["scenario"]["trajectories"]["hazard"].update(start_s=0, duration_s=3, delta=[0, -3])
    radius = scene["scenario"]["colliders"][0]["circle"][2]
    fraction = (2.6 - 1.78 / 2 - radius) / 3
    entry_time = 1 + (fraction - 0.1) / 0.7
    hit = assess_step(scene, (20, 0, 0), (20, 0, 0), 0, 3, contacts_only=True)
    assert hit["id"] == "hazard"
    lower, upper = hit["time_interval_s"]
    assert lower <= entry_time <= upper
    assert upper - lower < 0.03


def test_idle_once_retains_own_native_rate_instead_of_crossing_duration():
    clip = animation() | {
        "idle": {
            "frame_paths": ["idle0.png", "idle1.png", "idle2.png"],
            "fps": 2,
            "playback": "once",
        }
    }
    validate(clip)
    assert frame_fields(clip, 30.75, 30)["asset_path"] == "idle2.png"
    assert frame_fields(clip, -1, 30, idle_time=0.75)["asset_path"] == "idle2.png"
