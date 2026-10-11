import copy
import hashlib
import json

import numpy as np
import pytest
from drivingbench.replay.trolley_visuals import Scene, rays, render, scene_depth
from PIL import Image


def fixture_scene(tmp_path):
    camera = dict(width=96, height=64, focal_px=70, cx=48, cy=27)
    yy, xx = np.mgrid[:64, :96]
    pixels = np.stack([xx * 2, yy * 3, np.full_like(xx, 100)], -1).astype(np.uint8)
    reference = tmp_path / "reference.png"
    Image.fromarray(pixels).save(reference)
    actor = tmp_path / "actor.png"
    Image.new("RGBA", (16, 32), (255, 0, 0, 255)).save(actor)
    second = tmp_path / "second-reference.png"
    Image.fromarray(np.roll(pixels, 2, axis=1)).save(second)
    flow = tmp_path / "flow.npz"
    forward = np.zeros((8, 12, 2), dtype=np.float32)
    forward[..., 0] = 2
    np.savez(flow, forward=forward, backward=-forward)
    manifest = dict(
        camera=camera,
        photo_bank=dict(
            keyframes=[
                dict(
                    id="a",
                    path=str(reference),
                    sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    camera_position_m=[0, 1.25, -12],
                    camera_yaw_rad=0,
                ),
                dict(
                    id="b",
                    path=str(second),
                    sha256=hashlib.sha256(second.read_bytes()).hexdigest(),
                    camera_position_m=[0, 1.25, -10],
                    camera_yaw_rad=0,
                ),
            ],
            edges=[
                dict(
                    id="ab",
                    a="a",
                    b="b",
                    flow_path=str(flow),
                    flow_sha256=hashlib.sha256(flow.read_bytes()).hexdigest(),
                )
            ],
        ),
        actors={"1_adult": dict(path=str(actor), height_m=1.7)},
        max_uncovered_fraction=0.1,
    )
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(manifest))
    config = dict(
        visuals_manifest=str(path),
        left=dict(kind="adult", count=1),
        right=dict(kind="adult", count=0),
    )
    state = dict(camera_position_m=[0, 1.25, -12], camera_yaw_rad=0, obstacles_visible=False)
    return Scene(manifest), config, state, pixels


def test_ground_and_divider_have_distinct_depth():
    directions = np.array([[[0.1, -0.1, 1], [0, 0, 1], [0, -0.025, 1]]], dtype=np.float32)
    depth = scene_depth([0, 1.25, -12], directions)
    assert depth[0, 0] == pytest.approx(12.5)
    assert np.isinf(depth[0, 1])
    assert depth[0, 2] == pytest.approx(12)  # divider intercept before far ground


def test_reference_pose_reproduces_photo_without_global_zoom(tmp_path):
    scene, config, state, original = fixture_scene(tmp_path)
    image, quality = scene.render(config, state)
    assert np.max(np.abs(np.asarray(image).astype(int) - original)) <= 1
    assert quality["uncovered_fraction"] < 0.04
    state["camera_position_m"][2] += 1
    moved, _ = scene.render(config, state)
    assert not np.array_equal(np.asarray(moved), original)


def test_actors_are_exposure_gated_and_ground_anchored(tmp_path):
    scene, config, state, _ = fixture_scene(tmp_path)
    before, _ = scene.render(config, state)
    state["obstacles_visible"] = True
    after, _ = scene.render(config, state)
    assert np.any(np.asarray(before) != np.asarray(after))
    directions = rays(scene.camera, 0)
    assert directions.shape == (64, 96, 3)


def test_render_does_not_advance_clock_or_mutate_state(tmp_path):
    _, config, state, _ = fixture_scene(tmp_path)
    saved = copy.deepcopy(state)
    first = render(config, state, tmp_path / "first.png")
    second = render(config, state, tmp_path / "second.png")
    assert first.read_bytes() == second.read_bytes()
    assert state == saved
    assert (
        json.loads(first.with_suffix(".png.json").read_text())["camera_position_m"]
        == state["camera_position_m"]
    )


def test_bad_pose_and_uncovered_camera_fail_explicitly(tmp_path):
    scene, config, state, _ = fixture_scene(tmp_path)
    state["camera_yaw_rad"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        scene.render(config, state)
    state["camera_yaw_rad"] = np.pi
    with pytest.raises(ValueError, match="unsupported"):
        render(config, state, tmp_path / "bad.png")


def test_sidecar_distinguishes_engine_and_photographic_pose(tmp_path):
    _, config, state, _ = fixture_scene(tmp_path)
    state["camera_position_m"] = [0.4, 1.25, -11]
    output = render(config, state, tmp_path / "residual.png")
    metadata = json.loads(output.with_suffix(".png.json").read_text())
    assert metadata["camera_position_m"] == [0.4, 1.25, -11]
    assert metadata["represented_camera_position_m"] == [0, 1.25, -11]
    assert metadata["pose_residual_m"] == pytest.approx(0.4)
    assert metadata["interpolation"]["fraction"] == pytest.approx(0.5)
    assert len(metadata["photo_sources"]) == 2
    assert metadata["interpolation"]["flow_sha256"]


def test_unserved_pose_cannot_silently_fall_back_to_nearest_photo(tmp_path):
    scene, config, state, _ = fixture_scene(tmp_path)
    state["camera_position_m"][0] = 8
    with pytest.raises(ValueError, match="unsupported"):
        scene.render(config, state)


def test_fixed_crop_records_effective_projection(tmp_path):
    scene, config, state, _ = fixture_scene(tmp_path)
    scene.manifest["frame_crop_px"] = 4
    image, metadata = scene.render(config, state)
    assert image.size == (96, 64)
    assert metadata["frame_crop_px"] == 4
    effective = metadata["effective_camera"]
    assert effective["focal_x_px"] == pytest.approx(70 * 96 / 88)
    assert effective["focal_y_px"] == pytest.approx(70 * 64 / 56)
    assert effective["cx_px"] == 48
    assert effective["cy_px"] == pytest.approx(23 * 64 / 56)
