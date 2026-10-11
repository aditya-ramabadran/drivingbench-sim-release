import numpy as np
import pytest
from drivingbench.sim.visual_wide import (
    INTRINSICS,
    WideCalibration,
    native_maps,
    pair_frame,
    perspective_tiles,
    tile_map,
)


def calibration(**overrides):
    values = dict(
        intrinsics=INTRINSICS,
        distortion=[-0.01, -0.003, 0, 0],
        rotation_wide_from_narrow=np.eye(3).tolist(),
        translation_wide_from_narrow=[0.03, 0, 0],
    )
    return WideCalibration(**dict(values, **overrides))


def test_wide_projection_roundtrip_and_center():
    camera = calibration()
    pixels = np.array([[672, 380], [100, 200], [1250, 450], [500, 20]])
    rays = camera.rays(pixels)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1)
    np.testing.assert_allclose(camera.project(rays), pixels, atol=1e-7)
    np.testing.assert_allclose(rays[0], [0, 0, 1])
    assert np.isnan(camera.rays([[10000, 380]])).all()
    assert np.isnan(camera.project([0, 0, -1])).all()
    with pytest.raises(ValueError, match="zero ray"):
        camera.project([0, 0, 0])


def test_wide_fisheye_retains_visible_rays_beyond_ninety_degrees():
    camera = calibration()
    angle = np.radians(105)
    ray = np.array([np.sin(angle), 0, np.cos(angle)])
    recovered = camera.rays(camera.project(ray))
    np.testing.assert_allclose(recovered, ray, atol=1e-8)
    assert recovered[2] < 0


def test_wide_rig_translation_and_rotation_convention():
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    camera = calibration(rotation_wide_from_narrow=rotation.tolist())
    narrow = np.eye(4)
    narrow[:3, 3] = [5, 3, 2]
    wide = camera.camera_pose(narrow)
    world = np.array([7, 4, 8])
    pn = narrow[:3, :3].T @ (world - narrow[:3, 3])
    pw = wide[:3, :3].T @ (world - wide[:3, 3])
    np.testing.assert_allclose(pw, rotation @ pn + camera.translation)


@pytest.mark.parametrize(
    "override",
    [
        dict(distortion=[-1, 0, 0, 0]),
        dict(rotation_wide_from_narrow=np.diag([1, 1, -1]).tolist()),
        dict(translation_wide_from_narrow=[np.nan, 0, 0]),
    ],
)
def test_wide_rejects_invalid_calibration(override):
    with pytest.raises(ValueError):
        calibration(**override)


def test_tile_remapping_is_same_ray_and_weights_normalize():
    camera = calibration(
        intrinsics=dict(INTRINSICS, w=134, h=76, fl_x=42.525, fl_y=42.525, cx=67.2, cy=38)
    )
    tiles = perspective_tiles(size=90, focal=45)
    maps, weights = native_maps(camera, tiles)
    total = np.sum(weights, axis=0)
    assert np.allclose(total[total > 0], 1)
    for tile, mapping in zip(tiles, maps, strict=True):
        u, v = mapping[30, 60]
        ray = (
            np.array([(u - tile["cx"]) / tile["fl_x"], (v - tile["cy"]) / tile["fl_y"], 1])
            @ np.array(tile["rotation_wide_from_tile"]).T
        )
        np.testing.assert_allclose(camera.project(ray), [60, 30], atol=1e-5)
        sample = tile_map(camera, tile)[45, 45]
        np.testing.assert_allclose(
            sample, camera.project(np.array(tile["rotation_wide_from_tile"])[:, 2]), atol=1e-5
        )


def test_pairing_preserves_split_and_uses_wide_encode_index(tmp_path):
    narrow = dict(
        frame_id=9,
        segment=2,
        index=1,
        eof_ns=10000000,
        name="test.jpg",
        file_path="narrow.jpg",
        split="test",
    )
    wide = dict(frame_id=9, segment=2, index=4, eof_ns=10050000, sof_ns=10000000)
    paired = pair_frame(narrow, [wide], tmp_path)
    assert paired["split"] == "test"
    assert paired["index"] == 1 and paired["wide_index"] == 4
    assert paired["pair_delta_s"] == pytest.approx(0.00005)
    with pytest.raises(ValueError, match="exactly once"):
        pair_frame(narrow, [wide, wide], tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        pair_frame(narrow, [dict(wide, eof_ns=20000000)], tmp_path)
