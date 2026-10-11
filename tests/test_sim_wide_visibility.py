import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from drivingbench.sim.camera_response import response_profile
from drivingbench.sim.persistent import PersistentRenderer
from drivingbench.sim.visual_composite import compose
from drivingbench.sim.visual_ground import GroundCalibration
from drivingbench.sim.visual_objects import ObjectLayer, composite_objects
from drivingbench.sim.wide_renderer import WideRenderer, visible_billboards
from PIL import Image


def layers():
    rgba = np.zeros((4, 6, 4), np.uint8)
    rgba[1:3, 1:5] = (190, 110, 40, 255)
    actor = ObjectLayer(rgba, np.full((4, 6), 2.0), instance=np.where(rgba[..., 3] > 0, -1, 0))
    front = np.zeros_like(rgba)
    front[1:3, 2] = (30, 50, 80, 255)
    foreground = ObjectLayer(front, np.ones((4, 6)), instance=np.ones((4, 6), int))
    depth, alpha = np.full((4, 6), 10.0), np.ones((4, 6))
    depth[:, 3], alpha[:, 4] = 1, 0
    return np.full((4, 6, 3), 100, np.uint8), depth, alpha, [foreground, actor]


def test_visibility_collector_reuses_depth_alpha_and_nearer_object_occlusion():
    args = layers()
    original, counts = composite_objects(*args)
    masks = {}
    actual, same_counts = composite_objects(*args, instance_masks=masks)
    np.testing.assert_array_equal(actual, original)
    assert same_counts["instances"] == counts["instances"]
    expected = np.zeros((4, 6), bool)
    expected[1:3, 1] = True
    np.testing.assert_array_equal(masks[0], expected)
    assert masks[0].sum() == counts["instances"][0]["visible_pixels"] == 2


@pytest.mark.parametrize("profile", [None, "site1-splat-softness-v2"])
def test_tile_mask_export_preserves_composed_rgb_and_jpeg(profile):
    rgb, depth, alpha, values = layers()
    camera = dict(w=6, h=4, fl_x=4, fl_y=4, cx=3, cy=2)
    scene = SimpleNamespace(
        objects=[],
        ground=GroundCalibration(np.array([0, 0, 1]), -8, 1.21, np.eye(3)),
        course=SimpleNamespace(corridor=[[-100, -100], [100, -100], [100, 100], [-100, 100]]),
    )
    kwargs = dict(camera=camera, response=response_profile(profile), diagnostics=True)
    baseline, counts = compose(scene, np.eye(4), rgb, depth, alpha, values, **kwargs)
    masks = {}
    actual, same_counts = compose(
        scene, np.eye(4), rgb, depth, alpha, values, instance_masks=masks, **kwargs
    )
    np.testing.assert_array_equal(actual, baseline)
    assert counts == same_counts
    assert masks[0].shape == (4, 6) and masks[0].dtype == bool
    assert masks[0].sum() == counts[0]["visible_pixels"] == 2
    encoded = []
    for pixels in (baseline, actual):
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, "JPEG", quality=92)
        encoded.append(stream.getvalue())
    assert encoded[0] == encoded[1]


def test_native_visibility_uses_linear_warp_weights_body_and_unique_pixels():
    pytest.importorskip("cv2")
    mask = np.array([[1, 0, 0], [0, 0, 0]], bool)
    mapping = np.zeros((2, 4, 2), np.float32)
    mapping[..., 0] = [0, 0.5, 1, 99]
    mapping[1, :, 1] = 1
    body = np.zeros((2, 4))
    body[0, 0] = 1
    weights = [np.full((2, 4), 0.5)] * 2
    result = visible_billboards(
        [{"hazard": mask}, {"hazard": mask}], [mapping] * 2, weights, body, ["hazard", "absent"]
    )
    assert result == {"hazard": {"visible_pixels": 1}, "absent": {"visible_pixels": 0}}
    assert visible_billboards(
        [{"hazard": mask}], [mapping], [np.zeros((2, 4))], body, ["hazard"]
    ) == {"hazard": {"visible_pixels": 0}}


def test_transient_tile_masks_are_separate_from_json_metrics():
    renderer = PersistentRenderer.__new__(PersistentRenderer)
    renderer._draw_matrix = lambda matrix, **kwargs: matrix

    def composite(frame, *, visibility_masks, **kwargs):
        visibility_masks["hazard"] = np.full((2, 3), bool(frame))
        return np.full((2, 3, 3), frame, np.uint8), {
            "billboards": {"hazard": {"visible_pixels": frame * 6}}
        }

    renderer._compose_frame = composite
    results = renderer.render_matrices_rgb([(0, {}), (1, {})], visibility_masks=True)
    assert not np.any(results[0][2]["hazard"])
    assert np.all(results[1][2]["hazard"])
    assert results[0][2] is not results[1][2]
    for _, metrics, _ in results:
        json.dumps(metrics)


def test_wide_metadata_does_not_change_encoded_image():
    pytest.importorskip("cv2")
    renderer = WideRenderer.__new__(WideRenderer)
    camera = dict(w=6, h=4, fl_x=4, fl_y=4, cx=3, cy=2)
    renderer.calibration = SimpleNamespace(camera_pose=lambda matrix: matrix, intrinsics=camera)
    renderer.tiles = [{"name": "tile", "rotation_wide_from_tile": np.eye(3)}]
    renderer.cameras = [camera]
    yy, xx = np.indices((4, 6), dtype=np.float32)
    renderer.maps = [np.stack((xx, yy), axis=-1)]
    renderer.weights = [np.ones((4, 6))]
    renderer.body_alpha = np.zeros((4, 6))
    renderer.body_alpha[3] = 1
    renderer.body_rgb = np.full((4, 6, 3), 20, np.uint8)
    mask = np.zeros((4, 6), bool)
    image = np.full((4, 6, 3), 100, np.uint8)
    renderer.renderer = SimpleNamespace(
        scene=SimpleNamespace(camera_matrix=lambda pose: np.eye(4)),
        render_matrices_rgb=lambda views, **kwargs: [
            (
                image,
                {"billboards": {"hazard": {"visible_pixels": int(mask.sum())}}},
                {"hazard": mask},
            )
        ],
    )
    before = renderer.render((0, 0, 0))
    assert renderer.last_metrics["billboards"]["hazard"]["visible_pixels"] == 0
    mask[1, 1] = True
    mask[3] = True
    after = renderer.render((0, 0, 0))
    assert before == after
    assert renderer.last_metrics["billboards"]["hazard"]["visible_pixels"] == 1
    json.dumps(renderer.last_metrics)
