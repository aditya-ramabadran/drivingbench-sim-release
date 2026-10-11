"""Selective compositing must match full-frame geometry and color exactly."""

from types import SimpleNamespace

import numpy as np
import pytest
from drivingbench.sim.camera_response import response_profile
from drivingbench.sim.visual import ground_region
from drivingbench.sim.visual_composite import _support, compose
from drivingbench.sim.visual_ground import GroundCalibration
from drivingbench.sim.visual_objects import ObjectLayer, PlacedObject, composite_grounded
from drivingbench.sim.visual_pose import camera_to_world


@pytest.fixture
def camera(monkeypatch):
    value = dict(w=80, h=56, fl_x=68, fl_y=68, cx=40, cy=28)
    monkeypatch.setattr("drivingbench.sim.visual.CAMERA", value)
    return value


def setup_scene():
    normal = np.array([0.03, -0.015, 1.0])
    normal /= np.linalg.norm(normal)
    ground = GroundCalibration(normal, 0.2, 1.21, camera_to_world(0, 0, 0)[:3, :3])
    objects = []
    for x, y in ((4, 0), (4.1, 0.1), (7, -1.9), (-2, 0), (2, 4)):
        point = ground.map_from_ground @ [x, y, 0, 1]
        objects.append(
            PlacedObject(
                "dome_marker", *point[:3], diameter_m=0.2, height_m=0.06, color_srgb=(0.8, 0.3, 0.1)
            )
        )
    return SimpleNamespace(
        ground=ground,
        objects=objects,
        course=SimpleNamespace(corridor=((-2, -2), (10, -2), (10, 2), (-2, 2))),
    )


def reference(scene, matrix, rgb, depth, alpha, layer, camera):
    points, plane = scene.ground.ray_intersections(matrix, camera)
    transform = scene.ground.map_from_ground
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    region = ground_region(local, scene.course.corridor, margin=0.4)
    return composite_grounded(rgb, depth, alpha, layer, scene.objects, points, plane, region)[0]


@pytest.mark.parametrize("seed", range(6))
def test_selective_pixels_match_full_image_with_occlusion_unknowns_and_shadows(camera, seed):
    scene = setup_scene()
    rng = np.random.default_rng(seed)
    matrix = scene.ground.camera_from_ground(0, 0, rng.uniform(-0.35, 0.35))
    shape = (camera["h"], camera["w"])
    rgb = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    _, plane = scene.ground.ray_intersections(matrix, camera)
    depth = plane + 0.2
    depth[35:40, 20:50] = 2
    depth[42:46, 25:30] = np.nan
    alpha = np.ones(shape)
    alpha[::7] = 0.2
    rgba = rng.integers(0, 256, (*shape, 4), dtype=np.uint8)
    rgba[..., 3] = 0
    rgba[32:47, 25:45, 3] = rng.integers(1, 256, (15, 20), dtype=np.uint8)
    object_depth = plane - 0.2
    object_depth[36:39, 29:33] = np.nan
    layer = ObjectLayer(rgba, object_depth)
    expected = reference(scene, matrix, rgb, depth, alpha, layer, camera)
    actual = compose(scene, matrix, rgb, depth, alpha, layer)
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, rgb)
    assert np.isnan(depth[42:46, 25:30]).all()


def test_shadow_support_selected_even_without_object_alpha(camera):
    scene = setup_scene()
    matrix = scene.ground.camera_from_ground(0, 0, 0)
    shape = (camera["h"], camera["w"])
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    rgb = np.full((*shape, 3), 180, dtype=np.uint8)
    _, plane = scene.ground.ray_intersections(matrix, camera)
    alpha = np.ones(shape)
    layer = ObjectLayer(rgba, np.full(shape, np.nan))
    expected = reference(scene, matrix, rgb, plane, alpha, layer, camera)
    assert np.any(expected != rgb)
    np.testing.assert_array_equal(compose(scene, matrix, rgb, plane, alpha, layer), expected)
    y, _ = _support(matrix, scene.objects, rgba, camera, scene.ground)
    assert len(y) < np.prod(shape) / 4


def test_ground_shadow_crossing_camera_plane_stays_below_horizon(camera):
    scene = setup_scene()
    matrix = scene.ground.camera_from_ground(0, 0, 0)
    point = scene.ground.map_from_ground @ [0.2, 0, 0, 1]
    scene.objects = [PlacedObject("barrier", *point[:3], width_m=5, depth_m=2, height_m=1)]
    shape = (camera["h"], camera["w"])
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    y, x = _support(matrix, scene.objects, rgba, camera, scene.ground)
    assert 0 < len(y) < np.prod(shape) * 0.6
    rgb = np.full((*shape, 3), 180, np.uint8)
    _, plane = scene.ground.ray_intersections(matrix, camera)
    layer = ObjectLayer(rgba, np.full(shape, np.nan))
    expected = reference(scene, matrix, rgb, plane, np.ones(shape), layer, camera)
    np.testing.assert_allclose(
        compose(scene, matrix, rgb, plane, np.ones(shape), layer), expected, atol=1
    )


def test_ground_shadows_outside_frustum_are_rejected(camera):
    scene = setup_scene()
    matrix = scene.ground.camera_from_ground(0, 0, 0)
    objects = []
    for x, y in ((-3, 0), (3, -30), (3, 30)):
        point = scene.ground.map_from_ground @ [x, y, 0, 1]
        objects.append(
            PlacedObject(
                "dome_marker", *point[:3], diameter_m=0.2, height_m=0.06, color_srgb=(1, 0, 0)
            )
        )
    rgba = np.zeros((camera["h"], camera["w"], 4), dtype=np.uint8)
    y, x = _support(matrix, objects, rgba, camera, scene.ground)
    assert len(y) == len(x) == 0


def test_empty_scene_preserves_all_rgb_bytes_and_rejects_wrong_shape(camera):
    scene = setup_scene()
    scene.objects = []
    shape = (camera["h"], camera["w"])
    rgb = np.arange(np.prod(shape) * 3, dtype=np.uint8).reshape(*shape, 3)
    layer = ObjectLayer(np.zeros((*shape, 4), np.uint8), np.full(shape, np.nan))
    matrix = scene.ground.camera_from_ground(0, 0, 0)
    result = compose(scene, matrix, rgb, np.ones(shape), np.ones(shape), layer)
    np.testing.assert_array_equal(result, rgb)
    with pytest.raises(ValueError, match="native"):
        compose(scene, matrix, rgb[:-1], np.ones(shape), np.ones(shape), layer)


def test_camera_response_keeps_depth_visibility_and_foreground_bytes(camera):
    scene = setup_scene()
    scene.objects = []
    matrix = scene.ground.camera_from_ground(0, 0, 0)
    shape = (camera["h"], camera["w"])
    rgb = np.full((*shape, 3), 100, np.uint8)
    rgba = np.zeros((*shape, 4), np.uint8)
    rgba[20:40, 20:50] = (200, 100, 50, 255)
    depth = np.full(shape, 10.0)
    depth[:, 20:24] = 1
    layer = ObjectLayer(rgba, np.full(shape, 2.0), np.ones(shape, int))
    baseline, expected_visibility = compose(
        scene, matrix, rgb, depth, np.ones(shape), layer, diagnostics=True
    )
    actual, visibility = compose(
        scene,
        matrix,
        rgb,
        depth,
        np.ones(shape),
        layer,
        diagnostics=True,
        response=response_profile("site1-overcast-v1"),
    )
    np.testing.assert_array_equal(actual[:, 20:24], rgb[:, 20:24])
    np.testing.assert_array_equal(actual[:17], rgb[:17])
    assert visibility == expected_visibility
    assert not np.array_equal(actual, baseline)


def test_fully_hidden_inserted_surface_does_not_change_camera_response(camera):
    scene = setup_scene()
    scene.objects = []
    matrix = scene.ground.camera_from_ground(0, 0, 0)
    shape = (camera["h"], camera["w"])
    rgb = np.full((*shape, 3), 100, np.uint8)
    rgba = np.zeros((*shape, 4), np.uint8)
    rgba[20:40, 24:50] = (200, 100, 50, 255)
    depth = np.full(shape, 10.0)
    depth[:, 20:24] = 5
    front = ObjectLayer(rgba, np.full(shape, 2.0))
    hidden = ObjectLayer(rgba, np.full(shape, 8.0))
    kwargs = {"response": response_profile("site1-overcast-v1")}
    single = compose(scene, matrix, rgb, depth, np.ones(shape), front, **kwargs)
    together = compose(scene, matrix, rgb, depth, np.ones(shape), [front, hidden], **kwargs)
    np.testing.assert_array_equal(single, together)
    assert single[30, 23, 0] > rgb[30, 23, 0]


def test_optional_support_is_post_occlusion_and_excludes_response_halo(camera):
    scene = setup_scene()
    scene.objects = []
    matrix = scene.ground.camera_from_ground(0, 0, 0)
    shape = (camera["h"], camera["w"])
    rgb = np.full((*shape, 3), 100, np.uint8)
    rgba = np.zeros((*shape, 4), np.uint8)
    rgba[20:40, 20:50] = (200, 100, 50, 255)
    depth = np.full(shape, 10.0)
    depth[:, 20:24] = 1
    layer = ObjectLayer(rgba, np.full(shape, 2.0))
    args = (scene, matrix, rgb, depth, np.ones(shape), layer)
    baseline, counts = compose(*args, diagnostics=True)
    image, same_counts, support = compose(*args, return_support=True)
    np.testing.assert_array_equal(image, baseline)
    assert counts == same_counts
    assert support.dtype == bool
    assert support[20:40, 24:50].all()
    assert not support[:, :24].any()
    assert not support[:20].any()
    response, _, same_support = compose(
        *args, return_support=True, response=response_profile("site1-overcast-v1")
    )
    np.testing.assert_array_equal(support, same_support)
    assert np.any(response[~support] != image[~support])
    # A generator cannot repaint the real occluder even when its proposal changes everything.
    assert not support[:, 20:24].any()


def test_optional_support_contains_ground_shadows_and_empty_scene_zero_mask(camera):
    scene = setup_scene()
    matrix = scene.ground.camera_from_ground(0, 0, 0)
    shape = (camera["h"], camera["w"])
    rgb = np.full((*shape, 3), 180, np.uint8)
    points, plane = scene.ground.ray_intersections(matrix, camera)
    layer = ObjectLayer(np.zeros((*shape, 4), np.uint8), np.full(shape, np.nan))
    image, _, support = compose(
        scene, matrix, rgb, plane, np.ones(shape), layer, return_support=True
    )
    assert support.any()
    assert np.all(support[np.any(image != rgb, axis=-1)])
    scene.objects = []
    image, counts, support = compose(
        scene, matrix, rgb, plane, np.ones(shape), layer, return_support=True
    )
    np.testing.assert_array_equal(image, rgb)
    assert counts == {} and not support.any()
