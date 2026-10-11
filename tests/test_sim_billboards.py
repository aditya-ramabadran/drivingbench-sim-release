import json
from dataclasses import replace

import numpy as np
import pytest
from drivingbench.sim.billboards import BillboardCache
from drivingbench.sim.cameras import SceneRenderer
from drivingbench.sim.scene import Scene
from drivingbench.sim.visual_objects import PlacedObject, composite_objects, empty_layer
from PIL import Image

CAMERA = dict(w=64, h=64, fl_x=50, fl_y=50, cx=32, cy=32)
# Camera looks along +Y; image down is -Z.
MATRIX = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1], [0, 0, 0, 1]], dtype=float)


def asset(tmp_path):
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[2:14, 2:14] = [220, 80, 40, 255]
    pixels[6:10, 6:10] = 0
    path = tmp_path / "person.png"
    Image.fromarray(pixels).save(path)
    return PlacedObject("billboard", 0, 5, asset_path=str(path), width_m=2, height_m=2)


def test_alpha_holes_and_depth_order_preserve_foreground(tmp_path):
    obj = asset(tmp_path)
    cache = BillboardCache([obj])
    layer = cache.layers([obj], MATRIX, CAMERA)[0]
    assert layer.rgba[32, 32, 3] == 0
    assert np.isnan(layer.depth[32, 32])
    assert layer.rgba[25, 25, 3] == 255
    assert layer.depth[25, 25] == pytest.approx(5)
    background = np.full((64, 64, 3), 20, dtype=np.uint8)
    depth = np.full((64, 64), 8.0)
    depth[:, :32] = 3
    result, _ = composite_objects(background, depth, np.ones((64, 64)), layer)
    np.testing.assert_array_equal(result[:, :32], background[:, :32])
    assert (result[:, 32:] != background[:, 32:]).any()
    far = replace(obj, y=7)
    a, _ = composite_objects(
        background, depth, np.ones((64, 64)), cache.layers([obj, far], MATRIX, CAMERA)
    )
    b, _ = composite_objects(
        background, depth, np.ones((64, 64)), cache.layers([far, obj], MATRIX, CAMERA)
    )
    np.testing.assert_array_equal(a, b)


def test_partial_clipping_orientation_and_behind_camera(tmp_path):
    obj = asset(tmp_path)
    cache = BillboardCache([obj])
    partial = cache.layers([replace(obj, x=3)], MATRIX, CAMERA)[0]
    assert partial.rgba[:, -1, 3].any()
    edge = cache.layers([replace(obj, yaw=np.pi / 2)], MATRIX, CAMERA)[0]
    assert not edge.rgba.any()
    behind = cache.layers([replace(obj, y=-5)], MATRIX, CAMERA)[0]
    assert not behind.rgba.any()
    near = cache.layers([replace(obj, y=0.2, yaw=0.5)], MATRIX, CAMERA)[0]
    assert np.isfinite(near.depth[near.rgba[..., 3] > 0]).all()


def test_ground_anchor_accounts_for_transparent_canvas_padding(tmp_path):
    obj = asset(tmp_path)
    # The visible bottom ends at 14/16 of the image: ground anchor follows it.
    obj = replace(obj, anchor_uv=(0.5, 14 / 16))
    layer = BillboardCache([obj]).layers([obj], MATRIX, CAMERA)[0]
    rows = np.nonzero(layer.rgba[..., 3] > 128)[0]
    assert rows.max() == 41  # Ground projection is y=42; last pixel center is 41.5.


def test_assets_load_once(tmp_path, monkeypatch):
    obj = asset(tmp_path)
    cache = BillboardCache([obj])
    monkeypatch.setattr(Image, "open", lambda *_: pytest.fail("Asset reopened"))
    cache.layers([obj], MATRIX, CAMERA)[0]
    cache.layers([replace(obj, x=1)], MATRIX, CAMERA)
    assert len(cache.images) == 1


def test_material_saturation_changes_color_without_alpha_depth_or_cache_leak(tmp_path):
    obj = asset(tmp_path)
    cache = BillboardCache([obj])
    original = cache.project(obj, MATRIX, CAMERA)
    identity = cache.project(replace(obj, material_saturation=1), MATRIX, CAMERA)
    muted = cache.project(replace(obj, material_saturation=0.65), MATRIX, CAMERA)
    gray = cache.project(replace(obj, material_saturation=0), MATRIX, CAMERA)
    np.testing.assert_array_equal(original.rgba, identity.rgba)
    for layer in (muted, gray):
        np.testing.assert_array_equal(layer.rgba[..., 3], original.rgba[..., 3])
        np.testing.assert_array_equal(layer.depth, original.depth)
    covered = original.rgba[..., 3] > 0
    assert (
        np.ptp(muted.rgba[covered, :3], axis=1).mean()
        < np.ptp(original.rgba[covered, :3], axis=1).mean()
    )
    assert np.all(np.ptp(gray.rgba[covered, :3], axis=1) == 0)
    np.testing.assert_array_equal(cache.project(obj, MATRIX, CAMERA).rgba, original.rgba)
    assert len(cache.images) == 1


def test_empty_billboard_layers_preserve_all_background_bytes():
    background = np.full((64, 64, 3), 180, np.uint8)
    actual, _ = composite_objects(
        background, np.ones((64, 64)), np.ones((64, 64)), [empty_layer(CAMERA), empty_layer(CAMERA)]
    )
    np.testing.assert_array_equal(actual, background)


def test_scene_object_time_uses_same_ground_plane_and_trajectory(tmp_path):
    ground = tmp_path / "ground.json"
    ground.write_text(
        json.dumps(
            dict(normal=[0, 0, 1], offset=0.2, height=1.21, mount_rotation=np.eye(3).tolist())
        )
    )
    obj = asset(tmp_path)
    from dataclasses import asdict

    scene = dict(
        frame="pilot_ground",
        calibration=str(ground),
        background={"checkpoint": "x"},
        start=dict(x=0, y=0, heading_deg=0),
        markers=[],
        finish={"polygon": [[8, -2], [12, -2], [12, 2], [8, 2]]},
        boundaries={"left": [[-5, 3], [15, 3]], "right": [[-5, -3], [15, -3]]},
        objects=[asdict(obj) | {"id": "person"}],
        scenario={
            "time_limit_s": 60,
            "colliders": [{"id": "person", "circle": [0, 5, 0.3], "outcome": "hazard"}],
            "trajectories": {"person": {"start_s": 2, "duration_s": 4, "delta": [2, -4]}},
        },
    )
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    loaded = Scene(path, backend=lambda *_: b"")
    at = loaded.objects_at(4)[0]
    assert (at.x, at.y, at.z) == pytest.approx((1, 3, -0.2))
    assert loaded.objects[0].x == 0


def test_renderer_snapshot_carries_physics_time_without_relabeling(tmp_path):
    renderer = SceneRenderer.__new__(SceneRenderer)
    renderer.output = tmp_path
    received = []

    class Worker:
        def request(self, **payload):
            received.append(payload)
            path = tmp_path / "frame.jpg"
            path.write_bytes(b"jpeg")
            return {"metrics": {}, "paths": {"narrow": str(path)}}

    renderer.worker = Worker()
    snapshot = dict(pose=(1, 2, 3), sim_time_s=12.3, captured_at=99.4, actor_onsets={"hazard": 4.5})
    assert renderer.render_snapshot(snapshot) == b"jpeg"
    assert received[0]["sim_time_s"] == 12.3
    assert received[0]["actor_onsets"] == {"hazard": 4.5}
    assert snapshot["captured_at"] == 99.4


@pytest.mark.parametrize(
    "kind,args",
    [
        ("billboard", dict(asset_path="a.png", width_m=1, height_m=2)),
        ("barrier", dict(width_m=2, depth_m=0.5, height_m=1)),
        ("paint", dict(width_m=2, depth_m=0.1)),
    ],
)
def test_new_object_shapes_validate_positive_dimensions(kind, args):
    PlacedObject(kind, 0, 0, **args)
    with pytest.raises(ValueError):
        PlacedObject(kind, 0, 0, **(args | {"width_m": 0}))


def test_contact_shadow_follows_tilted_ground_and_is_withheld_behind_foreground(tmp_path):
    from drivingbench.sim.visual_ground import road_rotation
    from drivingbench.sim.visual_objects import composite_grounded

    normal = np.array([0.03, -0.02, 1])
    normal /= np.linalg.norm(normal)
    basis = road_rotation(normal, 0)
    obj = replace(asset(tmp_path), x=0, y=0, yaw=0.6)
    local = np.array([[[0, 0, 0], [0.1, 0.1, 0], [10, 10, 0]]])
    points = local @ basis.T
    background = np.full((1, 3, 3), 180, dtype=np.uint8)
    layer = empty_layer(dict(w=3, h=1))
    result, masks = composite_grounded(
        background,
        np.array([[5.0, 3.0, 5.0]]),
        np.ones((1, 3)),
        layer,
        [obj],
        points,
        np.full((1, 3), 5.0),
        np.ones((1, 3), bool),
        ground_normal=normal,
    )
    assert np.all(result[0, 0] < background[0, 0])
    np.testing.assert_array_equal(result[0, 1:], background[0, 1:])
    assert masks["shadow_opacity"][0, 0] == pytest.approx(0.3)


def test_visibility_diagnostics_match_visible_billboard_pixels(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from drivingbench.sim.visual_composite import compose
    from drivingbench.sim.visual_ground import GroundCalibration

    monkeypatch.setattr("drivingbench.sim.visual.CAMERA", CAMERA)
    obj = asset(tmp_path)
    layer = BillboardCache([obj]).layers([obj], MATRIX, CAMERA)[0]
    scene = SimpleNamespace(
        objects=[obj],
        ground=GroundCalibration(np.array([0.0, 0, 1]), 0, 1.21, np.eye(3)),
        course=SimpleNamespace(corridor=[[-10, -10], [10, -10], [10, 10], [-10, 10]]),
    )
    background = np.full((64, 64, 3), 20, dtype=np.uint8)
    raw = np.full((64, 64), 8.0)
    raw[:, :32] = 3
    _, counts = compose(scene, MATRIX, background, raw, np.ones((64, 64)), layer, diagnostics=True)
    assert counts[0]["visible_pixels"] > 0
    assert counts[0]["occluded_pixels"] > 0
    assert counts[0]["projected_pixels"] == sum(
        counts[0][key] for key in ("visible_pixels", "occluded_pixels", "unknown_pixels")
    )


@pytest.mark.parametrize("near_alpha", [128, 253])
@pytest.mark.parametrize("reverse", [False, True])
def test_semtransparent_front_object_never_reveals_object_behind_real_wall(near_alpha, reverse):
    from drivingbench.sim.visual_objects import ObjectLayer, composite_grounded

    background = np.array([[[20, 100, 30]]], np.uint8)
    near = ObjectLayer(
        np.array([[[220, 30, 10, near_alpha]]], np.uint8),
        np.array([[2.0]]),
        instance=np.array([[-1]]),
    )
    far = ObjectLayer(
        np.array([[[10, 0, 255, 255]]], np.uint8), np.array([[4.0]]), instance=np.array([[-2]])
    )
    layers = [near, far] if reverse else [far, near]
    args = (background, np.array([[3.0]]), np.ones((1, 1)))
    expected, _ = composite_objects(*args, near, margin_m=0)
    result, masks = composite_objects(*args, layers, margin_m=0)
    np.testing.assert_array_equal(result, expected)
    assert masks["instances"][1]["visible_pixels"] == 0
    assert masks["instances"][1]["occluded_pixels"] == 1
    grounded, _ = composite_grounded(
        *args, layers, [], np.zeros((1, 1, 3)), np.array([[8.0]]), np.ones((1, 1), bool)
    )
    np.testing.assert_array_equal(grounded, expected)


def test_visible_translucent_layers_are_depth_sorted_and_opaque_foreground_blocks_exposure():
    from drivingbench.sim.visual_objects import ObjectLayer

    background = np.zeros((1, 1, 3), np.uint8)
    front = ObjectLayer(
        np.array([[[255, 0, 0, 128]]], np.uint8), np.array([[2.0]]), instance=np.array([[-1]])
    )
    back = ObjectLayer(
        np.array([[[0, 0, 255, 255]]], np.uint8), np.array([[4.0]]), instance=np.array([[-2]])
    )
    args = (background, np.array([[8.0]]), np.ones((1, 1)))
    forward, masks = composite_objects(*args, [front, back])
    reverse, _ = composite_objects(*args, [back, front])
    np.testing.assert_array_equal(forward, reverse)
    assert forward[0, 0, 0] > 180 and forward[0, 0, 2] > 180
    assert masks["instances"][1]["visible_pixels"] == 1
    front.rgba[..., 3] = 255
    opaque, masks = composite_objects(*args, [front, back])
    np.testing.assert_array_equal(opaque, [[[255, 0, 0]]])
    assert masks["instances"][1]["visible_pixels"] == 0
    assert masks["instances"][1]["occluded_pixels"] == 1


def test_camera_facing_billboard_keeps_shape_and_ground_anchor_across_azimuths(tmp_path):
    obj = replace(asset(tmp_path), face_camera=True, yaw=np.pi / 2, anchor_uv=(0.5, 14 / 16))
    cache = BillboardCache([obj])
    baseline = cache.project(obj, MATRIX, CAMERA)
    origin = np.array([obj.x, obj.y, obj.z])
    for angle in (np.pi / 2, np.pi, -np.pi / 2):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        matrix = MATRIX.copy()
        matrix[:3, :3] = rotation @ MATRIX[:3, :3]
        matrix[:3, 3] = origin + rotation @ (MATRIX[:3, 3] - origin)
        layer = cache.project(obj, matrix, CAMERA)
        np.testing.assert_allclose(layer.rgba, baseline.rgba, atol=1)
        np.testing.assert_allclose(layer.depth, baseline.depth, equal_nan=True)
        assert np.nonzero(layer.rgba[..., 3] > 128)[0].max() == 41
    fixed = cache.project(replace(obj, face_camera=False), MATRIX, CAMERA)
    assert not fixed.rgba.any()  # Existing authored yaw remains edge-on.


def test_wide_tile_rotations_share_camera_facing_world_plane(tmp_path):
    obj = replace(asset(tmp_path), face_camera=True, yaw=0.8)
    cache = BillboardCache([obj])
    for angle in (-0.3, 0, 0.3):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        matrix = MATRIX.copy()
        matrix[:3, :3] = rotation @ matrix[:3, :3]
        layer = cache.project(obj, matrix, CAMERA)
        ys, xs = np.nonzero(layer.rgba[..., 3])
        assert len(xs) > 0
        rays = np.column_stack(
            (
                (xs + 0.5 - CAMERA["cx"]) / CAMERA["fl_x"],
                (ys + 0.5 - CAMERA["cy"]) / CAMERA["fl_y"],
                np.ones(len(xs)),
            )
        )
        points = (rays * layer.depth[ys, xs, None]) @ matrix[:3, :3].T + matrix[:3, 3]
        np.testing.assert_allclose(points[:, 1], obj.y, atol=1e-6)


def test_camera_facing_billboard_preserves_ground_anchor_on_tilted_plane(tmp_path):
    obj = replace(asset(tmp_path), face_camera=True, anchor_uv=(0.5, 14 / 16))
    cache = BillboardCache([obj])
    baseline = cache.project(obj, MATRIX, CAMERA)
    angle = 0.2
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    matrix = MATRIX.copy()
    matrix[:3, :3] = rotation @ matrix[:3, :3]
    matrix[:3, 3] = rotation @ matrix[:3, 3]
    position = rotation @ [obj.x, obj.y, obj.z]
    tilted = replace(obj, x=position[0], y=position[1], z=position[2])
    layer = cache.project(tilted, matrix, CAMERA, rotation[:, 2])
    np.testing.assert_allclose(layer.rgba, baseline.rgba, atol=1)
    np.testing.assert_allclose(layer.depth, baseline.depth, equal_nan=True)
