import numpy as np
import pytest
from drivingbench.sim.visual_objects import (
    ObjectLayer,
    PlacedObject,
    blender_depth,
    composite_grounded,
    composite_objects,
    contact_shadows,
    dome_mesh,
    marker_repair_mask,
    repair_background,
    repair_edge_depth,
)


def test_imported_material_saturation_validates_and_survives_object_job():
    from drivingbench.sim.visual_objects import build_object_job

    for value in (None, 0, 0.15, 1):
        obj = PlacedObject("asset", 0, 0, asset_path="drum.glb", material_saturation=value)
        assert build_object_job([obj])["objects"][0]["material_saturation"] == value
    for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.15"):
        with pytest.raises(ValueError, match="material_saturation"):
            PlacedObject("asset", 0, 0, asset_path="drum.glb", material_saturation=value)
    with pytest.raises(ValueError, match="static imported GLB"):
        PlacedObject("cone", 0, 0, material_saturation=0.15)
    obj = PlacedObject(
        "billboard", 0, 0, asset_path="actor.png", width_m=1, height_m=2, material_saturation=0.15
    )
    assert obj.material_saturation == 0.15


@pytest.mark.parametrize("textured", [False, True])
def test_material_saturation_changes_only_private_base_color_graph(textured):
    from types import SimpleNamespace
    from unittest.mock import MagicMock, call

    from drivingbench.sim.visual_objects import _blender_material_saturation

    source, material, shader, gray, blend = (MagicMock() for _ in range(5))
    source.copy.return_value = material
    source.use_nodes = True
    graph = material.node_tree
    shader.bl_idname = "ShaderNodeBsdfPrincipled"
    graph.nodes.__iter__.return_value = iter([shader])
    graph.nodes.new.side_effect = [gray, blend]
    blend.inputs = [MagicMock() for _ in range(3)]
    base = shader.inputs["Base Color"]
    base.default_value = (0.2, 0.3, 0.4, 1)
    original = MagicMock()
    base.links = [SimpleNamespace(from_socket=original)] if textured else []
    changed = SimpleNamespace(type="MESH", material_slots=[SimpleNamespace(material=source)])
    sibling = SimpleNamespace(type="MESH", material_slots=[SimpleNamespace(material=source)])
    _blender_material_saturation([changed], 0.15)
    assert changed.material_slots[0].material is material
    assert changed.material_slots[0].link == "OBJECT"
    assert sibling.material_slots[0].material is source
    assert blend.blend_type == "MIX" and blend.inputs[0].default_value == 0.15
    assert gray.inputs[0].default_value == blend.inputs[2].default_value == base.default_value
    expected = (
        [call(original, gray.inputs[0]), call(original, blend.inputs[2])] if textured else []
    ) + [call(gray.outputs[0], blend.inputs[1]), call(blend.outputs[0], base)]
    assert graph.links.new.call_args_list == expected
    assert all(c == call("Base Color") for c in shader.inputs.__getitem__.call_args_list)
    source.node_tree.nodes.new.assert_not_called()
    for value in (None, 1):
        _blender_material_saturation([sibling], value)
    source.copy.assert_called_once_with()


def test_placement_validation_and_rendering_only_metadata():
    cone = PlacedObject("cone", 4, -2, scale=2, yaw=0.3)
    assert cone.scale == 2 and cone.yaw == 0.3
    assert not hasattr(cone, "footprint")
    assert PlacedObject("asset", 0, 0, asset_path="dog.glb").asset_path == "dog.glb"
    for kwargs in ({"scale": 0}, {"x": float("nan")}, {"asset_path": "a.glb"}):
        with pytest.raises(ValueError):
            PlacedObject(**(dict(kind="cone", x=0, y=0) | kwargs))
    with pytest.raises(ValueError):
        PlacedObject("asset", 0, 0)


def test_depth_occlusion_distinguishes_known_foreground_from_unknown():
    background = np.full((1, 5, 3), 40, dtype=np.uint8)
    rgba = np.tile([200, 100, 0, 255], (1, 5, 1)).astype(np.uint8)
    layer = ObjectLayer(rgba, np.array([[3.0, 3, 3, 3, np.nan]]))
    depth = np.array([[10.0, 2, np.nan, 10, 10]])
    alpha = np.array([[1.0, 1, 1, 0.2, 1]])
    pixels, masks = composite_objects(background, depth, alpha, layer)
    np.testing.assert_array_equal(pixels[0, 0], [200, 100, 0])
    np.testing.assert_array_equal(pixels[0, 1:], background[0, 1:])
    np.testing.assert_array_equal(masks["visible"], [[True, False, False, False, False]])
    np.testing.assert_array_equal(masks["occluded"], [[False, True, False, False, False]])
    np.testing.assert_array_equal(masks["unknown"], [[False, False, True, True, True]])


def test_object_straight_alpha_and_empty_pixels_preserve_background():
    background = np.full((1, 2, 3), 100, dtype=np.uint8)
    background[0, 0] = 0
    layer = ObjectLayer(
        np.array([[[255, 255, 255, 128], [200, 0, 0, 0]]], dtype=np.uint8), np.full((1, 2), 3.0)
    )
    pixels, masks = composite_objects(background, np.full((1, 2), 5), np.ones((1, 2)), layer)
    np.testing.assert_array_equal(pixels, [[[188, 188, 188], [100, 100, 100]]])
    assert not any(mask[0, 1] for mask in masks.values())


def test_blender_depth_flips_rows_and_rejects_background_sentinel():
    depth = blender_depth([[2, 1e10], [4, 0]])
    np.testing.assert_allclose(depth[:, 0], [4, 2])
    assert np.isnan(depth[:, 1]).all()
    assert depth.dtype == np.float32


def test_edge_repair_never_propagates_or_fills_uncovered_background():
    rgba = np.zeros((1, 5, 4), dtype=np.uint8)
    rgba[0, 1:, 3] = [255, 128, 64, 128]
    depth = np.array([[np.nan, 3, np.nan, np.nan, np.nan]])
    layer = repair_edge_depth(rgba, depth, np.array([[0, 2, 0, 0, 0]]))
    repaired_depth, repaired = layer.depth, layer.repaired
    np.testing.assert_array_equal(layer.instance, [[0, 2, 2, 0, 0]])
    np.testing.assert_array_equal(repaired, [[False, False, True, False, False]])
    np.testing.assert_allclose(repaired_depth[0, 1:3], [3, 3])
    assert np.isnan(repaired_depth[0, [0, 3, 4]]).all()
    assert np.isnan(depth[0, 2])


def test_dome_parameters_and_mesh_have_explicit_outer_dimensions():
    args = dict(
        kind="dome_marker", x=1, y=2, diameter_m=0.16, height_m=0.08, color_srgb=(0.6, 0.4, 0.3)
    )
    marker = PlacedObject(**args)
    vertices, faces = dome_mesh(marker.diameter_m, marker.height_m)
    points = np.asarray(vertices)
    np.testing.assert_allclose(points.max(axis=0) - points.min(axis=0), [0.16, 0.16, 0.08])
    assert points[:, 2].min() == 0 and len(faces) == 384
    for invalid in (
        {"diameter_m": None},
        {"height_m": 0},
        {"color_srgb": None},
        {"color_srgb": (1.1, 0, 0)},
        {"color_srgb": (0, 0)},
        {"diameter_m": np.nan},
    ):
        with pytest.raises(ValueError):
            PlacedObject(**(args | invalid))


def test_background_repair_preserves_unmasked_and_refuses_nested_oversized_region():
    rgb = np.full((25, 25, 3), 110, dtype=np.uint8)
    mask = np.zeros((25, 25), dtype=bool)
    mask[3:22, 3] = mask[3:22, 21] = True
    mask[3, 3:22] = mask[21, 3:22] = True
    mask[6:19, 6:19] = True
    rgb[mask] = 240
    result, diagnostics = repair_background(rgb, mask, max_component_pixels=100)
    np.testing.assert_array_equal(result, rgb)
    assert not diagnostics["repaired"].any()
    assert diagnostics["unrepaired"].sum() == mask.sum()


def test_background_repair_only_uses_unmasked_splat_neighbors():
    rgb = np.full((12, 12, 3), 100, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:7, 4:7] = True
    rgb[mask] = [240, 0, 0]
    result, diagnostics = repair_background(rgb, mask)
    np.testing.assert_array_equal(result[~mask], rgb[~mask])
    assert (result[mask] == 100).all() and diagnostics["repaired"].sum() == 9
    edge = mask.copy()
    edge[0, 0] = True
    _, diagnostics = repair_background(rgb, edge)
    assert diagnostics["unrepaired"][0, 0]


def test_marker_repair_mask_is_bounded_and_preserves_foreground_unknowns():
    rgba = np.zeros((13, 13, 4), dtype=np.uint8)
    rgba[6, 6, 3] = 255
    z = np.full((13, 13), np.nan)
    z[6, 6] = 3
    layer = ObjectLayer(rgba, z)
    scene = np.full((13, 13), 10.0)
    alpha = np.ones_like(scene)
    scene[5, 5], scene[6, 6], alpha[7, 7] = 2, np.nan, 0.1
    mask = marker_repair_mask(layer, scene, alpha)
    assert mask.sum() == 49 - 3
    assert not mask[5, 5] and not mask[6, 6] and not mask[7, 7]
    assert not mask[:3].any() and not mask[:, :3].any()
    with pytest.raises(ValueError):
        marker_repair_mask(layer, scene, alpha, radius=5)


def test_edge_instance_repair_respects_known_ids_and_withholds_ambiguous_samples():
    rgba = np.full((1, 3, 4), 255, dtype=np.uint8)
    z = np.array([[2, np.nan, 8]])
    known = repair_edge_depth(rgba, z, np.array([[1, 2, 2]]))
    assert known.depth[0, 1] == 8 and known.instance[0, 1] == 2
    ambiguous = repair_edge_depth(rgba, z, np.array([[1, 0, 2]]))
    assert np.isnan(ambiguous.depth[0, 1]) and ambiguous.instance[0, 1] == 0
    assert not ambiguous.repaired[0, 1]


def test_ground_rotation_preserves_contact_on_sloped_plane():
    from drivingbench.sim.visual_ground import road_rotation

    n = np.array([0.1, -0.05, 1.0])
    n /= np.linalg.norm(n)
    rotation = road_rotation(n, 0.3)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(rotation[:, 2], n)
    vertices, _ = dome_mesh(0.2, 0.06)
    base = np.asarray(vertices)[:64] @ rotation.T
    np.testing.assert_allclose(base @ n, 0, atol=1e-14)
    for invalid in ([0, 0, 0], [0, 0, -1], [np.nan, 0, 1]):
        with pytest.raises(ValueError):
            road_rotation(invalid, 0)


def test_contact_shadow_is_soft_bounded_and_preserves_foreground_unknowns():
    obj = PlacedObject(
        "dome_marker", 0, 0, diameter_m=0.2, height_m=0.06, color_srgb=(0.8, 0.3, 0.1)
    )
    rgb = np.full((1, 6, 3), 150, dtype=np.uint8)
    points = np.array(
        [[[0, 0, 0], [0.1, 0, 0], [0.21, 0, 0], [0, 0, 0], [np.nan, 0, 0], [0, 0, 0]]]
    )
    visible = np.array([[True, True, True, False, True, False]])
    result, opacity = contact_shadows(rgb, [obj], points, visible)
    assert 0 < opacity[0, 1] < opacity[0, 0] == pytest.approx(0.3)
    assert result[0, 0, 0] < result[0, 1, 0] < 150
    np.testing.assert_array_equal(result[0, 2:], rgb[0, 2:])
    assert not opacity[0, 2:].any()
    same, none = contact_shadows(rgb, [], points, visible)
    np.testing.assert_array_equal(same, rgb)
    assert not none.any()


def test_contact_shadow_batch_handles_overlap_without_negative_light():
    dome = PlacedObject(
        "dome_marker", 0, 0, diameter_m=0.2, height_m=0.06, color_srgb=(0.8, 0.3, 0.1)
    )
    rgb = np.full((1, 1, 3), 100, dtype=np.uint8)
    result, opacity = contact_shadows(
        rgb, [dome, dome, PlacedObject("cone", 0, 0)], np.zeros((1, 1, 3)), np.ones((1, 1), bool)
    )
    assert opacity.item() == pytest.approx(1 - 0.7**2)
    assert 0 < result[0, 0, 0] < 100


def test_grounded_composite_keeps_pole_unknown_and_outside_corridor():
    rgb = np.full((1, 5, 3), 100, np.uint8)
    rgba = np.full((1, 5, 4), 255, np.uint8)
    layer = ObjectLayer(rgba, np.array([[10.0, 8.0, 8.0, 8.0, 10.06]]))
    raw = np.array([[9.8, 6.0, np.nan, 6.0, 11.0]])
    original = raw.copy()
    plane = np.full((1, 5), 10.0)
    region = np.array([[True, True, True, False, True]])
    result, masks = composite_grounded(
        rgb, raw, np.ones_like(raw), layer, [], np.zeros((1, 5, 3)), plane, region
    )
    assert masks["visible"].tolist() == [[True, False, False, False, False]]
    assert masks["foreground"].tolist() == [[False, True, False, False, False]]
    assert masks["ground_unknown"].tolist() == [[False, False, True, False, False]]
    assert masks["unknown"][0, 2] and masks["occluded"][0, 1]
    np.testing.assert_array_equal(result[0, 1:], rgb[0, 1:])
    np.testing.assert_array_equal(raw, original)


def test_face_camera_is_explicit_billboard_only_and_survives_job_serialization():
    from drivingbench.sim.visual_objects import build_object_job

    obj = PlacedObject(
        "billboard", 0, 0, asset_path="ball.png", width_m=0.2, height_m=0.2, face_camera=True
    )
    assert build_object_job([obj])["objects"][0]["face_camera"] is True
    assert not PlacedObject("cone", 0, 0).face_camera
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="face_camera"):
            PlacedObject(
                "billboard",
                0,
                0,
                asset_path="ball.png",
                width_m=0.2,
                height_m=0.2,
                face_camera=value,
            )
    with pytest.raises(ValueError, match="face_camera"):
        PlacedObject("cone", 0, 0, face_camera=True)
