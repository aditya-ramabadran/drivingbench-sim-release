from collections import Counter

import numpy as np
import pytest
from drivingbench.sim.visual_objects import (
    PlacedObject,
    barrel_mesh,
    build_object_job,
    contact_shadows,
    jersey_mesh,
)


def assert_closed_positive_mesh(vertices, faces):
    edges = Counter()
    signed_volume = 0.0
    points = np.asarray(vertices)
    for face in faces:
        for first, second in zip(face, (*face[1:], face[0])):
            edges[tuple(sorted((first, second)))] += 1
        for i in range(1, len(face) - 1):
            a, b, c = points[list((face[0], face[i], face[i + 1]))]
            signed_volume += np.dot(a, np.cross(b, c)) / 6
    assert set(edges.values()) == {2}
    assert signed_volume > 0


def test_jersey_closed_mesh_has_exact_collision_base_and_narrow_upper_face():
    vertices, faces = jersey_mesh(2.4, 0.60, 0.81)
    points = np.asarray(vertices)
    np.testing.assert_allclose(np.ptp(points, axis=0), [2.4, 0.60, 0.81])
    assert points[:, 2].min() == 0
    base = points[points[:, 2] == 0]
    assert len(base) == 4
    np.testing.assert_allclose(np.ptp(base, axis=0), [2.4, 0.6, 0])
    assert np.ptp(points[points[:, 2] == 0.81, 1]) < 0.2
    assert_closed_positive_mesh(vertices, faces)


def test_barrel_foot_matches_collision_circle_and_body_is_closed():
    vertices, faces = barrel_mesh(0.58, 0.95)
    points = np.asarray(vertices)
    np.testing.assert_allclose(np.ptp(points[:, :2], axis=0), [0.58, 0.58])
    assert points[:, 2].min() == 0
    assert points[:, 2].max() == pytest.approx(0.875 * 0.95)
    # The Blender carry handle occupies the remaining height and meets the top cap.
    assert 0.9375 * 0.95 - (0.05 + 0.0125) * 0.95 == pytest.approx(points[:, 2].max())
    assert (0.9375 + 0.05 + 0.0125) * 0.95 == pytest.approx(0.95)
    foot = points[points[:, 2] == 0]
    np.testing.assert_allclose(np.linalg.norm(foot[:, :2], axis=1), 0.29)
    assert_closed_positive_mesh(vertices, faces)


@pytest.mark.parametrize(
    "constructor,args", [(jersey_mesh, (2.4, 0.6, 0)), (barrel_mesh, (0.58, np.nan))]
)
def test_construction_mesh_rejects_invalid_dimensions(constructor, args):
    with pytest.raises(ValueError):
        constructor(*args)


def test_new_object_serialization_keeps_shape_and_shadow_declaration():
    jersey = PlacedObject(
        "jersey_barrier", 1, 2, width_m=2.4, depth_m=0.6, height_m=0.81, shadow_strength=0.45
    )
    barrel = PlacedObject(
        "traffic_barrel", 3, 4, diameter_m=0.58, height_m=0.95, shadow_strength=0.45
    )
    job = build_object_job([jersey, barrel])
    assert [row["kind"] for row in job["objects"]] == ["jersey_barrier", "traffic_barrel"]
    assert all(row["shadow_strength"] == 0.45 for row in job["objects"])
    for strength in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="shadow_strength"):
            PlacedObject("cone", 0, 0, shadow_strength=strength)


def test_stronger_contact_is_explicit_and_never_changes_legacy_cones():
    background = np.full((1, 3, 3), 180, np.uint8)
    points = np.array([[[0.0, 0, 0], [0.1, 0, 0], [2, 0, 0]]])
    visible = np.array([[True, False, True]])
    old, _ = contact_shadows(background, [PlacedObject("cone", 0, 0)], points, visible)
    np.testing.assert_array_equal(old, background)
    revised, opacity = contact_shadows(
        background, [PlacedObject("cone", 0, 0, shadow_strength=0.45)], points, visible
    )
    assert opacity[0, 0] == pytest.approx(0.45)
    assert np.all(revised[0, 0] < background[0, 0])
    np.testing.assert_array_equal(revised[0, 1:], background[0, 1:])
