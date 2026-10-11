import hashlib

import numpy as np
import pytest
from drivingbench.sim.visual_objects import PlacedObject, contact_shadows


def shadow_fixture():
    x, y = np.meshgrid(np.linspace(-4, 4, 101), np.linspace(-3, 3, 81))
    normal = np.array([0.03, -0.02, 1.0])
    normal /= np.linalg.norm(normal)
    z = -(normal[0] * x + normal[1] * y) / normal[2]
    points = np.stack((x, y, z), axis=-1)
    rgb = np.random.default_rng(314).integers(1, 255, (*x.shape, 3), dtype=np.uint8)
    allowed = np.ones(x.shape, bool)
    allowed[::7, ::3] = False
    points[0, 0] = np.nan
    objects = [
        PlacedObject(
            "jersey_barrier",
            v,
            0,
            z=-normal[0] * v / normal[2],
            yaw=v,
            width_m=2.4,
            depth_m=0.6,
            height_m=0.81,
            shadow_strength=0.45,
        )
        for v in (-1.2, 0, 1.2)
    ]
    objects.append(
        PlacedObject("dome_marker", 0, 0, diameter_m=0.2, height_m=0.06, color_srgb=(1, 0, 0))
    )
    return rgb, objects, points, allowed, normal


def test_common_road_basis_preserves_mixed_tilted_overlapping_shadows():
    """RGB golden and opacity samples came from the original per-object projection.

    Byte RGB is the rendering contract; opacity tolerates floating-point platform variation.
    """
    rgb, objects, points, allowed, normal = shadow_fixture()
    result, opacity = contact_shadows(rgb, objects, points, allowed, ground_normal=normal)
    assert (
        hashlib.sha256(result.tobytes()).hexdigest()
        == "3e2acb2223d5cf41b70be8ea9343bf5161ce952389e25e523298a7ef05b8505e"
    )
    assert np.isfinite(opacity).all()
    assert np.all((opacity >= 0) & (opacity <= 1))
    assert opacity[0, 0] == 0
    assert opacity[40, 50] == pytest.approx(1 - (1 - 0.45) * (1 - 0.30), abs=2e-7)
    np.testing.assert_allclose(
        opacity[[40, 50], [65, 40]], [0.5146104693, 0.0003690371], rtol=0, atol=2e-7
    )
    np.testing.assert_array_equal(result[~allowed], rgb[~allowed])
    np.testing.assert_array_equal(opacity[~allowed], 0)
