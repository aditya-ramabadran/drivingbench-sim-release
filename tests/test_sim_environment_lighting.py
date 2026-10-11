import math

import numpy as np
import pytest
from drivingbench.sim.visual_ground import road_rotation
from drivingbench.sim.visual_objects import build_object_job, environment_lighting


def test_optional_environment_does_not_change_default_job():
    assert build_object_job([], lighting=None) == build_object_job([])
    assert "lighting" not in build_object_job([])
    assert environment_lighting(None, (0, 0, 1)) is None


@pytest.mark.parametrize("normal", [(0, 0, 1), (0.02, -0.03, math.sqrt(1 - 0.0013))])
def test_panorama_cardinals_match_blender_negative_longitude(normal):
    """Blender's environment shader uses -atan(y,x), with UV origin at bottom left."""
    config = environment_lighting({"environment_path": "lot.png"}, normal)
    lookup = np.array(config["environment_lookup_matrix"])
    basis = road_rotation(normal, 0)
    directions = [basis[:, 0], basis[:, 1], -basis[:, 1], basis[:, 2], -basis[:, 2]]
    expected = [(0.5, 0.5), (0.75, 0.5), (0.25, 0.5), (None, 1), (None, 0)]
    for direction, (u, v) in zip(directions, expected):
        x, y, z = lookup @ direction
        actual_u = -math.atan2(y, x) / (2 * math.pi) + 0.5
        actual_v = math.atan2(z, math.hypot(x, y)) / math.pi + 0.5
        if u is not None:
            assert actual_u == pytest.approx(u)
        assert actual_v == pytest.approx(v)


def test_lighting_settings_are_copied_and_retained_in_job():
    settings = dict(
        environment_path="lot.png", strength=0.7, ao_distance_m=0.2, bake_irradiance=True
    )
    original = settings.copy()
    job = build_object_job([], lighting=settings)
    assert settings == original
    assert {k: job["lighting"][k] for k in original} == original
    assert job["lighting"]["environment_lookup_matrix"] == [[1, 0, 0], [0, -1, 0], [0, 0, 1]]


@pytest.mark.parametrize(
    "settings",
    [
        [],
        {},
        {"environment_path": "lot.exr"},
        {"environment_path": "https://example.test/lot.png"},
        {"environment_path": "lot.png", "strength": -1},
        {"environment_path": "lot.png", "strength": math.nan},
        {"environment_path": "lot.png", "strength": True},
        {"environment_path": "lot.png", "ao_distance_m": math.inf},
        {"environment_path": "lot.png", "ao_distance_m": -0.1},
        {"environment_path": "lot.png", "bake_irradiance": 1},
        {"environment_path": "lot.png", "unknown": True},
    ],
)
def test_invalid_lighting_fails_before_worker_start(settings):
    with pytest.raises(ValueError):
        build_object_job([], lighting=settings)


@pytest.mark.parametrize("engine", ["BLENDER_EEVEE_NEXT", "CYCLES"])
def test_billboard_only_scene_preserves_environment_without_baking(monkeypatch, engine):
    import json
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from drivingbench.sim.visual_objects import _blender_lighting

    bpy = MagicMock()
    scene = bpy.context.scene
    scene.objects = [SimpleNamespace(type="CAMERA"), SimpleNamespace(type="EMPTY")]
    scene.render.engine = engine
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", MagicMock())
    lighting = environment_lighting(
        dict(environment_path="lot.png", ao_distance_m=0.5, bake_irradiance=True), (0, 0, 1)
    )
    _blender_lighting({"lighting": lighting})
    bpy.data.images.load.assert_called_once_with("lot.png", check_existing=True)
    bpy.data.lightprobes.new.assert_not_called()
    bpy.ops.object.lightprobe_cache_bake.assert_not_called()
    key, value = scene.__setitem__.call_args.args
    assert key == "drivingbench_lighting"
    status = json.loads(value)
    assert status["environment_path"] == "lot.png"
    assert status["bake_irradiance"] is True
    assert status["irradiance_status"] == "not_needed_no_static_mesh"
    assert status["bake_cache"] == "none"
    assert status["material_ao_count"] == 0


def test_static_mesh_still_requires_successful_requested_bake(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from drivingbench.sim.visual_objects import _blender_lighting

    bpy = MagicMock()
    scene = bpy.context.scene
    scene.objects = [
        SimpleNamespace(
            type="MESH",
            material_slots=[],
            matrix_world=np.eye(4),
            bound_box=[[-1, -1, 0], [1, 1, 1]],
        )
    ]
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    bpy.ops.object.lightprobe_cache_bake.return_value = {"CANCELLED"}
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", MagicMock())
    lighting = environment_lighting(
        dict(environment_path="lot.png", bake_irradiance=True), (0, 0, 1)
    )
    with pytest.raises(RuntimeError, match="did not finish"):
        _blender_lighting({"lighting": lighting})
    bpy.ops.object.lightprobe_cache_bake.assert_called_once_with(subset="SELECTED")
    scene.__setitem__.assert_not_called()
