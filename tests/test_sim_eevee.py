from dataclasses import replace

import numpy as np
import pytest
from drivingbench.sim.visual_objects import (
    PlacedObject,
    asset_normalization,
    instance_moments,
    repair_edge_depth,
)


def test_moment_ids_reject_foreign_label_mixtures_and_background_edges():
    mean = np.array([1, 2, 2, 1, 1.5, 0, np.nan, 9, -1])
    second = np.array([1, 4, 5, 2, 2.5, 0, 1, 81, 1])
    np.testing.assert_array_equal(instance_moments(mean, second, 4), [1, 2, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="matching"):
        instance_moments([1], [1, 2], 4)


def test_mixed_surface_edge_stays_unknown_between_different_ids():
    rgba = np.full((1, 3, 4), 255, np.uint8)
    ids = instance_moments([[1, 2, 3]], [[1, 5, 9]], 3)
    depth = np.array([[2, 2, 2]], dtype=np.float32)
    depth[ids == 0] = np.nan
    layer = repair_edge_depth(rgba, depth, ids)
    assert np.isnan(layer.depth[0, 1])
    assert not layer.repaired[0, 1]
    assert layer.instance[0, 1] == 0


def test_normalized_asset_fits_declared_bounds_and_preserves_ground_anchor():
    lower, upper = np.array([-2, 1, -0.4]), np.array([3, 2, 1.6])
    transform = asset_normalization(lower, upper)
    bounds = np.stack((lower, upper)) @ transform[:3, :3].T + transform[:3, 3]
    target = np.array([2.4, 0.6, 0.81])
    bounds *= target
    np.testing.assert_allclose(bounds[0], [-1.2, -0.3, 0])
    np.testing.assert_allclose(bounds[1], [1.2, 0.3, 0.81])
    with pytest.raises(ValueError, match="positive"):
        asset_normalization([0, 0, 0], [1, 0, 1])


def test_construction_glb_requires_declared_dimensions_and_unambiguous_material():
    obj = PlacedObject(
        "jersey_barrier", 1, 2, width_m=2.4, depth_m=0.6, height_m=0.81, asset_path="barrier.glb"
    )
    assert obj.asset_path == "barrier.glb"
    with pytest.raises(ValueError, match="not both"):
        replace(obj, texture_path="tile.png")
    with pytest.raises(ValueError, match=".glb"):
        replace(obj, asset_path="barrier.obj")
    with pytest.raises(ValueError, match="positive"):
        replace(obj, depth_m=None)


def test_warm_worker_exports_returned_depth_without_intermediate_array_files(tmp_path):
    from types import SimpleNamespace

    from drivingbench.sim.object_worker import BlenderObjects
    from drivingbench.sim.visual_objects import pinhole_camera

    camera = dict(w=3, h=2, fl_x=2, fl_y=2, cx=1.5, cy=1)

    def frame(job, *, arrays):
        assert arrays
        return dict(
            empty=False,
            bounds=(0, 0, 3, 2),
            depth=np.full((2, 3), 7, np.float32),
            instance=np.ones((2, 3), np.int32),
        )

    worker = BlenderObjects.__new__(BlenderObjects)
    worker.lighting = {"mode": "test"}
    worker.job = {"camera": camera}
    worker.module = SimpleNamespace(
        pinhole_camera=pinhole_camera,
        _blender_frame=frame,
        _blender_pixels=lambda *a, **k: np.ones((2, 3, 4)),
        repair_edge_depth=repair_edge_depth,
    )
    response = worker({"transform_matrix": np.eye(4), "output": tmp_path})
    np.testing.assert_array_equal(np.load(response["paths"]["depth"]), 7)
    np.testing.assert_array_equal(np.load(response["paths"]["instance"]), 1)
    assert not (tmp_path / "objects-depth-raw.npy").exists()


def test_object_renderer_defaults_and_shared_sample_field():
    from drivingbench.sim.visual_objects import build_object_job, object_renderer_settings

    assert object_renderer_settings(None) == {"engine": "CYCLES", "samples": 128}
    assert object_renderer_settings({"engine": "BLENDER_EEVEE_NEXT"}) == {
        "engine": "BLENDER_EEVEE_NEXT",
        "samples": 16,
    }
    for engine, samples in [("CYCLES", 8), ("BLENDER_EEVEE_NEXT", 16)]:
        settings = {"engine": engine, "samples": samples}
        job = build_object_job([], object_renderer=settings)
        assert {key: job[key] for key in settings} == settings
        assert "eevee_samples" not in job
        assert settings == {"engine": engine, "samples": samples}


@pytest.mark.parametrize(
    "settings",
    [
        [],
        {"engine": "EEVEE"},
        {"engine": None},
        {"samples": True},
        {"samples": 16.0},
        {"samples": 0},
        {"samples": 257},
        {"samples": float("nan")},
        {"eevee_samples": 16},
        {"engine": "BLENDER_EEVEE_NEXT", "device": "METAL"},
    ],
)
def test_bad_object_renderer_fails_before_worker_start(settings):
    from drivingbench.sim.visual_objects import build_object_job

    with pytest.raises(ValueError):
        build_object_job([], object_renderer=settings)


@pytest.mark.parametrize(
    "engine,device,samples,expected_engine,expected_samples",
    [
        ("BLENDER_EEVEE_NEXT", "CPU", 16, "CYCLES", 128),
        ("CYCLES", "CPU", 32, "CYCLES", 32),
        ("BLENDER_EEVEE_NEXT", "METAL", 8, "BLENDER_EEVEE_NEXT", 8),
    ],
)
def test_engine_samples_and_cpu_recovery_before_scene_assembly(
    monkeypatch, engine, device, samples, expected_engine, expected_samples
):
    import sys
    from unittest.mock import MagicMock

    from drivingbench.sim.visual_objects import _blender_scene

    bpy = MagicMock()
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", MagicMock())
    job = {"engine": engine, "device": device, "samples": samples, "objects": []}
    # Stop at the first geometry input; only worker configuration is exercised here.
    with pytest.raises(KeyError, match="camera"):
        _blender_scene(job)
    assert bpy.context.scene.render.engine == expected_engine
    if expected_engine == "BLENDER_EEVEE_NEXT":
        assert bpy.context.scene.eevee.taa_render_samples == expected_samples
    else:
        assert bpy.context.scene.cycles.samples == expected_samples
        assert bpy.context.scene.cycles.device == "CPU"


def test_eevee_static_id_boundary_is_conservative_and_cycles_is_unrestricted():
    from drivingbench.sim.visual_objects import build_object_job

    objects = [PlacedObject("cone", 0, 0)] * 45
    assert (
        len(build_object_job(objects, object_renderer={"engine": "BLENDER_EEVEE_NEXT"})["objects"])
        == 45
    )
    with pytest.raises(ValueError, match="1–45.*Use CYCLES"):
        build_object_job(objects + objects[:1], object_renderer={"engine": "BLENDER_EEVEE_NEXT"})
    assert len(build_object_job(objects * 3)["objects"]) == 135
    ids = np.arange(1, 46)
    np.testing.assert_array_equal((ids**2).astype(np.float16).astype(int), ids**2)
    assert float(np.float16(53**2)) != 53**2


def test_eevee_guard_uses_actual_ids_not_number_of_static_objects():
    from drivingbench.sim.visual_objects import build_object_job

    direct = PlacedObject("billboard", 0, 0, asset_path="actor.png", width_m=1, height_m=2)
    decal = PlacedObject("ground_decal", 0, 0, asset_path="shadow.png", width_m=1, depth_m=2)
    assert (
        len(
            build_object_job(
                [direct, decal] * 40, object_renderer={"engine": "BLENDER_EEVEE_NEXT"}
            )["objects"]
        )
        == 80
    )
    with pytest.raises(ValueError, match="Use CYCLES"):
        build_object_job(
            [direct] * 45 + [PlacedObject("cone", 0, 0)],
            object_renderer={"engine": "BLENDER_EEVEE_NEXT"},
        )


def test_raw_unsafe_eevee_job_rejected_before_importing_blender(monkeypatch):
    import sys

    from drivingbench.sim.visual_objects import _blender_scene

    monkeypatch.setitem(sys.modules, "bpy", None)
    with pytest.raises(ValueError, match="Use CYCLES"):
        _blender_scene({"engine": "BLENDER_EEVEE_NEXT", "objects": [{"kind": "cone"}] * 46})
