from types import SimpleNamespace

import numpy as np
import pytest
from drivingbench.sim.object_worker import BlenderObjects
from drivingbench.sim.persistent import PersistentRenderer
from drivingbench.sim.visual import CAMERA, ground_region
from drivingbench.sim.visual_composite import _intersections, compose
from drivingbench.sim.visual_ground import GroundCalibration
from drivingbench.sim.visual_objects import (
    ObjectLayer,
    blender_camera_settings,
    build_object_job,
    composite_grounded,
    empty_layer,
    pinhole_camera,
)

TILE = dict(w=24, h=18, fl_x=17, fl_y=13, cx=9, cy=7)


def test_offcenter_anisotropic_intrinsics_define_actual_pixel_rays():
    ground = GroundCalibration(np.array([0, 0, 1]), -5, 1.21, np.eye(3))
    y, x = np.array([0, 7, 17]), np.array([0, 9, 23])
    points, depth = _intersections(ground, np.eye(4), y, x, TILE)
    np.testing.assert_allclose(depth, 5)
    np.testing.assert_allclose(points[:, 0], (x + 0.5 - 9) / 17 * 5)
    np.testing.assert_allclose(points[:, 1], (y + 0.5 - 7) / 13 * 5)
    np.testing.assert_allclose(points[:, 2], 5)


def test_tile_composite_matches_full_geometry_without_global_camera_mutation():
    ground = GroundCalibration(np.array([0, 0, 1]), -5, 1.21, np.eye(3))
    corridor = [[-100, -100], [100, -100], [100, 100], [-100, 100]]
    scene = SimpleNamespace(ground=ground, objects=[], course=SimpleNamespace(corridor=corridor))
    shape = (TILE["h"], TILE["w"])
    rgb = np.full((*shape, 3), 100, np.uint8)
    layer = empty_layer(TILE)
    layer.rgba[3:12, 7:16] = (220, 30, 80, 255)
    layer.depth[3:12, 7:16] = 3
    depth, alpha = np.full(shape, 10.0), np.ones(shape)
    depth[5:8, 9:12] = 2
    actual = compose(scene, np.eye(4), rgb, depth, alpha, layer, camera=TILE)
    points, plane = ground.ray_intersections(np.eye(4), TILE)
    expected, _ = composite_grounded(
        rgb, depth, alpha, layer, [], points, plane, ground_region(points, corridor)
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[5:8, 9:12], rgb[5:8, 9:12])
    assert CAMERA["w"] == 1344
    assert build_object_job([])["camera"] == CAMERA
    assert build_object_job([], camera=TILE)["camera"] == TILE


def test_blender_settings_represent_principal_point_and_nonsquare_pixels():
    for camera in (CAMERA, TILE, TILE | {"fl_x": 13, "fl_y": 17}):
        settings = blender_camera_settings(camera)
        ratio = settings["pixel_aspect_y"] / settings["pixel_aspect_x"]
        assert settings["lens"] / 36 * camera["w"] == pytest.approx(camera["fl_x"])
        assert camera["fl_x"] / ratio == pytest.approx(camera["fl_y"])
        assert camera["w"] * (0.5 - settings["shift_x"]) == pytest.approx(camera["cx"])
        assert camera["h"] / 2 + settings["shift_y"] * camera["w"] / ratio == pytest.approx(
            camera["cy"]
        )
    assert blender_camera_settings(CAMERA)["shift_x"] == 0
    assert blender_camera_settings(CAMERA)["shift_y"] == 0


@pytest.mark.parametrize("change", [{"w": 0}, {"h": 1.5}, {"fl_y": -1}, {"cx": np.nan}])
def test_bad_camera_fails_before_worker_request(change):
    with pytest.raises(ValueError):
        pinhole_camera(TILE | change)


def test_one_object_worker_changes_empty_frame_dimensions_per_request(tmp_path):
    worker = BlenderObjects.__new__(BlenderObjects)
    worker.lighting = {"mode": "test"}
    worker.job = {"camera": CAMERA, "objects": []}
    jobs = []

    def frame(job, *, arrays=False):
        assert arrays
        jobs.append(job)
        from pathlib import Path

        Path(job["output"]).mkdir()
        return {"empty": True, "bounds": None}

    worker.module = SimpleNamespace(
        pinhole_camera=pinhole_camera, _blender_frame=frame, empty_layer=empty_layer
    )
    for index, camera in enumerate((TILE, TILE | {"w": 12, "h": 10, "cx": 6, "cy": 5})):
        response = worker(
            {"camera": camera, "transform_matrix": np.eye(4), "output": tmp_path / str(index)}
        )
        assert np.load(response["paths"]["rgba"]).shape == (camera["h"], camera["w"], 4)
        assert np.load(response["paths"]["depth"]).shape == (camera["h"], camera["w"])
    assert jobs[0]["camera"] == TILE
    assert worker.job["camera"] == CAMERA


def test_raw_tile_path_never_encodes_jpeg_and_shares_camera_with_both_workers(
    tmp_path, monkeypatch
):
    value = PersistentRenderer.__new__(PersistentRenderer)
    value.output, value.camera = tmp_path, CAMERA
    value.response, value.restarts, value.device = None, 0, "CPU"
    value.scene = SimpleNamespace(
        objects_at=lambda *a, **k: [],
        ground=SimpleNamespace(normal=(0, 0, 1)),
        data={"markers": [], "objects": []},
    )
    value.billboards = SimpleNamespace(layers=lambda *a: [])
    calls = []
    shape = (TILE["h"], TILE["w"])
    rgb = np.full((*shape, 3), 127, np.uint8)

    def request(kind, matrix, output, *, camera):
        calls.append((kind, camera))
        output.mkdir()
        if kind == "splat":
            for name, data in (("rgb", rgb), ("depth", np.ones(shape)), ("alpha", np.ones(shape))):
                np.save(output / f"{name}.npy", data)
            return {}
        layer = ObjectLayer(np.zeros((*shape, 4), np.uint8), np.ones(shape))
        paths = {}
        for name in ("rgba", "depth"):
            paths[name] = output / f"{name}.npy"
            np.save(paths[name], getattr(layer, name))
        return {"paths": paths}

    value._request = request
    monkeypatch.setattr("drivingbench.sim.persistent.compose", lambda *a, **k: (rgb, {}))
    monkeypatch.setattr("PIL.Image.Image.save", lambda *a, **k: pytest.fail("Unexpected JPEG"))
    result = value.render_matrix_rgb(np.eye(4), camera=TILE)
    np.testing.assert_array_equal(result, rgb)
    assert calls == [("splat", TILE), ("objects", TILE)]
    assert value.camera == CAMERA
    assert value.last_metrics["camera"] == TILE
