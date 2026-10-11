import io

import numpy as np
import pytest
from drivingbench.sim.visual import Renderer, novel_queries
from drivingbench.sim.visual_pose import camera_to_world
from PIL import Image


def jpeg(size=(1344, 760)):
    output = io.BytesIO()
    Image.new("RGB", size).save(output, "JPEG", quality=92)
    return output.getvalue()


def test_renderer_registers_sim_pose_and_rejects_unsupported_camera():
    poses = []

    def capture(pose, objects):
        assert not objects
        poses.append(pose)
        return jpeg()

    renderer = Renderer(capture, origin=(10, 20, np.pi / 2), rpy=(0, 0, 0))
    from drivingbench.sim.car import SimNative

    native = SimNative()
    native.car.x, native.car.y, native.car.heading = 2, 1, -np.pi / 2
    renderer.render(*native.pose)
    np.testing.assert_allclose(poses[0][:3, 3], [9, 22, 1.21])
    np.testing.assert_allclose(poses[0][:3, 2], [1, 0, 0], atol=1e-15)
    with pytest.raises(ValueError, match="narrow"):
        renderer.render(0, 0, 0, camera="wide")
    with pytest.raises(ValueError, match="finite"):
        renderer.render(np.nan, 0, 0)
    assert len(poses) == 1


def test_renderer_enforces_observe_image_dimensions():
    renderer = Renderer(lambda pose, objects: jpeg((672, 380)), origin=(0, 0, 0), rpy=(0, 0, 0))
    with pytest.raises(ValueError, match="native-resolution"):
        renderer.render(0, 0, 0)


def test_renderer_preserves_objects_in_map_coordinates():
    from drivingbench.sim.visual_objects import PlacedObject

    cone = PlacedObject("cone", 10, 20)
    calls = []

    def capture(matrix, objects):
        calls.append((matrix, objects))
        return jpeg()

    renderer = Renderer(capture, origin=(3, 4, 0), rpy=(0, 0, 0))
    renderer.render(1, 2, 15, objects=[cone])
    np.testing.assert_allclose(calls[0][0][:3, 3], [4, 6, 1.21])
    assert calls[0][1] == (cone,)
    assert (cone.x, cone.y) == (10, 20)


def test_backend_repairs_only_dome_residuals_when_explicitly_enabled(monkeypatch, tmp_path):
    from drivingbench.sim import visual, visual_objects, visual_splat

    monkeypatch.setattr(visual, "REPO", tmp_path)
    shape = (760, 1344)
    background = np.full((*shape, 3), 100, dtype=np.uint8)
    background[199:202, 199:202] = [200, 50, 0]
    background[399:402, 399:402] = [200, 50, 0]
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    instance = np.zeros(shape, dtype=np.int32)
    for center, index in ((200, 2), (400, 1)):
        rgba[center - 5 : center + 6, center - 5 : center + 6] = [180, 80, 20, 255]
        instance[center - 5 : center + 6, center - 5 : center + 6] = index
    layer = visual_objects.ObjectLayer(rgba, np.full(shape, 5.0), instance=instance)

    def render_frames(checkpoint, cameras, out, *, depth):
        assert depth
        path = out / "query.jpg"
        Image.fromarray(background).save(path)
        Image.fromarray(background).save(out / "query_color.png")
        np.save(out / "query_depth.npy", np.full(shape, 10.0))
        np.save(out / "query_depth_alpha.npy", np.ones(shape))
        return [path]

    captured = []
    composite = visual_objects.composite_objects

    def capture(pixels, depth, alpha, objects):
        np.testing.assert_array_equal(depth, np.full(shape, 10.0))
        np.testing.assert_array_equal(alpha, np.ones(shape))
        captured.append(pixels.copy())
        return composite(pixels, depth, alpha, objects)

    monkeypatch.setattr(visual_splat, "render_frames", render_frames)
    monkeypatch.setattr(visual_objects, "render_objects", lambda *args, **kwargs: layer)
    monkeypatch.setattr(visual_objects, "composite_objects", capture)
    objects = [
        visual_objects.PlacedObject("cone", 0, 0),
        visual_objects.PlacedObject(
            "dome_marker", 1, 0, diameter_m=0.16, height_m=0.07, color_srgb=(0.7, 0.2, 0.1)
        ),
    ]
    for enabled in (False, True):
        result = visual.splat_backend("unused.ply", repair_markers=enabled)(np.eye(4), objects)
        with Image.open(io.BytesIO(result)) as output:
            assert output.format == "JPEG" and output.size == (1344, 760)
    np.testing.assert_array_equal(captured[0], background)
    np.testing.assert_array_equal(captured[1][200, 200], [100, 100, 100])
    np.testing.assert_array_equal(captured[1][399:402, 399:402], background[399:402, 399:402])
    np.testing.assert_array_equal(layer.rgba, rgba)


def test_novel_queries_are_metric_and_cannot_be_mistaken_for_real_test_frames():
    reference = {
        "name": "real.jpg",
        "x": 0,
        "y": 0,
        "heading": 0.0,
        "split": "test",
        "transform_matrix": camera_to_world(0, 0, 0).tolist(),
    }
    rows = novel_queries(reference)
    assert all(row["split"] == "novel" and row["name"] != reference["name"] for row in rows)
    assert rows[0]["y"] == 2 and rows[1]["y"] == -2
    assert rows[2]["heading"] == pytest.approx(-np.pi / 12)
    assert all(row["transform_matrix"][2][3] == 1.21 for row in rows)
