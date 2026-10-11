"""Data-free checks for bounded Gaussian cleanup."""

import numpy as np
import pytest
from drivingbench.sim.visual_clean import in_course, in_region, read_ply, selection, write_ply


def test_ply_round_trip_preserves_binary_fields_and_updates_count(tmp_path):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment test",
        "element vertex 3",
        "property float x",
        "property float opacity",
        "end_header",
    ]
    vertices = np.array([(1, 0.4), (2, 0.5), (3, 0.6)], dtype=[("x", "<f4"), ("opacity", "<f4")])
    path = tmp_path / "scene.ply"
    write_ply(path, header, vertices[[0, 2]])
    saved, data = read_ply(path)
    assert "element vertex 2" in saved
    assert data.tobytes() == vertices[[0, 2]].tobytes()
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload"):
        read_ply(path)


def test_course_orientation_and_bounded_margin():
    polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    points = np.array([[1, 1], [2.1, 1], [2.3, 1]])
    assert in_course(points, polygon, 0.2).tolist() == [True, True, False]
    assert np.array_equal(in_course(points, polygon, 0.2), in_course(points, polygon[::-1], 0.2))
    with pytest.raises(ValueError):
        in_course(points, [[0, 0], [0, 0], [1, 1]])


def test_selection_keeps_neutral_ground_high_colored_objects_and_outside_course():
    pytest.importorskip("skimage")
    vertices = np.zeros(
        5, dtype=[(k, "<f4") for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2")]
    )
    vertices["x"] = [1, 1, 1, 5, 1]
    vertices["y"] = 1
    vertices["z"] = [0.1, 0.1, 2, 0.1, -2]
    rgb = np.array(
        [[0.8, 0.3, 0.1], [0.3, 0.3, 0.3], [0.8, 0.3, 0.1], [0.8, 0.3, 0.1], [0.8, 0.3, 0.1]]
    )
    for index in range(3):
        vertices[f"f_dc_{index}"] = (rgb[:, index] - 0.5) / 0.28209479177387814
    mask, report = selection(vertices, [[0, 0], [2, 0], [2, 2], [0, 2]], [0, 0, 1, 0], margin=0)
    assert mask.tolist() == [True, False, False, False, False]
    assert report["selected"] == 1


def test_corridor_union_does_not_fill_large_convex_hull_gap():
    region = {
        "polygon_xy": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "corridor_xy": [[0, 0], [10, 0], [10, 10]],
        "corridor_radius_m": 1,
    }
    assert in_region(np.array([[0.5, 0.5], [10, 5], [5, 5]]), region, 0).tolist() == [
        True,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="convex"):
        in_course(np.array([[0, 0]]), [[0, 0], [2, 0], [1, 0.5], [2, 2], [0, 2]])


def test_halo_is_bounded_and_keeps_large_neutral_splats():
    pytest.importorskip("skimage")
    pytest.importorskip("scipy")
    fields = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1", "scale_2")
    vertices = np.zeros(4, dtype=[(k, "<f4") for k in fields])
    vertices["x"] = [1, 1.05, 1.3, 1.05]
    vertices["y"] = 1
    vertices["z"] = 0.1
    rgb = np.array([[0.8, 0.3, 0.1], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])
    for i in range(3):
        vertices[f"f_dc_{i}"] = (rgb[:, i] - 0.5) / 0.28209479177387814
        vertices[f"scale_{i}"] = np.log([0.05, 0.05, 0.05, 0.6])
    mask, _ = selection(vertices, [[0, 0], [2, 0], [2, 2], [0, 2]], [0, 0, 1, 0], halo=0.12)
    assert mask.tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="Halo"):
        selection(vertices, [[0, 0], [2, 0], [2, 2], [0, 2]], [0, 0, 1, 0], halo=0.3)
