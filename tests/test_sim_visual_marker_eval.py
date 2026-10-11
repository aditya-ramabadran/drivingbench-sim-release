"""Synthetic checks for calibration geometry and frozen annotation semantics."""

import numpy as np
import pytest
from drivingbench.sim.visual_marker_eval import annotation_digest, apparent_size, matched_markers


def camera():
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1.2], [0, 0, 0, 1.0]])


def test_apparent_size_accounts_for_ground_foreshortening():
    diameter, height, focal, depth = 0.2, 0.08, 1141.5, 10
    ground_half = diameter / 2 * focal * 1.2 / depth**2
    pixels_high = ground_half + np.hypot(ground_half, focal * height / depth)
    assert apparent_size(
        [0, 0, focal * diameter / depth, pixels_high], [0, depth, 0], camera()
    ) == pytest.approx((diameter, height))


def test_degenerate_height_or_behind_camera_rejected():
    with pytest.raises(ValueError):
        apparent_size([0, 0, 5, 5], [0, -5, 0], camera())
    overhead = np.diag([1, -1, -1, 1.0])
    overhead[2, 3] = 1.2
    with pytest.raises(ValueError, match="unobservable"):
        apparent_size([0, 0, 5, 5], [0, 0, 0], overhead)


def test_paths_can_change_but_frozen_boxes_cannot():
    row = dict(id=1, file_path="/old/image.jpg", bbox=[1, 2, 3, 4], kind="dome_marker")
    assert annotation_digest([row]) == annotation_digest(
        [dict(row, file_path="relative/image.jpg")]
    )
    assert annotation_digest([row]) != annotation_digest([dict(row, bbox=[1, 2, 4, 4])])


def test_projected_assignments_are_one_to_one_and_color_matched():
    pytest.importorskip("scipy")
    frames = {"view": {"transform_matrix": camera().tolist()}}
    annotations = [
        dict(id=i, name="view", bbox=[668, 512, 677, 518], color="orange_red", kind="dome_marker")
        for i in range(2)
    ]
    markers = [
        dict(id="orange", position_enu_m=[0, 10, 0], kind="dome_marker", color="orange_red"),
        dict(id="yellow", position_enu_m=[0, 10, 0], kind="dome_marker", color="yellow_green"),
    ]
    result = matched_markers(annotations, frames, markers)
    assert len(result) == 1 and result[0]["marker_id"] == "orange"
    assert len(annotations) == 2


def test_distance_gate_prevents_invalid_pair_stealing_valid_match(monkeypatch):
    pytest.importorskip("scipy")
    import drivingbench.sim.visual_marker_eval as study

    monkeypatch.setattr(study, "project", lambda frame, point: (np.array(point[:2]), 10))
    annotations = [
        dict(
            id=i, name="view", bbox=[x, y - 1, x + 1, y + 1], color="orange_red", kind="dome_marker"
        )
        for i, (x, y) in enumerate(((1, 23), (7, 18)))
    ]
    markers = [
        dict(id=str(i), position_enu_m=[x, y, 0], kind="dome_marker", color="orange_red")
        for i, (x, y) in enumerate(((10, 7), (12, 15)))
    ]
    matches = matched_markers(annotations, {"view": {}}, markers)
    assert {(m["annotation_id"], m["marker_id"]) for m in matches} == {(0, "1"), (1, "0")}


def test_persistent_asset_fit_uses_actual_mesh_and_shared_marker_identity():
    pytest.importorskip("scipy")
    from drivingbench.sim.visual_marker_eval import K, fit_marker_assets
    from drivingbench.sim.visual_objects import dome_mesh

    frames, annotations, rows = {}, [], []
    for index, distance in enumerate((8, 12)):
        pose = camera()
        pose[1, 3] = 10 - distance
        name = str(index)
        frames[name] = {"transform_matrix": pose.tolist()}
        vertices = np.asarray(dome_mesh(0.18, 0.065)[0]) + [0, 10, 0]
        local = (vertices - pose[:3, 3]) @ pose[:3, :3]
        pixels = local @ K.T
        extent = np.ptp(pixels[:, :2] / pixels[:, 2:], axis=0)
        annotations.append({"id": index, "bbox": [0, 0, *extent]})
        rows.append(
            dict(
                annotation_id=index,
                marker_id="one",
                name=name,
                position_enu_m=[0, 10, 0],
                observed_color_srgb=[0.5, 0.3, 0.2],
            )
        )
    result = fit_marker_assets(
        {"diameter_m": 0.16, "height_m": 0.08, "rows": rows}, annotations, frames
    )
    assert len(result) == 1 and result["one"]["observations"] == 2
    assert result["one"]["diameter_m"] == pytest.approx(0.18, abs=1e-6)
    assert result["one"]["height_m"] == pytest.approx(0.065, abs=1e-6)
