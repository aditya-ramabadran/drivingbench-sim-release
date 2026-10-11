"""Geometry conventions and split isolation without image data or COLMAP."""

import numpy as np
import pytest
from drivingbench.sim.visual_sfm import candidate_pairs, opengl_pose, similarity


def test_similarity_recovers_metric_planar_transform():
    source = np.array([[0, 0, 0], [3, 0, 0], [2, 5, 0], [-1, 4, 0]], dtype=float)
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    target = 2.7 * source @ rotation.T + [10, -3, 1.21]
    scale, fitted, translation = similarity(source, target)
    np.testing.assert_allclose(scale, 2.7)
    np.testing.assert_allclose(fitted, rotation, atol=1e-14)
    np.testing.assert_allclose(translation, [10, -3, 1.21])
    assert np.linalg.det(fitted) == pytest.approx(1)


def test_similarity_rejects_unobservable_line():
    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="non-collinear"):
        similarity(line, line)


def test_opengl_pose_camera_center_and_forward():
    world_to_camera = np.array([[1, 0, 0, -2], [0, 0, -1, 1.21], [0, 1, 0, -3]])
    pose = opengl_pose(world_to_camera)
    np.testing.assert_allclose(pose[:3, 3], [2, 3, 1.21])
    np.testing.assert_allclose(pose[:3, :3] @ [0, 0, -1], [0, 1, 0])
    np.testing.assert_allclose(pose[:3, :3] @ [0, 1, 0], [0, 0, 1])


def test_localization_pairs_never_connect_test_images():
    train = [dict(name=f"train-{i}", x=i, y=0, heading=0) for i in range(5)]
    test = [dict(name=f"test-{i}", x=i, y=0, heading=0) for i in range(3)]
    pairs = candidate_pairs(test, train, neighbors=2)
    assert len(pairs) == 6
    assert all(sum(name.startswith("test") for name in pair) == 1 for pair in pairs)
    assert all(sum(name.startswith("train") for name in pair) == 1 for pair in pairs)


def test_training_pairs_are_unique_and_have_no_self_edges():
    train = [dict(name=str(i), x=i, y=0, heading=0) for i in range(10)]
    pairs = candidate_pairs(train)
    assert len(pairs) == len(set(pairs))
    assert all(a != b for a, b in pairs)


def test_refined_manifest_fields_follow_pose_and_keep_seed():
    from drivingbench.sim.visual_pose import camera_to_world
    from drivingbench.sim.visual_sfm import _brush_frame, refined_frame

    rpy = (0.015, -0.025, 0.011)
    row = dict(
        name="a.jpg",
        split="test",
        file_path="/images/a.jpg",
        rpy=rpy,
        x=1,
        y=2,
        heading=0.3,
        transform_matrix=camera_to_world(1, 2, 0.3, rpy).tolist(),
    )
    expected = camera_to_world(10, -4, 1.1, rpy, height=1.23)
    refined = refined_frame(row, expected @ np.diag([1, -1, -1, 1]))
    assert refined["x"] == 10
    assert refined["y"] == -4
    assert refined["heading"] == pytest.approx(1.1)
    assert refined["file_path"] == "/images/a.jpg"
    assert refined["seed_pose"]["heading"] == 0.3
    np.testing.assert_allclose(refined["transform_matrix"], expected)
    brush = _brush_frame(refined)
    assert brush["heading"] == refined["heading"]
    assert brush["file_path"] == "images/a.jpg"
    np.testing.assert_allclose(brush["transform_matrix"], expected @ np.diag([1, -1, -1, 1]))


def test_keyframes_preserve_metric_and_heading_coverage():
    from drivingbench.sim.visual_sfm import keyframes

    rows = [
        dict(name=str(i), split="train", x=x, y=0, heading=h)
        for i, (x, h) in enumerate(
            [(0, 0), (0.5, 2 * np.pi - 0.01), (2.1, 0), (2.2, np.pi / 2), (2.3, np.pi / 2)]
        )
    ]
    selected = keyframes(rows, 2, 15)
    assert [r["name"] for r in selected] == ["0", "2", "3"]
    for row in rows:
        assert any(
            np.hypot(row["x"] - ref["x"], row["y"] - ref["y"]) < 2
            and abs(
                np.arctan2(
                    np.sin(row["heading"] - ref["heading"]), np.cos(row["heading"] - ref["heading"])
                )
            )
            < np.radians(15)
            for ref in selected
        )
    with pytest.raises(ValueError, match="training split"):
        keyframes(rows + [dict(rows[0], split="test")], 2, 15)
    with pytest.raises(ValueError, match="positive"):
        keyframes(rows, 0, 15)


def test_feature_cache_requires_matching_manifest_and_camera(tmp_path):
    import json
    import sqlite3

    from drivingbench.sim.visual_sfm import reuse_features

    source = tmp_path / "database.db"
    destination = tmp_path / "copy.db"
    inputs = {"/manifest.jsonl": "frames-hash", "/metadata.json": "camera-hash"}
    with pytest.raises(ValueError, match="does not exist"):
        reuse_features(source, destination, inputs)
    with sqlite3.connect(source) as db:
        db.execute("create table images (name text)")
        db.execute("insert into images values ('train.jpg')")
    with pytest.raises(ValueError, match="provenance"):
        reuse_features(source, destination, inputs)
    (tmp_path / "inputs.json").write_text(json.dumps(dict(inputs, keyframes={"spacing_m": 3})))
    with pytest.raises(ValueError, match="hashes"):
        reuse_features(source, destination, dict(inputs, **{"/metadata.json": "changed"}))
    assert not destination.exists()
    reuse_features(source, destination, dict(inputs, keyframes={"spacing_m": 2}))
    with sqlite3.connect(destination) as db:
        assert db.execute("select name from images").fetchall() == [("train.jpg",)]
