import io
import json
import math

import numpy as np
import pytest
from drivingbench.sim.course import CAMERA_OFFSET_M
from drivingbench.sim.scene import Scene
from drivingbench.sim.visual import ground_region
from PIL import Image


def write_scene(tmp_path):
    normal = np.array([0.02, -0.01, 1])
    normal /= np.linalg.norm(normal)
    ground = tmp_path / "ground.json"
    ground.write_text(
        json.dumps(
            dict(normal=normal.tolist(), offset=0.1, height=1.21, mount_rotation=np.eye(3).tolist())
        )
    )
    value = dict(
        frame="pilot_ground",
        calibration=str(ground),
        background={"checkpoint": "not-needed.ply"},
        start=dict(x=0, y=0, heading_deg=90),
        finish={"polygon": [[-4, 10], [4, 10], [4, 18], [-4, 18]]},
        boundaries={"left": [[-5, -5], [-5, 20]], "right": [[5, -5], [5, 20]]},
        markers=[
            dict(
                id="left",
                kind="dome_marker",
                x=-5,
                y=3,
                z=0,
                diameter_m=0.2,
                height_m=0.06,
                color_srgb=[0.8, 0.2, 0.1],
                collision_radius_m=0.1,
            )
        ],
        objects=[dict(kind="asset", x=7, y=8, z=0, asset_path="tree.glb")],
    )
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(value))
    return path


def test_scene_shares_camera_body_registration_and_plants_objects(tmp_path):
    captured = []

    def backend(matrix, objects):
        captured.append((matrix, objects))
        output = io.BytesIO()
        Image.new("RGB", (1344, 760)).save(output, "JPEG", quality=92)
        return output.getvalue()

    scene = Scene(write_scene(tmp_path), backend=backend)
    scene.render((2, 1, 15))
    pose = scene.course.camera_pose(scene.course.map_pose((2, 1, 15)))
    expected = scene.ground.camera_from_ground(pose[0], pose[1], math.radians(pose[2]))
    np.testing.assert_allclose(captured[0][0], expected)
    assert scene.renderer.camera_offset == CAMERA_OFFSET_M
    for obj in scene.objects:
        assert np.dot(
            scene.ground.normal, [obj.x, obj.y, obj.z]
        ) + scene.ground.offset == pytest.approx(0)
    assert scene.course.assess(scene.course.start)["collision_ids"] == []
    assert len(captured[0][1]) == 2


def test_scene_rejects_object_typo_instead_of_silently_ignoring_it(tmp_path):
    path = write_scene(tmp_path)
    value = json.loads(path.read_text())
    value["objects"][0]["diamter_m"] = 0.3
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="diamter_m"):
        Scene(path, backend=lambda *args: b"")


def test_ground_region_handles_concavity_and_invalid_rays():
    polygon = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
    points = np.array([[0.5, 2, 0], [2, 0.5, 0], [2, 2, 0], [np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(ground_region(points, polygon), [True, True, False, False])


def test_ground_collar_keeps_boundary_marker_shadows_whole_without_moving_physics():
    polygon = [[0, 0], [4, 0], [4, 4], [0, 4]]
    points = np.array([[2, -0.2, 0], [2, -0.5, 0], [-0.3, -0.3, 0], [np.nan, 0, 0]])
    np.testing.assert_array_equal(ground_region(points, polygon), [False] * 4)
    np.testing.assert_array_equal(
        ground_region(points, polygon, margin=0.4), [True, False, False, False]
    )
    with pytest.raises(ValueError, match="margin"):
        ground_region(points, polygon, margin=-1)
