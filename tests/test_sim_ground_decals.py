"""Natural shadow decals share calibrated ground and actor timing, not collision geometry."""

import json
from dataclasses import replace

import numpy as np
import pytest
from drivingbench.sim.billboards import BillboardCache
from drivingbench.sim.scene import Scene, scene_asset_paths
from drivingbench.sim.visual_ground import GroundCalibration
from drivingbench.sim.visual_objects import PlacedObject, composite_grounded, shadow_radius
from drivingbench.sim.visual_pose import camera_to_world
from PIL import Image
from test_sim_scene import write_scene


def decal(tmp_path):
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[2:18, 2:18, 3] = 100
    path = tmp_path / "shadow.png"
    Image.fromarray(rgba).save(path)
    return PlacedObject("ground_decal", 5, 0, asset_path=str(path), width_m=2, depth_m=3)


def test_decal_depth_is_ground_plane_and_real_foreground_stays_opaque(tmp_path):
    camera = dict(w=100, h=80, fl_x=85, fl_y=85, cx=50, cy=40)
    normal = np.array([0.08, -0.04, 1])
    normal /= np.linalg.norm(normal)
    ground = GroundCalibration(normal, 0.2, 1.21, camera_to_world(0, 0, 0)[:3, :3])
    obj = decal(tmp_path)
    position = ground.map_from_ground @ [obj.x, obj.y, 0, 1]
    obj = replace(obj, x=position[0], y=position[1], z=position[2], yaw=0.3)
    matrix = ground.camera_from_ground(0, 0, 0)
    layer = BillboardCache([obj]).layers([obj], matrix, camera, normal)[0]
    points, plane = ground.ray_intersections(matrix, camera)
    selected = layer.rgba[..., 3] > 0
    assert selected.sum() > 100
    np.testing.assert_allclose(layer.depth[selected], plane[selected], rtol=1e-6)
    assert shadow_radius(obj) == 0
    background = np.full((80, 100, 3), 150, np.uint8)
    depth = plane.copy()
    foreground = selected & (np.indices(selected.shape)[1] < 50)
    depth[foreground] = 1.0
    rendered, masks = composite_grounded(
        background,
        depth,
        np.ones(plane.shape),
        layer,
        [obj],
        points,
        plane,
        np.ones(plane.shape, bool),
        ground_normal=normal,
    )
    np.testing.assert_array_equal(rendered[foreground], background[foreground])
    assert np.all(rendered[selected & ~foreground] <= background[selected & ~foreground])
    assert np.any(rendered[selected & ~foreground] < background[selected & ~foreground])
    assert not masks["shadow_opacity"].any()


def test_decal_anchor_projects_authored_contact_point(tmp_path):
    camera = dict(w=200, h=160, fl_x=170, fl_y=170, cx=100, cy=80)
    obj = replace(decal(tmp_path), anchor_uv=(5.5 / 20, 15.5 / 20))
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[15, 5, 3] = 255
    Image.fromarray(rgba).save(obj.asset_path)
    matrix = camera_to_world(0, 0, 0, height=1.21)
    layer = BillboardCache([obj]).layers([obj], matrix, camera)[0]
    weights = layer.rgba[..., 3]
    assert weights.sum() > 0
    yy, xx = np.indices(weights.shape)
    centroid = [np.average(xx + 0.5, weights=weights), np.average(yy + 0.5, weights=weights)]
    point = (np.array([obj.x, obj.y, obj.z]) - matrix[:3, 3]) @ matrix[:3, :3]
    expected = [170 * point[0] / point[2] + 100, 170 * point[1] / point[2] + 80]
    np.testing.assert_allclose(centroid, expected, atol=0.75)
    assert np.isfinite(layer.depth[weights > 0]).all()
    with pytest.raises(ValueError, match="normalized"):
        replace(obj, anchor_uv=(2, 0.5))
    with pytest.raises(ValueError, match="not used"):
        replace(obj, height_m=1)


def test_linked_shadow_follows_latched_actor_without_duplicate_analytic_shadow(tmp_path):
    path = write_scene(tmp_path)
    data = json.loads(path.read_text())
    data["objects"] = [
        dict(
            id="child", kind="billboard", x=5, y=2, asset_path="child.png", width_m=1, height_m=1.2
        ),
        dict(
            id="shadow",
            kind="ground_decal",
            x=5.1,
            y=2.2,
            asset_path="shadow.png",
            width_m=1.4,
            depth_m=1.4,
            shadow_of="child",
        ),
    ]
    data["scenario"] = dict(
        type="hazard",
        time_limit_s=60,
        colliders=[dict(id="child", outcome="hazard", circle=[5, 2, 0.3])],
        trajectories={
            "child": dict(
                duration_s=2,
                delta=[0, -2],
                trigger=dict(kind="front_distance", point=[5, 0], heading_deg=0, distance_m=3),
            )
        },
    )
    path.write_text(json.dumps(data))
    scene = Scene(path, backend=lambda *a: b"")
    before = scene.objects_at(10)
    after = scene.objects_at(10, actor_onsets={"child": 9})
    delta = np.array([0, -1, 0]) @ scene.ground.map_from_ground[:3, :3].T
    for initial, moved in zip(before[1:], after[1:], strict=True):
        np.testing.assert_allclose(
            np.array([moved.x, moved.y, moved.z]) - [initial.x, initial.y, initial.z], delta
        )
    assert before[1].shadow_strength == after[1].shadow_strength == 0
    assert scene.course.assess((5, 2, 0))["collision_ids"] == []
    assert scene_asset_paths(data) == ["child.png", "shadow.png"]
    assert data["objects"][1]["y"] == 2.2
    data["objects"][1]["shadow_of"] = "missing"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="shadow_of"):
        Scene(path, backend=lambda *a: b"")


def test_scene_provenance_includes_environment_and_decal_assets_once():
    data = {
        "objects": [
            {"kind": "ground_decal", "asset_path": "shadow.png"},
            {"kind": "ground_decal", "asset_path": "shadow.png"},
        ],
        "lighting": {"environment_path": "lot.png", "strength": 0.8},
        "scenario": {"animations": {"child": {"frame_paths": ["step0.png", "step1.png"]}}},
    }
    assert scene_asset_paths(data) == ["lot.png", "shadow.png", "step0.png", "step1.png"]


def test_scene_resolves_one_lighting_source_for_warm_and_one_shot_rendering(tmp_path, monkeypatch):
    from drivingbench.sim.logs import REPO

    path = write_scene(tmp_path)
    data = json.loads(path.read_text())
    data["lighting"] = {"environment_path": "runs/test-environment.png", "strength": 0.6}
    data["object_renderer"] = {"engine": "BLENDER_EEVEE_NEXT", "samples": 16}
    path.write_text(json.dumps(data))
    calls = []

    def backend(*args, **kwargs):
        calls.append(kwargs)
        return lambda *args: b"unused"

    monkeypatch.setattr("drivingbench.sim.scene.splat_backend", backend)
    scene = Scene(path)
    assert scene.lighting == {
        "environment_path": str(REPO / "runs/test-environment.png"),
        "strength": 0.6,
    }
    assert calls[0]["lighting"] is scene.lighting
    assert scene.object_renderer == {"engine": "BLENDER_EEVEE_NEXT", "samples": 16}
    assert calls[0]["object_renderer"] is scene.object_renderer
    assert scene.data["lighting"]["environment_path"] == "runs/test-environment.png"


def test_warm_object_job_uses_scene_renderer_selection(tmp_path, monkeypatch):
    from drivingbench.sim.persistent import PersistentRenderer

    path = write_scene(tmp_path)
    data = json.loads(path.read_text())
    data["object_renderer"] = {"engine": "BLENDER_EEVEE_NEXT", "samples": 16}
    path.write_text(json.dumps(data))
    jobs = []

    class Worker:
        def __init__(self, command, *args, **kwargs):
            jobs.append(json.loads(command[-1].read_text()))

        def start(self):
            return {"ready": True, "device": "METAL"}

    monkeypatch.setattr("drivingbench.sim.persistent.RenderProcess", Worker)
    renderer = PersistentRenderer(path, output=tmp_path / "output")
    renderer._objects_start(renderer.scene.camera_matrix((0, 0, 0)))
    assert jobs[0]["engine"] == "BLENDER_EEVEE_NEXT"
    assert jobs[0]["samples"] == 16
