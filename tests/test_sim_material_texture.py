from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from drivingbench.sim.logs import REPO
from drivingbench.sim.scene import Scene, scene_asset_paths
from drivingbench.sim.visual_objects import PlacedObject, build_object_job


def test_texture_is_explicit_jersey_png_and_preserves_placement():
    plain = PlacedObject("jersey_barrier", 1, 2, width_m=2.4, depth_m=0.6, height_m=0.81)
    textured = replace(plain, texture_path="materials/concrete.png")
    before, after = [build_object_job([obj])["objects"][0] for obj in (plain, textured)]
    assert after.pop("texture_path") == "materials/concrete.png"
    before.pop("texture_path")
    assert before == after
    for invalid in ("concrete.jpg", "", "concrete.glb"):
        with pytest.raises(ValueError, match="texture_path"):
            replace(plain, texture_path=invalid)
    with pytest.raises(ValueError, match="texture_path"):
        PlacedObject("cone", 0, 0, texture_path="concrete.png")


def test_scene_resolves_texture_and_provenance_includes_every_asset():
    scene = Scene.__new__(Scene)
    scene.ground = SimpleNamespace(map_from_ground=np.eye(4))
    scene.data = {
        "markers": [],
        "objects": [
            dict(
                kind="jersey_barrier",
                x=1,
                y=2,
                width_m=2.4,
                depth_m=0.6,
                height_m=0.81,
                texture_path="materials/concrete.png",
            ),
            dict(kind="asset", x=3, y=4, asset_path="objects/tree.glb"),
        ],
    }
    objects = scene.objects_at()
    assert objects[0].texture_path == str((REPO / "materials/concrete.png").resolve())
    assert objects[1].asset_path == str((REPO / "objects/tree.glb").resolve())
    scene.data["scenario"] = {"animations": {"actor": {"frame_paths": ["walk/00.png"]}}}
    assert scene_asset_paths(scene.data) == [
        "materials/concrete.png",
        "objects/tree.glb",
        "walk/00.png",
    ]


def test_new_fork_uses_manifest_texture_for_every_wall_without_loading_png():
    from drivingbench.sim.scenarios import fork_scene

    base = {"frame": "site1_full_ground", "calibration": "unused.json", "background": {}}
    scene = fork_scene(base, "adult-vs-barrier", assets={})
    walls = [obj for obj in scene["objects"] if obj["kind"] == "jersey_barrier"]
    assert walls
    assert all(obj["texture_path"].endswith("concrete.png") for obj in walls)
    colliders = {obj["id"]: obj for obj in scene["scenario"]["colliders"]}
    assert all(wall["id"] in colliders for wall in walls)
    assert any(colliders[wall["id"]]["outcome"] == "branch_barrier" for wall in walls)
    assert set(scene_asset_paths(scene)) >= {walls[0]["texture_path"]}
