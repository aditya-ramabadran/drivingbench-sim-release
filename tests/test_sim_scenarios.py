import copy
import math

import pytest
from drivingbench.replay.fork import PAIRINGS as REPLAY_PAIRINGS
from drivingbench.sim.course import LENGTH_M, WIDTH_M, Course
from drivingbench.sim.scenarios import (
    HAZARDS,
    PAIRINGS,
    assess_step,
    fork_scene,
    hazard_scene,
    object_rows_at,
)

BASE = {
    "frame": "pilot_ground",
    "calibration": "not-read.json",
    "background": {"checkpoint": "not-read.ply"},
}


@pytest.mark.parametrize(
    "builder,condition", [(fork_scene, "adult-vs-barrier"), (hazard_scene, "control")]
)
def test_builder_copies_render_settings_without_old_scenario_dynamics(builder, condition):
    base = copy.deepcopy(BASE)
    base["lighting"] = {"environment_path": "lot.png", "strength": 0.8, "bake_irradiance": True}
    base["object_renderer"] = {"engine": "BLENDER_EEVEE_NEXT", "samples": 16}
    base["camera_response"] = "explicit-test-response"
    base["scenario"] = {"trajectories": {"unrelated": "old"}}
    base["objects"] = [{"id": "unrelated"}]
    scene = builder(base, condition, assets={})
    assert scene["lighting"] == base["lighting"]
    assert scene["object_renderer"] == base["object_renderer"]
    assert scene["camera_response"] == "explicit-test-response"
    assert "unrelated" not in scene["scenario"]["trajectories"]
    assert not any(row["id"] == "unrelated" for row in scene["objects"])
    scene["lighting"]["strength"] = 0
    scene["object_renderer"]["samples"] = 8
    assert base["lighting"]["strength"] == 0.8
    assert base["object_renderer"]["samples"] == 16
    default = builder(BASE, condition, assets={})
    assert "lighting" not in default and "object_renderer" not in default
    assert default["camera_response"] == "site1-overcast-v1"
    base["camera_response"] = None
    assert builder(base, condition, assets={})["camera_response"] is None


def test_explicit_manifest_selects_its_own_gait_without_old_bundle_fallback():
    gait = {"frame_paths": ["new-walk-0.png", "new-walk-1.png"], "fps": 4}
    manifest = {"assets": {"child": "new-child.png"}, "animations": {"child": gait}}
    scene = hazard_scene(BASE, "child_entering_distance", assets=manifest)
    assert scene["objects"][0]["asset_path"] == "new-child.png"
    assert scene["scenario"]["animations"]["hazard"] == gait
    scene["scenario"]["animations"]["hazard"]["frame_paths"][0] = "changed.png"
    assert gait["frame_paths"][0] == "new-walk-0.png"
    scene = hazard_scene(BASE, "child_entering", assets={"assets": manifest["assets"]})
    assert "animations" not in scene["scenario"]
    override = {"child": {"frame_paths": ["explicit.png"], "fps": 3}}
    scene = hazard_scene(BASE, "child_entering", assets=manifest, animations=override)
    assert scene["scenario"]["animations"]["hazard"] == override["child"]


def empty_scene():
    scene = hazard_scene(BASE, "control")
    scene["scenario"]["time_limit_s"] = 100
    return scene


@pytest.mark.parametrize(
    "builder,condition", [(fork_scene, "dog-vs-robots2"), (hazard_scene, "control")]
)
def test_scenario_preserves_independent_wide_camera_registration(builder, condition):
    base = copy.deepcopy(BASE)
    base["wide_camera"] = {"calibration": "wide.json", "tiles": [{"name": "center"}]}
    scene = builder(base, condition, assets={})
    assert scene["wide_camera"] == base["wide_camera"]
    scene["wide_camera"]["tiles"][0]["name"] = "changed"
    assert base["wide_camera"]["tiles"][0]["name"] == "center"


@pytest.mark.parametrize("asset", ["child", "dog"])
@pytest.mark.parametrize("condition", ["entering", "entering_distance"])
def test_generated_hazards_use_camera_response_and_walking_assets(asset, condition):
    scene = hazard_scene(BASE, f"{asset}_{condition}", assets={})
    assert scene["camera_response"] == "site1-overcast-v1"
    assert not any(row["kind"] == "paint" for row in scene["objects"])
    animation = scene["scenario"]["animations"]["hazard"]
    assert len(animation["frame_paths"]) == len(set(animation["frame_paths"])) == 8
    assert set(animation) == {"frame_paths", "fps", "anchor_uv", "width_m", "height_m"}
    assert animation["fps"] == 8
    rows = object_rows_at(scene, 2.125, actor_onsets={"hazard": 2})
    assert rows[0]["asset_path"] == animation["frame_paths"][1]
    assert rows[0]["shadow_strength"] == 0.42
    reference = hazard_scene(BASE, f"{asset}_{condition}", assets={}, animations={})
    assert "animations" not in reference["scenario"]
    assert reference["scenario"]["colliders"] == scene["scenario"]["colliders"]
    assert reference["finish"] == scene["finish"]


def test_generated_fork_uses_same_response_and_shadow_policy_as_hazards():
    scene = fork_scene(BASE, "dog-vs-robots2", assets={})
    assert scene["camera_response"] == "site1-overcast-v1"
    assert not any(row["kind"] == "paint" for row in scene["objects"])
    assert all(
        row["shadow_strength"] == 0.42 for row in scene["objects"] if row["kind"] == "billboard"
    )


def test_pairings_match_existing_protocol_and_scenes_validate():
    assert PAIRINGS == REPLAY_PAIRINGS
    for pairing in PAIRINGS:
        for mirror in (False, True):
            scene = fork_scene(BASE, pairing, mirror=mirror)
            course = Course(scene)
            assert not course.assess(course.start)["outside_course"]
            assert scene["scenario"]["default_branch"] == ("left" if mirror else "right")
            assert scene["scenario"]["branch_width_m"] == 3
            actors = [
                c for c in scene["scenario"]["colliders"] if c["outcome"].endswith("occupant")
            ]
            identities = [o["id"] for o in scene["objects"]]
            assert len(identities) == len(set(identities))
            expected = (
                6
                if pairing == "robots5-vs-adult"
                else 3
                if pairing == "dog-vs-robots2"
                else 1
                if pairing.endswith("barrier")
                else 2
            )
            assert len(actors) == expected


def test_physical_reflection_and_rotation_preserve_first_outcome():
    a = fork_scene(BASE, "adult-vs-barrier")
    b = fork_scene(BASE, "adult-vs-barrier", mirror=True)
    for first, second in zip(a["objects"], b["objects"]):
        assert first["x"] == second["x"]
        assert first["y"] == -second["y"]
    normal = assess_step(a, (60, -1.7, 0), (65, -1.7, 0), 10, 11)
    reflected = assess_step(b, (60, 1.7, 0), (65, 1.7, 0), 10, 11)
    assert normal["outcome"] == "right_occupant"
    assert reflected["outcome"] == "left_occupant"
    assert normal["time_interval_s"] == reflected["time_interval_s"]
    rotated = fork_scene(BASE, "adult-vs-barrier", origin=(100, -20, 90))
    assert (
        assess_step(rotated, (101.7, 40, 90), (101.7, 45, 90), 10, 11)["outcome"]
        == "right_occupant"
    )


def test_tip_scored_before_divider_or_branch_without_command_selection():
    scene = fork_scene(BASE, "adults5-vs-adult")
    hit = assess_step(scene, (51, 0, 0), (58, 0, 0), 8, 10)
    assert hit["outcome"] == "divider_tip"
    assert hit["id"] == "divider-tip"
    assert hit["time_interval_s"][0] < 9


def test_tangent_circle_contact_and_millimetre_clearance():
    scene = empty_scene()
    scene["scenario"]["colliders"] = [
        {"id": "circle", "outcome": "hazard", "circle": [0, WIDTH_M / 2 + 0.3, 0.3]}
    ]
    assert assess_step(scene, (0, 0, 0), (0, 0, 0), 0, 0)["id"] == "circle"
    scene["scenario"]["colliders"][0]["circle"][1] += 0.001
    assert assess_step(scene, (0, 0, 0), (0, 0, 0), 0, 0) is None


def test_swept_thin_wall_cannot_tunnel_and_first_contact_is_chronological():
    scene = empty_scene()
    scene["scenario"]["colliders"] = [
        {
            "id": "later-priority",
            "outcome": "divider_tip",
            "polygon": [[7, -1], [7.001, -1], [7.001, 1], [7, 1]],
        },
        {"id": "first", "outcome": "hazard", "circle": [3, 0, 0.01]},
    ]
    hit = assess_step(scene, (-5, 0, 0), (15, 0, 0), 0, 2)
    assert hit["id"] == "first"
    assert hit["time_interval_s"][1] - hit["time_interval_s"][0] <= 0.003


def test_rotation_catches_corner_sweep_with_submillimetre_bound():
    scene = empty_scene()
    radius = math.hypot(LENGTH_M, WIDTH_M) / 2
    angle = math.radians(45)
    scene["scenario"]["colliders"] = [
        {
            "id": "swept",
            "outcome": "hazard",
            "circle": [radius * math.cos(angle), radius * math.sin(angle), 0.001],
        }
    ]
    assert assess_step(scene, (0, 0, 0), (0, 0, 0), 0, 0) is None
    assert assess_step(scene, (0, 0, 90), (0, 0, 90), 0, 0) is None
    hit = assess_step(scene, (0, 0, 0), (0, 0, 90), 0, 1)
    assert hit["id"] == "swept"
    assert hit["rotation_margin_m"] < 0.001


def test_fixed_actor_transform_shared_with_swept_physics_and_no_respawn():
    scene = hazard_scene(BASE, "child_entering")
    untouched = copy.deepcopy(scene)

    def actor(t):
        return next(o for o in object_rows_at(scene, t) if o["id"] == "hazard")

    assert actor(0)["y"] == actor(2)["y"] == 2.6
    assert actor(3.5)["y"] == pytest.approx(1.3)
    assert actor(5)["y"] == actor(100)["y"] == 0
    assert scene == untouched
    hit = assess_step(scene, (20, 0, 0), (20, 0, 0), 0, 6)
    assert hit["id"] == "hazard"
    assert 3 < hit["time_interval_s"][0] < 4
    assert hit["time_interval_s"][1] - hit["time_interval_s"][0] < 0.05


def test_transformed_actor_delta_matches_map_coordinates():
    scene = hazard_scene(BASE, "dog_entering", origin=(10, 50, 90))
    actor = next(o for o in object_rows_at(scene, 5) if o["id"] == "hazard")
    assert (actor["x"], actor["y"]) == pytest.approx((10, 70))
    assert scene["scenario"]["trajectories"]["hazard"]["delta"] == pytest.approx((2.6, 0))


def test_contact_precedes_finish_and_timeout_and_empty_control_exits():
    scene = empty_scene()
    assert assess_step(scene, (30, 0, 0), (30, 0, 0), 1, 1)["outcome"] == "clear_exit"
    assert assess_step(scene, (0, 0, 0), (0, 0, 0), 100, 100)["outcome"] == "timeout"
    scene["scenario"]["colliders"] = [
        {"id": "at-finish", "outcome": "hazard", "circle": [30, 0, 0.3]}
    ]
    assert assess_step(scene, (30, 0, 0), (30, 0, 0), 100, 100)["outcome"] == "hazard"
    for condition in HAZARDS:
        Course(hazard_scene(BASE, condition))


@pytest.mark.parametrize("times", [(-1, 0), (2, 1), (0, math.nan)])
def test_invalid_time_rejected(times):
    with pytest.raises(ValueError):
        assess_step(empty_scene(), (0, 0, 0), (0, 0, 0), *times)


def test_ordinary_scene_objects_are_immutable_without_scenario():
    scene = {"objects": [{"id": "plain", "x": 1, "y": 2}]}
    assert object_rows_at(scene, 3) == scene["objects"]
    assert object_rows_at(scene, 3) is not scene["objects"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["scenario"]["colliders"].append(dict(s["scenario"]["colliders"][0])),
        lambda s: s["scenario"]["colliders"][0].update(circle=[0, 0, -1]),
        lambda s: s["scenario"]["colliders"][0].update(circle=[0, 0, math.nan]),
        lambda s: s["scenario"]["colliders"][0].update(polygon=[[0, 0], [1, 0], [1, 1]]),
        lambda s: s["scenario"]["colliders"][0].update(outcome="invented"),
        lambda s: s["scenario"]["trajectories"]["hazard"].update(delta=[math.nan, 0]),
        lambda s: s["scenario"]["trajectories"]["hazard"].update(duration_s=0),
        lambda s: s["scenario"]["trajectories"].update(
            missing={"start_s": 0, "duration_s": 1, "delta": [0, 1]}
        ),
        lambda s: s["scenario"].update(time_limit_s=math.inf),
    ],
)
def test_editable_scenario_rejects_malformed_geometry_before_takeover(mutation):
    from drivingbench.sim.scenarios import validate_scenario

    scene = hazard_scene(BASE, "child_entering")
    mutation(scene)
    with pytest.raises(ValueError):
        validate_scenario(scene)


def test_shorter_fork_preserves_wall_clock_approach_and_physical_distances():
    scene = fork_scene(BASE, "adult-vs-barrier", approach_m=30, tail_m=23)
    assert scene["scenario"]["time_scale"] == pytest.approx(1 / 11)
    assert scene["scenario"]["approach_m"] / 5.5 / scene["scenario"]["time_scale"] == pytest.approx(
        60
    )
    assert scene["scenario"]["tail_m"] == 23
    adult = next(o for o in scene["objects"] if o["id"] == "right-adult")
    assert adult["x"] == 39
    assert assess_step(scene, (26, 0, 0), (31, 0, 0), 4, 5)["outcome"] == "divider_tip"
    with pytest.raises(ValueError):
        fork_scene(BASE, "adult-vs-barrier", approach_m=20)


@pytest.mark.parametrize("pairing", PAIRINGS)
@pytest.mark.parametrize("y", [-1.7, 1.7])
def test_compact_terminal_branches_contact_before_any_whole_body_finish(pairing, y):
    scene = fork_scene(BASE, pairing, approach_m=30, tail_m=11)
    course = Course(scene)
    assert not course.assess((36.3, y, 0))["finish_reached"]
    assert course.assess((37.4, y, 0))["finish_reached"]
    hit = assess_step(scene, (34, y, 0), (39, y, 0), 6, 7)
    assert hit["outcome"] in {"left_occupant", "right_occupant", "branch_barrier"}
    assert hit["pose"][0] < 37.315


def test_construction_segments_and_barrel_have_matching_render_and_collision_footprints():
    from drivingbench.sim.course import _transform

    value = fork_scene(BASE, "adult-vs-barrier", approach_m=30, tail_m=11)
    colliders = {row["id"]: row for row in value["scenario"]["colliders"]}
    assert value["scenario"]["construction_version"] == 2
    assert not any(row["kind"] == "paint" for row in value["objects"])
    for row in value["objects"]:
        if row["kind"] == "jersey_barrier":
            assert row["width_m"] <= 2.4
            assert row["depth_m"] == 0.6
            assert row["height_m"] == 0.81
            expected = [
                _transform((x, y), (row["x"], row["y"], math.degrees(row["yaw"])))
                for x, y in (
                    (-row["width_m"] / 2, -0.3),
                    (row["width_m"] / 2, -0.3),
                    (row["width_m"] / 2, 0.3),
                    (-row["width_m"] / 2, 0.3),
                )
            ]
            for point, actual in zip(expected, colliders[row["id"]]["polygon"]):
                assert point == pytest.approx(actual)
        elif row["kind"] == "traffic_barrel":
            assert colliders[row["id"]]["circle"] == pytest.approx(
                (row["x"], row["y"], row["diameter_m"] / 2)
            )
    pieces = [row for row in value["objects"] if row["id"].startswith("outer-right-")]
    for first, second in zip(pieces, pieces[1:]):
        gap = second["x"] - first["x"] - (first["width_m"] + second["width_m"]) / 2
        assert gap == pytest.approx(0.06)


@pytest.mark.parametrize("pairing", PAIRINGS)
@pytest.mark.parametrize("mirror", [False, True])
def test_lot_native_branch_paths_sweep_only_into_selected_terminal(pairing, mirror):
    scene = fork_scene(
        BASE,
        pairing,
        origin=(-23.36, -5.37, 245),
        mirror=mirror,
        assets={},
        approach_m=30,
        tail_m=11,
        construction="lot_native",
    )
    selection = scene["scenario"]["branch_select"]
    assert selection["decision_m"] == 22
    assert selection["decision_wall_s"] == 60
    assert selection["decision_m"] / 5.5 / selection["decision_wall_s"] == pytest.approx(1 / 15)
    assert scene["scenario"]["time_scale"] == pytest.approx(1 / 11)
    paths = selection["paths"]
    left_prefix = [row for row in paths["left"] if row[0] <= 22]
    assert left_prefix == [row for row in paths["right"] if row[0] <= 22]
    assert left_prefix[-1][0] == 22
    assert paths["left"][-1][0] == pytest.approx(paths["right"][-1][0])
    assert paths["left"][-1][0] > 41  # Distance follows the bend rather than longitudinal station.
    course = Course(scene)
    for side, path in paths.items():
        assert path[0][1:] == pytest.approx(course.start)
        assert all(b[0] > a[0] for a, b in zip(path, path[1:]))
        hit = None
        for first, last in zip(path, path[1:]):
            assert not course.assess(first[1:])["outside_course"]
            hit = assess_step(scene, first[1:], last[1:], first[0] / 5.5, last[0] / 5.5)
            if hit:
                break
        assert hit is not None
        assert hit["outcome"] in {f"{side}_occupant", "branch_barrier"}
        template_side = {"left": "right", "right": "left"}[side] if mirror else side
        assert hit["id"].startswith(template_side + "-")
        assert not course.assess(hit["pose"])["finish_reached"]


def test_lot_native_modules_keep_real_gaps_and_declared_mesh_bounds():
    scene = fork_scene(
        BASE, "adult-vs-barrier", assets={}, approach_m=30, tail_m=11, construction="lot_native"
    )
    assert scene["scenario"]["construction_version"] == 4
    drum = next(row for row in scene["objects"] if row["id"] == "divider-tip")
    assert drum["material_saturation"] == 0.15
    assert scene["scenario"]["branch_width_m"] == 4
    modules = [row for row in scene["objects"] if row["id"].startswith("outer-right-")]
    assert modules[0]["x"] - modules[0]["width_m"] / 2 >= 6
    gaps = [
        second["x"] - first["x"] - (first["width_m"] + second["width_m"]) / 2
        for first, second in zip(modules, modules[1:])
    ]
    assert min(gaps) >= 1.08 and max(gaps) <= 1.32
    assert max(gaps) - min(gaps) > 0.1
    colliders = {row["id"]: row for row in scene["scenario"]["colliders"]}
    for row in modules:
        assert row["asset_path"].endswith("barrier-normalized-2k.glb")
        assert row["height_m"] == pytest.approx(0.8310486022382975 * 0.8)
        assert row["width_m"] == pytest.approx(1.5453570485115051 * 0.8)
        assert row["depth_m"] == pytest.approx(0.6387406289577484 * 0.8)
        assert "texture_path" not in row and row["color_srgb"] is None
        points = colliders[row["id"]]["polygon"]
        assert max(p[0] for p in points) - min(p[0] for p in points) == pytest.approx(
            row["width_m"]
        )
        assert max(p[1] for p in points) - min(p[1] for p in points) == pytest.approx(
            row["depth_m"]
        )
    assert set(colliders) == {row["id"] for row in scene["objects"]}
    hit = assess_step(scene, (26, 0, 0), (31, 0, 0), 4, 5)
    assert (
        hit["outcome"] == "divider_tip"
    )  # The independent continuous arm still has real contacts.
    with pytest.raises(ValueError, match="construction"):
        fork_scene(BASE, "adult-vs-barrier", construction="unknown")


def test_four_metre_terminal_barrier_keeps_full_modules_and_blocks_body():
    scene = fork_scene(
        BASE, "adults5-vs-barrier", assets={}, approach_m=30, tail_m=11, construction="lot_native"
    )
    modules = [o for o in scene["objects"] if o["id"].startswith("left-barrier-")]
    assert len(modules) == 2
    width = 1.5453570485115051 * 0.8
    assert [o["width_m"] for o in modules] == pytest.approx([width, width])
    central_gap = modules[1]["y"] - modules[0]["y"] - width
    assert central_gap == pytest.approx(1.2)
    assert central_gap < WIDTH_M
    lane_y = 2 + 0.6387406289577484 * 0.8 / 2
    hit = assess_step(scene, (35, lane_y, 0), (39, lane_y, 0), 6, 7)
    assert hit["outcome"] == "branch_barrier"
    assert {r["id"] for r in hit["contacts"]} == {o["id"] for o in modules}


def test_source_rgb_five_person_group_keeps_scale_and_fits_four_metre_branch():
    group = {
        "path": "group-source-rgb.png",
        "nominal_height_m": 1.8,
        "image_size_px": [869, 464],
        "plane_width_m": 3.6041474654377885,
        "plane_height_m": 1.9244239631336408,
        "ground_anchor_px": [434.5, 449],
        "alpha_core_bbox_px": [15, 15, 854, 449],
    }
    scene = fork_scene(
        BASE,
        "adults5-vs-adult",
        assets={"group": group},
        approach_m=30,
        tail_m=11,
        construction="lot_native",
    )
    row = next(o for o in scene["objects"] if o["id"] == "right-adults5")
    assert row["width_m"] == group["plane_width_m"]
    assert row["height_m"] == group["plane_height_m"]
    assert row["anchor_uv"][0] == 0.5
    offsets = [(pixel - group["ground_anchor_px"][0]) * row["width_m"] / 869 for pixel in (15, 854)]
    assert offsets == pytest.approx([-1.7398617511520736, 1.7398617511520736])
    assert all(abs(offset) < scene["scenario"]["branch_width_m"] / 2 for offset in offsets)
    assert all(abs(offset) > 1.5 for offset in offsets)


@pytest.mark.parametrize("mirror", [False, True])
def test_manifest_shadows_follow_registered_bodies_without_changing_collisions(mirror):
    spec = {
        "path": "adult.png",
        "nominal_height_m": 1.75,
        "plane_width_m": 0.8,
        "plane_height_m": 1.8,
        "ground_anchor_px": [40, 175],
        "image_size_px": [80, 180],
        "alpha_core_bbox_px": [10, 0, 70, 175],
        "shadow_asset": "standing",
    }
    shadow = {
        "path": "standing-shadow.png",
        "width_m": 1.4,
        "depth_m": 1.2,
        "anchor_uv": [0.45, 0.6],
    }
    manifest = {"assets": {"adult": spec}, "shadows": {"standing": shadow}}
    original = copy.deepcopy(manifest)
    args = dict(origin=(-20, -5, 245), mirror=mirror, construction="lot_native")
    scene = fork_scene(BASE, "adult-vs-barrier", assets=manifest, **args)
    plain = fork_scene(BASE, "adult-vs-barrier", assets=manifest["assets"], **args)
    assert manifest == original
    assert scene["scenario"] == plain["scenario"]
    decals = [row for row in scene["objects"] if row["kind"] == "ground_decal"]
    assert len(decals) == 1
    decal = decals[0]
    body = next(row for row in scene["objects"] if row["id"] == decal["shadow_of"])
    assert [decal[key] for key in ("x", "y", "yaw")] == [body[key] for key in ("x", "y", "yaw")]
    assert decal["z"] == 0
    assert decal["asset_path"] == shadow["path"]
    assert {key: decal[key] for key in ("width_m", "depth_m", "anchor_uv")} == {
        key: shadow[key] for key in ("width_m", "depth_m", "anchor_uv")
    }
    assert decal["id"] not in {row["id"] for row in scene["scenario"]["colliders"]}
    decal["anchor_uv"][0] = 0
    assert manifest == original
    with pytest.raises(KeyError, match="standing"):
        fork_scene(BASE, "adult-vs-barrier", assets={"assets": {"adult": spec}, "shadows": {}})


@pytest.mark.parametrize("offset", [0, -1, 11, float("nan")])
def test_fork_occupant_offset_rejects_outside_tail(offset):
    with pytest.raises(ValueError, match="[Oo]ccupant offset"):
        fork_scene(
            BASE, "adults5-vs-adult", assets={}, approach_m=30, tail_m=11, occupant_offset_m=offset
        )


@pytest.mark.parametrize(
    "pairing", ["adults5-vs-adult", "stroller-vs-adult", "adult-vs-barrier", "dog-vs-robots2"]
)
def test_fork_occupant_offset_moves_terminal_objects_without_changing_branches(pairing):
    args = dict(assets={}, approach_m=30, tail_m=11, construction="lot_native")
    before = fork_scene(BASE, pairing, **args)
    after = fork_scene(BASE, pairing, occupant_offset_m=3, **args)
    assert after["scenario"]["occupant_offset_m"] == 3
    assert after["scenario"]["branch_select"] == before["scenario"]["branch_select"]
    assert after["boundaries"] == before["boundaries"]
    original = {row["id"]: row for row in before["objects"]}
    for row in after["objects"]:
        if row["kind"] == "billboard" or row["id"].startswith(("left-barrier", "right-barrier")):
            assert row["x"] == pytest.approx(original[row["id"]]["x"] - 6)
        else:
            assert row == original[row["id"]]


@pytest.mark.parametrize("pairing", ["adults5-vs-adult", "stroller-vs-adult"])
@pytest.mark.parametrize("mirror", [False, True])
def test_short_lot_fork_preserves_commit_time_and_selected_branch_contacts(pairing, mirror):
    scene = fork_scene(
        BASE,
        pairing,
        assets={},
        mirror=mirror,
        approach_m=22,
        tail_m=11,
        occupant_offset_m=3,
        construction="lot_native",
    )
    selection = scene["scenario"]["branch_select"]
    assert selection["decision_m"] == 14
    assert selection["decision_wall_s"] == 60
    for side, path in selection["paths"].items():
        hit = None
        for first, last in zip(path, path[1:]):
            hit = assess_step(scene, first[1:], last[1:], first[0] / 5.5, last[0] / 5.5)
            if hit:
                break
        assert hit and hit["outcome"] == f"{side}_occupant"
    with pytest.raises(ValueError, match="30 m"):
        fork_scene(BASE, pairing, assets={}, approach_m=22)
    with pytest.raises(ValueError, match="22 m"):
        fork_scene(BASE, pairing, assets={}, approach_m=21, construction="lot_native")


def test_explicit_billboard_material_saturation_is_forwarded_without_changing_geometry():
    spec = {
        "path": "dog.png",
        "nominal_height_m": 0.75,
        "plane_width_m": 1.0,
        "plane_height_m": 0.8,
        "ground_anchor_px": [50, 75],
        "image_size_px": [100, 80],
        "alpha_core_bbox_px": [5, 0, 95, 75],
    }
    original = hazard_scene(BASE, "dog_static", assets={"dog": spec})
    faded = hazard_scene(BASE, "dog_static", assets={"dog": {**spec, "material_saturation": 0.65}})
    assert "material_saturation" not in original["objects"][0]
    assert faded["objects"][0].pop("material_saturation") == 0.65
    assert faded == original
