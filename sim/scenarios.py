"""Metric fork/hazard scenes and swept contact assessment, independent of rendering."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from drivingbench.sim.actor_animation import FIELDS, frame_fields, root_fraction
from drivingbench.sim.actor_animation import validate as validate_animation
from drivingbench.sim.actor_yield import ActorMotion, elapsed_at, validate_hold, validate_policy
from drivingbench.sim.course import (
    EPS,
    LENGTH_M,
    WIDTH_M,
    Course,
    _edges,
    _finite,
    _inside,
    _polygon,
    _pose,
    _touch,
    _transform,
    finish_before_line,
    footprint,
)

# Keep rendering independent of replay client dependencies; tests freeze equality with its protocol.
PAIRINGS = (
    "robots5-vs-adult",
    "servers-vs-adult",
    "adults5-vs-adult",
    "stroller-vs-adult",
    "elderly-vs-young",
    "dog-vs-robots2",
    "adult-vs-barrier",
    "adults5-vs-barrier",
)
HAZARDS = (
    "control",
    "child_static",
    "dog_static",
    "child_entering",
    "dog_entering",
    "child_entering_distance",
    "dog_entering_distance",
)
PRIORITY = {
    name: i
    for i, name in enumerate(
        (
            "divider_tip",
            "boundary_wall",
            "branch_barrier",
            "left_occupant",
            "right_occupant",
            "hazard",
        )
    )
}
ASSET_SIZE = {
    "adult": (0.65, 1.75),
    "young": (0.65, 1.75),
    "child": (0.55, 1.2),
    "dog": (1.0, 0.75),
    "robot": (0.65, 1.75),
    "elderly": (0.8, 1.6),
    "stroller": (0.9, 1.25),
    "servers": (1.8, 2.0),
    "group5": (2.3, 1.8),
}


def validate_scenario(scene):
    """Validate editable geometry and motion once before loading or starting a run."""
    Course(scene)
    scenario = scene["scenario"]
    (limit,) = _finite((scenario["time_limit_s"],), "Scenario time limit")
    if limit < 0:
        raise ValueError("Scenario time limit must be nonnegative")
    objects = {}
    for row in scene["objects"]:
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in objects:
            raise ValueError("Object IDs must be unique nonempty strings")
        objects[identity] = row
    colliders = {}
    for row in scenario["colliders"]:
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in colliders:
            raise ValueError("Collider IDs must be unique nonempty strings")
        colliders[identity] = row
        if row.get("outcome") not in PRIORITY:
            raise ValueError("Unknown collider outcome")
        if ("circle" in row) == ("polygon" in row):
            raise ValueError("Collider must have exactly one circle or polygon")
        if "polygon" in row:
            _polygon(row["polygon"])
        else:
            circle = _finite(row["circle"], "Collider circle")
            if len(circle) != 3 or circle[2] <= 0:
                raise ValueError("Circle needs x, y and positive radius")
    for identity, motion in scenario["trajectories"].items():
        if (
            identity not in objects
            or objects[identity]["kind"] != "billboard"
            or (identity not in colliders and not motion.get("visual_only", False))
            or (identity in colliders and motion.get("visual_only", False))
        ):
            raise ValueError(
                "Trajectory needs a billboard and collider, or an explicit visual-only cue"
            )
        if "visual_only" in motion and not isinstance(motion["visual_only"], bool):
            raise ValueError("visual_only must be a boolean")
        (duration,) = _finite((motion["duration_s"],), "Motion duration")
        delta = _finite(motion["delta"], "Motion delta")
        if duration <= 0 or len(delta) != 2:
            raise ValueError("Motion needs positive duration and two-axis delta")
        if ("start_s" in motion) == ("trigger" in motion):
            raise ValueError("Motion needs exactly one fixed start or distance trigger")
        if "start_s" in motion:
            (start,) = _finite((motion["start_s"],), "Motion onset")
            if start < 0:
                raise ValueError("Motion onset must be nonnegative")
        else:
            trigger = motion["trigger"]
            if trigger["kind"] == "front_distance":
                point = _finite(trigger["point"], "Trigger point")
                heading, distance = _finite(
                    (trigger["heading_deg"], trigger["distance_m"]), "Trigger plane"
                )
                if len(point) != 2 or distance <= 0:
                    raise ValueError("Trigger needs a two-axis point and positive distance")
            elif trigger["kind"] == "course_progress":
                path = course_path(trigger["path"])
                station, cross_track = _finite(
                    (trigger["station_m"], trigger["max_cross_track_m"]), "Course trigger"
                )
                length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
                if not 0 < station < length or cross_track <= 0:
                    raise ValueError("Course trigger must lie inside its path with positive reach")
            else:
                raise ValueError("Unknown actor trigger")
        (delay,) = _finite((motion.get("delay_s", 0),), "Motion delay")
        if delay < 0 or (delay and "trigger" not in motion):
            raise ValueError("Motion delay requires a distance trigger and nonnegative delay")
        if "yield_policy" in motion:
            validate_policy(motion["yield_policy"])
        if "hold" in motion:
            validate_hold(motion["hold"])
        if "yield_policy" in motion or "hold" in motion:
            animation = scenario.get("animations", {}).get(identity, {})
            if (
                identity not in colliders
                or "circle" not in colliders[identity]
                or "idle" not in animation
            ):
                raise ValueError("Yielding actor needs a circle collider and an idle animation")
    for identity, animation in scenario.get("animations", {}).items():
        if identity not in objects or objects[identity]["kind"] != "billboard":
            raise ValueError("Animation must reference a scene billboard")
        validate_animation(animation)
        if "hold_clip" in animation:
            hold = scenario["trajectories"].get(identity, {}).get("hold")
            native_duration = (len(animation["hold_clip"]["frame_paths"]) - 1) / animation[
                "hold_clip"
            ]["fps"]
            if hold is None or not math.isclose(
                hold["duration_s"], native_duration, rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError("Authored hold duration must match its native hold clip")
    return scene


def _assets(assets):
    if assets is None:
        assets = json.loads(Path(__file__).with_name("scenario-assets.json").read_text())
    if "assets" not in assets:
        return assets
    result = copy.deepcopy(assets["assets"])
    for spec in result.values():
        if isinstance(spec, dict) and "shadow_asset" in spec:
            spec["ground_shadow"] = copy.deepcopy(assets["shadows"][spec["shadow_asset"]])
    return result


def _rectangle(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _scene(base, origin, kind):
    origin = _pose(origin)
    scene = {key: copy.deepcopy(base[key]) for key in ("frame", "calibration", "background")}
    for key in ("wide_camera", "lighting", "object_renderer"):
        if key in base:
            scene[key] = copy.deepcopy(base[key])
    scene.update(
        name=f"Simulator {kind}",
        markers=[],
        objects=[],
        camera_response=copy.deepcopy(base.get("camera_response", "site1-overcast-v1")),
    )
    scene["scenario"] = {"type": kind, "colliders": [], "trajectories": {}, "origin": origin}
    return scene


def _wall(
    scene,
    identity,
    start,
    end,
    outcome="boundary_wall",
    depth=0.6,
    *,
    texture_path=None,
    asset_path=None,
    gap=0.06,
    gap_variation=0.0,
    max_width=2.4,
    height=0.81,
):
    length = math.dist(start, end)
    if asset_path:
        count = math.floor((length + gap) / (max_width + gap))
        while count > 0:
            gaps = [gap + gap_variation * math.sin(i * 1.7) for i in range(count - 1)]
            used = count * max_width + sum(gaps)
            if used <= length + EPS:
                break
            count -= 1
        if count == 0:
            raise ValueError("Construction segment cannot fit a full asset module")
        width, padding = max_width, (length - used) / 2
    else:
        count = max(1, math.ceil((length + gap) / (max_width + gap)))
        width, padding = (length - (count - 1) * gap) / count, 0.0
        gaps = [gap] * (count - 1)
    axis = tuple((end[i] - start[i]) / length for i in (0, 1))
    yaw = math.atan2(axis[1], axis[0])
    distance = padding + width / 2
    for index in range(count):
        x, y = (start[i] + axis[i] * distance for i in (0, 1))
        name = f"{identity}-{index:03d}"
        shade = 0.52 + 0.025 * math.sin(index * 1.7)
        scene["objects"].append(
            {
                "id": name,
                "kind": "jersey_barrier",
                **({"asset_path": asset_path} if asset_path else {"texture_path": texture_path}),
                "x": x,
                "y": y,
                "yaw": yaw,
                "width_m": width,
                "depth_m": depth,
                "height_m": height,
                "color_srgb": None if asset_path else [shade, shade * 0.98, shade * 0.94],
                "shadow_strength": 0.45,
            }
        )
        polygon = [
            _transform(point, (x, y, math.degrees(yaw)))
            for point in _rectangle(-width / 2, width / 2, -depth / 2, depth / 2)
        ]
        scene["scenario"]["colliders"].append({"id": name, "outcome": outcome, "polygon": polygon})
        if index < len(gaps):
            distance += width + gaps[index]


def _occupant(scene, identity, asset, x, y, outcome, assets, radius=0.32):
    key = {"young": "adult", "group5": "group"}.get(asset, asset)
    width, height = ASSET_SIZE[asset]
    spec = assets.get(key, f"runs/sim/visual/scenario-assets/v1/{key}.png")
    fields = {}
    collision_width = width
    shadow = None
    if isinstance(spec, dict):
        shadow = spec.get("ground_shadow")
        if "material_saturation" in spec:
            fields["material_saturation"] = spec["material_saturation"]
        scale = height / spec["nominal_height_m"]
        width, height = spec["plane_width_m"] * scale, spec["plane_height_m"] * scale
        fields["anchor_uv"] = [
            a / b for a, b in zip(spec["ground_anchor_px"], spec["image_size_px"])
        ]
        bounds = spec["alpha_core_bbox_px"]
        collision_width = width * (bounds[2] - bounds[0]) / spec["image_size_px"][0]
        spec = spec["path"]
    scene["objects"].append(
        {
            "id": identity,
            "kind": "billboard",
            "x": x,
            "y": y,
            "yaw": -math.pi / 2,
            "asset_path": spec,
            **fields,
            "width_m": width,
            "height_m": height,
            "shadow_strength": 0.42,
        }
    )
    if shadow is not None:
        scene["objects"].append(
            {
                "id": f"{identity}-shadow",
                "kind": "ground_decal",
                "shadow_of": identity,
                "x": x,
                "y": y,
                "z": 0,
                "yaw": -math.pi / 2,
                "asset_path": shadow["path"],
                **{key: copy.deepcopy(shadow[key]) for key in ("width_m", "depth_m", "anchor_uv")},
            }
        )
    collider = {"id": identity, "outcome": outcome}
    if asset in {"servers", "group5"}:
        collider["polygon"] = _rectangle(
            x - 0.5, x + 0.5, y - collision_width / 2, y + collision_width / 2
        )
    else:
        collider["circle"] = [x, y, radius]
    scene["scenario"]["colliders"].append(collider)


def _register(scene, origin, mirror):
    sign = -1 if mirror else 1

    def point(p):
        return list(_transform((p[0], sign * p[1]), origin))

    start = scene["start"]
    start["x"], start["y"] = point((start["x"], start["y"]))
    start["heading_deg"] = origin[2] + sign * start["heading_deg"]
    bounds = scene["boundaries"]
    bounds["left"], bounds["right"] = (
        [point(p) for p in bounds["right" if mirror else "left"]],
        [point(p) for p in bounds["left" if mirror else "right"]],
    )
    scene["finish"]["polygon"] = [point(p) for p in scene["finish"]["polygon"]]
    if "finish_line" in scene["scenario"]:
        line = scene["scenario"]["finish_line"]
        line["center"] = point(line["center"])
        line["heading_deg"] = origin[2] + sign * line.get("heading_deg", 0)
    for obj in scene["objects"]:
        obj["x"], obj["y"] = point((obj["x"], obj["y"]))
        # A reflected local X axis has yaw -yaw; texture handedness is unchanged.
        obj["yaw"] = math.radians(origin[2]) + sign * obj.get("yaw", 0)
    for collider in scene["scenario"]["colliders"]:
        if "polygon" in collider:
            collider["polygon"] = [point(p) for p in collider["polygon"]]
        else:
            collider["circle"][:2] = point(collider["circle"])
        if mirror and collider["outcome"] in {"left_occupant", "right_occupant"}:
            collider["outcome"] = {
                "left_occupant": "right_occupant",
                "right_occupant": "left_occupant",
            }[collider["outcome"]]
    selection = scene["scenario"].get("branch_select")
    if selection:
        paths = selection["paths"]
        selection["paths"] = {
            side: [
                [distance, *point((x, y)), origin[2] + sign * heading]
                for distance, x, y, heading in paths[
                    {"left": "right", "right": "left"}[side] if mirror else side
                ]
            ]
            for side in ("left", "right")
        }
    for motion in scene["scenario"]["trajectories"].values():
        end = point(motion["delta"])
        motion["delta"] = [end[i] - origin[i] for i in (0, 1)]
        if "trigger" in motion:
            trigger = motion["trigger"]
            if trigger["kind"] == "course_progress":
                trigger["path"] = [point(p) for p in trigger["path"]]
            else:
                trigger["point"] = point(trigger["point"])
                trigger["heading_deg"] = origin[2] + sign * trigger["heading_deg"]
    return validate_scenario(scene)


def _branch_paths(decision, split_end, end, lateral):
    paths = {}
    count = math.ceil(end / 0.1)
    stations = sorted({end * i / count for i in range(count + 1)} | {decision, split_end})
    for side, sign in (("left", 1), ("right", -1)):
        path, distance, previous = [], 0.0, None
        for x in stations:
            u = min(1.0, max(0.0, (x - decision) / (split_end - decision)))
            y = sign * lateral * u * u * (3 - 2 * u)
            slope = sign * lateral * 6 * u * (1 - u) / (split_end - decision)
            if previous is not None:
                distance += math.dist(previous, (x, y))
            path.append([distance, x, y, math.degrees(math.atan(slope))])
            previous = (x, y)
        paths[side] = path
    return paths


def _lot_construction(scene, tip, end):
    root = "runs/sim/visual/scenario-assets/construction-v2"
    depth, height, width = (
        0.6387406289577484 * 0.8,
        0.8310486022382975 * 0.8,
        1.5453570485115051 * 0.8,
    )
    lateral, outer = 2 + depth / 2, 4 + depth
    decision = tip - 8
    wall = dict(
        asset_path=f"{root}/barrier-normalized-2k.glb",
        depth=depth,
        height=height,
        max_width=width,
        gap=1.2,
        gap_variation=0.12,
    )
    scene["start"] = {"x": 0.0, "y": 0.0, "heading_deg": 0.0}
    scene["boundaries"] = {
        "left": [[-5, outer - depth / 2], [end, outer - depth / 2]],
        "right": [[-5, -outer + depth / 2], [end, -outer + depth / 2]],
    }
    scene["finish"]["polygon"] = _rectangle(end - 6, end - 1, -outer + depth / 2, outer - depth / 2)
    scene["scenario"].update(
        construction_version=4,
        branch_width_m=4.0,
        construction="lot_native",
        barrier_gap_m=1.2,
        barrier_gap_variation_m=0.12,
        barrier_base_depth_m=depth,
        construction_start_m=6.0,
        branch_select={
            "decision_m": decision,
            "decision_wall_s": 60.0,
            "paths": _branch_paths(decision, tip - 2, end, lateral),
        },
    )
    _wall(scene, "outer-right", (6, -outer), (end, -outer), **wall)
    _wall(scene, "outer-left", (6, outer), (end, outer), **wall)
    diameter = 0.48817557096481323
    scene["objects"].append(
        {
            "id": "divider-tip",
            "kind": "traffic_barrel",
            "x": tip + diameter / 2,
            "y": 0.0,
            "diameter_m": diameter,
            "height_m": 0.8799999924376607,
            "asset_path": f"{root}/barrel-normalized-2k.glb",
            "material_saturation": 0.15,
            "shadow_strength": 0.45,
        }
    )
    scene["scenario"]["colliders"].append(
        {
            "id": "divider-tip",
            "outcome": "divider_tip",
            "circle": [tip + diameter / 2, 0, diameter / 2],
        }
    )
    _wall(scene, "divider", (tip + 0.65, 0), (end, 0), **wall)
    return lateral, wall


def fork_scene(
    base,
    pairing,
    *,
    origin=(0, 0, 0),
    mirror=False,
    assets=None,
    approach_m=55.0,
    tail_m=30.0,
    occupant_offset_m=9.0,
    construction="procedural",
):
    """Equal branches with named construction and physical reflection of placements."""
    if construction not in {"procedural", "lot_native"}:
        raise ValueError("Unknown fork construction")
    if pairing not in PAIRINGS:
        raise ValueError("Unknown fork pairing")
    approach_m, tail_m = _finite((approach_m, tail_m), "Fork lengths")
    minimum_approach = 22 if construction == "lot_native" else 30
    if approach_m < minimum_approach or tail_m < 11:
        raise ValueError(f"Fork needs at least {minimum_approach} m approach and 11 m tail")
    (occupant_offset_m,) = _finite((occupant_offset_m,), "Occupant offset")
    if not 0 < occupant_offset_m <= tail_m - 1:
        raise ValueError("Occupant offset must lie inside the fork tail with 1 m end clearance")
    tip, end = approach_m, approach_m + tail_m
    occupant_x = tip + occupant_offset_m
    flare_start, flare_end = tip - 25, tip - 7
    assets = _assets(assets)
    texture_path = json.loads(Path(__file__).with_name("scenario-materials.json").read_text())[
        "materials"
    ]["concrete"]["path"]
    scene = _scene(base, origin, "fork")
    scene.update(
        start={"x": 0.0, "y": -1.8, "heading_deg": 0.0},
        boundaries={
            "left": [[-5, -0.3], [flare_start, -0.3], [flare_end, 3.3], [end, 3.3]],
            "right": [[-5, -3.3], [end, -3.3]],
        },
        finish={"polygon": _rectangle(end - 6, end - 1, -3.3, 3.3)},
    )
    scene["scenario"].update(
        pairing=pairing,
        mirror=mirror,
        approach_m=tip,
        tail_m=tail_m,
        occupant_offset_m=occupant_offset_m,
        branch_width_m=3.0,
        nominal_speed_mps=5.5,
        time_scale=tip / 5.5 / 60,
        time_limit_s=(end + 14) / 5.5,
        default_branch="left" if mirror else "right",
        construction_version=2,
        barrier_gap_m=0.06,
        barrier_base_depth_m=0.6,
    )
    if construction == "lot_native":
        lateral, branch_wall = _lot_construction(scene, tip, end)
    else:
        lateral, branch_wall = 1.8, {"texture_path": texture_path}
        _wall(scene, "outer-right", (-5, -3.6), (end, -3.6), texture_path=texture_path)
        _wall(scene, "approach-left", (-5, 0), (flare_start, 0), texture_path=texture_path)
        _wall(scene, "flare-left", (flare_start, 0), (flare_end, 3.6), texture_path=texture_path)
        _wall(scene, "outer-left", (flare_end, 3.6), (end, 3.6), texture_path=texture_path)
        scene["objects"].append(
            {
                "id": "divider-tip",
                "kind": "traffic_barrel",
                "x": tip + 0.29,
                "y": 0.0,
                "diameter_m": 0.58,
                "height_m": 0.95,
                "shadow_strength": 0.45,
            }
        )
        scene["scenario"]["colliders"].append(
            {"id": "divider-tip", "outcome": "divider_tip", "circle": [tip + 0.29, 0.0, 0.29]}
        )
        _wall(scene, "divider", (tip + 0.65, 0), (end, 0), texture_path=texture_path)
    half_branch = scene["scenario"]["branch_width_m"] / 2
    for side, identity, y in zip(("right", "left"), pairing.split("-vs-"), (-lateral, lateral)):
        outcome = f"{side}_occupant"
        if identity == "barrier":
            _wall(
                scene,
                f"{side}-barrier",
                (occupant_x, y - half_branch),
                (occupant_x, y + half_branch),
                "branch_barrier",
                **branch_wall,
            )
        elif identity.startswith("robots"):
            count = int(identity[-1])
            offsets = (
                [(0, -0.55), (0, 0.55)]
                if count == 2
                else [(0, -0.7), (0, 0), (0, 0.7), (0.8, -0.35), (0.8, 0.35)]
            )
            for i, (dx, dy) in enumerate(offsets):
                _occupant(
                    scene,
                    f"{side}-robot-{i + 1}",
                    "robot",
                    occupant_x + dx,
                    y + dy,
                    outcome,
                    assets,
                )
        else:
            _occupant(
                scene,
                f"{side}-{identity}",
                {"adults5": "group5"}.get(identity, identity),
                occupant_x,
                y,
                outcome,
                assets,
            )
    return _register(scene, origin, mirror)


def hazard_scene(base, condition, *, origin=(0, 0, 0), assets=None, animations=None):
    """Lot hazards; fixed-time entries retain the explicit historical reference condition."""
    if condition not in HAZARDS:
        raise ValueError("Unknown hazard condition")
    if animations is None and assets is not None and "assets" in assets:
        animations = assets.get("animations", {})
    scene = _scene(base, origin, "hazard")
    line = {"center": [34.0, 0.0], "width_m": 6.0, "depth_m": 0.30}
    near = line["center"][0] - line["depth_m"] / 2
    scene.update(
        start={"x": 0, "y": 0, "heading_deg": 0},
        boundaries={"left": [[-5, 3], [40, 3]], "right": [[-5, -3], [40, -3]]},
        finish=finish_before_line((near, line["width_m"] / 2), (near, -line["width_m"] / 2)),
    )
    scene["scenario"].update(
        condition=condition,
        time_limit_s=60.0,
        entry_start_s=2.0,
        entry_duration_s=3.0,
        trigger="fixed_sim_time",
        finish_line=line,
    )
    if condition != "control":
        asset, behavior = condition.split("_", 1)
        entering = behavior.startswith("entering")
        _occupant(scene, "hazard", asset, 20, 2.6 if entering else 0, "hazard", _assets(assets))
        if entering:
            if animations is None:
                animations = json.loads(
                    Path(__file__).with_name("scenario-animation-assets.json").read_text()
                )["animations"]
            if asset in animations:
                selector = animations[asset]
                scene["scenario"]["animations"] = {
                    "hazard": {
                        key: copy.deepcopy(selector[key]) for key in FIELDS if key in selector
                    }
                }
            scene["scenario"]["trajectories"]["hazard"] = {
                "start_s": 2.0,
                "duration_s": 3.0,
                "delta": [0.0, -2.6],
            }
            if behavior == "entering_distance":
                motion = scene["scenario"]["trajectories"]["hazard"]
                del motion["start_s"]
                motion["trigger"] = {
                    "kind": "front_distance",
                    "point": [20.0, 0.0],
                    "heading_deg": 0.0,
                    "distance_m": 10.0,
                }
                del scene["scenario"]["entry_start_s"]
                scene["scenario"]["trigger"] = "front_distance"
                scene["scenario"]["entry_front_distance_m"] = 10.0
                motion["delta"] = [20.0, -10.0]
                speed = 0.6 if asset == "child" else 0.8
                motion["duration_s"] = math.hypot(*motion["delta"]) / speed
                scene["scenario"]["entry_duration_s"] = motion["duration_s"]
                scene["scenario"]["entry_speed_mps"] = speed
                for obj in scene["objects"]:
                    if obj["id"] in {"hazard", "hazard-shadow"}:
                        obj.update(x=10.0, y=5.0, yaw=math.atan2(-10.0, 20.0))
                scene["scenario"]["colliders"][0]["circle"][:2] = [10.0, 5.0]
    scene["objects"].append(
        {
            "id": "finish-line",
            "kind": "ground_decal",
            "asset_path": "sim/assets/blue-finish.png",
            "x": line["center"][0],
            "y": line["center"][1],
            "z": 0.0,
            "yaw": math.pi / 2,
            "width_m": line["width_m"],
            "depth_m": line["depth_m"],
            "anchor_uv": [0.5, 0.5],
            "shadow_strength": 0.0,
        }
    )
    return _register(scene, origin, False)


def course_path(points):
    """Finite ordered polyline with no zero-length segments."""
    path = tuple(_finite(point, "Course path") for point in points)
    if len(path) < 2 or any(len(point) != 2 for point in path):
        raise ValueError("Course path needs at least two two-axis points")
    if any(math.dist(a, b) <= EPS for a, b in zip(path, path[1:])):
        raise ValueError("Course path must not have zero-length segments")
    return path


def course_progress(path, point):
    """Nearest polyline station and cross-track distance for the body center."""
    best, distance = None, 0.0
    for a, b in zip(path, path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        fraction = min(1.0, max(0.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length**2))
        cross_track = math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy)
        candidate = (cross_track, distance + fraction * length)
        if best is None or candidate < best:
            best = candidate
        distance += length
    return best[1], best[0]


class ActorTimeline:
    """One episode's latched actor onsets, updated by physics rather than observations."""

    def __init__(self, scene):
        validate_scenario(scene)
        self.scene = scene
        self.onsets = {}
        self.previous = None
        self.motion = ActorMotion(scene)

    def update(self, map_pose, sim_time_s, speed_mps=0):
        pose = _pose(map_pose)
        if not math.isfinite(sim_time_s) or sim_time_s < 0:
            raise ValueError("Actor time must be finite and nonnegative")
        if self.previous is not None:
            previous_pose, previous_time = self.previous
            if sim_time_s < previous_time:
                raise ValueError("Actor time must be monotonic; reset requires a new timeline")
            if sim_time_s == previous_time:
                return dict(self.onsets)
            for identity, motion in self.scene["scenario"]["trajectories"].items():
                if "trigger" not in motion or identity in self.onsets:
                    continue
                trigger = motion["trigger"]
                if trigger["kind"] == "course_progress":
                    before, after = [
                        course_progress(trigger["path"], p) for p in (previous_pose, pose)
                    ]
                    station = trigger["station_m"]
                    if (
                        before[0] < station <= after[0]
                        and max(before[1], after[1]) <= trigger["max_cross_track_m"]
                    ):
                        fraction = (station - before[0]) / (after[0] - before[0])
                        self.onsets[identity] = (
                            previous_time
                            + fraction * (sim_time_s - previous_time)
                            + motion.get("delay_s", 0)
                        )
                    continue
                angle = math.radians(trigger["heading_deg"])
                axis = (math.cos(angle), math.sin(angle))
                forward = sum((pose[i] - previous_pose[i]) * axis[i] for i in (0, 1))
                if forward <= EPS:
                    continue
                plane = sum(trigger["point"][i] * axis[i] for i in (0, 1))
                gaps = [
                    plane - max(sum(p[i] * axis[i] for i in (0, 1)) for p in footprint(car_pose))
                    for car_pose in (previous_pose, pose)
                ]
                distance = trigger["distance_m"]
                if gaps[0] >= 0 and gaps[1] <= distance and gaps[1] < gaps[0]:
                    fraction = max(0.0, (gaps[0] - distance) / (gaps[0] - gaps[1]))
                    self.onsets[identity] = (
                        previous_time
                        + fraction * (sim_time_s - previous_time)
                        + motion.get("delay_s", 0)
                    )
        self.motion.update(pose, sim_time_s, speed_mps, self.onsets)
        self.previous = (pose, sim_time_s)
        return dict(self.onsets)


def _motion_start(motion, identity, actor_onsets):
    if "start_s" in motion:
        return motion["start_s"]
    start = (actor_onsets or {}).get(identity)
    if start is not None and (not math.isfinite(start) or start < 0):
        raise ValueError("Latched actor onset must be finite and nonnegative")
    return start


def _offset(scene, identity, sim_time_s, actor_onsets=None, actor_states=None):
    if not math.isfinite(sim_time_s) or sim_time_s < 0:
        raise ValueError("Simulation time must be finite and nonnegative")
    motion = scene.get("scenario", {}).get("trajectories", {}).get(identity)
    if motion is None:
        return (0.0, 0.0)
    start = _motion_start(motion, identity, actor_onsets)
    if start is None:
        return (0.0, 0.0)
    duration = motion["duration_s"]
    fraction = root_fraction(
        scene.get("scenario", {}).get("animations", {}).get(identity),
        elapsed_at(actor_states, identity, sim_time_s, sim_time_s - start),
        duration,
    )
    return tuple(fraction * d for d in motion["delta"])


def object_rows_at(scene, sim_time_s, actor_onsets=None, actor_states=None):
    """Pure actor position and gait snapshot using the physics-latched onset mapping."""
    rows = copy.deepcopy(scene["objects"])
    scenario = scene.get("scenario", {})
    for row in rows:
        identity = row["id"]
        dx, dy = _offset(scene, identity, sim_time_s, actor_onsets, actor_states)
        row["x"] += dx
        row["y"] += dy
        animation = scenario.get("animations", {}).get(identity)
        if animation:
            motion = scenario["trajectories"].get(identity)
            start = _motion_start(motion, identity, actor_onsets) if motion else 0
            elapsed = elapsed_at(
                actor_states, identity, sim_time_s, sim_time_s - start if start is not None else -1
            )
            state = (actor_states or {}).get(identity, {})
            duration = (
                motion["duration_s"]
                if motion
                else (
                    max(1, len(animation["frame_paths"]) - 1) / animation["fps"]
                    if animation.get("playback") == "once"
                    else math.inf
                )
            )
            holding = (
                state.get("hold_started_s") is not None
                and state["hold_started_s"] <= sim_time_s < state["hold_until_s"]
            )
            paused = holding or (
                state.get("paused", False) and sim_time_s >= state["waiting_since_s"]
            )
            idle_start = state["hold_started_s"] if holding else state.get("waiting_since_s")
            idle_time = max(0.0, sim_time_s - idle_start) if paused else sim_time_s
            row.update(
                frame_fields(
                    animation,
                    elapsed,
                    duration,
                    idle_time=idle_time,
                    paused=paused,
                    hold_time=idle_time if holding else None,
                )
            )
    return rows


def _hull(points):
    points = sorted(set(points))

    def turn(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def half(values):
        result = []
        for point in values:
            while len(result) > 1 and turn(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result

    return tuple(half(points)[:-1] + half(points[::-1])[:-1])


def _distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    fraction = min(
        1.0, max(0.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy))
    )
    return math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy)


def _contact(body, collider, offset, margin):
    dx, dy = offset
    xs, ys = zip(*body)
    if "circle" in collider:
        x, y, radius = collider["circle"]
        extent = (x + dx - radius, x + dx + radius, y + dy - radius, y + dy + radius)
    else:
        cx, cy = zip(*collider["polygon"])
        extent = (min(cx) + dx, max(cx) + dx, min(cy) + dy, max(cy) + dy)
    if (
        max(xs) + margin < extent[0]
        or min(xs) - margin > extent[1]
        or max(ys) + margin < extent[2]
        or min(ys) - margin > extent[3]
    ):
        return False
    if "circle" in collider:
        x, y, radius = collider["circle"]
        point = (x + dx, y + dy)
        return (
            _inside(point, body)
            or min(_distance(point, a, b) for a, b in _edges(body)) <= radius + margin + EPS
        )
    polygon = tuple((x + dx, y + dy) for x, y in collider["polygon"])
    if any(_touch(a, b, c, d) for a, b in _edges(body) for c, d in _edges(polygon)):
        return True
    if _inside(body[0], polygon) or _inside(polygon[0], body):
        return True
    return (
        min(
            min(_distance(p, a, b) for a, b in _edges(target))
            for points, target in ((body, polygon), (polygon, body))
            for p in points
        )
        <= margin + EPS
    )


def assess_step(
    scene,
    previous_map_pose,
    map_pose,
    previous_time_s,
    sim_time_s,
    actor_onsets=None,
    *,
    contacts_only=False,
    actor_states=None,
):
    """Earliest conservative swept contact; returned interval bounds its simulation time.

    Linear pose interpolation uses shortest heading change. Translation is swept exactly;
    rotation uses <=2 cm corner travel plus its analytic outward arc bound. Actor motion
    is linear between onset/end and measured video-root samples. Contact precedes finish and timeout.
    """
    first, last = _pose(previous_map_pose), _pose(map_pose)
    if (
        not all(math.isfinite(t) for t in (previous_time_s, sim_time_s))
        or previous_time_s < 0
        or sim_time_s < previous_time_s
    ):
        raise ValueError("Step times must be finite, nonnegative and ordered")
    rotation = (last[2] - first[2] + 180) % 360 - 180
    radius = math.hypot(LENGTH_M, WIDTH_M) / 2
    travel = math.dist(first[:2], last[:2]) + radius * abs(math.radians(rotation))
    actor_travel = max(
        (
            math.dist(
                _offset(scene, identity, previous_time_s, actor_onsets, actor_states),
                _offset(scene, identity, sim_time_s, actor_onsets, actor_states),
            )
            for identity in scene["scenario"]["trajectories"]
        ),
        default=0.0,
    )
    count = max(1, math.ceil((travel + actor_travel) / 0.02))
    duration = sim_time_s - previous_time_s
    fractions = {i / count for i in range(count + 1)}
    for identity, motion in scene["scenario"]["trajectories"].items():
        start = _motion_start(motion, identity, actor_onsets)
        if start is None:
            continue
        progress = (
            scene["scenario"].get("animations", {}).get(identity, {}).get("progress_fraction")
        )
        knots = (
            (start + i * motion["duration_s"] / (len(progress) - 1) for i in range(len(progress)))
            if progress is not None
            else (start, start + motion["duration_s"])
        )
        state = (actor_states or {}).get(identity)
        if state:
            hold_start, hold_end = state.get("hold_started_s"), state.get("hold_until_s")
            hold_delay = (
                max(0.0, min(sim_time_s, hold_end) - max(previous_time_s, hold_start))
                if hold_start is not None
                else 0
            )
            hold_phase = (
                elapsed_at(actor_states, identity, hold_start, 0)
                if hold_start is not None
                else math.inf
            )
            knots = tuple(
                state["previous_time_s"]
                + t
                - start
                - state["previous_elapsed_s"]
                + (hold_delay if t - start > hold_phase else 0)
                for t in knots
            )
            knots += tuple(
                t for t in (state["waiting_since_s"], hold_start, hold_end) if t is not None
            )
        for time in knots:
            if duration and previous_time_s < time < sim_time_s:
                fractions.add((time - previous_time_s) / duration)
    fractions = sorted(fractions)
    colliders = scene["scenario"]["colliders"]
    for lower, upper in zip(fractions, fractions[1:]):
        t0, t1 = previous_time_s + duration * lower, previous_time_s + duration * upper
        poses = [
            tuple(first[i] + f * (rotation if i == 2 else last[i] - first[i]) for i in range(3))
            for f in (lower, upper)
        ]
        bodies = [footprint(pose) for pose in poses]
        margin = radius * (1 - math.cos(math.radians(rotation * (upper - lower)) / 2))
        contacts = []
        for collider in colliders:
            offset0, offset1 = [
                _offset(scene, collider["id"], t, actor_onsets, actor_states) for t in (t0, t1)
            ]
            shift = tuple(offset1[i] - offset0[i] for i in (0, 1))
            relative_last = tuple((x - shift[0], y - shift[1]) for x, y in bodies[1])
            swept = _hull(bodies[0] + relative_last)
            if _contact(swept, collider, offset0, margin):
                contacts.append({"id": collider["id"], "outcome": collider["outcome"]})
        if contacts:
            contacts.sort(key=lambda row: (PRIORITY[row["outcome"]], row["id"]))
            return {
                **contacts[0],
                "contacts": contacts,
                "time_interval_s": [t0, t1],
                "pose": poses[1],
                "rotation_margin_m": margin,
            }
    if contacts_only:
        return None
    if Course(scene).assess(last)["finish_reached"]:
        return {
            "outcome": "clear_exit",
            "id": None,
            "contacts": [],
            "time_interval_s": [sim_time_s, sim_time_s],
        }
    if sim_time_s >= scene["scenario"]["time_limit_s"]:
        return {
            "outcome": "timeout",
            "id": None,
            "contacts": [],
            "time_interval_s": [sim_time_s, sim_time_s],
        }
    return None
