"""Explicit pinned authored tidy C/D variant; original pilot defaults remain separate."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path

from drivingbench.sim.logs import REPO
from drivingbench.sim.producer import file_hash
from drivingbench.sim.real_course_study.coverage import CoverageCertificate
from drivingbench.sim.real_course_study.refined import RefinedPolicy, runtime_factory

VARIANT = "tidy-authored-v3-CD-v1"
BINDINGS = REPO / "runs/sim/naturalistic-CD-tidy-preparation-v1/scene-bindings.json"
BINDINGS_SHA = "f660d1af3bf49f748da2ca1920b5d87c39927c5a6a4cb94cce7000de03f0d5af"
BASE_SHA = "de4b2da622f6e9d9646604340974d94a22c2a753286d1aa7ce5f93b04c321906"
EXPOSURE_VARIANT = "tidy-authored-v3-CD-exposure-v2"
C_CENTER_VARIANT = "tidy-authored-v3-C-center-hold-clear-v3"
C_NEAR_VARIANT = "tidy-authored-v3-C-center-hold-clear-v4"
D_NEAR_VARIANT = "tidy-authored-v3-D-near-roadside-v3"
D_NEAR_NEXT_VARIANT = "tidy-authored-v3-D-near-roadside-v4"
B_VARIANT = "tidy-authored-v3-B-known-car-v1"
# Root approved this exact prospective scene source only; actual qualification is separate.
B_BINDINGS = REPO / "runs/sim/oct5-B-tidy-actual-asset-binding-preparation-v1/scene-bindings.json"
B_BINDINGS_SHA = "5ccde7ea1d3cbab519b1f56b88c9d945b48f26c303a34d3e0e4e77c2eff721f0"
B_GEOMETRY_SHA = "59ddd99fee5cb5063bac269aac5a23b1f1b888238268b22627173ef493424853"
B_REFERENCES = {
    "tidy-authored-v3-B-known-bollard-v1": (
        REPO / "runs/sim/oct6-B-bollard-station16-actual-binding-v2/scene-bindings.json",
        "68484eb6c835d0aa0699a2a26c1bfe48142873fd952529b7c1e972b1c17bb8fe",
        "ab60f9119cda83a8d420bc2ae3a2afb0a00aaec3a5ece90324d94d985ebe7ef8",
    ),
    B_VARIANT: (B_BINDINGS, B_BINDINGS_SHA, B_GEOMETRY_SHA),
    "tidy-authored-v3-B-known-car-v2": (
        REPO / "runs/sim/oct6-B-tidy-actual-asset-binding-v2/scene-bindings.json",
        "f7d1775a18455a578ec81c436cf31b1ba7b0424d888d55dd98452df2b1f8e176",
        "c07ee2afa967114a7340663ef9e05b898e9bca7a6e80d8360377b988800f7e2f",
    ),
}
VARIANTS = (
    VARIANT,
    EXPOSURE_VARIANT,
    C_CENTER_VARIANT,
    C_NEAR_VARIANT,
    D_NEAR_VARIANT,
    D_NEAR_NEXT_VARIANT,
    *B_REFERENCES,
)
EXPOSURE_BINDINGS = (
    REPO / "runs/sim/oct5-CD-exposure-implementation-candidate-v1/scene-bindings.json"
)
EXPOSURE_BINDINGS_SHA = "9356ae99b922f426de2376c5a1ea6e547164e3ad2ad455ad783f23ba3be02841"

C_CENTER_BINDINGS = REPO / "runs/sim/oct5-CD-center-and-roadside-candidates-v1/scene-bindings.json"
C_CENTER_BINDINGS_SHA = "06800f0c2c2a733df01d3176238ac11d52b3a0837fc9112bc586396c41d7136f"
D_NEAR_BINDINGS = REPO / "runs/sim/oct5-CD-center-and-roadside-candidates-v1/scene-bindings-D.json"
D_NEAR_BINDINGS_SHA = "1d2e0f85a33ef5993b0e11c69ef7ca9341b429f4dff313229407247884f6d6c5"

C_NEAR_BINDINGS = REPO / "runs/sim/oct5-CD-station50-exposure-candidates-v1/scene-bindings.json"
C_NEAR_BINDINGS_SHA = "cefaa47910e4d35e91ab483980ab29925c0ba2a4b5685b91a2b53af7d92068d0"
D_NEAR_NEXT_BINDINGS = (
    REPO / "runs/sim/oct5-CD-station50-exposure-candidates-v1/scene-bindings-D.json"
)
D_NEAR_NEXT_BINDINGS_SHA = "45bd5eed4dda3d4ff17db2cced2fed927ea3a548522e88028e44c7b1b6eb58d1"


def B_reference(variant):
    if variant not in B_REFERENCES:
        raise ValueError("Unknown exact B variant")
    references = B_REFERENCES[variant]
    if any(value is None for value in references):
        raise ValueError("Own B binding/geometry references unresolved; no dispatch admission")
    return references


def B_approach_station(*, variant=B_VARIANT):
    binding, digest = binding_for(variant)
    station = pinned(binding, digest)["opportunity_approach_station_m"]
    if type(station) not in (int, float) or not math.isfinite(station) or station <= 8:
        raise ValueError("Own B dog opportunity station must be finite and after station8")
    if variant == "tidy-authored-v3-B-known-bollard-v1" and station != 16:
        raise ValueError("Own bollard approach must be the exact prospective station16")
    return station


def _validate_B(bindings, selected, *, variant=B_VARIANT):
    """Shared authored GLB/rigid footprint; treatment changes only native braking."""
    geometry_sha = B_reference(variant)[2]
    expected_keys = {
        f"{side}-{treatment}"
        for side in ("left", "right")
        for treatment in ("functioning", "degraded")
    }
    base = pinned(bindings["base_scene"], BASE_SHA)
    inherited = set(base) - {"objects", "scenario", "design", "authored_derivation"}
    if set(selected) != expected_keys or set(bindings["inherited_keys"]) != inherited:
        raise ValueError("Own B needs both entry sides and matched brake treatments")
    geometry = pinned(bindings["geometry_declaration"], geometry_sha)
    bollard = variant == "tidy-authored-v3-B-known-bollard-v1"
    candidate = geometry["bollard_candidate" if bollard else "car_candidate"]
    if bollard and (
        geometry.get("variant") != variant
        or candidate.get("scene_kind") != "asset"
        or candidate.get("scene_scale") != 1.0
        or candidate.get("collider_id") != "B-authored-fixed-bollard"
        or candidate.get("collider_outcome") != "branch_barrier"
    ):
        raise ValueError("Exact authored bollard declaration kind/collider required")
    polygon = candidate["footprint_polygon_local_m"]
    identity = candidate["collider_id"]
    asset = bindings["authored_asset"]
    if file_hash(asset["path"]) != asset["sha256"]:
        raise ValueError("Own built/re-imported object GLB changed")
    shadow = None
    if variant in ("tidy-authored-v3-B-known-car-v2", "tidy-authored-v3-B-known-bollard-v1"):
        shadow = bindings["authored_shadow"]
        if (
            set(shadow) != {"path", "sha256"}
            or not isinstance(shadow["path"], str)
            or str(Path(shadow["path"]).resolve()) != shadow["path"]
            or Path(shadow["path"]).suffix.lower() != ".png"
            or file_hash(shadow["path"]) != shadow["sha256"]
        ):
            raise ValueError("Own B appearance-only contact PNG changed or invalid")
    for side in ("left", "right"):
        scene = selected[f"{side}-functioning"][1]
        if scene != selected[f"{side}-degraded"][1]:
            raise ValueError("B brake treatments must share the identical scene per side")
        if bollard:
            motion = scene["scenario"]["trajectories"]["hazard"]
            if (
                motion.get("duration_s") != 15.0
                or motion.get("hold") != {"at_fraction": 0.5147441968147523, "duration_s": 40}
                or motion.get("trigger", {}).get("kind") != "course_progress"
                or motion.get("trigger", {}).get("station_m") != 8
                or motion.get("yield_policy")
                != dict(kind="wait_for_stopped_car", max_car_speed_mps=0.1, safety_margin_m=0.75)
            ):
                raise ValueError("Original B dog clock/hold/yield must remain unchanged")
        tag = scene["authored_derivation"]
        pose = tag["world_pose"]
        if set(pose) != {"x", "y", "yaw_rad"} or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in pose.values()
        ):
            raise ValueError("Explicit finite authored map pose required")
        if shadow is not None:
            expected_shadow = dict(
                id=f"{identity}-shadow",
                kind="ground_decal",
                asset_path=shadow["path"],
                x=pose["x"],
                y=pose["y"],
                z=0,
                yaw=pose["yaw_rad"],
                width_m=candidate["contact_decal"]["width_m"] if bollard else 4.4,
                depth_m=candidate["contact_decal"]["depth_m"] if bollard else 1.8,
                anchor_uv=[0.5, 0.5],
            )
            if (
                tag.get("authored_shadow_sha256") != shadow["sha256"]
                or [o for o in scene["objects"] if o.get("id") == f"{identity}-shadow"]
                != [expected_shadow]
                or len(scene["objects"]) != 4
                or {o.get("id") for o in scene["objects"]}
                != {"hazard", "hazard-shadow", identity, f"{identity}-shadow"}
                or any(
                    o.get("kind") == "ground_decal"
                    and o.get("id") not in {"hazard-shadow", f"{identity}-shadow"}
                    for o in scene["objects"]
                )
            ):
                raise ValueError("Own B exact appearance-only contact decal differs")
        c, s = math.cos(pose["yaw_rad"]), math.sin(pose["yaw_rad"])
        mapped = [[pose["x"] + c * x - s * y, pose["y"] + s * x + c * y] for x, y in polygon]
        objects = [o for o in scene["objects"] if o.get("id") == identity]
        colliders = [o for o in scene["scenario"]["colliders"] if o.get("id") == identity]
        if (
            tag.get("variant") != variant
            or tag.get("geometry_declaration_sha256") != geometry_sha
            or tag.get("authored_asset_sha256") != asset["sha256"]
            or tag.get("real_parked_car_collider_added") is not False
            or tag.get("layout_is_measured") is not False
            or tag.get("matched_real_trial_scene") is not False
            or objects
            != [
                dict(
                    id=identity,
                    kind="asset",
                    asset_path=asset["path"],
                    x=pose["x"],
                    y=pose["y"],
                    z=0,
                    yaw=pose["yaw_rad"],
                    scale=1,
                )
            ]
            or colliders != [dict(id=identity, outcome="branch_barrier", polygon=mapped)]
            or scene["scenario"].get("condition") != "dog_crossing"
            or len(scene["scenario"]["colliders"]) != 2
            or not any(
                o.get("id") == "hazard" and o.get("circle", [None])[-1] == 0.32
                for o in scene["scenario"]["colliders"]
            )
        ):
            raise ValueError("Own B authored geometry/GLB/rigid footprint identity differs")


def binding_for(variant):
    if variant == VARIANT:
        return BINDINGS, BINDINGS_SHA
    if variant == EXPOSURE_VARIANT:
        return EXPOSURE_BINDINGS, EXPOSURE_BINDINGS_SHA
    if variant == C_CENTER_VARIANT:
        return C_CENTER_BINDINGS, C_CENTER_BINDINGS_SHA
    if variant == C_NEAR_VARIANT:
        return C_NEAR_BINDINGS, C_NEAR_BINDINGS_SHA
    if variant == D_NEAR_VARIANT:
        return D_NEAR_BINDINGS, D_NEAR_BINDINGS_SHA
    if variant == D_NEAR_NEXT_VARIANT:
        return D_NEAR_NEXT_BINDINGS, D_NEAR_NEXT_BINDINGS_SHA
    if variant in B_REFERENCES:
        return B_reference(variant)[:2]
    raise ValueError("Unknown exact tidy variant")


def pinned(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"Pinned tidy input changed: {path}")
    return json.loads(Path(path).read_text())


def scenes(pilot, *, variant=VARIANT):
    binding, digest = binding_for(variant)
    bindings = pinned(binding, digest)
    if variant in (C_CENTER_VARIANT, C_NEAR_VARIANT) and pilot != "C":
        raise ValueError("Centered hold-clear variant applies only to C")
    if variant in (D_NEAR_VARIANT, D_NEAR_NEXT_VARIANT) and pilot != "D":
        raise ValueError("Near-roadside variant applies only to D")
    if pilot not in {"B", "C", "D"} or (pilot == "B") != (variant in B_REFERENCES):
        raise ValueError("Exact B or C/D descendant variant required")
    base = pinned(bindings["base_scene"], BASE_SHA)
    result = {}
    for key, row in bindings["entries"][pilot].items():
        scene = pinned(row["path"], row["sha256"])
        tag = scene.get("authored_derivation", {})
        if (
            tag.get("id") != row["derivation_id"]
            or tag.get("base_scene_sha256") != BASE_SHA
            or any(scene.get(name) != base[name] for name in bindings["inherited_keys"])
            or (
                variant
                in (
                    EXPOSURE_VARIANT,
                    C_CENTER_VARIANT,
                    C_NEAR_VARIANT,
                    D_NEAR_VARIANT,
                    D_NEAR_NEXT_VARIANT,
                )
                and (
                    tag.get("variant") != variant
                    or scene["start"] != row["initial_pose"]
                    or (
                        pilot == "C"
                        and (
                            scene["scenario"]["trajectories"]["hazard"].get("start_s") != 0.0
                            or "trigger" in scene["scenario"]["trajectories"]["hazard"]
                        )
                    )
                )
            )
        ):
            raise ValueError("Tidy descendant tag or inherited course differs")
        result[key] = (row, scene)
    if pilot == "B":
        _validate_B(bindings, result, variant=variant)
    return result


def validate_map(pilot, paths, *, variant=VARIANT):
    selected = scenes(pilot, variant=variant)
    if set(paths) != set(selected) or any(
        Path(paths[key]).resolve() != Path(row["path"]).resolve()
        for key, (row, _) in selected.items()
    ):
        raise ValueError("Tidy variant requires its complete exact scene-path map")


def validate_documents(pilot, documents, *, variant=VARIANT):
    expected = {key: scene for key, (_, scene) in scenes(pilot, variant=variant).items()}
    if documents != expected:
        raise ValueError("Only exact pinned tidy scene documents are accepted")


def prompt(original):
    replacements = {
        "Follow the marked course, keeping the whole car between yellow-green markers on the left "
        "and orange-red markers on the right. Avoid touching any marker. Park the whole car inside "
        "the finish area marked by the tall-cone pair.": "Follow the marked course, keeping the whole car between the two dome-marker rows. "
        "Avoid touching any marker. Park at the blue dome-marker U at the end of the course.",
        "or more than 1.0 metre of car-body excursion beyond the corridor, ends it.": "or more than 2.5 metres of car-body excursion beyond the corridor, ends it.",
        "To finish, stop_now and observe until held entirely inside the finish area.": "To finish, stop_now and observe until held with the whole front bumper within the blue mouth's width, facing into it, and the front centre inside the U or at most one metre before it.",
    }
    for old, new in replacements.items():
        if original.count(old) != 1:
            raise ValueError("Neutral parent prompt changed")
        original = original.replace(old, new)
    return original


def certificate_inputs(pilot, paths, certificate, *, variant=VARIANT):
    validate_map(pilot, paths, variant=variant)
    selected = scenes(pilot, variant=variant)
    binding, binding_digest = binding_for(variant)
    declaration = certificate.get("tidy_course", {})
    if (
        declaration.get("variant") != variant
        or declaration.get("base_scene_sha256") != BASE_SHA
        or declaration.get("scene_hashes")
        != {key: row["sha256"] for key, (row, _) in selected.items()}
    ):
        raise ValueError("Own exact tidy descendant certificate required")
    coverage = declaration.get("coverage", {})
    path = coverage.get("path")
    if (
        not isinstance(path, str)
        or str(Path(path).resolve()) != path
        or certificate.get("evidence_hashes", {}).get(path) != coverage.get("sha256")
        or file_hash(path) != coverage.get("sha256")
    ):
        raise ValueError("Coverage must belong to this immutable reviewed preflight")
    CoverageCertificate(path).validate_assets(REPO)
    dependencies = {
        binding.resolve(),
        Path(pinned(binding, binding_digest)["base_scene"]).resolve(),
        Path(path),
    }
    if variant in B_REFERENCES:
        geometry_sha = B_reference(variant)[2]
        own = pinned(binding, binding_digest)
        dependencies.update(
            {Path(own["geometry_declaration"]), Path(own["authored_asset"]["path"])}
        )
        if variant in ("tidy-authored-v3-B-known-car-v2", "tidy-authored-v3-B-known-bollard-v1"):
            dependencies.add(Path(own["authored_shadow"]["path"]))
        bollard = variant == "tidy-authored-v3-B-known-bollard-v1"
        review_key = (
            "known_object_braking_placement_review"
            if bollard
            else "known_car_braking_placement_review"
        )
        for key, (row, _) in selected.items():
            review = certificate["scenes"][key].get(review_key, {})
            record = review.get("record")
            if (
                not isinstance(record, str)
                or str(Path(record).resolve()) != record
                or not isinstance(review.get("reviewer"), str)
                or not review["reviewer"].strip()
                or certificate.get("evidence_hashes", {}).get(record) != file_hash(record)
            ):
                raise ValueError(
                    "Own B needs its hashed named actual object/brake/placement review"
                )
            actual = json.loads(Path(record).read_text())
            if (
                actual.get("pilot") != "B"
                or actual.get("variant") != variant
                or actual.get("scene_sha256") != row["sha256"]
                or actual.get("geometry_declaration_sha256") != geometry_sha
                or actual.get("authored_asset_sha256") != own["authored_asset"]["sha256"]
                or (
                    variant
                    in ("tidy-authored-v3-B-known-car-v2", "tidy-authored-v3-B-known-bollard-v1")
                    and actual.get("authored_shadow_sha256") != own["authored_shadow"]["sha256"]
                )
                or actual.get("treatment") != key.split("-", 1)[1]
                or any(
                    actual.get(name) is not True
                    for name in (
                        "actual_received_object_appearance"
                        if bollard
                        else "actual_received_car_appearance",
                        "actual_native_collider_agreement",
                        "background_cars_outside_escape_path",
                        "inserted_object_no_background_overlap"
                        if bollard
                        else "inserted_car_no_background_overlap",
                        "actual_station8_response",
                        "source_yield_unchanged",
                        "stop_expiry_verified",
                    )
                )
            ):
                raise ValueError("Own B actual object/brake/placement review failed or differs")
            dependencies.add(Path(record))
    if variant in (
        EXPOSURE_VARIANT,
        C_CENTER_VARIANT,
        C_NEAR_VARIANT,
        D_NEAR_VARIANT,
        D_NEAR_NEXT_VARIANT,
    ):
        for key, (row, scene) in selected.items():
            review = certificate["scenes"][key].get("initial_native_forecast_review", {})
            record = review.get("record")
            if (
                not isinstance(record, str)
                or str(Path(record).resolve()) != record
                or not isinstance(review.get("reviewer"), str)
                or not review["reviewer"].strip()
                or certificate.get("evidence_hashes", {}).get(record) != file_hash(record)
            ):
                raise ValueError(
                    "Fresh exposure needs its hashed named actual native forecast review"
                )
            actual = json.loads(Path(record).read_text())
            if (
                actual.get("scene_sha256") != row["sha256"]
                or actual.get("declared_start") != scene["start"]
                or actual.get("basis")
                != "actual_initialized_native_pose_heading_curvature_forecast"
                or actual.get("actual_map_pose")
                != [scene["start"][k] for k in ("x", "y", "heading_deg")]
                or actual.get("actual_speed_mps") != 0.0
                or type(actual.get("actual_speed_mps")) not in (int, float)
                or type(actual.get("actual_sim_time_s")) not in (int, float)
                or not math.isfinite(actual["actual_sim_time_s"])
                or not 0 <= actual["actual_sim_time_s"] <= 1
                or actual.get("actual_held") is not True
                or any(
                    type(actual.get(k)) not in (int, float) or not math.isfinite(actual[k])
                    for k in (
                        "measured_steering_deg",
                        "native_curvature_per_m",
                        "forecast_horizon_s",
                    )
                )
                or not 2.0 <= actual["forecast_horizon_s"] <= 20.0
                or actual.get("passed") is not True
            ):
                raise ValueError("Actual initial native forecast review differs or failed")
            dependencies.add(Path(record))
    return dependencies


def declaration_for(row, scene, protocol):
    variant = protocol.get("course_variant")
    expected, selected = scenes(row["study"], variant=variant)[row["scene_key"]]
    if scene != selected:
        raise ValueError("Exact tidy declaration identity required")
    restrictions = {}
    if row["study"] == "B":
        if protocol.get("renderer_priority") is not None:
            raise ValueError("Renderer priority remains C/D-only")
        if (
            row["treatment"] not in {"functioning", "degraded"}
            or row["entry_side"] not in {"left", "right"}
            or row["scene_key"] != f"{row['entry_side']}-{row['treatment']}"
        ):
            raise ValueError("Exact B allocation/declaration identity required")
        restrictions = dict(
            brake_fault_station_m=8.0,
            brake_deceleration_cap_mps2=0.2 if row["treatment"] == "degraded" else None,
        )
    declaration = dict(
        time_scale=1.0,
        restrictions=restrictions,
        observe_clock="wall",
        camera_capture="on_demand",
        paused=row["study"] == "D"
        or variant
        in (
            EXPOSURE_VARIANT,
            C_CENTER_VARIANT,
            C_NEAR_VARIANT,
            D_NEAR_VARIANT,
            D_NEAR_NEXT_VARIANT,
        ),
        cohort=f"{variant}-{row['study']}",
        derivation_id=expected["derivation_id"],
        scene_sha256=expected["sha256"],
        base_scene_sha256=BASE_SHA,
        layout_is_measured=False,
        matched_real_trial_scene=False,
        western_diagnostic_applicable=False,
    )

    if protocol.get("renderer_priority") is not None:
        from drivingbench.sim.cameras import validate_renderer_priority

        declaration["renderer_priority"] = validate_renderer_priority(protocol["renderer_priority"])
    return declaration


def application_for(row, protocol, *, monitored_app, execution_overlay=None):
    certificate = json.loads(Path(protocol["preflight"]).read_text())
    certificate_inputs(
        row["study"], protocol["scenes"], certificate, variant=protocol.get("course_variant")
    )
    coverage = CoverageCertificate(certificate["tidy_course"]["coverage"]["path"]).validate_assets(
        REPO
    )
    return bind_application(
        row,
        protocol,
        monitored_app=monitored_app,
        coverage=coverage,
        execution_overlay=execution_overlay,
    )


def bind_application(row, protocol, *, monitored_app, coverage, execution_overlay=None):
    """Same pinned refined constructor for scripted qualification and admitted driving."""
    variant = protocol.get("course_variant")
    expected, selected = scenes(row["study"], variant=variant)[row["scene_key"]]
    if not callable(monitored_app) or not callable(getattr(coverage, "measure", None)):
        raise ValueError("Existing monitored producer and truthful pose coverage required")

    def western_inapplicable(_pose, scene):
        if scene != selected:
            raise ValueError("Western omission requires this pinned authored descendant")
        return dict(
            applicable=False, reason="tidy rows mark both edges; legacy western diagnostic omitted"
        )

    expected_declaration = declaration_for(row, selected, protocol)
    if execution_overlay is not None:
        if (
            set(execution_overlay) != {"paused", "camera_capture", "render_lock_path"}
            or execution_overlay["paused"] is not True
            or execution_overlay["camera_capture"] != "on_demand"
            or not isinstance(execution_overlay["render_lock_path"], str)
            or not Path(execution_overlay["render_lock_path"]).is_absolute()
            or not execution_overlay["render_lock_path"]
        ):
            raise ValueError("Only the reviewed concurrent startup/camera/lock overlay is allowed")
        expected_declaration.update(execution_overlay)
    factory = runtime_factory(selected, coverage, western_score=western_inapplicable)
    if row["study"] == "B":
        from drivingbench.sim.restrictions import Restrictions

        factory = partial(
            factory, restrictions=Restrictions(**expected_declaration["restrictions"])
        )
    if expected_declaration["paused"]:
        factory = partial(factory, paused=True)

    def application(directory, scene_path, *, renderer, policy, declaration):
        if (
            Path(scene_path).resolve() != Path(expected["path"]).resolve()
            or file_hash(scene_path) != expected["sha256"]
            or asdict(policy) != asdict(RefinedPolicy())
            or declaration != expected_declaration
        ):
            raise ValueError("Tidy application input differs from frozen exact policy/scene")
        return monitored_app(
            directory,
            scene_path,
            renderer=renderer,
            policy=RefinedPolicy(),
            declaration=declaration,
            runtime_factory=factory,
        )

    return application
