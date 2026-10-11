"""Training-only COLMAP geometry, metric registration, and frozen-map test localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from drivingbench.sim.visual_pose import camera_to_world


def similarity(source, target):
    """Least-squares proper Sim(3), including planar trajectories."""
    source, target = np.asarray(source), np.asarray(target)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("expected matching N×3 positions")
    a, b = source - source.mean(0), target - target.mean(0)
    if len(a) < 3 or np.linalg.matrix_rank(a) < 2:
        raise ValueError("registration needs at least three non-collinear cameras")
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    scale = float(d @ signs / np.mean(np.sum(a * a, axis=1)))
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return scale, rotation, translation


def opengl_pose(cam_from_world):
    """COLMAP world-to-OpenCV camera to world-from-OpenGL camera."""
    pose = np.eye(4)
    pose[:3] = cam_from_world
    return np.linalg.inv(pose) @ np.diag([1, -1, -1, 1])


def refined_frame(row, opengl):
    """Keep seed evidence explicit; export a consistent refined OpenCV camera pose."""
    pose = np.asarray(opengl) @ np.diag([1, -1, -1, 1])
    road_from_camera = camera_to_world(0, 0, 0, row["rpy"])[:3, :3]
    world_from_road = pose[:3, :3] @ road_from_camera.T
    return dict(
        row,
        seed_pose={key: row[key] for key in ("x", "y", "heading", "transform_matrix")},
        x=float(pose[0, 3]),
        y=float(pose[1, 3]),
        heading=float(np.arctan2(world_from_road[1, 0], world_from_road[0, 0])),
        transform_matrix=pose.tolist(),
    )


def _brush_frame(row):
    return dict(
        row,
        file_path=f"images/{row['name']}",
        transform_matrix=(np.asarray(row["transform_matrix"]) @ np.diag([1, -1, -1, 1])).tolist(),
    )


def keyframes(rows, spacing, angle_deg):
    """Greedy position-and-heading coverage; only training views may define geometry."""
    if any(r["split"] != "train" for r in rows):
        raise ValueError("geometry keyframes must come from the training split")
    if spacing is None:
        return rows
    if spacing <= 0 or not 0 < angle_deg <= 180:
        raise ValueError("keyframe spacing and angle must be positive; angle cannot exceed 180")
    chosen, poses = [], []
    for row in rows:
        pose = np.array([row["x"], row["y"], row["heading"]])
        if poses:
            delta = np.array(poses) - pose
            near = np.linalg.norm(delta[:, :2], axis=1) < spacing
            aligned = np.abs(np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))) < np.radians(
                angle_deg
            )
            if np.any(near & aligned):
                continue
        chosen.append(row)
        poses.append(pose)
    return chosen


def candidate_pairs(rows, references=None, neighbors=12):
    """Temporal continuity plus same-facing neighboring passes from seed poses."""
    same = references is None
    references = rows if same else references
    if not rows or not references:
        return []
    xy = np.array([[r["x"], r["y"]] for r in references])
    headings = np.array([r["heading"] for r in references])
    pairs = set()
    for i, row in enumerate(rows):
        distance = np.linalg.norm(xy - [row["x"], row["y"]], axis=1)
        angle = np.abs(
            np.arctan2(np.sin(headings - row["heading"]), np.cos(headings - row["heading"]))
        )
        indices = set(np.argsort(distance + 10 * angle)[:neighbors])
        if same:
            indices.update(range(max(0, i - 5), min(len(rows), i + 6)))
        for j in indices:
            other = references[j]
            if row["name"] != other["name"]:
                pairs.add(tuple(sorted((row["name"], other["name"]))))
    return sorted(pairs)


def _match(pycolmap, database, pairs, path, threads):
    path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    pycolmap.match_image_pairs(
        database,
        matching_options=pycolmap.FeatureMatchingOptions(num_threads=threads, use_gpu=False),
        pairing_options=pycolmap.ImportedPairingOptions(match_list_path=str(path)),
        device=pycolmap.Device.cpu,
    )


def _extract(pycolmap, database, images, names, camera, threads):
    pycolmap.extract_features(
        database,
        images,
        image_names=names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(
            camera_model="PINHOLE",
            camera_params=",".join(str(camera[k]) for k in ("fl_x", "fl_y", "cx", "cy")),
        ),
        extraction_options=pycolmap.FeatureExtractionOptions(
            num_threads=threads,
            max_image_size=1000,
            use_gpu=False,
            sift=pycolmap.SiftExtractionOptions(max_num_features=2048),
        ),
        device=pycolmap.Device.cpu,
    )


def _localize(pycolmap, database, model, row, train_by_name, candidates):
    with pycolmap.Database.open(database) as db:
        image = db.read_image_with_name(row["name"])
        keypoints = db.read_keypoints(image.image_id)[:, :2]
        observations = {}
        for a, b in candidates:
            other_name = b if a == row["name"] else a
            if row["name"] not in (a, b) or other_name not in train_by_name:
                continue
            other = train_by_name[other_name]
            if not db.exists_matches(image.image_id, other.image_id):
                continue
            matches = db.read_two_view_geometry(image.image_id, other.image_id).inlier_matches
            for query_index, reference_index in matches:
                point = other.points2D[int(reference_index)]
                if point.has_point3D():
                    observations.setdefault(int(query_index), set()).add(point.point3D_id)
        # Ambiguous matches to multiple landmarks cannot provide a trustworthy PnP constraint.
        observations = {k: next(iter(v)) for k, v in observations.items() if len(v) == 1}
        if len(observations) < 20:
            return None, {"correspondences": len(observations), "reason": "too few matches"}
        indices, point_ids = zip(*observations.items(), strict=True)
        camera = db.read_camera(image.camera_id)
    estimation = pycolmap.AbsolutePoseEstimationOptions()
    estimation.ransac.max_error = 4.0
    estimation.ransac.random_seed = 0
    result = pycolmap.estimate_and_refine_absolute_pose(
        keypoints[list(indices)],
        np.array([model.points3D[p].xyz for p in point_ids]),
        camera,
        estimation_options=estimation,
    )
    stats = {"correspondences": len(observations), "inliers": 0}
    if result is None or result["num_inliers"] < 20:
        return None, stats
    stats["inliers"] = result["num_inliers"]
    return opengl_pose(result["cam_from_world"].matrix()), stats


def reuse_features(source, destination, inputs):
    """Reuse descriptors only when their manifest and calibration provenance match."""
    if not source.is_file():
        raise ValueError("reused database does not exist")
    record = source.parent / "inputs.json"
    expected = {key: value for key, value in inputs.items() if key != "keyframes"}
    if not record.exists():
        raise ValueError("reused database has no inputs.json provenance")
    actual = json.loads(record.read_text())
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("reused database manifest or camera hashes do not match")
    with (
        sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db_source,
        sqlite3.connect(destination) as db_destination,
    ):
        db_source.backup(db_destination)


def run(
    manifest,
    camera_path,
    out,
    threads=8,
    keyframe_spacing=None,
    keyframe_angle=15,
    reuse_database=None,
):
    import pycolmap

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("manifest image names must be unique across splits")
    all_train = [r for r in rows if r["split"] == "train"]
    train = keyframes(all_train, keyframe_spacing, keyframe_angle)
    test = [r for r in rows if r["split"] == "test"]
    metadata = json.loads(camera_path.read_text())
    camera = metadata.get("camera", metadata)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (manifest, camera_path)
    }
    if keyframe_spacing is not None:
        inputs["keyframes"] = {"spacing_m": keyframe_spacing, "angle_deg": keyframe_angle}
    input_record = out / "inputs.json"
    if input_record.exists() and json.loads(input_record.read_text()) != inputs:
        raise ValueError("SfM output belongs to different inputs; choose a fresh output directory")
    input_record.write_text(json.dumps(inputs, indent=2) + "\n")
    images = out / "images"
    images.mkdir(exist_ok=True)
    for row in all_train + test:
        link = images / row["name"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(Path(row["file_path"]).resolve())
    database = out / "database.db"
    if reuse_database is not None and not database.exists():
        reuse_features(reuse_database, database, inputs)
    sparse = out / "sparse"
    sparse.mkdir(exist_ok=True)
    if not (sparse / "cameras.bin").exists():
        _extract(pycolmap, database, images, [r["name"] for r in train], camera, threads)
        _match(pycolmap, database, candidate_pairs(train), out / "train-pairs.txt", threads)
        options = pycolmap.IncrementalPipelineOptions(
            num_threads=threads,
            random_seed=0,
            image_names=[r["name"] for r in train],
            ba_refine_focal_length=False,
            ba_refine_extra_params=False,
            ba_refine_sensor_from_rig=False,
        )
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
        models = pycolmap.incremental_mapping(database, images, out / "models", options=options)
        if not models:
            raise RuntimeError("COLMAP could not reconstruct training frames")
        (out / "components.json").write_text(
            json.dumps(
                [
                    {"id": key, "registered": m.num_reg_images(), "points": m.num_points3D()}
                    for key, m in models.items()
                ],
                indent=2,
            )
            + "\n"
        )
        model = max(models.values(), key=lambda m: m.num_reg_images())
        by_name = {r["name"]: r for r in train}
        registered = [i for i in model.images.values() if i.has_pose]
        source = np.array([i.projection_center() for i in registered])
        target = np.array(
            [np.asarray(by_name[i.name]["transform_matrix"])[:3, 3] for i in registered]
        )
        scale, rotation, translation = similarity(source, target)
        errors = np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)
        model.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation), translation))
        model.write(sparse)
        model.export_PLY(out / "init.ply")
        (out / "registration.json").write_text(
            json.dumps(
                {
                    "source": "COLMAP training cameras",
                    "target": "manifest metric ENU camera centers",
                    "scale": scale,
                    "rotation": rotation.tolist(),
                    "translation": translation.tolist(),
                    "train_camera_count": len(registered),
                    "rms_m": float(np.sqrt(np.mean(errors**2))),
                    "median_m": float(np.median(errors)),
                    "max_m": float(errors.max()),
                },
                indent=2,
            )
            + "\n"
        )
    model = pycolmap.Reconstruction(sparse)
    registered = {i.name: i for i in model.images.values() if i.has_pose}
    train_frames = [
        refined_frame(r, opengl_pose(registered[r["name"]].cam_from_world().matrix()))
        for r in train
        if r["name"] in registered
    ]
    registered_rows = [r for r in train if r["name"] in registered]
    queries = [r for r in all_train if r["name"] not in registered] + test
    candidates = candidate_pairs(queries, registered_rows, neighbors=20)
    if queries:
        _extract(pycolmap, database, images, [r["name"] for r in queries], camera, threads)
        _match(pycolmap, database, candidates, out / "localization-pairs.txt", threads)
    test_frames, localization = [], []
    for row in queries:
        pose, stats = _localize(pycolmap, database, model, row, registered, candidates)
        stats.update(name=row["name"], split=row["split"], localized=pose is not None)
        if pose is not None:
            stats["seed_position_error_m"] = float(
                np.linalg.norm(pose[:3, 3] - np.asarray(row["transform_matrix"])[:3, 3])
            )
            target = train_frames if row["split"] == "train" else test_frames
            target.append(refined_frame(row, pose))
        localization.append(stats)
    (out / "transforms_train.json").write_text(
        json.dumps(dict(camera, frames=[_brush_frame(r) for r in train_frames]), indent=2)
    )
    (out / "transforms_val.json").write_text(
        json.dumps(dict(camera, frames=[_brush_frame(r) for r in test_frames]), indent=2)
    )
    refined = train_frames + test_frames
    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in refined))
    heights = [r["transform_matrix"][2][3] for r in refined]
    rotation_errors = []
    for row in refined:
        seed_rotation = np.asarray(row["seed_pose"]["transform_matrix"])[:3, :3]
        rotation = np.asarray(row["transform_matrix"])[:3, :3]
        rotation_errors.append(
            float(
                np.degrees(
                    np.arccos(np.clip((np.trace(seed_rotation.T @ rotation) - 1) / 2, -1, 1))
                )
            )
        )
    report = {
        "train_input": len(all_train),
        "geometry_train_input": len(train),
        "train_registered": len(registered),
        "train_localized": len(train_frames) - len(registered),
        "train_output": len(train_frames),
        "test_input": len(test),
        "test_localized": len(test_frames),
        "points": model.num_points3D(),
        "reprojection_error_px": model.compute_mean_reprojection_error(),
        "camera_height_range_m": [float(min(heights)), float(max(heights))],
        "seed_rotation_error_median_deg": float(np.median(rotation_errors)),
        "seed_rotation_error_p90_deg": float(np.percentile(rotation_errors, 90)),
        "localization": localization,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("camera", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--keyframe-spacing", type=float)
    parser.add_argument("--keyframe-angle", type=float, default=15)
    parser.add_argument("--reuse-database", type=Path)
    args = parser.parse_args()
    report = run(
        args.manifest,
        args.camera,
        args.out,
        args.threads,
        args.keyframe_spacing,
        args.keyframe_angle,
        args.reuse_database,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "localization"}, indent=2))


if __name__ == "__main__":
    main()
