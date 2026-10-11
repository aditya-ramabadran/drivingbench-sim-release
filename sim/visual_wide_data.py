"""Stage masked perspective tiles without changing the frozen paired-camera split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from drivingbench.sim.visual_wide import WideCalibration, perspective_tiles, read_rows, tile_map


def stage_tiles(manifest, calibration_path, mask_path, init_cloud, output, *, workers=6):
    import cv2

    output = Path(output)
    if output.exists():
        raise ValueError("tile output must be a new directory")
    rows = read_rows(manifest)
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("source names must be nonempty and unique")
    if any(r["split"] not in ("train", "test") for r in rows):
        raise ValueError("only explicit train/test paired frames may be staged")
    calibration = WideCalibration.load(calibration_path)
    tiles = perspective_tiles()
    maps = [tile_map(calibration, tile) for tile in tiles]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (calibration.intrinsics["h"], calibration.intrinsics["w"]):
        raise ValueError("loss mask must match native wide dimensions")
    masks = [
        cv2.erode(
            cv2.remap(mask, m[..., 0], m[..., 1], cv2.INTER_NEAREST), np.ones((7, 7), np.uint8)
        )
        for m in maps
    ]
    (output / "images").mkdir(parents=True)
    (output / "masks").mkdir()
    cv2.setNumThreads(1)

    def convert(row):
        image = cv2.imread(row["file_path"])
        if image is None or image.shape[:2] != mask.shape:
            raise ValueError(f"invalid source image {row['name']}")
        wide = calibration.camera_pose(row["transform_matrix"])
        result = []
        for tile, mapping, loss_mask in zip(tiles, maps, masks, strict=True):
            name = Path(row["name"]).stem + "-" + tile["name"]
            rectified = cv2.remap(image, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR)
            if not cv2.imwrite(
                str(output / "images" / f"{name}.jpg"), rectified, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise OSError("tile JPEG write failed")
            if not cv2.imwrite(str(output / "masks" / f"{name}.png"), loss_mask):
                raise OSError("tile mask write failed")
            pose = wide.copy()
            pose[:3, :3] = wide[:3, :3] @ np.asarray(tile["rotation_wide_from_tile"])
            pose[:3, 1:3] *= -1
            result.append(
                dict(
                    file_path=f"images/{name}.jpg",
                    transform_matrix=pose.tolist(),
                    source_name=row["name"],
                    split=row["split"],
                )
            )
        return result

    with ThreadPoolExecutor(workers) as pool:
        frames = [frame for group in pool.map(convert, rows) for frame in group]
    camera = {key: tiles[0][key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
    for label, split in (("train", "train"), ("val", "test")):
        (output / f"transforms_{label}.json").write_text(
            json.dumps(dict(camera, frames=[f for f in frames if f["split"] == split])) + "\n"
        )
    shutil.copyfile(init_cloud, output / "init.ply")
    hashes = {
        str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (manifest, calibration_path, mask_path, init_cloud)
    }
    (output / "recipe.json").write_text(
        json.dumps(
            dict(
                inputs_sha256=hashes,
                tiles=tiles,
                source_counts={
                    split: sum(r["split"] == split for r in rows) for split in ("train", "test")
                },
                tile_counts={
                    split: sum(f["split"] == split for f in frames) for split in ("train", "test")
                },
                mask_erosion_pixels=7,
                lpips_training_loss_weight=0,
            ),
            indent=2,
        )
        + "\n"
    )
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "calibration", "mask", "init_cloud", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    stage_tiles(
        args.manifest,
        args.calibration,
        args.mask,
        args.init_cloud,
        args.output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
