"""Pinned Brush training and arbitrary-camera rendering on the local Metal GPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from drivingbench.sim.logs import REPO

TOOLS = REPO / "runs/sim/visual/tools"
SOURCE = TOOLS / "brush-source/lpips-convert-0.3.0"
BRUSH = TOOLS / "brush-app-aarch64-apple-darwin/brush_app"
RENDER = SOURCE / "target/release/drivingbench-brush-render"
CARGO_MANIFEST = """[package]
name = "drivingbench-brush-render"
version = "0.1.0"
edition.workspace = true
[dependencies]
brush-render.path = "../../crates/brush-render"
brush-serde.path = "../../crates/brush-serde"
anyhow.workspace = true
burn.workspace = true
glam.workspace = true
image.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio = { workspace = true, features = ["rt-multi-thread", "macros", "fs"] }
"""


def build_render_helper():
    """Compile the small helper against the downloaded, pinned Brush 0.3.0 source."""
    crate = SOURCE / "examples/drivingbench-render"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    (crate / "Cargo.toml").write_text(CARGO_MANIFEST)
    shutil.copyfile(Path(__file__).with_name("brush_render.rs"), crate / "src/main.rs")
    env = dict(os.environ, CARGO_HOME=str(TOOLS / "cargo"), RUSTUP_HOME=str(TOOLS / "rustup"))
    subprocess.run(
        [str(TOOLS / "cargo/bin/cargo"), "build", "--release", "--manifest-path",
         str(SOURCE / "Cargo.toml"), "-p", "drivingbench-brush-render"],
        env=env, check=True,
    )  # fmt: skip
    return RENDER


def prepare_dataset(source, out):
    """Stage only explicit splits and their images so COLMAP cannot override them."""
    source, out = Path(source).resolve(), Path(out).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        data = json.loads((source / f"transforms_{split}.json").read_text())
        for frame in data["frames"]:
            image = (source / frame["file_path"]).resolve()
            target = out / "images" / image.name
            if target.exists():
                if not target.samefile(image):
                    raise ValueError(f"Image filename collision: {target.name}")
            else:
                os.link(image, target)
            frame["file_path"] = str(target.relative_to(out))
        (out / f"transforms_{split}.json").write_text(json.dumps(data) + "\n")
    shutil.copyfile(source / "init.ply", out / "init.ply")
    train_command(out, out / "output")
    return out


def train_command(
    dataset,
    out,
    *,
    steps=2000,
    resolution=672,
    max_splats=500_000,
    start_iter=None,
    growth_stop_iter=None,
    growth_grad_threshold=None,
    refine_every=None,
):
    """Reject intersecting train/validation images before invoking the trainer."""
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    splits = [
        json.loads((dataset / f"transforms_{split}.json").read_text()) for split in ("train", "val")
    ]
    names = [{(dataset / f["file_path"]).resolve() for f in s["frames"]} for s in splits]
    if not all(names) or names[0] & names[1]:
        raise ValueError("Training and validation must be nonempty and disjoint")
    if (
        (dataset / "transforms.json").exists()
        or any(dataset.rglob("cameras.bin"))
        or any(dataset.rglob("cameras.txt"))
    ):
        raise ValueError("Brush would prefer another dataset over the explicit split")
    if min(steps, resolution, max_splats) <= 0:
        raise ValueError("Training limits must be positive")
    extra = {
        "start-iter": start_iter,
        "growth-stop-iter": growth_stop_iter,
        "growth-grad-threshold": growth_grad_threshold,
        "refine-every": refine_every,
    }
    for key, value in extra.items():
        if value is not None:
            positive = key in ("growth-grad-threshold", "refine-every")
            if not np.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"Invalid {key}: {value}")
    if start_iter is not None and start_iter >= steps:
        raise ValueError("start_iter must precede total steps")
    command = [
        str(BRUSH), str(dataset), "--total-steps", str(steps),
        "--max-resolution", str(resolution), "--max-splats", str(max_splats),
        "--eval-every", str(steps), "--eval-save-to-disk", "--export-every", str(steps),
        "--export-path", str(out), "--seed", "42",
    ]  # fmt: skip
    for key, value in extra.items():
        if value is not None:
            command.extend([f"--{key}", str(value)])
    return command


def train(dataset, out, **options):
    command = train_command(dataset, out, **options)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.perf_counter()
    with (out / "train.log").open("w") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=dict(
                os.environ, RUST_LOG="warn,brush_cli=info,brush_process=info,cubecl_wgpu=info"
            ),
            cwd=TOOLS,
        )
    (out / "runtime.json").write_text(
        json.dumps({"train_seconds": time.perf_counter() - started}) + "\n"
    )


def camera_manifest(data):
    """Normalize Nerfstudio/OpenGL camera-to-world poses for the native renderer."""
    intrinsics = {key: data[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
    if not np.isfinite(list(intrinsics.values())).all():
        raise ValueError("Camera intrinsics must be finite")
    if any(intrinsics[k] <= 0 for k in ("w", "h", "fl_x", "fl_y")):
        raise ValueError("Image dimensions and focal lengths must be positive")
    if any(not isinstance(intrinsics[k], int) for k in ("w", "h")):
        raise ValueError("Image dimensions must be integers")
    frames, names = [], set()
    for frame in data["frames"]:
        name = frame.get("name", Path(frame.get("file_path", "")).stem)
        if not name or name in (".", "..") or Path(name).name != name:
            raise ValueError("Render names must be unique, nonempty basenames")
        name = Path(name).stem if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png") else name
        if name in names:
            raise ValueError("Render names must be unique")
        pose = np.asarray(frame["transform_matrix"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Camera transform must be a finite 4x4 matrix")
        rotation = pose[:3, :3]
        if (
            not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
        ):
            raise ValueError("Camera transform must be rigid and right-handed")
        frames.append({"name": name, "transform_matrix": pose.tolist()})
        names.add(name)
    if not frames:
        raise ValueError("At least one camera is required")
    return dict(intrinsics, frames=frames)


def serve_command(checkpoint):
    """Start the persistent native worker; supervision belongs to the caller."""
    return [str(RENDER), str(Path(checkpoint).resolve()), "--serve"]


def stream_request(request_id, transform_matrix, camera, output):
    """Build one raw-array render request using an OpenCV camera-to-world pose."""
    manifest = camera_manifest(
        dict(camera, frames=[{"name": "request", "transform_matrix": transform_matrix}])
    )
    return {
        "id": request_id,
        "transform_matrix": manifest["frames"][0]["transform_matrix"],
        "camera": {key: value for key, value in manifest.items() if key != "frames"},
        "output": str(Path(output).resolve()),
    }


def render_frames(checkpoint, cameras, out, *, depth=False):
    """Render a fixed checkpoint to JPEG92 and premultiplied RGB/alpha PNGs."""
    data = json.loads(Path(cameras).read_text()) if not isinstance(cameras, dict) else cameras
    manifest = camera_manifest(data)
    manifest["depth"] = depth
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "cameras.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    started = time.perf_counter()
    with (out / "render.log").open("w") as log:
        subprocess.run(
            [str(RENDER), str(Path(checkpoint).resolve()), str(manifest_path), str(out)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            cwd=TOOLS,
        )
    (out / "runtime.json").write_text(
        json.dumps(
            {"render_seconds": time.perf_counter() - started, "frames": len(manifest["frames"])}
        )
        + "\n"
    )
    return [out / f"{frame['name']}.jpg" for frame in manifest["frames"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("build-renderer")
    p = sub.add_parser("prepare")
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p = sub.add_parser("train")
    p.add_argument("dataset", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--resolution", type=int, default=672)
    p.add_argument("--max-splats", type=int, default=500_000)
    p.add_argument("--start-iter", type=int, help="Warm-start iteration; optimizer state resets")
    p.add_argument("--growth-stop-iter", type=int)
    p.add_argument("--growth-grad-threshold", type=float)
    p.add_argument("--refine-every", type=int)
    p = sub.add_parser("render")
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--depth", action="store_true", help="Export expected camera-z depth and float alpha"
    )
    p.add_argument("cameras", type=Path)
    p.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.action == "build-renderer":
        build_render_helper()
    elif args.action == "prepare":
        prepare_dataset(args.source, args.out)
    elif args.action == "train":
        train(
            args.dataset,
            args.out,
            steps=args.steps,
            resolution=args.resolution,
            max_splats=args.max_splats,
            start_iter=args.start_iter,
            growth_stop_iter=args.growth_stop_iter,
            growth_grad_threshold=args.growth_grad_threshold,
            refine_every=args.refine_every,
        )
    else:
        render_frames(args.checkpoint, args.cameras, args.out, depth=args.depth)


if __name__ == "__main__":
    main()
