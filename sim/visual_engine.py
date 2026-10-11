"""A textured Blender lot with Gaussian and layered-image hybrid composition.

    python -m drivingbench.sim.visual_engine --manifest frames.jsonl --queries test.jsonl

Coordinates are ENU metres; transform_matrix maps OpenCV camera coordinates to ENU.
The similar-looking lot is procedural, not surveyed geometry. The layered baseline assumes
flat ground and infinitely distant backgrounds: vertical objects can smear and distant
trees/buildings have no translation parallax. Gaussian composition uses rendered opacity.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

WIDTH, HEIGHT, FOCAL = 1344, 760, 1141.5


def ground_projection(target, source, width=WIDTH, height=HEIGHT, focal=FOCAL):
    """Source pixel coordinates and visibility for target rays hitting z=0."""
    target, source = np.asarray(target, dtype=float), np.asarray(source, dtype=float)
    y, x = np.mgrid[:height, :width]
    rays = np.stack(((x - width / 2) / focal, (y - height / 2) / focal, np.ones_like(x)))
    direction = np.einsum("ij,jhw->ihw", target[:3, :3], rays)
    distance = np.divide(
        -target[2, 3],
        direction[2],
        out=np.zeros_like(x, dtype=float),
        where=np.abs(direction[2]) > 1e-8,
    )
    points = target[:3, 3, None, None] + direction * distance
    local = np.einsum("ij,jhw->ihw", source[:3, :3].T, points - source[:3, 3, None, None])
    z = np.where(np.abs(local[2]) > 1e-8, local[2], 1e-8)
    u, v = focal * local[0] / z + width / 2, focal * local[1] / z + height / 2
    visible = (
        (distance > 0)
        & (distance < 60)
        & (local[2] > 0)
        & (u >= 0)
        & (u < width - 1)
        & (v >= 0)
        & (v < height * 0.9)
    )
    return u, v, visible


def nearest_training(query, training):
    """Select a source using pose alone, without reading held-out pixels."""
    q = np.asarray(query["transform_matrix"])

    def cost(row):
        p = np.asarray(row["transform_matrix"])
        angle = math.acos(float(np.clip(q[:3, 2] @ p[:3, 2], -1, 1)))
        return float(np.linalg.norm(p[:3, 3] - q[:3, 3])) + 10 * angle

    if not training or any(row["split"] != "train" for row in training):
        raise ValueError("Texture sources must contain only training frames")
    return min(training, key=cost)


def background_projection(target, source, width=WIDTH, height=HEIGHT, focal=FOCAL):
    """Rotation-only pixels near and above the world horizon, ignoring translation."""
    target, source = np.asarray(target), np.asarray(source)
    y, x = np.mgrid[:height, :width]
    rays = np.stack(((x - width / 2) / focal, (y - height / 2) / focal, np.ones_like(x)))
    world = np.einsum("ij,jhw->ihw", target[:3, :3], rays)
    local = np.einsum("ij,jhw->ihw", source[:3, :3].T, world)
    z = np.where(np.abs(local[2]) > 1e-8, local[2], 1e-8)
    u, v = focal * local[0] / z + width / 2, focal * local[1] / z + height / 2
    visible = (
        (world[2] > -0.04)
        & (local[2] > 0)
        & (u >= 0)
        & (u < width - 1)
        & (v >= 0)
        & (v < height - 1)
    )
    return u, v, visible


def hybrid_image(engine, source, target_pose, source_pose):
    """Bilinear ground/infinity reprojection with engine fallback and soft edges."""
    result = engine.astype(float)
    for projection, bottom in (
        (background_projection, HEIGHT - 1),
        (ground_projection, HEIGHT * 0.9),
    ):
        u, v, visible = projection(target_pose, source_pose)
        x, y = np.clip(u, 0, WIDTH - 2).astype(int), np.clip(v, 0, HEIGHT - 2).astype(int)
        dx, dy = np.clip(u - x, 0, 1)[..., None], np.clip(v - y, 0, 1)[..., None]
        sampled = (
            source[y, x] * (1 - dx) * (1 - dy)
            + source[y, x + 1] * dx * (1 - dy)
            + source[y + 1, x] * (1 - dx) * dy
            + source[y + 1, x + 1] * dx * dy
        )
        edge = np.minimum.reduce((u, WIDTH - 1 - u, v, bottom - v))
        weight = (visible * np.clip(edge / 20, 0, 1))[..., None]
        result = sampled * weight + result * (1 - weight)
    return np.clip(result, 0, 255).astype(np.uint8)


def splat_engine_image(splat_rgb, alpha, engine):
    """Composite premultiplied Brush RGB (0–255) over engine RGB using opacity (0–1)."""
    splat_rgb, alpha, engine = np.asarray(splat_rgb), np.asarray(alpha), np.asarray(engine)
    if splat_rgb.shape != engine.shape or alpha.shape != engine.shape[:2]:
        raise ValueError("RGB images and opacity must have matching dimensions")
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("Opacity must be finite and between zero and one")
    return np.clip(splat_rgb + (1 - alpha[..., None]) * engine, 0, 255).astype(np.uint8)


def build_scene(job):
    """Called inside Blender; all procedural placement derives from training poses."""
    import bpy
    from mathutils import Matrix

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.cycles.device = "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        metal = [d for d in prefs.devices if d.type == "METAL"]
        if metal:
            for device in prefs.devices:
                device.use = device.type == "METAL"
            scene.cycles.device = "GPU"
    except TypeError:
        pass
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    scene.view_settings.view_transform = "Standard"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (*job["sky"], 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.8

    def material(name, color):
        result = bpy.data.materials.new(name)
        result.diffuse_color = (*color, 1)
        result.use_nodes = True
        result.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1)
        result.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.85
        return result

    asphalt = material("training asphalt", (0.12, 0.12, 0.12))
    nodes, links = asphalt.node_tree.nodes, asphalt.node_tree.links
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(job["asphalt"])
    image.extension = "REPEAT"
    coords = nodes.new("ShaderNodeTexCoord")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = 0.4
    links.new(coords.outputs["Object"], scale.inputs[0])
    links.new(scale.outputs[0], image.inputs[0])
    links.new(image.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    paint = material("faded parking paint", (0.46, 0.47, 0.43))
    facade = material("concrete", (0.37, 0.34, 0.29))
    glass = material("windows", (0.12, 0.15, 0.14))
    grass = material("dry verge", (0.18, 0.20, 0.11))
    bark = material("bark", (0.14, 0.12, 0.09))
    leaves = material("foliage", (0.07, 0.12, 0.055))
    orange = material("cone orange", (0.9, 0.12, 0.025))
    lime = material("marker lime", (0.43, 0.61, 0.07))
    cx, cy, yaw = job["origin"]
    c, s = math.cos(yaw), math.sin(yaw)

    def location(x, y, z):
        return cx + c * x - s * y, cy + s * x + c * y, z

    def box(name, xyz, size, mat):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location(*xyz))
        obj = bpy.context.object
        obj.name, obj.dimensions, obj.rotation_euler.z = name, size, yaw
        obj.data.materials.append(mat)
        return obj

    box("lot", (0, 0, -0.05), (220, 220, 0.1), asphalt)
    for x in range(-70, 90, 9):
        for y in (-17, 0, 17, 34):
            box("parking stripe", (x, y, 0.006), (0.08, 5.0, 0.008), paint)
    for y in (-31, 51):
        box("verge", (0, y, 0.12), (200, 4, 0.25), grass)
        box("curb", (0, y - math.copysign(2, y), 0.1), (200, 0.2, 0.2), facade)
        for x in range(-80, 100, 13):
            box("trunk", (x, y, 3), (0.35, 0.35, 6), bark)
            bpy.ops.mesh.primitive_ico_sphere_add(
                subdivisions=2, radius=1, location=location(x, y, 6)
            )
            tree = bpy.context.object
            tree.scale = (3, 3, 4)
            tree.data.materials.append(leaves)
    for x, y, height in ((55, -65, 35), (10, -67, 40), (-70, 72, 27)):
        box("office tower", (x, y, height / 2), (26, 22, height), facade)
        for z in np.arange(3, height - 1, 3):
            for offset in range(-10, 12, 3):
                for face in (-1, 1):
                    box("window", (x + offset, y + face * 11.01, z), (1.7, 0.03, 1.9), glass)
    for x in range(-12, 21, 8):
        for y in (-4, 4):
            box("cone foot", (x, y, 0.025), (0.35, 0.35, 0.05), orange)
            bpy.ops.mesh.primitive_cone_add(
                vertices=16,
                radius1=0.15,
                radius2=0.025,
                depth=0.55,
                location=location(x, y, 0.3),
            )
            bpy.context.object.data.materials.append(orange)
    for x in range(-16, 25, 4):
        bpy.ops.mesh.primitive_cone_add(
            vertices=16, radius1=0.12, radius2=0, depth=0.07, location=location(x, 0, 0.035)
        )
        bpy.context.object.data.materials.append(lime)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36
    camera.data.lens = FOCAL * 36 / WIDTH
    camera.data.clip_start, camera.data.clip_end = 0.05, 1000
    scene.camera = camera
    scene.render.film_transparent = False
    times = []
    for query in job["queries"]:
        pose = np.asarray(query["transform_matrix"], dtype=float)
        camera.matrix_world = Matrix((pose @ np.diag([1, -1, -1, 1])).tolist())
        scene.render.filepath = str(Path(job["output"]) / query["name"])
        start = time.monotonic()
        if "FINISHED" not in bpy.ops.render.render(write_still=True):
            raise RuntimeError(f"Blender cancelled rendering {query['name']}")
        times.append({"name": query["name"], "seconds": time.monotonic() - start})
    Path(job["output"], "timings.json").write_text(
        json.dumps(
            {
                "blender": bpy.app.version_string,
                "device": scene.cycles.device,
                "run_id": job["run_id"],
                "renders": times,
            },
            indent=2,
        )
        + "\n"
    )


def render_batch(manifest, queries, output, blender, hybrid=False):
    from PIL import Image

    training = [row for row in manifest if row["split"] == "train"]
    if not training:
        raise ValueError("A training frame is required for texture extraction")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = training[len(training) // 2]
    with Image.open(source["file_path"]) as image:
        image.crop((WIDTH * 0.25, HEIGHT * 0.72, WIDTH * 0.75, HEIGHT * 0.85)).save(
            output / "asphalt.jpg", quality=92
        )
        sky = np.asarray(image.crop((WIDTH * 0.65, 0, WIDTH * 0.85, HEIGHT * 0.08)))
    positions = np.asarray([row["transform_matrix"] for row in training])[:, :3, 3]
    center = np.median(positions, axis=0)
    rotation = np.asarray(source["transform_matrix"])
    job = {
        "run_id": uuid.uuid4().hex,
        "origin": [*center[:2], math.atan2(rotation[1, 2], rotation[0, 2])],
        "sky": (np.median(sky, axis=(0, 1)) / 255).tolist(),
        "asphalt": str(output / "asphalt.jpg"),
        "output": str(output),
        "queries": queries,
        "texture_source": source["name"],
    }
    job_file = output / "job.json"
    job_file.write_text(json.dumps(job, indent=2) + "\n")
    subprocess.run(
        [
            str(Path(blender).resolve()),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--job",
            str(job_file),
        ],
        check=True,
    )
    completion_file = output / "timings.json"
    completion = json.loads(completion_file.read_text()) if completion_file.exists() else {}
    names = [query["name"] for query in queries]
    if (
        completion.get("run_id") != job["run_id"]
        or [row["name"] for row in completion.get("renders", [])] != names
        or any(not (output / name).is_file() for name in names)
    ):
        raise RuntimeError(
            "Blender did not complete this render job; stale images are not accepted"
        )
    if hybrid:
        dest = output.parent / "hybrid"
        dest.mkdir(exist_ok=True)
        sources = []
        for query in queries:
            source = nearest_training(query, training)
            with Image.open(output / query["name"]) as image:
                engine = np.asarray(image, dtype=float)
            with Image.open(source["file_path"]) as image:
                pixels = np.asarray(image, dtype=float)
            result = hybrid_image(
                engine, pixels, query["transform_matrix"], source["transform_matrix"]
            )
            Image.fromarray(result).save(dest / query["name"], quality=92)
            sources.append({"query": query["name"], "source": source["name"]})
        (dest / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/sim/visual/engine"))
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path("runs/sim/visual/tools/Blender.app/Contents/MacOS/Blender"),
    )
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    if args.job:
        build_scene(json.loads(args.job.read_text()))
    else:
        if not args.manifest or not args.queries:
            parser.error("--manifest and --queries are required")

        def read(path):
            return [json.loads(line) for line in path.read_text().splitlines() if line]

        render_batch(
            read(args.manifest), read(args.queries), args.output, args.blender, args.hybrid
        )


if __name__ == "__main__":
    main()
