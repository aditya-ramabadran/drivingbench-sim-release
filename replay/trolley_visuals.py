"""Bounded photographic camera approximation with independently grounded actors.

The engine owns motion and time. Photo poses and their residual from the engine pose
are recorded, never represented as calibrated photographs or physical ground truth.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float32)


def rays(camera, yaw):
    y, x = np.mgrid[: camera["height"], : camera["width"]]
    return np.stack(
        [
            (x - camera["cx"]) / camera["focal_px"],
            -(y - camera["cy"]) / camera["focal_px"],
            np.ones_like(x),
        ],
        axis=-1,
    ).astype(np.float32) @ rotation(yaw)


def scene_surfaces(position, directions):
    """Ray parameter to the nearest ground/barrier plane; infinity denotes sky."""
    position = np.asarray(position, dtype=np.float32)
    depth = np.full(directions.shape[:2], np.inf, dtype=np.float32)

    surfaces = np.full(directions.shape[:2], -1, dtype=np.int8)
    surface_id = 0

    def plane(normal, constant, inside):
        nonlocal surface_id
        normal = np.asarray(normal, dtype=np.float32)
        denominator = directions @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (constant - position @ normal) / denominator
            points = position + directions * t[..., None]
        valid = (t > 0.02) & np.isfinite(t) & inside(points)
        closer = valid & (t < depth)
        surfaces[closer] = surface_id
        np.minimum(depth, np.where(valid, t, np.inf), out=depth)
        surface_id += 1

    plane(
        [0, 1, 0],
        0,
        lambda p: (
            (np.abs(p[..., 0]) <= 3.8 + 0.28 * np.maximum(p[..., 2], 0))
            & ((p[..., 2] < 0) | (np.abs(p[..., 0]) >= 0.2 + 0.28 * p[..., 2]))
        ),
    )
    for side in [-1, 1]:
        for intercept, slope, low, high in [
            (3.8, 0, -100, 0),
            (3.8, 0.28, 0, 150),
            (0.2, 0.28, 0, 150),
        ]:
            plane(
                [1, 0, -side * slope],
                side * intercept,
                lambda p, low=low, high=high: (
                    (p[..., 2] >= low)
                    & (p[..., 2] <= high)
                    & (p[..., 1] >= 0)
                    & (p[..., 1] <= 1.05)
                ),
            )
    plane(
        [0, 0, 1], 0, lambda p: (np.abs(p[..., 0]) <= 0.2) & (p[..., 1] >= 0) & (p[..., 1] <= 1.05)
    )
    plane(
        [0, 1, 0],
        1.05,
        lambda p: (
            (np.abs(p[..., 0]) >= 3.8 + 0.28 * np.maximum(p[..., 2], 0))
            & (np.abs(p[..., 0]) <= 4.02 + 0.28 * np.maximum(p[..., 2], 0))
        ),
    )
    plane(
        [0, 1, 0], 1.05, lambda p: (p[..., 2] >= 0) & (np.abs(p[..., 0]) <= 0.2 + 0.28 * p[..., 2])
    )
    return depth, surfaces


def scene_depth(position, directions):
    return scene_surfaces(position, directions)[0]


def sample(array, u, v):
    """Bilinear sampling with an explicit in-bounds mask (never wrap texture)."""
    height, width = array.shape[:2]
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= -1e-4)
        & (u <= width - 1 + 1e-4)
        & (v >= -1e-4)
        & (v <= height - 1 + 1e-4)
    )
    u = np.clip(np.nan_to_num(u), 0, width - 1)
    v = np.clip(np.nan_to_num(v), 0, height - 1)
    x, y = u.astype(np.int32), v.astype(np.int32)
    x1, y1 = np.minimum(x + 1, width - 1), np.minimum(y + 1, height - 1)
    dx, dy = (u - x).astype(np.float32), (v - y).astype(np.float32)
    if array.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    output = (
        (1 - dx) * (1 - dy) * array[y, x]
        + dx * (1 - dy) * array[y, x1]
        + (1 - dx) * dy * array[y1, x]
        + dx * dy * array[y1, x1]
    )
    return output, valid


@lru_cache(maxsize=12)
def photo_pixels(path):
    return np.asarray(Image.open(path).convert("RGB"))


@lru_cache(maxsize=6)
def photo_flow(path):
    with np.load(path) as data:
        return data["forward"].astype(np.float32), data["backward"].astype(np.float32)


def pose_vector(position, yaw):
    return np.array([position[0], position[2], 8 * yaw], dtype=float)


def inverse_warp(image, flow, fraction):
    """Invert a smooth, offline-computed flow; report uncovered borders explicitly."""
    height, width = image.shape[:2]
    fh, fw = flow.shape[:2]
    gy, gx = np.mgrid[:fh, :fw].astype(np.float32)
    gx *= (width - 1) / (fw - 1)
    gy *= (height - 1) / (fh - 1)
    u, v = gx - fraction * flow[..., 0], gy - fraction * flow[..., 1]
    for _ in range(6):
        delta, _ = sample(flow, u * (fw - 1) / (width - 1), v * (fh - 1) / (height - 1))
        u, v = gx - fraction * delta[..., 0], gy - fraction * delta[..., 1]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    maps, _ = sample(
        np.stack([u, v], -1), xx * (fw - 1) / (width - 1), yy * (fh - 1) / (height - 1)
    )
    return sample(image, maps[..., 0], maps[..., 1])


class Scene:
    def __init__(self, manifest):
        self.manifest = manifest
        self.camera = manifest["camera"]
        self.frames = {r["id"]: r for r in manifest["photo_bank"]["keyframes"]}
        self.edges = manifest["photo_bank"]["edges"]
        if not self.frames:
            raise ValueError("photographic keyframes required")

    def select(self, position, yaw):
        actual = pose_vector(position, yaw)
        choices = []
        for edge in self.edges:
            a, b = self.frames[edge["a"]], self.frames[edge["b"]]
            av = pose_vector(a["camera_position_m"], a["camera_yaw_rad"])
            bv = pose_vector(b["camera_position_m"], b["camera_yaw_rad"])
            delta = bv - av
            fraction = float(np.clip((actual - av) @ delta / max(delta @ delta, 1e-12), 0, 1))
            represented = av + fraction * delta
            choices.append((float(np.linalg.norm(actual - represented)), edge, fraction, a, b))
        # Exact keyframe support also works for a one-frame test fixture.
        for frame in self.frames.values():
            residual = np.linalg.norm(
                actual - pose_vector(frame["camera_position_m"], frame["camera_yaw_rad"])
            )
            choices.append((float(residual), None, 0.0, frame, frame))
        _, edge, fraction, a, b = min(choices, key=lambda c: c[0])
        represented = (1 - fraction) * np.array(a["camera_position_m"]) + fraction * np.array(
            b["camera_position_m"]
        )
        represented_yaw = (1 - fraction) * a["camera_yaw_rad"] + fraction * b["camera_yaw_rad"]
        residual = float(np.linalg.norm(np.array(position) - represented))
        yaw_residual = float(abs(np.degrees(yaw - represented_yaw)))
        if residual > self.manifest.get(
            "max_pose_residual_m", 1.25
        ) or yaw_residual > self.manifest.get("max_yaw_residual_deg", 10):
            raise ValueError(
                f"photographic pose unsupported: residual {residual:.3f}m/{yaw_residual:.2f}deg"
            )
        return edge, fraction, a, b, represented, represented_yaw, residual, yaw_residual

    def background(self, position, yaw):
        edge, t, a, b, represented, represented_yaw, residual, yaw_residual = self.select(
            position, yaw
        )
        first = photo_pixels(a["path"])
        second = photo_pixels(b["path"])
        expected = (self.camera["height"], self.camera["width"])
        if first.shape[:2] != expected or second.shape[:2] != expected:
            raise ValueError("photo resolution does not match declared camera")
        primary = a if t <= 0.5 else b
        if edge is None or t < 1e-6 or t > 1 - 1e-6:
            result = (first if t <= 0.5 else second).astype(np.float32).copy()
            valid = np.ones(expected, dtype=bool)
        else:
            forward, backward = photo_flow(edge["flow_path"])
            wa, va = inverse_warp(first, forward, t)
            wb, vb = inverse_warp(second, backward, 1 - t)
            # One complete photograph supplies the interior; the other fills newly exposed borders.
            result = np.where((va if t <= 0.5 else ~vb)[..., None], wa, wb)
            valid = va | vb
        crop = self.manifest.get("frame_crop_px", 0)
        inspected = valid[crop:-crop, crop:-crop] if crop else valid
        uncovered = float(np.mean(~inspected))
        directions = rays(self.camera, represented_yaw)
        depth = scene_depth(represented, directions)
        metadata = dict(
            renderer_version="photographic-bank-v1",
            represented_camera_position_m=represented.tolist(),
            represented_camera_yaw_rad=float(represented_yaw),
            pose_residual_m=residual,
            yaw_residual_deg=yaw_residual,
            uncovered_fraction=uncovered,
            photo_sources=[
                {k: r[k] for k in ["id", "path", "sha256"]}
                for r in ([a] if a["id"] == b["id"] else [a, b])
            ],
            interpolation=dict(
                edge_id=edge["id"] if edge else None,
                fraction=t,
                method="regularized single-photo flow with opposite-photo border coverage",
                primary_source_id=primary["id"],
                flow_path=edge["flow_path"] if edge else None,
                flow_sha256=edge["flow_sha256"] if edge else None,
            ),
        )
        return result, depth, directions, metadata

    def render(self, config, state):
        position = np.asarray(state["camera_position_m"], dtype=np.float32)
        yaw = float(state["camera_yaw_rad"])
        if position.shape != (3,) or not np.all(np.isfinite(position)) or not np.isfinite(yaw):
            raise ValueError("finite 3D camera pose required")
        image, depth, directions, quality = self.background(position, yaw)
        position = np.asarray(quality["represented_camera_position_m"], dtype=np.float32)
        yaw = quality["represented_camera_yaw_rad"]
        image = image.copy()
        actors = []
        shadows = []
        if state.get("obstacles_visible", False):
            for side, sign in [("left", -1), ("right", 1)]:
                group = config[side]
                if not group["count"]:
                    continue
                asset = self.manifest["actors"][f"{group['count']}_{group['kind']}"]
                for card in asset.get("instances", [asset]):
                    source = Image.open(card["path"]).convert("RGBA")
                    foot = card.get("footpoint_px")
                    if foot is None:
                        source = source.crop(source.getchannel("A").getbbox())
                        foot = [source.width / 2, source.height - 1]
                    z = self.manifest.get("actor_z_m", 12.0) + card.get("depth_offset_m", 0)
                    lateral = card.get("lateral_m", 0)
                    if config.get("mirror_scene", False):
                        source = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        foot = [source.width - 1 - foot[0], foot[1]]
                        lateral *= -1
                    center = sign * (2 + 0.28 * z) + lateral
                    # Premultiplied-alpha downsampling prevents dark fringes and tiny-subject aliasing.
                    height = card["height_m"]
                    scale = height / max(foot[1], 1)
                    projected = self.camera["focal_px"] * height / max(z - position[2], 1)
                    factor = min(1, max(8, projected * 2) / source.height)
                    newsize = (
                        max(1, round(source.width * factor)),
                        max(1, round(source.height * factor)),
                    )
                    rgba = np.asarray(
                        source.convert("RGBa").resize(newsize, Image.Resampling.LANCZOS),
                        dtype=np.float32,
                    )
                    sx, sy = newsize[0] / source.width, newsize[1] / source.height
                    world_corners = np.array(
                        [
                            [center - foot[0] * scale, 0, z],
                            [center + (source.width - foot[0]) * scale, 0, z],
                            [center - foot[0] * scale, height, z],
                            [center + (source.width - foot[0]) * scale, height, z],
                        ]
                    )
                    q = (world_corners - position) @ rotation(yaw).T
                    if np.any(q[:, 2] <= 0):
                        continue
                    px = self.camera["cx"] + self.camera["focal_px"] * q[:, 0] / q[:, 2]
                    py = self.camera["cy"] - self.camera["focal_px"] * q[:, 1] / q[:, 2]
                    x0 = max(0, int(np.floor(px.min())) - 2)
                    x1 = min(image.shape[1], int(np.ceil(px.max())) + 3)
                    y0 = max(0, int(np.floor(py.min())) - 2)
                    y1 = min(image.shape[0], int(np.ceil(py.max())) + 3)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    ds = directions[y0:y1, x0:x1]
                    t = (z - position[2]) / ds[..., 2]
                    points = position + ds * t[..., None]
                    u = (foot[0] + (points[..., 0] - center) / scale) * sx
                    v = (foot[1] - points[..., 1] / scale) * sy
                    color, valid = sample(rgba, u, v)
                    valid &= (t > 0) & (t <= depth[y0:y1, x0:x1] + 0.001)
                    alpha = color[..., 3] / 255 * valid
                    premult = color[..., :3] * valid[..., None]
                    actors.append(
                        (
                            float(np.linalg.norm([center - position[0], z - position[2]])),
                            (x0, y0, x1, y1),
                            premult,
                            alpha,
                        )
                    )
                    shadows.append((center, z, 0.18 if group["kind"] != "dog" else 0.3, 0.12))
        # Soft overcast contact shadows are fixed ground decals, occluded by the same scene geometry.
        ground_t = np.divide(
            -position[1],
            directions[..., 1],
            out=np.full(depth.shape, np.inf),
            where=directions[..., 1] < 0,
        )
        visible_ground = (ground_t > 0) & (ground_t <= depth + 0.001)
        pixel_y, pixel_x = np.mgrid[: image.shape[0], : image.shape[1]]
        for x, z, rx, rz in shadows:
            q = (np.array([x, 0, z]) - position) @ rotation(yaw).T
            if q[2] <= 0:
                continue
            foot_x = self.camera["cx"] + self.camera["focal_px"] * q[0] / q[2]
            foot_y = self.camera["cy"] - self.camera["focal_px"] * q[1] / q[2]
            radius_x = max(1.0, self.camera["focal_px"] * rx / q[2])
            # Subpixel ground shadows need a pixel-footprint filter; point sampling erased them.
            radius_y = max(0.65, self.camera["focal_px"] * 0.5 * position[1] / q[2] ** 2)
            radius = ((pixel_x - foot_x) / radius_x) ** 2 + ((pixel_y - foot_y) / radius_y) ** 2
            opacity = 0.32 * np.exp(-radius * 2) * visible_ground
            image *= 1 - opacity[..., None]
        for _, (x0, y0, x1, y1), color, alpha in sorted(actors, key=lambda a: -a[0]):
            image[y0:y1, x0:x1] = image[y0:y1, x0:x1] * (1 - alpha[..., None]) + color
        result = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        crop = self.manifest.get("frame_crop_px", 0)
        if crop:
            result = result.crop((crop, crop, result.width - crop, result.height - crop)).resize(
                result.size, Image.Resampling.LANCZOS
            )
        quality["frame_crop_px"] = crop
        width, height = self.camera["width"], self.camera["height"]
        quality["effective_camera"] = {
            "width": width,
            "height": height,
            "focal_x_px": self.camera["focal_px"] * width / (width - 2 * crop),
            "focal_y_px": self.camera["focal_px"] * height / (height - 2 * crop),
            "cx_px": (self.camera["cx"] - crop) * width / (width - 2 * crop),
            "cy_px": (self.camera["cy"] - crop) * height / (height - 2 * crop),
        }
        return result, quality


@lru_cache(maxsize=4)
def load_scene(manifest_path):
    return Scene(json.loads(Path(manifest_path).read_text()))


def render(config, state, output_path):
    """Render the supplied camera pose; no elapsed time, controls, or state mutation."""
    scene = load_scene(str(Path(config["visuals_manifest"]).resolve()))
    image, quality = scene.render(config, state)
    limit = scene.manifest.get("max_uncovered_fraction", 0.1)
    if quality["uncovered_fraction"] > limit:
        raise ValueError(f"uncovered novel view exceeds declared limit: {quality}")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(
            {
                "camera_position_m": state["camera_position_m"],
                "camera_yaw_rad": state["camera_yaw_rad"],
                **quality,
            },
            indent=2,
        )
        + "\n"
    )
    return target
