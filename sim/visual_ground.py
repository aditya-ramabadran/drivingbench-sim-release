"""Training-observed road plane and a fixed camera mount in the reconstruction frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def road_rotation(normal, heading):
    """Forward/left/up basis with heading projected onto the plane."""
    normal = np.asarray(normal, dtype=float)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or not np.isfinite(heading)
        or not np.isclose(np.linalg.norm(normal), 1)
        or normal[2] < 0.95
    ):
        raise ValueError("Expected an upward unit road normal and finite heading")
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    forward -= normal * (forward @ normal)
    forward /= np.linalg.norm(forward)
    return np.column_stack((forward, np.cross(normal, forward), normal))


def plane_fit(points, threshold=0.04):
    """Robust orthogonal plane fit; points must be independently selected road observations."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
        raise ValueError("Plane fitting needs at least six 3D road points")
    if not np.isfinite(points).all() or threshold <= 0:
        raise ValueError("Plane inputs must be finite and threshold positive")
    rng = np.random.default_rng(42)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(512):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if abs(normal[2]) < 0.95:
            continue
        inside = np.abs((points - a) @ normal) < threshold
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < 6:
        raise ValueError("No sufficiently supported road plane")
    for _ in range(5):
        center = points[best].mean(0)
        _, singular, vt = np.linalg.svd(points[best] - center, full_matrices=False)
        if singular[1] < 1e-6:
            raise ValueError("Road points must span an area")
        normal = vt[-1] * np.sign(vt[-1, 2])
        offset = -normal @ center
        inside = np.abs(points @ normal + offset) < threshold
        if np.array_equal(inside, best):
            break
        if inside.sum() < 6:
            raise ValueError("Road plane refinement lost support")
        best = inside
    center = points[best].mean(0)
    _, _, vt = np.linalg.svd(points[best] - center, full_matrices=False)
    normal = vt[-1] * np.sign(vt[-1, 2])
    offset = -normal @ center
    return normal, float(offset), best


@dataclass(frozen=True)
class GroundCalibration:
    """Plane n·p+d=0; mount maps OpenCV camera axes into forward/left/up car axes."""

    normal: np.ndarray
    offset: float
    height: float
    mount_rotation: np.ndarray

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float)
        mount = np.asarray(self.mount_rotation, dtype=float)
        if normal.shape != (3,) or mount.shape != (3, 3):
            raise ValueError("Expected a three-vector normal and 3×3 mount rotation")
        if not np.isfinite([*normal, self.offset, self.height, *mount.ravel()]).all():
            raise ValueError("Ground calibration must be finite")
        if not np.isclose(np.linalg.norm(normal), 1) or normal[2] < 0.95 or self.height <= 0:
            raise ValueError("Expected an upward unit road normal and positive camera height")
        if not np.allclose(mount.T @ mount, np.eye(3), atol=1e-7) or not np.isclose(
            np.linalg.det(mount), 1
        ):
            raise ValueError("Mount must be a proper rotation")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "mount_rotation", mount)

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        return cls(**{key: values[key] for key in cls.__dataclass_fields__})

    def ground_height(self, x, y):
        return (
            -(self.normal[0] * np.asarray(x) + self.normal[1] * np.asarray(y) + self.offset)
            / self.normal[2]
        )

    def road_rotation(self, heading):
        return road_rotation(self.normal, heading)

    @property
    def map_from_ground(self):
        """Rigid map transform; ground origin is ENU x=y=0 projected vertically onto the plane."""
        result = np.eye(4)
        result[:3, :3] = self.road_rotation(0)
        result[:3, 3] = (0, 0, self.ground_height(0, 0))
        return result

    def camera_pose(self, x, y, heading):
        """ENU ground x/y and heading radians → OpenCV camera-to-map pose."""
        if not np.isfinite([x, y, heading]).all():
            raise ValueError("Camera pose must be finite")
        pose = np.eye(4)
        pose[:3, :3] = self.road_rotation(heading) @ self.mount_rotation
        pose[:3, 3] = np.array([x, y, self.ground_height(x, y)]) + self.height * self.normal
        return pose

    def camera_from_ground(self, x, y, heading):
        """Flat ground-frame x/y and heading radians → OpenCV camera-to-map pose."""
        if not np.isfinite([x, y, heading]).all():
            raise ValueError("Camera pose must be finite")
        c, s = np.cos(heading), np.sin(heading)
        pose = np.eye(4)
        pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.mount_rotation
        pose[:3, 3] = (x, y, self.height)
        return self.map_from_ground @ pose

    def ray_intersections(self, camera, intrinsics):
        """Map points and camera-forward depth at pixel centers; invalid rays return NaN."""
        camera = np.asarray(camera, dtype=float)
        if camera.shape != (4, 4) or not np.isfinite(camera).all():
            raise ValueError("Expected finite 4×4 OpenCV camera pose")
        w, h = intrinsics["w"], intrinsics["h"]
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        directions = (
            np.stack(
                (
                    (u - intrinsics["cx"]) / intrinsics["fl_x"],
                    (v - intrinsics["cy"]) / intrinsics["fl_y"],
                    np.ones_like(u),
                ),
                axis=-1,
            )
            @ camera[:3, :3].T
        )
        denominator = directions @ self.normal
        numerator = -(camera[:3, 3] @ self.normal + self.offset)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = numerator / denominator
        depth[(np.abs(denominator) < 1e-8) | (depth <= 0)] = np.nan
        return camera[:3, 3] + depth[..., None] * directions, depth


def fit_camera(normal, offset, frames):
    """Fit one height and mount from training camera poses; no held-out pose fitting."""
    if not frames or any(row["split"] != "train" for row in frames):
        raise ValueError("Camera calibration requires training frames only")
    temporary = GroundCalibration(normal, offset, 1.21, np.eye(3))
    cameras = [np.asarray(row["transform_matrix"]) for row in frames]
    mounts = np.array(
        [
            temporary.road_rotation(row["heading"]).T @ pose[:3, :3]
            for row, pose in zip(frames, cameras)
        ]
    )
    height = float(np.median([pose[:3, 3] @ normal + offset for pose in cameras]))
    return GroundCalibration(normal, offset, height, _rotation(mounts.mean(0)))
