"""Planar course physics in authored ground-map metres, independent of rendering."""

from __future__ import annotations

import math

LENGTH_M = 4.63
WIDTH_M = 1.78
CAMERA_OFFSET_M = (1.0, 0.0)
# Nominal CoG/body centre; symmetric body and camera forward/left offset are assumptions.
EPS = 1e-9
BOUNDARY_RELATIVE_ALLOWANCE = 0.0001
BOUNDARY_QUAD_SEGS = 128


def _finite(values, label):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite numbers") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def _point(value):
    result = _finite(value, "Point")
    if len(result) != 2:
        raise ValueError("Point must have two coordinates")
    return result


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(p, a, b):
    return abs(_cross(a, b, p)) <= EPS and all(
        min(a[i], b[i]) - EPS <= p[i] <= max(a[i], b[i]) + EPS for i in (0, 1)
    )


def _touch(a, b, c, d):
    turns = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    if turns[0] * turns[1] < 0 and turns[2] * turns[3] < 0:
        return True
    return any(
        _on_segment(p, first, last)
        for p, first, last in ((a, c, d), (b, c, d), (c, a, b), (d, a, b))
    )


def _edges(polygon):
    return zip(polygon, polygon[1:] + polygon[:1])


def _inside(point, polygon):
    """Strict inclusion; touching an edge is outside."""
    inside = False
    x, y = point
    for a, b in _edges(polygon):
        if _on_segment(point, a, b):
            return False
        if (a[1] > y) != (b[1] > y):
            crossing_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < crossing_x:
                inside = not inside
    return inside


def _polygon(values):
    points = tuple(_point(value) for value in values)
    if len(points) < 3:
        raise ValueError("Polygon needs at least three vertices")
    if any(math.dist(a, b) <= EPS for a, b in _edges(points)):
        raise ValueError("Polygon has a zero-length edge")
    area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges(points))
    if abs(area2) <= EPS:
        raise ValueError("Polygon has zero area")
    edges = list(_edges(points))
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _touch(a, b, *edges[j]):
                raise ValueError("Polygon intersects itself")
    return points


def _contains_body(polygon, body):
    return all(_inside(point, polygon) for point in body) and not any(
        _touch(a, b, c, d) for a, b in _edges(body) for c, d in _edges(polygon)
    )


def _pose(values):
    result = _finite(values, "Pose")
    if len(result) != 3:
        raise ValueError("Pose must be x, y, heading_deg")
    return result


def _transform(point, pose):
    x, y, heading = pose
    c, s = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    return x + c * point[0] - s * point[1], y + s * point[0] + c * point[1]


def footprint(pose):
    """Nominal CoG-centred body corners for a ground-map x/y/heading-degrees pose."""
    pose = _pose(pose)
    return tuple(
        _transform(point, pose)
        for point in (
            (LENGTH_M / 2, WIDTH_M / 2),
            (-LENGTH_M / 2, WIDTH_M / 2),
            (-LENGTH_M / 2, -WIDTH_M / 2),
            (LENGTH_M / 2, -WIDTH_M / 2),
        )
    )


def finish_before_line(left, right, *, max_front_gap_m=3.0):
    """Whole-body receiver behind the near paint edge, viewed from the approach.

    An aligned body stops strictly between zero and max_front_gap_m before paint.
    Within a 20-degree approach envelope, yaw makes this allowance conservative.
    The existing strict corridor and stop/dwell requirements remain independent.
    """
    left, right = _point(left), _point(right)
    (gap,) = _finite((max_front_gap_m,), "Finish front gap")
    width = math.dist(left, right)
    if gap <= 0 or not math.isfinite(width) or width <= WIDTH_M:
        raise ValueError("Finish needs a positive gap and a line wider than the car")
    forward = ((left[1] - right[1]) / width, (right[0] - left[0]) / width)
    depth = LENGTH_M + gap

    def back(point):
        return [point[i] - depth * forward[i] for i in range(2)]

    return {"polygon": [list(left), list(right), back(right), back(left)]}


def _boundary_buffer(corridor, tolerance):
    rule = {"tolerance_m": float(tolerance), "numerical_allowance_m": 0.0}
    if not tolerance:
        return None, None, rule
    try:
        import shapely
        from shapely.geometry import Polygon
        from shapely.prepared import prep
    except ImportError as error:
        raise RuntimeError(
            "Boundary tolerance requires the simulator extra: uv sync --extra sim"
        ) from error
    allowance = tolerance * BOUNDARY_RELATIVE_ALLOWANCE
    allowed = prep(Polygon(corridor).buffer(tolerance + allowance, quad_segs=BOUNDARY_QUAD_SEGS))
    rule.update(
        numerical_allowance_m=allowance,
        quad_segs=BOUNDARY_QUAD_SEGS,
        shapely=shapely.__version__,
        geos=shapely.geos_version_string,
    )
    return allowed, Polygon, rule


class Course:
    """Validate an authored scene and assess instantaneous body/marker geometry.

    SimNative's unlabelled pose is assigned a nominal CoG/body-centre convention.
    The camera is assumed 1 m forward, centred laterally; neither offset is surveyed.
    No plant or controller behavior changes. Reverse motion needs no special case.
    Generic scene objects are visual only until their collision geometry is defined.
    """

    def __init__(self, scene, *, boundary_tolerance_m=0.0, strict_boundary_tolerance_m=None):
        if not math.isfinite(boundary_tolerance_m) or boundary_tolerance_m < 0:
            raise ValueError("Boundary tolerance must be finite and nonnegative")
        self.boundary_tolerance_m = float(boundary_tolerance_m)
        if scene.get("frame") not in {"pilot_ground", "site1_full_ground", "video_metric"}:
            raise ValueError("Course needs a declared metric frame")
        self.start = _pose(scene["start"][key] for key in ("x", "y", "heading_deg"))
        left = tuple(_point(p) for p in scene["boundaries"]["left"])
        right = tuple(_point(p) for p in scene["boundaries"]["right"])
        if min(len(left), len(right)) < 2:
            raise ValueError("Each boundary needs at least two points, start to finish")
        self.corridor = _polygon(left + right[::-1])
        if not _inside(self.start[:2], self.corridor):
            raise ValueError("Start reference point must be strictly inside the corridor")
        self.finish = _polygon(scene["finish"]["polygon"])
        self._allowed_body, self._body_polygon, self.boundary_rule = _boundary_buffer(
            self.corridor, self.boundary_tolerance_m
        )
        self._strict_allowed_body = None
        self.strict_boundary_rule = None
        if strict_boundary_tolerance_m is not None:
            if not math.isfinite(strict_boundary_tolerance_m) or strict_boundary_tolerance_m < 0:
                raise ValueError("Strict boundary tolerance must be finite and nonnegative")
            self._strict_allowed_body, polygon, self.strict_boundary_rule = _boundary_buffer(
                self.corridor, strict_boundary_tolerance_m
            )
            if polygon is not None:
                self._body_polygon = polygon
        self.obstacle_ids = set()
        self.markers = []
        ids = set()
        for marker in scene["markers"]:
            marker_id = marker["id"]
            if not isinstance(marker_id, str) or not marker_id or marker_id in ids:
                raise ValueError("Marker IDs must be unique nonempty strings")
            ids.add(marker_id)
            if marker.get("kind") == "cone":
                self.obstacle_ids.add(marker_id)
            x, y, radius = _finite(
                (marker["x"], marker["y"], marker["collision_radius_m"]), "Marker circle"
            )
            if radius <= 0:
                raise ValueError("Marker collision radius must be positive")
            self.markers.append((marker_id, x, y, radius))

    def map_pose(self, local_pose):
        """Register SimNative local x/y/heading-degrees to the scene's start pose."""
        x, y, heading = _pose(local_pose)
        return (*_transform((x, y), self.start), heading + self.start[2])

    def camera_pose(self, map_pose):
        """Camera ground point and heading, using the declared mounting assumption."""
        pose = _pose(map_pose)
        return (*_transform(CAMERA_OFFSET_M, pose), pose[2])

    def assess(self, map_pose):
        """Independent collision/exit/finish flags at one ground-map pose; no swept test."""
        pose = _pose(map_pose)
        body = footprint(pose)
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        hits = []
        for marker_id, x, y, radius in self.markers:
            dx, dy = x - pose[0], y - pose[1]
            longitudinal, lateral = c * dx + s * dy, -s * dx + c * dy
            distance = math.hypot(
                max(abs(longitudinal) - LENGTH_M / 2, 0),
                max(abs(lateral) - WIDTH_M / 2, 0),
            )
            if distance <= radius + EPS:
                hits.append(marker_id)
        outside = not _contains_body(self.corridor, body)
        violation = (
            not self._allowed_body.covers(self._body_polygon(body))
            if self._allowed_body is not None
            else outside
        )
        finish = _contains_body(self.finish, body)
        result = {
            "collision_ids": hits,
            "obstacle_collision_ids": [
                identity for identity in hits if identity in self.obstacle_ids
            ],
            "outside_course": outside,
            "boundary_violation": violation,
            "finish_reached": finish,
            "valid_finish": finish and not hits and not outside,
        }
        if self.strict_boundary_rule is not None:
            result["strict_boundary_violation"] = (
                not self._strict_allowed_body.covers(self._body_polygon(body))
                if self._strict_allowed_body is not None
                else outside
            )
        return result
