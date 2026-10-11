import math

import pytest
from drivingbench.sim.course import CAMERA_OFFSET_M, LENGTH_M, WIDTH_M, Course, footprint


def scene():
    return {
        "name": "test",
        "frame": "pilot_ground",
        "start": {"x": 0, "y": 0, "heading_deg": 0},
        "boundaries": {"left": [[-10, 5], [10, 5]], "right": [[-10, -5], [10, -5]]},
        "finish": {"polygon": [[3, -4], [9, -4], [9, 4], [3, 4]]},
        "markers": [],
        "objects": [],
    }


def marker(x, y, radius=0.1, marker_id="dome"):
    return {
        "id": marker_id,
        "kind": "dome_marker",
        "x": x,
        "y": y,
        "z": 0,
        "yaw": 0,
        "diameter_m": 0.2,
        "height_m": 0.08,
        "collision_radius_m": radius,
        "color_srgb": [0.8, 0.3, 0.1],
    }


@pytest.mark.parametrize("frame", ["pilot_ground", "site1_full_ground"])
def test_registration_and_camera_offset_rotate_with_body(frame):
    value = scene()
    value["frame"] = frame
    value["start"] = {"x": 1, "y": 2, "heading_deg": 90}
    course = Course(value)
    assert course.map_pose((3, 1, -20)) == pytest.approx((0, 5, 70))
    assert course.camera_pose((1, 2, 90)) == pytest.approx((1, 3, 90))
    assert CAMERA_OFFSET_M == (1, 0)


def test_rotated_body_dimensions_and_reverse_pose():
    body = footprint((0, 0, 90))
    assert max(p[0] for p in body) - min(p[0] for p in body) == pytest.approx(WIDTH_M)
    assert max(p[1] for p in body) - min(p[1] for p in body) == pytest.approx(LENGTH_M)
    course = Course(scene())
    assert course.assess((-3, 0, 180)) == course.assess((-3, 0, 0))


@pytest.mark.parametrize("heading", [0, 90, 180, -90])
def test_circle_tangency_collision_and_one_millimetre_clearance(heading):
    angle = math.radians(heading)
    distance = LENGTH_M / 2 + 0.1
    value = scene()
    value["markers"] = [marker(distance * math.cos(angle), distance * math.sin(angle))]
    assert Course(value).assess((0, 0, heading))["collision_ids"] == ["dome"]
    value["markers"][0]["x"] += 0.001 * math.cos(angle)
    value["markers"][0]["y"] += 0.001 * math.sin(angle)
    assert Course(value).assess((0, 0, heading))["collision_ids"] == []


def test_circle_inside_body_and_corner_tangency():
    value = scene()
    offset = 0.1 / math.sqrt(2)
    value["markers"] = [
        marker(0, 0, marker_id="under_body"),
        marker(LENGTH_M / 2 + offset, WIDTH_M / 2 + offset, marker_id="corner"),
    ]
    assert Course(value).assess((0, 0, 0))["collision_ids"] == ["under_body", "corner"]


def test_boundary_grazing_and_start_endcaps_count_as_exit():
    course = Course(scene())
    assert not course.assess((0, 5 - WIDTH_M / 2 - 0.001, 0))["outside_course"]
    assert course.assess((0, 5 - WIDTH_M / 2, 0))["outside_course"]
    assert course.assess((-10 + LENGTH_M / 2, 0, 0))["outside_course"]
    assert course.assess((10 - LENGTH_M / 2, 0, 0))["outside_course"]


def test_reversing_out_of_start_does_not_require_a_heading_flip():
    course = Course(scene())
    results = [course.assess((x, 0, 0))["outside_course"] for x in (0, -7, -8)]
    assert results == [False, False, True]


def test_concave_corridor_gap_crosses_body_with_all_corners_inside():
    value = scene()
    value["start"]["x"] = -5
    value["boundaries"]["left"] = [[-10, 5], [-0.3, 5], [-0.3, 0], [0.3, 0], [0.3, 5], [10, 5]]
    course = Course(value)
    assert not course.assess((-5, 0, 0))["outside_course"]
    assert course.assess((0, 0, 0))["outside_course"]


def test_finish_requires_whole_body_and_reports_collision_independently():
    value = scene()
    value["markers"] = [marker(6, 0)]
    course = Course(value)
    assert not course.assess((3, 0, 0))["finish_reached"]
    assert not course.assess((3 + LENGTH_M / 2, 0, 0))["finish_reached"]
    result = course.assess((6, 0, 0))
    assert result == {
        "collision_ids": ["dome"],
        "obstacle_collision_ids": [],
        "outside_course": False,
        "boundary_violation": False,
        "finish_reached": True,
        "valid_finish": False,
    }
    value["markers"] = []
    assert Course(value).assess((6, 0, 0))["valid_finish"] is True


def test_finish_cannot_hide_course_exit():
    value = scene()
    value["finish"]["polygon"] = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
    result = Course(value).assess((10, 0, 0))
    assert result["finish_reached"] and result["outside_course"]
    assert not result["valid_finish"]


@pytest.mark.parametrize(
    "polygon",
    [
        [[0, 0], [1, 1], [0, 1], [1, 0]],
        [[0, 0], [1, 0], [1, 0]],
        [[0, 0], [1, 0], [2, 0]],
        [[0, 0], [float("nan"), 1], [2, 0]],
    ],
)
def test_invalid_polygons_rejected(polygon):
    value = scene()
    value["finish"]["polygon"] = polygon
    with pytest.raises(ValueError):
        Course(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("collision_radius_m", -1),
        ("x", float("inf")),
        ("collision_radius_m", 0),
        ("y", float("nan")),
    ],
)
def test_invalid_marker_geometry_rejected(field, value):
    data = scene()
    data["markers"] = [marker(0, 0)]
    data["markers"][0][field] = value
    with pytest.raises(ValueError):
        Course(data)


def test_duplicate_ids_start_outside_and_nonfinite_pose_rejected():
    value = scene()
    value["markers"] = [marker(0, 0), marker(1, 0)]
    with pytest.raises(ValueError, match="unique"):
        Course(value)
    value["markers"] = []
    value["start"]["x"] = 10
    with pytest.raises(ValueError, match="Start"):
        Course(value)
    with pytest.raises(ValueError, match="finite"):
        Course(scene()).assess((0, 0, float("nan")))


def test_physics_uses_marker_circles_without_visual_fields_and_ignores_generic_objects():
    value = scene()
    value["markers"] = [{"id": "circle", "x": 0, "y": 0, "collision_radius_m": 0.1}]
    value["objects"] = [{"kind": "tree", "x": 0, "y": 0}, {"kind": "dog", "x": 1, "y": 0}]
    result = Course(value).assess((0, 0, 0))
    assert result["collision_ids"] == ["circle"]
    value["markers"] = []
    assert Course(value).assess((0, 0, 0))["collision_ids"] == []


@pytest.mark.parametrize("heading", [0, 180])
def test_half_metre_excursion_keeps_raw_touch_and_strict_finish(heading):
    course = Course(scene(), boundary_tolerance_m=0.5)
    for overrun in (0, 0.2, 0.5):
        result = course.assess((0, 5 - WIDTH_M / 2 + overrun, heading))
        assert result["outside_course"]
        assert not result["boundary_violation"]
        assert not result["valid_finish"]
    assert course.assess((0, 5 - WIDTH_M / 2 + 0.500051, heading))["boundary_violation"]
    assert course.boundary_rule["numerical_allowance_m"] == pytest.approx(0.00005)
    assert course.boundary_rule["quad_segs"] == 128


def test_reverse_endcap_and_rotated_body_use_whole_polygon():
    course = Course(scene(), boundary_tolerance_m=0.5)
    assert not course.assess((-10 + LENGTH_M / 2 - 0.5, 0, 0))["boundary_violation"]
    assert course.assess((-10 + LENGTH_M / 2 - 0.501, 0, 0))["boundary_violation"]
    extent = (LENGTH_M + WIDTH_M) / (2 * math.sqrt(2))
    assert not course.assess((0, 5 - extent + 0.5, 45))["boundary_violation"]
    assert course.assess((0, 5 - extent + 0.501, 45))["boundary_violation"]


def test_buffer_preserves_outside_bay_hole_inside_body():
    value = scene()
    value["start"]["x"] = -5
    value["boundaries"]["left"] = [
        [-10, 5],
        [-0.05, 5],
        [-0.05, 0.7],
        [-0.7, 0.7],
        [-0.7, -0.7],
        [0.7, -0.7],
        [0.7, 0.7],
        [0.05, 0.7],
        [0.05, 5],
        [10, 5],
    ]
    course = Course(value, boundary_tolerance_m=0.5)
    result = course.assess((0, 0, 0))
    assert result["outside_course"] and result["boundary_violation"]
    assert len(course._allowed_body.context.interiors) == 1


@pytest.mark.parametrize("exterior_angle", [0.01832, 0.03, 0.31, 1.57, 2.1])
def test_round_corner_threshold_includes_arbitrary_geos_arc_angles(exterior_angle):
    from shapely.geometry import Point

    interior = math.pi - exterior_angle
    value = scene()
    value["boundaries"] = {
        "left": [[0, 0], [100, 0]],
        "right": [[100 * math.cos(interior), 100 * math.sin(interior)], [50, 1]],
    }
    value["start"].update(x=50, y=0.25)
    course = Course(value, boundary_tolerance_m=0.5)
    # Vertex (0,0) has a convex round offset spanning this exterior angle.
    angle = -math.pi / 2 - exterior_angle / 2
    assert course._allowed_body.covers(Point(0.5 * math.cos(angle), 0.5 * math.sin(angle)))
    assert not course._allowed_body.covers(
        Point(0.500051 * math.cos(angle), 0.500051 * math.sin(angle))
    )


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_boundary_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="Boundary tolerance"):
        Course(scene(), boundary_tolerance_m=tolerance)


def test_optional_geometry_dependency_fails_before_takeover(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_shapely(name, *args, **kwargs):
        if name == "shapely" or name.startswith("shapely."):
            raise ImportError("test unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_shapely)
    assert Course(scene()).boundary_rule["tolerance_m"] == 0
    with pytest.raises(RuntimeError, match="simulator extra"):
        Course(scene(), boundary_tolerance_m=0.5)


@pytest.mark.parametrize("heading", [0, 90, 180, -90])
def test_blue_line_receiver_rotates_and_uses_front_bumper(heading):
    from drivingbench.sim.course import _transform, finish_before_line

    value = scene()
    pose = (0, 0, heading)
    value["boundaries"] = {
        "left": [_transform(p, pose) for p in ((-10, 5), (20, 5))],
        "right": [_transform(p, pose) for p in ((-10, -5), (20, -5))],
    }
    value["finish"] = finish_before_line(_transform((9, 4), pose), _transform((9, -4), pose))
    course = Course(value)
    for gap in (0.001, 0.5, 1.5, 2.8, 2.999):
        point = _transform((9 - LENGTH_M / 2 - gap, 0), pose)
        assert course.assess((*point, heading))["valid_finish"]
        assert course.assess((*point, heading + 180))["valid_finish"]
    for gap in (-0.001, 0, 3, 3.001):
        point = _transform((9 - LENGTH_M / 2 - gap, 0), pose)
        assert not course.assess((*point, heading))["finish_reached"]


@pytest.mark.parametrize("heading", [-20, 0, 20])
def test_blue_line_whole_body_yaw_lateral_edge_and_contact(heading):
    from drivingbench.sim.course import finish_before_line

    value = scene()
    value["finish"] = finish_before_line((9, 4), (9, -4))
    course = Course(value)
    projected = max(p[0] for p in footprint((0, 0, heading)))
    centre = (9 - projected - 0.5, 0, heading)
    assert course.assess(centre)["valid_finish"]
    too_far = (9 - projected - 3.001, 0, heading)
    assert not course.assess(too_far)["finish_reached"]
    lateral = max(p[1] for p in footprint((0, 0, heading)))
    assert not course.assess((centre[0], 4 - lateral, heading))["finish_reached"]
    value["markers"] = [marker(*centre[:2])]
    assert not Course(value).assess(centre)["valid_finish"]


@pytest.mark.parametrize(
    "left,right,gap",
    [
        ((9, 4), (9, -4), 0),
        ((9, 4), (9, -4), float("nan")),
        ((9, 4), (9, -4), float("inf")),
        ((float("nan"), 4), (9, -4), 3),
        ((9, 0), (9, 0), 3),
        ((9, WIDTH_M / 2), (9, -WIDTH_M / 2), 3),
    ],
)
def test_blue_line_invalid_geometry(left, right, gap):
    from drivingbench.sim.course import finish_before_line

    with pytest.raises(ValueError):
        finish_before_line(left, right, max_front_gap_m=gap)


@pytest.mark.parametrize("heading", [0, 180])
def test_lenient_and_strict_excursion_scores_are_independent(heading):
    course = Course(scene(), boundary_tolerance_m=1, strict_boundary_tolerance_m=0.5)
    for excursion, strict, live in (
        (0.5, False, False),
        (0.501, True, False),
        (1, True, False),
        (1.001, True, True),
    ):
        result = course.assess((0, 5 - WIDTH_M / 2 + excursion, heading))
        assert result["strict_boundary_violation"] is strict
        assert result["boundary_violation"] is live
    assert course.strict_boundary_rule["numerical_allowance_m"] == pytest.approx(0.00005)
    assert course.boundary_rule["numerical_allowance_m"] == pytest.approx(0.0001)


def test_tall_cone_collision_is_tagged_separately_from_domes():
    value = scene()
    value["markers"] = [marker(0, 0), dict(marker(1, 0, marker_id="tall"), kind="cone")]
    result = Course(value).assess((0, 0, 0))
    assert result["collision_ids"] == ["dome", "tall"]
    assert result["obstacle_collision_ids"] == ["tall"]


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_strict_boundary_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="Strict boundary tolerance"):
        Course(scene(), strict_boundary_tolerance_m=tolerance)
