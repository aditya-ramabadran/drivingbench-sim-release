import copy
import math

import pytest
from drivingbench.sim.course import LENGTH_M, WIDTH_M, Course
from drivingbench.sim.episode import Episode
from drivingbench.sim.real_course_study.refined import (
    RefinedCourse,
    RefinedPolicy,
    runtime_factory,
)
from drivingbench.sim.real_course_study.runtime import (
    RealCourse,
    RealCoursePolicy,
)
from drivingbench.sim.real_course_study.runtime import (
    runtime_factory as original_factory,
)
from test_sim_real_course_study import Coverage, scene, western
from test_sim_runtime import Clock, move


def refined_scene():
    data = scene()
    data["refinement"] = {"raw_boundaries": copy.deepcopy(data["boundaries"])}
    return data


def course(data=None, score=western):
    return RefinedCourse(data or refined_scene(), western_score=score)


def body_for_front(x, y=0, heading=0):
    angle = math.radians(heading)
    return x - LENGTH_M / 2 * math.cos(angle), y - LENGTH_M / 2 * math.sin(angle), heading


@pytest.mark.parametrize(
    "front,expected",
    [
        ((-1, 0, 0), True),
        ((-1.01, 0, 0), False),
        ((0, 0, 0), True),
        ((1, 0, 0), True),
        ((6.01, 0, 0), False),
        ((0.5, 0, 180), False),
        ((-0.5, 20, 0), False),
    ],
)
def test_parking_requires_finite_front_region_and_facing(front, expected):
    assert course().parking(body_for_front(*front))["finish_geometry"] is expected


def test_whole_bumper_must_fit_even_when_front_centre_enters():
    data = refined_scene()
    measured = course(data)
    tangent = body_for_front(0.5, 2.5 - WIDTH_M / 2)
    inside = body_for_front(0.5, 2.5 - WIDTH_M / 2 - 0.001)
    over = body_for_front(0.5, 2.0)
    assert measured.parking(tangent)["front_inside_U"]
    assert not measured.parking(tangent)["finish_geometry"]
    assert measured.parking(inside)["finish_geometry"]
    assert not measured.parking(over)["finish_geometry"]
    data["finish"]["mouth"]["left"], data["finish"]["mouth"]["right"] = (
        data["finish"]["mouth"]["right"],
        data["finish"]["mouth"]["left"],
    )
    assert course(data).parking(inside)["finish_geometry"]
    assert not course(data).parking(over)["finish_geometry"]


def test_parking_rotates_with_map_frame():
    data = refined_scene()

    def transform(p):
        return [8 - p[1], 3 + p[0]]

    for side in ("left", "right"):
        data["boundaries"][side] = [transform(p) for p in data["boundaries"][side]]
        data["refinement"]["raw_boundaries"][side] = [
            transform(p) for p in data["refinement"]["raw_boundaries"][side]
        ]
        data["finish"]["mouth"][side] = transform(data["finish"]["mouth"][side])
    data["finish"]["polygon"] = [transform(p) for p in data["finish"]["polygon"]]
    data["start"] = {"x": 8, "y": -7, "heading_deg": 90}
    for marker in data["markers"]:
        marker["x"], marker["y"] = transform([marker["x"], marker["y"]])
    original_pose = body_for_front(-0.5)
    assert course(data).parking((*transform(original_pose), 90))["finish_geometry"]


def test_live_and_strict_scores_use_smooth_corridor_and_raw_circles():
    measured = course(score=lambda *_: {"virtual_backstop": True})
    tangent_y = 4 + 2.5 - WIDTH_M / 2
    assert not measured.assess((-10, tangent_y, 0))["boundary_violation"]
    departed = measured.assess((-10, tangent_y + 0.001, 180))
    assert departed["boundary_violation"]
    assert departed["western"]["virtual_backstop"] and departed["western"]["score_only"]
    only_strict = measured.assess((-10, 4, 0))
    assert only_strict["strict_boundary_violation"] and not only_strict["boundary_violation"]
    assert measured.assess((-10, 0, 0))["collision_ids"] == ["dome"]
    assert measured.assess((10, 0, 0))["obstacle_collision_ids"] == ["tall"]
    assert not measured.assess((10, 0, 0))["valid_finish"]


def test_smooth_boundaries_not_raw_reference_control_live_rule():
    data = refined_scene()
    data["refinement"]["raw_boundaries"]["left"] = [[-20, 2], [20, 2]]
    result = course(data).assess((-10, 4, 0))
    assert result["strict_boundary_violation"] and not result["boundary_violation"]
    assert "offline" in course(data).boundary_rule["raw_reference"]
    del data["refinement"]
    with pytest.raises(ValueError, match="raw boundary reference"):
        course(data)


def test_policy_requires_verified_hold_and_no_positive_motion():
    evidence = dict(course().assess(body_for_front(0.5)), held=True, speed_mps=0)
    policy = RefinedPolicy()
    assert policy.boundary_tolerance_m == 2.5
    assert policy.strict_boundary_tolerance_m == 0.5
    assert policy.finish_eligible(evidence)
    for update in (
        {"held": False},
        {"positive_motion_command": True},
        {"speed_mps": 0.05},
        {"obstacle_collision_ids": ["tall"]},
        {"boundary_violation": True},
    ):
        assert not policy.finish_eligible(dict(evidence, **update))


def test_default_factory_behavior_identical_to_explicit_original_course():
    data = scene()
    options = {"western_score": western, "registered_start": (-2, 0, 0)}
    implicit = original_factory(data, Coverage(False, 5.001), **options)
    explicit = original_factory(data, Coverage(False, 5.001), course_factory=RealCourse, **options)
    runtimes = [
        factory(Course(data), clock=lambda: 1000.0, frame_clock=lambda: 1000.0)
        for factory in (implicit, explicit)
    ]
    try:
        assert runtimes[0].pose_snapshot() == runtimes[1].pose_snapshot()
        assert runtimes[0].course.boundary_rule == runtimes[1].course.boundary_rule
        assert type(runtimes[0].course) is RealCourse
        assert type(runtimes[1].course) is RealCourse
    finally:
        for runtime in runtimes:
            runtime.close()


def test_new_runtime_initializes_new_course_and_retains_legacy_evidence():
    data = refined_scene()
    factory = runtime_factory(
        data, Coverage(False, 5.001), registered_start=body_for_front(0.5), western_score=western
    )
    runtime = factory(Course(data), clock=lambda: 1000.0, frame_clock=lambda: 1000.0)
    try:
        snapshot = runtime.pose_snapshot()
        assert isinstance(runtime.course, RefinedCourse)
        assert snapshot["map_pose"] == body_for_front(0.5)
        assert snapshot["finish_reached"]
        assert snapshot["lot_safety_backstop"] is None
        assert snapshot["legacy_lot_safety_backstop"] == snapshot["rendered_coverage"]
        assert not snapshot["rendered_coverage"]["supported"]
        assert snapshot["coverage_exit"] is None
    finally:
        runtime.close()


def test_held_dwell_and_client_done_confirmation_reuse_original_lifecycle(tmp_path):
    data = refined_scene()
    start = body_for_front(0.5)
    clock = Clock()
    runtime = runtime_factory(
        data, Coverage(False, 6), registered_start=start, western_score=western
    )(Course(data), clock=clock)
    episode = Episode(runtime, tmp_path, RefinedPolicy(), metadata={})
    try:
        move(runtime, clock, 0.5)
        assert episode.summary()["outcome"] == "unfinished"
        runtime.finish("client_done")
        assert runtime.end_reason is None
        move(runtime, clock, 0.6)
        assert episode.summary()["outcome"] == "success"
        assert runtime.end_reason == "finish"
        assert episode.summary()["strict_score"]["first_cause"] is None
    finally:
        runtime.close()
        episode.close()


def test_original_parent_backstop_still_disqualifies_baseline():
    evidence = dict(
        RealCourse(scene(), western_score=western).assess((-2, 0, 0)),
        held=True,
        speed_mps=0,
        lot_safety_backstop={"distance": 6},
    )
    assert not RealCoursePolicy().finish_eligible(evidence)


# Frozen stop geometry from operator-refinements-v1/report.json and tracks.npz;
# source hashes and the conditional body-reference convention are in the runtime doc.
PHOTO_STOP = (-74.15556605879611, 1.9247618897983154, 57.404235)
CORNER_STOP = (-71.68889665775882, -0.495221914979286, 82.23009711751463)
BLUE_FINISH = {
    "polygon": [
        [-71.64171336741512, 9.1151795894019],
        [-74.78474542069623, 4.661086728617357],
        [-70.76411027178477, 1.6960403538766786],
        [-67.92133060811776, 5.479007533071567],
        [-68.07711935322361, 6.074711712345587],
        [-70.82571648180124, 8.497301941718861],
        [-71.19709205736288, 8.796367350004502],
    ],
    "mouth": {
        "left": [-74.78474542069623, 4.661086728617357],
        "right": [-70.76411027178477, 1.6960403538766786],
        "conditional_sigma_m": 0.07278452919567618,
    },
}


def test_approved_photo_stop_and_tiny_corner_overlap_regression():
    data = refined_scene()
    data["finish"] = copy.deepcopy(BLUE_FINISH)
    data["boundaries"] = {"left": [[-100, 30], [0, 30]], "right": [[-100, -30], [0, -30]]}
    measured = course(data)
    photo = measured.assess(PHOTO_STOP)
    corner = measured.assess(CORNER_STOP)
    assert photo["finish_reached"] and not photo["strict_whole_body_finish"]
    assert photo["parking"]["body_overlap_m2"] == pytest.approx(0.8581902503397055)
    assert corner["parking"]["body_overlap_m2"] == pytest.approx(0.0004235134636564121)
    assert corner["parking"]["legacy_area_or_centre_geometry"]
    assert not corner["finish_reached"]
    policy = RefinedPolicy()
    assert policy.finish_eligible(dict(photo, held=True, speed_mps=0))
    assert not policy.finish_eligible(dict(corner, held=True, speed_mps=0))
    counts = []
    for pose in (PHOTO_STOP, CORNER_STOP):
        heading = math.radians(pose[2])
        variants = [
            (
                pose[0] + shift * math.cos(heading),
                pose[1] + shift * math.sin(heading),
                pose[2] + yaw,
            )
            for shift in (-0.5, 0, 0.5)
            for yaw in (-3, 0, 3)
        ]
        counts.append(sum(measured.parking(value)["finish_geometry"] for value in variants))
    assert counts == [9, 2]
