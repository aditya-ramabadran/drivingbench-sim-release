"""Registered real starts need not lie inside the constructed score-only corridor."""

import copy

import pytest
from drivingbench.sim.course import Course
from drivingbench.sim.real_course_study.batch import PoseRendererLease, preflight_starts
from drivingbench.sim.real_course_study.runtime import runtime_factory
from test_sim_real_course_study import Coverage, scene, western

supported = Coverage()


def test_registered_start_preserves_scene_and_records_actual_geometry():
    data = scene()
    original = copy.deepcopy(data)
    actual = (-10, 5, 90)
    factory = runtime_factory(data, supported, registered_start=actual, western_score=western)
    runtime = factory(Course(data))
    try:
        assert runtime.pose_snapshot()["map_pose"] == actual
        assert runtime.pose_snapshot()["strict_boundary_violation"]
        assert not runtime.pose_snapshot()["boundary_violation"]
        assert runtime.course.start == actual
        assert data == original
    finally:
        runtime.close()
    outside = copy.deepcopy(data)
    outside["start"]["y"] = 5
    with pytest.raises(ValueError, match="strictly inside"):
        Course(outside)


@pytest.mark.parametrize("pose", [(0, 0, float("nan")), (0, float("inf"), 0), (0, 0)])
def test_registered_start_is_finite_three_coordinate_pose(pose):
    with pytest.raises(ValueError, match="Pose"):
        runtime_factory(scene(), supported, registered_start=pose, western_score=western)


def test_preflight_rejects_lot_backstop_before_model_or_renderer():
    data = measured_scene()
    rows = {
        "astra": [
            dict(
                attempt=1,
                start=dict(x=-10, y=5, heading_deg=90),
                native_state=dict(speed=0, steer_deg=0, hold=True),
            )
        ]
    }
    with pytest.raises(ValueError, match="lot safety backstop"):
        preflight_starts(data, rows, Coverage(False, 5.001))


def test_map_frame_and_renderer_transform_agree():
    data = scene()
    actual = (-10, 5, 90)
    runtime = runtime_factory(data, supported, registered_start=actual, western_score=western)(
        Course(data)
    )

    class Renderer:
        def render(self, pose):
            return pose

    try:
        lease = PoseRendererLease(Renderer(), Course(data).start, actual)
        assert lease.render((1, 2, 15)) == pytest.approx((-2, 6, 105))
        assert runtime.course.map_pose((1, 2, 15)) == pytest.approx((-12, 6, 105))
    finally:
        runtime.close()


def measured_scene():
    data = scene()
    data["boundaries"]["right"] = [[-20, -4], [-10, -4], [10, -4], [20, -4]]
    data["markers"].extend(
        [
            dict(id="west-first", kind="dome_marker", x=-10, y=-4, collision_radius_m=0.1),
            dict(id="west-last", kind="dome_marker", x=10, y=-4, collision_radius_m=0.1),
        ]
    )
    data["measurement"] = {
        "virtual_boundary": {
            "first_marker_id": "west-first",
            "last_marker_id": "west-last",
            "vertices": [],
        }
    }
    return data


def rows_at(pose):
    return {
        "astra": [
            dict(
                attempt=1,
                start=dict(zip(("x", "y", "heading_deg"), pose)),
                native_state=dict(speed=0, steer_deg=0, hold=True),
            )
        ]
    }


def test_preflight_accepts_registered_start_outside_score_only_corridor():
    result = preflight_starts(measured_scene(), rows_at((-10, 5, 90)), supported)
    assert result[0]["map_pose"] == (-10, 5, 90)
    assert result[0]["strict_boundary_violation"] and result[0]["held"]


def test_preflight_accepts_nearby_unreviewed_start_without_moving_it():
    result = preflight_starts(measured_scene(), rows_at((-10, 5, 90)), Coverage(False, 0.01))
    assert result[0]["map_pose"] == (-10, 5, 90)
    assert not result[0]["rendered_coverage"]["supported"]
    assert result[0]["rendered_coverage"]["distance_to_any_paired_pass_parent_m"] == 0.01


@pytest.mark.parametrize("pose", [(10, 0, 0), (-5, -10, 0)])
def test_preflight_checks_actual_tall_obstacle_and_western_backstop(pose):
    with pytest.raises(ValueError, match="terminal start geometry"):
        preflight_starts(measured_scene(), rows_at(pose), supported)


def test_run_preflight_fails_before_auth_or_renderer(monkeypatch, tmp_path):
    import json
    from dataclasses import asdict
    from datetime import datetime, timedelta, timezone

    from drivingbench.sim.real_course_study import batch

    scene_path = tmp_path / "scene.json"
    data = measured_scene()
    data["wide_camera"] = {}
    scene_path.write_text(json.dumps(data))
    declaration = dict(
        protocol=batch.PROTOCOL,
        client_profile="normal",
        release_passed=True,
        models=[asdict(batch.CLIENTS[k]) for k in batch.MODELS],
        stop_after_success=False,
        client_done_parking_confirmation_s=5.0,
        coverage_policy=batch.COVERAGE_POLICY,
        hashes={},
        scene=str(scene_path),
        settings={"speed_limit_mps": 3.5},
        deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        coverage_certificate="unused",
        starts=rows_at((-10, 5, 90)),
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(declaration))

    class Coverage:
        def validate_assets(self, root):
            return self

        def measure(self, pose):
            return {"supported": False, "cells": [], "distance_to_any_paired_pass_parent_m": 6.0}

    monkeypatch.setattr(
        batch, "SceneRenderer", lambda *a, **kw: pytest.fail("Renderer must not start")
    )
    monkeypatch.setattr(batch, "auth_preflight", lambda *a, **kw: pytest.fail("Auth must not run"))
    with pytest.raises(ValueError, match="lot safety backstop"):
        batch.run(protocol, tmp_path / "fresh-output", coverage_factory=lambda p: Coverage())
    assert not (tmp_path / "fresh-output").exists()
