import json
from types import SimpleNamespace

import pytest
from drivingbench.sim.course import Course
from drivingbench.sim.real_course_study.batch import (
    PoseRendererLease,
    checkpoint,
    producer_paired_observations,
    registered_starts,
)
from drivingbench.sim.real_course_study.runtime import (
    RealCourse,
    RealCoursePolicy,
    runtime_factory,
)


def scene():
    return {
        "frame": "site1_full_ground",
        "start": {"x": -10, "y": 0, "heading_deg": 0},
        "boundaries": {"left": [[-20, 4], [20, 4]], "right": [[-20, -4], [20, -4]]},
        "finish": {
            "polygon": [[0, -2.5], [6, -2.5], [6, 2.5], [0, 2.5]],
            "mouth": {"left": [0, 2.5], "right": [0, -2.5]},
        },
        "markers": [
            {"id": "dome", "kind": "dome_marker", "x": -10, "y": 0, "collision_radius_m": 0.1},
            {"id": "tall", "kind": "cone", "x": 10, "y": 0, "collision_radius_m": 0.18},
        ],
    }


def western(pose, scene):
    return {"virtual_excursion_m": 0, "virtual_backstop": False}


class Coverage:
    def __init__(self, supported=True, distance_m=0.0):
        self.supported, self.distance_m = supported, distance_m

    def measure(self, pose):
        return {
            "supported": self.supported,
            "cells": ["test"] if self.supported else [],
            "distance_to_any_paired_pass_parent_m": self.distance_m,
        }


def test_domes_and_interpolated_boundary_remain_strict_scores():
    course = RealCourse(scene(), western_score=western)
    overlap = course.assess((-10, 0, 0))
    assert overlap["collision_ids"] == ["dome"]
    assert overlap["obstacle_collision_ids"] == []
    departure = course.assess((-10, 4, 0))
    assert departure["strict_boundary_violation"]
    assert departure["outside_course"]
    assert not departure["boundary_violation"]
    assert course.assess((10, 0, 0))["obstacle_collision_ids"] == ["tall"]


def test_operator_like_finish_does_not_require_strict_containment():
    result = RealCourse(scene(), western_score=western).assess((-2, 0, 0))
    assert result["finish_reached"] and not result["strict_whole_body_finish"]
    evidence = dict(result, speed_mps=0, held=True)
    assert RealCoursePolicy().finish_eligible(evidence)
    assert not RealCoursePolicy().finish_eligible(dict(evidence, held=False))
    assert not RealCoursePolicy().finish_eligible(dict(evidence, positive_motion_command=True))
    assert not RealCoursePolicy().finish_eligible(dict(evidence, speed_mps=0.05))
    assert not RealCoursePolicy().finish_eligible(
        dict(evidence, coverage_exit={"unsupported": True})
    )
    assert not RealCoursePolicy().finish_eligible(dict(evidence, obstacle_collision_ids=["tall"]))


def test_western_backstop_independent_of_strict_score():
    course = RealCourse(scene(), western_score=lambda *_: {"virtual_backstop": True})
    result = course.assess((-2, 0, 0))
    assert result["boundary_violation"]
    assert not RealCoursePolicy().finish_eligible(dict(result, speed_mps=0, held=True))


def test_runtime_requires_explicit_rendered_coverage_and_records_sensor_hold():
    with pytest.raises(ValueError, match="rendered-coverage"):
        runtime_factory(scene(), None)
    factory = runtime_factory(scene(), Coverage(), western_score=western)
    runtime = factory(Course(scene()), clock=lambda: 1000.0, frame_clock=lambda: 1000.0)
    try:
        evidence = runtime.pose_snapshot()
        assert evidence["map_pose"] == (-10, 0, 0)
        assert evidence["held"] == bool(
            runtime.status().get("held") and runtime.status().get("standstill")
        )
        assert evidence["coverage_exit"] is None
    finally:
        runtime.close()
    factory = runtime_factory(scene(), Coverage(False, 0.001), western_score=western)
    runtime = factory(Course(scene()), clock=lambda: 1000.0, frame_clock=lambda: 1000.0)
    try:
        evidence = runtime.pose_snapshot()
        assert not evidence["rendered_coverage"]["supported"]
        assert evidence["coverage_exit"] is None
        assert evidence["lot_safety_backstop"] is None
    finally:
        runtime.close()
    with pytest.raises(ValueError, match="explicit supported"):
        runtime_factory(scene(), SimpleNamespace(measure=lambda _: {}), western_score=western)(
            Course(scene()), clock=lambda: 1000.0, frame_clock=lambda: 1000.0
        )


def test_renderer_lease_registers_attempt_local_pose_and_does_not_close_owner():
    class Renderer:
        def render_frames(self, value):
            self.snapshot = value
            return {"narrow": b"n", "wide": b"w"}

        def render(self, pose):
            return pose

        def close(self):
            pytest.fail("Attempt does not own warm renderer")

    renderer = Renderer()
    lease = PoseRendererLease(renderer, (10, 0, 90), (13, 2, 0))
    assert lease.render((1, 0, 10)) == pytest.approx((2, -4, -80))
    assert lease.render_frames({"pose": (1, 0, 10), "sim_time_s": 4}) == {
        "narrow": b"n",
        "wide": b"w",
    }
    assert renderer.snapshot["pose"] == pytest.approx((2, -4, -80))
    assert renderer.snapshot["sim_time_s"] == 4
    lease.close()


def test_lease_keeps_captured_coverage_and_owns_just_rendered_metrics_without_changing_bytes():
    snapshot = {
        "pose": (1, 0, 10),
        "map_pose": (14, 2, 10),
        "sim_time_s": 4,
        "rendered_coverage": {"supported": False, "distance_to_any_paired_pass_parent_m": 0.002},
    }
    jpeg = b"original JPEG bytes"
    frames = {"narrow": jpeg, "wide": jpeg}

    class Renderer:
        metrics = {"wide": {"elapsed_s": 0.4}}

        def render_frames(self, captured):
            snapshot["map_pose"] = (99, 99, 99)
            snapshot["rendered_coverage"]["supported"] = True
            return frames

        def render(self, pose):
            self.metrics = {"elapsed_s": 0.2}
            return jpeg

    renderer = Renderer()
    first = PoseRendererLease(renderer, (10, 0, 90), (13, 2, 0))
    assert first.render_frames(snapshot) is frames
    assert frames["narrow"] is jpeg and frames["wide"] is jpeg
    evidence = first.metrics["captured_coverage"]
    assert evidence["map_pose"] == (14, 2, 10)
    assert not evidence["measurement"]["supported"]
    assert evidence["outside_reviewed"] is True
    assert "body_reference/paired_narrow_wide" in evidence["reference"]
    renderer.metrics["wide"]["elapsed_s"] = 99
    second = PoseRendererLease(renderer, (10, 0, 90), (13, 2, 0))
    second.render_frames(snapshot)
    assert first.metrics["wide"]["elapsed_s"] == 0.4
    assert not first.metrics["captured_coverage"]["measurement"]["supported"]
    assert second.metrics["captured_coverage"]["measurement"]["supported"]
    assert "captured_coverage" not in renderer.metrics
    assert first.render((0, 0, 0)) is jpeg
    assert first.metrics == {"elapsed_s": 0.2}
    assert "captured_coverage" in second.metrics


@pytest.mark.parametrize("distance", [None, float("nan"), float("inf"), -1, True])
def test_runtime_rejects_missing_or_invalid_spatial_evidence(distance):
    with pytest.raises(ValueError, match="finite nonnegative"):
        runtime_factory(scene(), Coverage(False, distance), western_score=western)(Course(scene()))


def test_reported_outside_review_does_not_stop_or_disqualify_parking(tmp_path):
    from drivingbench.sim.episode import Episode
    from test_sim_runtime import Clock, move

    data = scene()
    data["start"]["x"] = -2
    coverage = Coverage(False, 5.0)
    clock = Clock()
    runtime = runtime_factory(data, coverage, western_score=western)(Course(data), clock=clock)
    episode = Episode(runtime, tmp_path, RealCoursePolicy(), metadata={})
    try:
        move(runtime, clock, 1.1)
        assert episode.summary()["outcome"] == "success"
        assert runtime.end_reason == "finish"
        assert not runtime.pose_snapshot()["rendered_coverage"]["supported"]
    finally:
        runtime.close()
        episode.close()


def test_lot_backstop_applies_physical_braking_and_a_distinct_cause(tmp_path):
    from drivingbench.sim.car import SimNative
    from drivingbench.sim.episode import Episode
    from test_sim_runtime import Clock, move

    coverage = Coverage(False, 5.001)
    clock = Clock()
    runtime = runtime_factory(
        scene(),
        coverage,
        western_score=western,
        native_factory=lambda **kw: SimNative(speed=0.8, hold=False, **kw),
    )(Course(scene()), clock=clock)
    episode = Episode(runtime, tmp_path, RealCoursePolicy(), metadata={})
    try:
        move(runtime, clock, 0.01)
        summary = episode.summary()
        assert summary["outcome"] == "failure"
        assert summary["first_cause"]["reason"] == "lot_safety_backstop"
        assert summary["first_cause"]["evidence"]["speed_mps"] > 0.05
        assert runtime.pending_end == "lot_safety_backstop"
        move(runtime, clock, 5)
        assert runtime.end_reason == "lot_safety_backstop"
        assert runtime.pose_snapshot()["held"] and runtime.pose_snapshot()["speed_mps"] == 0
        assert episode.summary()["first_coverage_exit"] is None
    finally:
        runtime.close()
        episode.close()


def test_paired_receipt_gate_uses_delivered_frames_not_rendered_frames(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [
        {"kind": "sim_frame", "frame_ids": {"narrow": "n", "wide": "w"}},
        {"kind": "sim_observe_delivery", "frames": [{"camera": "narrow"}]},
        {"kind": "sim_observe_delivery", "frames": [{"camera": "narrow"}, {"camera": "wide"}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert producer_paired_observations(path) == 1
    reports = [
        dict(
            model=model,
            paired_observations=1,
            client_errors=[],
            episode={"infrastructure_failure": None},
        )
        for model in ("fable", "astra", "sol", "grok")
    ]
    assert checkpoint(reports)["passed"]
    reports[-1]["paired_observations"] = 0
    assert not checkpoint(reports)["passed"]
    assert not checkpoint(reports[:3])["passed"]


def test_registered_start_selection_is_primary_only_and_requires_each_attempt():
    rows = [
        dict(
            scope="primary",
            model=model,
            label=f"attempt-{attempt}--fail",
            segment=f"s-{model}-{attempt}",
            start_pose={"x": attempt, "y": 0, "heading": 0},
            registration={"independent": True},
        )
        for model in ("gpt-6-astra", "gpt-5.6-sol", "claude-fable-5.1", "grok-4.6")
        for attempt in (1, 2, 3)
    ]
    rows.append({"scope": "testing"})
    result = registered_starts({"attempts": rows})
    assert result["astra"][1]["start"] == {"x": 2, "y": 0, "heading_deg": 0}
    with pytest.raises(ValueError, match="exactly three"):
        registered_starts({"attempts": rows[1:]})


def test_polygonal_western_guard_tangency_finite_caps_and_opposite_side():
    from drivingbench.sim.real_course_study.western import WesternBackstop

    guard = WesternBackstop([(0, -10), (0, 10)])
    assert not guard((3 - 4.63 / 2, 0, 0))["virtual_backstop"]
    assert guard((3 - 4.63 / 2 + 0.001, 0, 0))["virtual_backstop"]
    assert guard((2, 0, 0))["backstop_outward_overlap_m2"] > 0
    assert not guard((-10, 0, 0))["virtual_backstop"]
    assert not guard((10, -12, 0))["virtual_backstop"]
    # A corner over the finite cap still belongs to the span; centre-only gating misses it.
    assert guard((10, -10.5, 0))["virtual_backstop"]
    assert guard((0, 0, 0))["backstop_arc_allowance_m"] < 0.00023


def test_polygonal_guard_covers_curve_arc_without_inscribed_buffer_false_positive():
    from drivingbench.sim.real_course_study.western import WesternBackstop

    guard = WesternBackstop([(0, -10), (0, 0), (-10, 0)])
    assert guard((5, 5, 45))["virtual_backstop"]
    assert not guard((-5, -5, 0))["virtual_backstop"]
    with pytest.raises(ValueError, match="simple polyline"):
        WesternBackstop([(0, 0), (1, 1), (0, 1), (1, 0)])


def test_episode_parking_dwell_and_strict_score_are_independent(tmp_path):
    from drivingbench.sim.episode import Episode
    from test_sim_episode import Runtime

    runtime = Runtime()
    episode = Episode(runtime, tmp_path / "episode", RealCoursePolicy(), metadata={})
    try:
        for t, held in ((0, True), (0.9, False), (1, True), (1.9, True), (2, True)):
            runtime.tick(
                t,
                finish=True,
                held=held,
                hits=("dome",),
                outside=True,
                violation=False,
                strict_boundary_violation=True,
            )
        summary = episode.summary()
        assert summary["outcome"] == "success"
        assert summary["strict_score"]["failed"]
        assert runtime.requests == ["finish"]
    finally:
        episode.close()


async def test_real_mcp_paired_frames_and_midcourse_stop_do_not_end_measured_trial(tmp_path):
    from contextlib import AsyncExitStack
    from types import SimpleNamespace

    import httpx
    from drivingbench.gateway.app import create_app as gateway_app
    from drivingbench.sim.naturalistic_run import audit_attempt
    from drivingbench.sim.producer import simulator_app
    from drivingbench.sim.real_course_study.batch import received_pairs
    from test_e2e import http_server, summary, wait_status
    from test_sim_live import Renderer, connect, scene_file

    class Paired(Renderer):
        def render_frames(self, snapshot):
            return {"narrow": self.render(snapshot["pose"]), "wide": self.render(snapshot["pose"])}

    path = scene_file(tmp_path)
    data = json.loads(path.read_text())
    data.update(scene())
    data["wide_camera"] = {}
    for key in ("checkpoint", "calibration", "vehicle_layer"):
        asset = tmp_path / key
        asset.write_bytes(key.encode())
        data["wide_camera"][key] = str(asset)
    path.write_text(json.dumps(data))
    app = simulator_app(
        tmp_path / "run",
        path,
        renderer=PoseRendererLease(Paired(), Course(data).start, Course(data).start),
        policy=RealCoursePolicy(),
        runtime_factory=runtime_factory(data, Coverage(False, 0.002), western_score=western),
    )
    app.state.settings.patch({"road_camera_mode": "narrow_and_wide", "speed_limit_mps": 3.5})
    with http_server(app) as producer, http_server(gateway_app(producer)) as gateway:
        async with AsyncExitStack() as stack, httpx.AsyncClient(base_url=gateway) as http:
            client = await connect(stack, gateway, tmp_path)
            first = await client.call_tool("observe", {})
            assert summary(first)["cameras"] == ["narrow", "wide"]
            assert len(first.content) == 3
            assert "western" not in summary(first)
            assert (
                summary(
                    await client.call_tool(
                        "set_motion",
                        dict(direction="straight", steering_percent=0, speed_mps=0.8, duration_s=5),
                    )
                )["status"]
                == "accepted"
            )
            await wait_status(http, lambda value: value.get("speed_mps", 0) > 0.2, timeout=5)
            await client.call_tool("stop_now", {})
            await wait_status(http, lambda value: value.get("held"), timeout=6)
            assert app.state.sim_native.end_reason is None
            assert app.state.sim_episode.summary()["strict_score"]["failed"]
            calls = app.state.recording.history()
    assert producer_paired_observations(tmp_path / "run/episode/events.jsonl") == 1
    directory = tmp_path / "run"
    (directory / "drive-calls.json").write_text(json.dumps(calls))
    (directory / "client").mkdir()
    row = dict(
        type="item.completed",
        item=dict(
            type="mcp_tool_call",
            id="one",
            server="drivingbench_sandbox",
            tool="observe",
            arguments={},
            status="completed",
            result=first.model_dump(mode="json"),
        ),
    )
    (directory / "client/stdout.jsonl").write_text(json.dumps(row) + "\n")
    audit = audit_attempt(directory, SimpleNamespace(state_dir=tmp_path))
    assert received_pairs(audit["receipt"]) == 1
    for frame in audit["receipt"]["observations"][0]["frames"]:
        assert frame["captured_coverage"]["map_pose"] == [-10, 0, 0]
        assert not frame["captured_coverage"]["measurement"]["supported"]
        assert frame["captured_coverage"]["outside_reviewed"] is True
        assert (
            frame["captured_coverage"]["measurement"]["distance_to_any_paired_pass_parent_m"]
            == 0.002
        )
        assert frame["exact"]
    assert "captured_coverage" not in summary(first)
    assert audit["response_timing"] == {"narrow": None, "wide": None}


def test_received_pair_count_requires_two_verified_distinct_cameras():
    from drivingbench.sim.real_course_study.batch import received_pairs

    def frame(camera, accepted=True):
        return dict(camera=camera, accepted=accepted)

    receipt = {
        "observations": [
            {"frames": [frame("narrow"), frame("wide")]},
            {"frames": [frame("narrow"), frame("wide", False)]},
            {"frames": [frame("narrow"), frame("narrow")]},
            {"frames": [frame("narrow"), frame("wide")], "error": "ambiguous"},
        ]
    }
    assert received_pairs(receipt) == 1


def test_checkpoint_retains_provider_failures_but_stops_pipeline_faults():
    reports = [
        dict(model=model, status="complete", paired_observations=1) for model in ("astra", "sol")
    ]
    reports += [
        dict(model="fable", status="failed_capacity", paired_observations=0),
        dict(model="grok", status="failed_auth", paired_observations=0),
    ]
    assert checkpoint(reports)["passed"]
    reports[0]["paired_observations"] = 0
    assert not checkpoint(reports)["passed"]
    reports[0].update(status="infrastructure_failure", paired_observations=1)
    assert not checkpoint(reports)["passed"]


def test_driving_provider_failure_does_not_hide_renderer_fault():
    from drivingbench.sim.real_course_study.batch import driving_status

    report = dict(
        session_id=None,
        failure_kind="model_client",
        client_errors=["quota exceeded"],
        episode={
            "outcome": "infrastructure_failure",
            "infrastructure_failure": {"source": "model_client"},
        },
    )
    assert driving_status(report) == "failed_capacity"
    report["episode"]["infrastructure_failure"]["source"] = "sim_render_error"
    assert driving_status(report) == "infrastructure_failure"


def test_each_attempt_hash_validator_rejects_later_source_changes(tmp_path):
    from drivingbench.sim.producer import file_hash
    from drivingbench.sim.real_course_study.batch import validate_hashes

    source = tmp_path / "source.py"
    source.write_text("before")
    declaration = {"hashes": {str(source): file_hash(source)}}
    validate_hashes(declaration)
    source.write_text("after")
    with pytest.raises(ValueError, match="Frozen input changed"):
        validate_hashes(declaration)


def test_client_done_keeps_braking_ticks_until_held_parking_dwell(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from drivingbench.sim.car import SimNative
    from drivingbench.sim.episode import Episode
    from drivingbench.sim.validate import request
    from test_sim_runtime import Clock, move, queued

    data = scene()
    data["start"]["x"] = -2
    clock = Clock()
    factory = runtime_factory(
        data,
        Coverage(),
        native_factory=lambda **kw: SimNative(speed=0.8, hold=False, **kw),
        western_score=western,
    )
    runtime = factory(Course(data), clock=clock, frame_clock=clock)
    episode = Episode(runtime, tmp_path / "episode", RealCoursePolicy(), metadata={})
    try:
        with ThreadPoolExecutor() as pool:
            stop = request(runtime.native.bridge.controller, {"tool": "stop_now", "arguments": {}})
            future = queued(runtime, pool, stop)
            move(runtime, clock, 0.01)
            assert future.result()["status"] == "accepted"
        assert runtime.pose_snapshot()["speed_mps"] > 0.05
        runtime.finish("client_done")
        assert runtime.pending_end is None and runtime.end_reason is None
        move(runtime, clock, 5.1)
        assert episode.summary()["outcome"] == "success"
        assert runtime.end_reason == "finish"
        assert episode.summary()["first_cause"]["evidence"]["held"]
        assert not episode.summary()["first_cause"]["evidence"]["positive_motion_command"]
    finally:
        runtime.close()
        episode.close()


def test_cleanup_is_bounded_cannot_extend_or_admit_motion_and_failure_wins():
    from concurrent.futures import ThreadPoolExecutor

    from drivingbench.sim.real_course_study.runtime import PARKING_CONFIRMATION_S
    from test_sim_runtime import Clock, motion, move, queued

    clock = Clock()
    runtime = runtime_factory(scene(), Coverage(), western_score=western)(
        Course(scene()), clock=clock, frame_clock=clock
    )
    try:
        runtime.finish("client_done")
        move(runtime, clock, 2)
        runtime.finish("client_done")
        with ThreadPoolExecutor() as pool:
            future = queued(runtime, pool, motion(runtime))
            move(runtime, clock, 0.01)
            assert future.result() == {"status": "rejected", "reason": "operator_brake"}
        move(runtime, clock, PARKING_CONFIRMATION_S - 2 + 0.01)
        assert runtime.end_reason == "client_done"
    finally:
        runtime.close()
    runtime = runtime_factory(scene(), Coverage(), western_score=western)(
        Course(scene()), clock=clock, frame_clock=clock
    )
    try:
        runtime.finish("client_done")
        runtime.finish("infrastructure_failure")
        move(runtime, clock, 0.01)
        assert runtime.end_reason == "infrastructure_failure"
    finally:
        runtime.close()


def test_launch_wait_in_parking_does_not_count_as_a_held_finish(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from drivingbench.sim.car import SimNative
    from drivingbench.sim.episode import Episode
    from test_sim_runtime import Clock, motion, move, queued

    data = scene()
    data["start"]["x"] = -2
    clock = Clock()
    runtime = runtime_factory(
        data,
        Coverage(),
        native_factory=lambda **kw: SimNative(hold=True, **kw),
        western_score=western,
    )(Course(data), clock=clock, frame_clock=clock)
    episode = Episode(runtime, tmp_path / "episode", RealCoursePolicy(), metadata={})
    try:
        with ThreadPoolExecutor() as pool:
            future = queued(runtime, pool, motion(runtime))
            move(runtime, clock, 0.01)
            assert future.result()["status"] == "accepted"
        move(runtime, clock, 1.1)
        assert runtime.pose_snapshot()["held"]
        assert runtime.pose_snapshot()["positive_motion_command"]
        assert episode.summary()["outcome"] != "success"
        assert runtime.end_reason is None
    finally:
        runtime.close()
        episode.close()
