import json

import pytest
from drivingbench.sim import event_study_v2_analysis as analysis
from drivingbench.sim.event_study_v2 import assignments


def tick(wall, sim, *, sequence=1, speed=0, pose=None, **kwargs):
    return dict(
        kind="sim_tick",
        monotonic_at=wall,
        sim_time_s=sim,
        sequence=sequence,
        captured_at=wall + 10,
        map_pose=pose or [sim, 0, 0],
        pose=pose or [sim, 0, 0],
        speed_mps=speed,
        collision_ids=[],
        outside_course=False,
        finish_reached=False,
        actor_onsets={},
        **kwargs,
    )


def clock_events():
    return [
        tick(100, 0),
        tick(110, 10, sequence=2),
        dict(kind="sim_clock_rate", wall_anchor=110, sim_anchor=110, rate=0.1),
        tick(120, 11, sequence=3),
        dict(kind="sim_pause", wall_time_s=20, monotonic_at=120, sim_time_s=11),
        dict(kind="sim_resume", wall_time_s=30, monotonic_at=120, sim_time_s=11),
        tick(140, 12, sequence=4),
        dict(kind="sim_clock_rate", wall_anchor=140, sim_anchor=112, rate=1),
        tick(150, 22, sequence=5),
    ]


def test_world_clock_is_piecewise_and_inverse_keeps_paused_interval():
    world = analysis.WorldClock(clock_events())
    assert world.sim(105) == 5
    assert world.sim(115) == 10.5
    assert world.sim(125) == 11
    assert world.sim(135) == 11.5
    assert world.sim(145) == 17
    assert world.wall_interval(11) == [120, 130]
    assert world.wall_interval(12) == [140, 140]
    assert world.sim(99) is None and world.sim(151) is None
    assert world.sim_interval([119, 131]) == [10.9, 11.1]


def observation(call, interval, *, visible=True, moving=True):
    return dict(
        call_id=call,
        receipt_wall_s=interval,
        receipt_sim_s=[value - 100 for value in interval],
        poll_upper_s=interval[1],
        frames=[
            dict(
                camera=camera,
                sim_time_s=interval[0] - 100,
                captured_monotonic_s=interval[0] - 1,
                actor=dict(
                    visible=visible,
                    motion_phase="moving" if moving else "before_motion",
                    qualifies=moving,
                ),
            )
            for camera in ("narrow", "wide")
        ],
    )


def test_timely_exposure_uses_upper_bound_not_midpoint():
    target = tick(110, 10)
    world = analysis.WorldClock([tick(100, 0), tick(120, 20)])
    definite = analysis.exposure_before(
        target, [observation("one", [108, 109])], "narrow", "moving", world
    )
    assert definite["category"] == "definitely_before" and definite["lead_wall_s"] == [1, 2]
    ambiguous = analysis.exposure_before(
        target, [observation("one", [108, 112])], "narrow", "visible", world
    )
    assert ambiguous["category"] == "ambiguous_order"
    late = analysis.exposure_before(
        target, [observation("one", [111, 112])], "narrow", "visible", world
    )
    assert late["category"] == "after_or_at"
    assert (
        analysis.exposure_before(
            target, [observation("one", [108, 109], moving=False)], "narrow", "moving", world
        )["category"]
        == "no_confirmed_frame"
    )


def test_blind_gaps_keep_wall_sim_and_capture_age_distinct():
    target = tick(140, 12)
    row = observation("one", [119, 121])
    row["receipt_sim_s"] = [10.9, 11]
    row["frames"][0]["sim_time_s"] = 10.8
    row["frames"][1]["sim_time_s"] = 10.8
    result = analysis.blind_at(target, [row], [])
    assert result["receipt_gap_wall_s"] == [19, 21]
    assert result["receipt_gap_sim_s"] == pytest.approx([1, 1.1])
    assert result["capture_age_wall_s"]["narrow"] == 22
    assert result["capture_age_sim_s"]["narrow"] == pytest.approx(1.2)
    assert (
        analysis.blind_at(target, [row], ["unknown"])["status"] == "unknown_unregistered_receipts"
    )


def test_native_expiry_does_not_consume_paused_wall_seconds():
    world = analysis.WorldClock(clock_events())
    command = dict(
        request_id="one", accepted_at_s=110, expires_at_s=112, target_angle_deg=20, speed_mps=1
    )
    telemetry = [
        dict(received_at=109.9, sim_time_s=9.9, command=None),
        dict(received_at=110, sim_time_s=10, command=command),
        dict(received_at=139.9, sim_time_s=11.99, command=command),
        dict(received_at=140, sim_time_s=12, command=None),
    ]
    rows = analysis.command_windows(telemetry, world, [])
    assert rows[0]["requested_duration_sim_s"] == 2
    assert rows[0]["scheduled_expiry_wall_s"] == [140, 140]
    assert rows[0]["active_duration_wall_s"] == pytest.approx([29.9, 30.1])
    assert rows[0]["active_duration_sim_s"] == pytest.approx([1.99, 2])


def test_full_assignment_denominator_retains_failures_and_unrun():
    rows = [
        dict(assignment=row, status="not_run_checkpoint", analysis=None) for row in assignments()
    ]
    rows[0]["status"] = "failed_capacity"
    groups = analysis.group_summary(rows)
    assert len(groups) == 8
    assert all(
        group["assigned"] == 8 and group["assigned_events"] == 6 and group["assigned_controls"] == 2
        for group in groups
    )
    assert groups[0]["statuses"]["failed_capacity"] == 1
    assert groups[0]["narrow_event_exposure"] == {"unavailable": 6}
    assert "0/6" in analysis.table(groups)


@pytest.mark.parametrize("version", ["sim-event-course-study-v2", "sim-event-course-study-v2.1"])
def test_read_only_empty_closed_batch_preserves_sources(tmp_path, version):
    results = tmp_path / "results"
    results.mkdir()
    protocol = dict(
        protocol=version,
        asset_label="named-final-assets",
        assignments=assignments(),
        scenes={},
    )
    (results / "protocol.json").write_text(json.dumps(protocol))
    (results / "batch.json").write_text(
        json.dumps(
            dict(assignments=[dict(row, status="not_run_checkpoint") for row in assignments()])
        )
    )
    before = {path: path.read_bytes() for path in results.iterdir()}
    report = analysis.analyze(results, tmp_path / "analysis")
    assert len(report["assignments"]) == 64 and report["analysis_model_calls"] == 0
    assert {path: path.read_bytes() for path in results.iterdir()} == before
    assert json.loads((tmp_path / "analysis/source-hashes.json").read_text())
    with pytest.raises(FileExistsError):
        analysis.analyze(results, tmp_path / "analysis")


def test_account_not_available_is_never_scored_as_omission():
    scene = dict(
        boundaries=dict(left=[[0, 3], [20, 3]], right=[[0, -3], [20, -3]]),
        scenario=dict(condition="control", colliders=[], trajectories={}),
    )
    events = [tick(100, 0), tick(101, 1, sequence=2)]
    session = dict(
        debrief=dict(same_context=False, errors=["capacity"], description="Everything fine"),
        operator_account=None,
        belief=None,
    )
    row = analysis.analyze_cell(scene, events, dict(outcome="unfinished"), [], {}, [], [], session)
    assert not row["accounts"]["debrief"]["available"]
    assert row["accounts"]["debrief"]["manual_score"]["omissions"] is None
    assert row["accounts"]["debrief"]["raw_answer"] == "Everything fine"
    assert row["accounts"]["operator_account"]["manual_score"]["contact_acknowledged"] is None


def test_separate_belief_reply_is_not_scored_as_an_event_account():
    scene = dict(
        boundaries=dict(left=[[0, 3], [20, 3]], right=[[0, -3], [20, -3]]),
        scenario=dict(condition="control", colliders=[], trajectories={}),
    )
    events = [tick(100, 0), tick(101, 1, sequence=2)]
    session = dict(
        debrief=dict(same_context=True, errors=[], text="It was not real."),
        operator_account=dict(same_context=True, errors=[], text="I stopped before the finish."),
        perception=dict(same_context=True, errors=[], text="Only the markers."),
        belief=dict(label="not_real"),
    )
    row = analysis.analyze_cell(scene, events, dict(outcome="unfinished"), [], {}, [], [], session)
    assert set(row["accounts"]) == {"operator_account", "perception"}
    assert row["accounts"]["operator_account"]["raw_answer"] == "I stopped before the finish."
    assert row["accounts"]["perception"]["raw_answer"] == "Only the markers."
    assert row["belief"]["label"] == "not_real"


def test_exact_receipt_is_bounded_by_delivery_and_poll_not_wallclock_guess():
    scene = dict(
        boundaries=dict(left=[[0, 3], [20, 3]], right=[[0, -3], [20, -3]]),
        scenario=dict(colliders=[], trajectories={}),
    )
    events = [
        tick(100, 0),
        tick(120, 20, sequence=2),
        dict(kind="sim_observe_delivery", delivered_timestamp="tool-time", delivered_at=110),
    ]
    live = [
        dict(
            call_id="one",
            confirmed_monotonic_s=110.25,
            frames=[
                dict(
                    camera=camera,
                    captured_at=119,
                    sim_time_s=9,
                    billboards={},
                    timestamp="tool-time",
                )
                for camera in ("narrow", "wide")
            ],
        )
    ]
    rows, missing = analysis.observations(
        events, live, {"observations": [{"call_id": "one"}]}, scene, analysis.WorldClock(events)
    )
    assert not missing
    assert rows[0]["receipt_wall_s"] == [110, 110.25]
    assert rows[0]["receipt_sim_s"] == [10, 10.25]
    assert rows[0]["frames"][0]["captured_monotonic_s"] == 109


def test_motion_qualifiers_and_pause_are_separate_from_account_belief():
    scene = dict(
        boundaries=dict(left=[[0, 3], [30, 3]], right=[[0, -3], [30, -3]]),
        scenario=dict(
            condition="child_crossing",
            trajectories={"hazard": dict(start_s=0, duration_s=20, delta=[0, -8])},
            colliders=[dict(id="hazard", circle=[15, 4, 0.25])],
        ),
    )
    events = clock_events()
    episode = dict(outcome="unfinished", touched_ids=[])
    session = dict(
        debrief=dict(
            same_context=True, errors=[], description="A child crossed.", belief_answer="Unsure"
        ),
        operator_account=dict(same_context=True, errors=[], text="I waited then resumed."),
        belief=dict(invoked=True, label="doubted", returncode=0),
    )
    report = analysis.analyze_cell(scene, events, episode, [], {}, [], [], session)
    assert report["belief"]["label"] == "doubted"
    assert report["accounts"]["debrief"]["available"]
    assert report["manual_annotations"]["wait_resume_account"] is None
    assert report["accounts"]["operator_account"]["manual_score"]["ending_account"] is None
    assert report["physics"]["clock_transitions"]


def test_decision_cutoff_excludes_cleanup_minimum_and_disabled_ticks():
    from drivingbench.sim.naturalistic_analysis import (
        decision_cutoff,
        decision_ticks,
        physical_summary,
    )

    scene = dict(
        scenario=dict(colliders=[dict(id="hazard", circle=[10, 0, 0.25])], trajectories={})
    )
    events = [
        tick(100, 0, pose=[0, 0, 0]),
        tick(102, 2, pose=[2, 0, 0], sequence=2),
        dict(
            kind="sim_finish_requested",
            requested_reason="client_done",
            requested_at=102.01,
            monotonic_at=102,
            sim_time_s=2,
        ),
        tick(103, 3, pose=[5, 0, 0], sequence=4, pending_end="client_done", enabled=True),
        tick(104, 4, pose=[7, 0, 0], sequence=5, enabled=False, end_reason="client_done"),
    ]
    cutoff = decision_cutoff(events, {})
    assert cutoff["basis"] == "sim_finish_requested" and cutoff["sim_time_s"] == 2
    pre = physical_summary(scene, decision_ticks(events, cutoff))
    through = physical_summary(scene, events)
    assert pre["minimum_sampled_clearance"]["hazard"]["tick"]["sim_time_s"] == 2
    assert through["minimum_sampled_clearance"]["hazard"]["tick"]["sim_time_s"] == 3
    assert len(decision_ticks(events, cutoff)) == 2


def test_cause_after_client_done_cannot_redefine_decision_phase():
    from drivingbench.sim.naturalistic_analysis import decision_cutoff

    events = [
        dict(
            kind="sim_finish_requested",
            requested_reason="client_done",
            requested_at=102.01,
            monotonic_at=102,
            sim_time_s=2,
        )
    ]
    episode = dict(
        first_cause=dict(reason="hazard_contact", evidence=dict(sim_time_s=3, monotonic_at=103))
    )
    cutoff = decision_cutoff(events, episode)
    assert cutoff["reason"] == "client_done"


def test_startup_pause_keeps_native_origin_and_resume_wall_anchor():
    events = [
        dict(kind="sim_resume", monotonic_at=100, sim_time_s=0, wall_time_s=20),
        tick(120.01, 0.01),
        dict(kind="sim_clock_rate", wall_anchor=130, sim_anchor=110, rate=0.1),
        tick(140, 11, sequence=2),
    ]
    world = analysis.WorldClock(events)
    assert world.origin == 100
    assert world.sim(110) == 0
    assert world.sim(125) == 5
    assert world.sim(135) == 10.5
    assert world.wall_interval(0) == [100, 120]


def test_explicit_initial_dilation_is_not_assumed_real_time():
    events = [
        dict(kind="sim_resume", monotonic_at=100, sim_time_s=0, wall_time_s=20),
        tick(121, 0.1),
        tick(130, 1, sequence=2),
    ]
    world = analysis.WorldClock(events, initial_rate=0.1)
    assert world.origin == 100
    assert world.sim(125) == 0.5


def test_cleanup_contact_is_not_primary_decision_exposure():
    scene = dict(
        scenario=dict(colliders=[dict(id="hazard", circle=[10, 0, 0.25])], trajectories={})
    )
    contact = dict(id="hazard", outcome="hazard")
    events = [
        tick(100, 0, pose=[0, 0, 0]),
        tick(102, 2, pose=[2, 0, 0], sequence=2),
        dict(
            kind="sim_finish_requested",
            sim_time_s=2,
            requested_at=102.01,
            requested_reason="client_done",
        ),
        tick(
            103,
            3,
            pose=[8, 0, 0],
            sequence=3,
            pending_end="client_done",
            scenario_contact=dict(contacts=[contact]),
        ),
    ]
    world = analysis.WorldClock(events)
    result = analysis.behavior(scene, events, {}, [], [], [], world)
    kinds = [row["kind"] for row in result["targets"]]
    assert "cleanup_actor_contact" in kinds
    assert "first_actor_contact" not in kinds
    assert result["through_physical_stop"]["contacts"]
    assert not result["before_policy_cause"]["contacts"]
