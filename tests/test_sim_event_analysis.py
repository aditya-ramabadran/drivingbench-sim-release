import base64
import hashlib
import json

import pytest
from drivingbench.sim.event_analysis import (
    _output_identity,
    account_review,
    command_windows,
    encounter_exposure,
    qualifies,
    registered_observations,
    summarize_attempt,
)


def scene():
    return dict(
        frame="site1_full_ground",
        start=dict(x=0, y=0, heading_deg=0),
        boundaries=dict(left=[[-10, 3], [20, 3]], right=[[-10, -3], [20, -3]]),
        markers=[],
        finish=dict(polygon=[[10, -2], [19, -2], [19, 2], [10, 2]]),
        scenario=dict(
            trajectories={"hazard": dict(duration_s=10, delta=[0, 10])},
            colliders=[dict(id="hazard", circle=[5, -5, 0.3])],
        ),
    )


def observation(identity="a", receipt=119, time=4, pixels=5):
    return dict(
        call_id=identity,
        received_utc_s=receipt,
        frames=[
            dict(
                camera=camera,
                frame_id=identity + camera,
                captured_monotonic_s=time,
                sim_time_s=time,
                actor_onsets={"hazard": 1},
                billboards={"hazard": dict(visible_pixels=pixels)},
            )
            for camera in ("narrow", "wide")
        ],
    )


def test_qualification_keeps_roadside_before_motion_distinct():
    s = scene()
    frame = observation(time=0)["frames"][0]
    assert not qualifies(s, "hazard", frame)
    frame["sim_time_s"] = 2
    assert qualifies(s, "hazard", frame)
    frame["billboards"]["hazard"]["visible_pixels"] = 0
    assert not qualifies(s, "hazard", frame)
    frame = observation(time=11)["frames"][0]
    assert not qualifies(s, "hazard", frame)  # Finished on the opposite roadside.
    s["scenario"]["trajectories"]["hazard"]["delta"] = [0, 5]
    assert qualifies(s, "hazard", frame)  # Finished in the corridor.


def test_primary_before_encounter_is_not_one_second_or_stopping_claim():
    result = encounter_exposure(
        scene(),
        "hazard",
        dict(monotonic_at=20),
        [observation(receipt=119.8)],
        {"offset_s": [100, 100]},
    )
    assert result["category"] == "received_before_encounter"
    assert result["receipt_lead_s"] == pytest.approx([0.2, 0.2])
    assert result["one_second_response_window"] is False
    assert result["receipt_before_encounter"] is True
    assert result["blind"]["capture_age_s"] == {"narrow": 16, "wide": 16}


def test_ambiguous_unknown_and_no_exposure_never_count_as_timely():
    target = dict(monotonic_at=20)
    clock = {"offset_s": [99, 101]}
    result = encounter_exposure(scene(), "hazard", target, [observation(receipt=120)], clock)
    assert result["category"] == "ambiguous_receipt_order"
    assert result["receipt_before_encounter"] is None
    assert (
        encounter_exposure(scene(), "hazard", target, [observation(receipt=None)], clock)[
            "category"
        ]
        == "unknown_timing"
    )
    assert (
        encounter_exposure(scene(), "hazard", target, [observation(pixels=0)], clock)["category"]
        == "no_received_qualifying_frame"
    )
    rows = [observation(receipt=None), observation("late", receipt=122)]
    assert (
        encounter_exposure(scene(), "hazard", target, rows, clock)["category"]
        == "ambiguous_receipt_order"
    )


def test_accepted_server_response_before_encounter_excludes_rejections_and_late_calls():
    calls = [
        dict(tool="stop_now", timestamp="1970-01-01T00:01:55Z", outcome=dict(status="rejected")),
        dict(tool="set_motion", timestamp="1970-01-01T00:01:56Z", outcome=dict(status="accepted")),
        dict(tool="stop_now", timestamp="1970-01-01T00:02:00Z", outcome=dict(status="stopping")),
    ]
    result = encounter_exposure(
        scene(),
        "hazard",
        dict(monotonic_at=20),
        [observation(receipt=115)],
        {"offset_s": [100, 100]},
        calls=calls,
    )
    assert result["accepted_response_before_encounter"] == calls[1]
    assert result["receipt_to_first_accepted_server_record_s"] == 1


def test_codex_receipt_requires_exact_unique_outer_output():
    raw = b"jpeg"
    status = dict(timestamp="2026-01-01T00:00:00Z")
    payload = dict(
        output=[
            dict(type="input_text", text=json.dumps(status)),
            dict(
                type="input_image",
                image_url="data:image/jpeg;base64," + base64.b64encode(raw).decode(),
            ),
        ]
    )
    assert _output_identity(payload) == (status, [hashlib.sha256(raw).hexdigest()])
    receipt = dict(
        observations=[
            dict(
                call_id="a",
                event_index=1,
                frames=observation()["frames"],
                client_timestamp="1970-01-01T00:01:55Z",
            )
        ]
    )
    decoded = [dict(call_id="a", status=status, images=[raw])]
    outer = [
        dict(
            identity=_output_identity(payload),
            timestamp="1970-01-01T00:01:57Z",
            source=dict(path="raw", line=1),
        )
    ]
    rows = registered_observations(receipt, [{}], outer, "astra", 0, decoded)
    assert rows[0]["received_utc_s"] == 117
    assert (
        registered_observations(receipt, [{}], outer * 2, "astra", 0, decoded)[0]["received_utc_s"]
        is None
    )
    assert (
        registered_observations(receipt, [{}], [], "astra", 0, decoded)[0]["received_utc_s"] is None
    )


def test_command_replacement_and_episode_cutoff_are_bounded():
    first = dict(request_id="a", accepted_at_s=10, expires_at_s=30)
    second = dict(request_id="b", accepted_at_s=12, expires_at_s=20)
    telemetry = [
        dict(sim_time_s=t, received_at=t, command=c)
        for t, c in [(10, first), (11, first), (12, second), (13, second)]
    ]
    commands = command_windows(telemetry, 14)
    assert commands[0]["end_monotonic_s"] == 12
    assert commands[0]["early_clearance_interval_s"] == [11, 12]
    assert commands[1]["end_monotonic_s"] == 14


def test_physical_terminal_freeze_and_missing_account_are_not_outcomes():
    events = [
        dict(
            kind="sim_tick",
            sequence=t,
            sim_time_s=t,
            monotonic_at=t,
            map_pose=[0, 0, 0],
            speed_mps=0,
            actor_onsets={},
        )
        for t in (0, 1, 3)
    ]
    events.append(dict(kind="sim_terminal_state", sim_time_s=2))
    result = summarize_attempt(scene(), events, {}, dict(observations=[]), [], [], [], [])
    assert result["through_physical_stop"]["last_tick"]["sim_time_s"] == 1
    assert result["clock"]["status"] == "no_anchor"
    review = account_review(
        None,
        scene_condition="dog_crossing",
        physical=result["through_physical_stop"],
        episode={},
        opportunity=[],
    )
    assert review["available"] is False
    assert all(value is None for value in review["manual_score"].values())


def test_unverified_correspondence_cannot_prove_no_exposure_or_late_exposure():
    row = observation(receipt=122)
    row["receipt_verified"] = False
    target, clock = dict(monotonic_at=20), {"offset_s": [100, 100]}
    assert (
        encounter_exposure(scene(), "hazard", target, [row], clock)["category"]
        == "ambiguous_receipt_order"
    )
    row["frames"] = []
    assert (
        encounter_exposure(scene(), "hazard", target, [row], clock)["category"]
        == "unknown_image_correspondence"
    )


def test_closed_batch_retains_failed_and_checkpoint_unrun_cells(tmp_path):
    from drivingbench.sim.event_analysis import analyze

    results = tmp_path / "results"
    results.mkdir()
    assignments = [
        dict(
            id=f"model-attempt-{attempt}",
            model="astra",
            session="astra-A",
            attempt=attempt,
            condition="control",
        )
        for attempt in (1, 2, 3)
    ]
    (results / "protocol.json").write_text(json.dumps(dict(assignments=assignments)))
    statuses = ["infrastructure_failure", "failed_session_cell", "not_run_checkpoint"]
    (results / "batch.json").write_text(
        json.dumps(
            dict(
                assignments=[
                    dict(row, status=status)
                    for row, status in zip(assignments, statuses, strict=True)
                ]
            )
        )
    )
    rows = analyze(results, tmp_path / "closed-analysis")
    assert [row["status"] for row in rows] == statuses
    assert all(row["status_basis"] == "closed_batch_assignment" for row in rows)
    assert not any(row["attempt_artifact_available"] for row in rows)


def test_open_batch_without_attempt_record_stays_not_closed(tmp_path):
    from drivingbench.sim.event_analysis import analyze

    results = tmp_path / "results"
    results.mkdir()
    assignment = dict(id="a", model="astra", session="astra-A", attempt=1, condition="control")
    (results / "protocol.json").write_text(json.dumps(dict(assignments=[assignment])))
    rows = analyze(results, tmp_path / "partial-analysis")
    assert rows[0]["status"] == "not_closed"
    assert rows[0]["status_basis"] == "no_closed_record"
    folder = results / "astra-A/attempt-1"
    folder.mkdir(parents=True)
    (folder / "attempt.json").write_text(
        json.dumps(dict(status="infrastructure_failure", error="capacity"))
    )
    rows = analyze(results, tmp_path / "attempt-analysis")
    assert rows[0]["status"] == "infrastructure_failure"
    assert rows[0]["status_basis"] == "closed_attempt_record"
    assert rows[0]["attempt_artifact_available"] is True


def test_stop_acceptance_is_stopping_not_motion_accepted_status():
    call = dict(tool="stop_now", timestamp="1970-01-01T00:01:56Z", outcome=dict(status="stopping"))
    result = encounter_exposure(
        scene(),
        "hazard",
        dict(monotonic_at=20),
        [observation(receipt=115)],
        {"offset_s": [100, 100]},
        calls=[call],
    )
    assert result["accepted_response_before_encounter"] == call
