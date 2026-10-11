import copy
import io

import pytest
from drivingbench.sim.studies import pilot_exposure as exposure
from PIL import Image


def metrics():
    return dict(
        valid=True,
        projected_total_pixels=4000,
        in_frame_pixels=4000,
        visible_pixels=4000,
        visible_centroid_px=[672, 380],
        visible_height_px=120,
        in_frame_fraction=1,
        survival_fraction=1,
        edge_margin_px=40,
        region_survival=dict(head=1, body=1, feet=1),
    )


def fixture(pilot="A", *, receipt_s=3, cutoff_s=10, held=False, call_id="call-one"):
    """Synthetic native/source evidence; A/C legacy gates and D actual completion clocks."""
    scene = dict(
        design=dict(path=[[0, 0, 0], [40, 0, 0]]),
        scenario=dict(
            colliders=[dict(id="hazard", circle=[24, 4, 0.32])], trajectories={}, animations={}
        ),
    )
    if pilot != "D":
        scene["scenario"]["trajectories"]["hazard"] = dict(
            duration_s=100, delta=[0, 0], yield_policy={"kind": "wait_for_stopped_car"}
        )
    events = []
    for i in range(round(max(10, cutoff_s) * 100) + 1):
        t = i / 100
        tick = dict(
            kind="sim_tick",
            sequence=i,
            sim_time_s=t,
            monotonic_at=100 + t,
            map_pose=[10 if held else 2 * t, 0, 0],
            speed_mps=0 if held else 2,
            enabled=True,
            actor_onsets={} if pilot == "D" else {"hazard": 0.1},
        )
        if pilot != "D":
            tick["actor_states"] = {
                "hazard": dict(
                    time_s=t,
                    previous_time_s=max(0, t - 0.01),
                    elapsed_s=max(0, t - 0.1),
                    previous_elapsed_s=max(0, t - 0.11),
                    paused=False,
                    waiting_since_s=None,
                )
            }
        events.append(tick)
    events.append(
        dict(
            kind="sim_finish_requested",
            sim_time_s=cutoff_s,
            requested_at=100 + cutoff_s,
            requested_reason="client_done",
        )
    )
    snapshot = copy.deepcopy(events[round(receipt_s * 100)])
    frames = []
    for camera, letter in (("narrow", "a"), ("wide", "b")):
        digest = letter * 64
        identity = f"{call_id}-capture-{camera}"
        frames.append(
            dict(
                camera=camera,
                sha256=digest,
                producer_sha256=digest,
                received_sha256=digest,
                frame_id=identity,
                delivered_frame_id=identity + "-" + digest[:12],
                artifact_verified=True,
                captured_at=430000 + receipt_s,
                sim_time_s=receipt_s,
                actor_onsets=snapshot["actor_onsets"],
                **(
                    {"actor_states": copy.deepcopy(snapshot["actor_states"])}
                    if "actor_states" in snapshot
                    else {}
                ),
                timestamp=f"timestamp-{call_id}",
                accepted=True,
                producer_size=[1344, 760],
                received_size=[1344, 760],
                billboards={"hazard": {"whole_actor_visibility": metrics()}},
            )
        )
    worker = dict(
        kind="sim_frame",
        sim_time_s=receipt_s,
        captured_at=430000 + receipt_s,
        actor_onsets=snapshot["actor_onsets"],
        frame_ids={f["camera"]: f["frame_id"] for f in frames},
        worker={
            "billboards": copy.deepcopy(frames[0]["billboards"]),
            "wide": {"billboards": copy.deepcopy(frames[1]["billboards"])},
        },
    )
    if "actor_states" in snapshot:
        worker["actor_states"] = copy.deepcopy(snapshot["actor_states"])
    delivery = dict(
        kind="sim_observe_delivery",
        delivered_timestamp=f"timestamp-{call_id}",
        frames=[
            dict(
                camera=f["camera"],
                frame_id=f["delivered_frame_id"],
                sha256=f["sha256"],
                captured_at=f["captured_at"],
            )
            for f in frames
        ],
    )
    events.extend([worker, delivery])
    receipts = [dict(call_id=call_id, confirmed_monotonic_s=100 + receipt_s, frames=frames)]
    return dict(
        pilot=pilot,
        scene=scene,
        events=events,
        episode={},
        receipts=receipts,
        rule=exposure.rule_for(pilot),
        certificate={"no_response_conflict_delay_after_onset_s": 8.9},
    )


def d_fixture(*, receipt_s=6, cutoff_s=10, held=False, **options):
    """D-only actual completion/handback fields; generic static fixture stays reusable."""
    data = fixture("D", receipt_s=receipt_s, cutoff_s=cutoff_s, held=held, **options)
    data["scene"]["scenario"]["colliders"][0]["circle"] = [32, 4, 0.32]
    for tick in data["events"]:
        if tick["kind"] == "sim_tick":
            if held:
                tick["map_pose"] = [12, 0, 0]
            tick["pose"] = [tick["sim_time_s"] * 2, 0, 0]
            tick["native_completed_monotonic_s"] = 100 + tick["sim_time_s"]
    worker = next(event for event in data["events"] if event["kind"] == "sim_frame")
    worker["pose"] = [receipt_s * 2, 0, 0]
    finish = next(event for event in data["events"] if event["kind"] == "sim_finish_requested")
    index = data["events"].index(finish)
    data["events"][index:index] = [
        dict(
            kind="sim_client_handback_requested",
            sim_time_s=cutoff_s,
            requested_at=100 + cutoff_s,
            requested_reason="client_done",
            clock_basis="actual_host_finish_entry",
        ),
        dict(kind="sim_parking_confirmation", sim_time_s=cutoff_s),
    ]
    finish.update(sim_time_s=cutoff_s + 5, requested_at=105 + cutoff_s)
    return data


def result(data):
    return exposure.qualify(**data)


def candidate(data):
    return result(data)["candidates"][0]


def sync_worker(data):
    frames = data["receipts"][0]["frames"]
    worker = next(e for e in data["events"] if e["kind"] == "sim_frame")
    worker["worker"] = {
        "billboards": copy.deepcopy(frames[0]["billboards"]),
        "wide": {"billboards": copy.deepcopy(frames[1]["billboards"])},
    }
    for k in ("actor_states", "actor_onsets"):
        if k in frames[0]:
            worker[k] = copy.deepcopy(frames[0][k])
        else:
            worker.pop(k, None)


@pytest.mark.parametrize("pilot", ["A", "C", "D"])
def test_actual_received_opportunity_without_capture_wall_clock_comparison(pilot):
    data = d_fixture() if pilot == "D" else fixture(pilot)
    before = copy.deepcopy(data)
    report = result(data)
    assert report["candidates"][0]["normal"]["eligible"]
    assert data == before
    assert (
        data["receipts"][0]["frames"][0]["captured_at"]
        > data["receipts"][0]["confirmed_monotonic_s"]
    )
    assert report["approach_crossing_sim_s"] == (9 if pilot == "A" else None)
    if pilot == "D":
        assert report["candidates"][0]["captured_body_station_m"] == 12
        assert not report["candidates"][0]["safe_held"]["eligible"]
    assert report["counterfactual_conflict_sim_s"] == (9 if pilot == "C" else None)


@pytest.mark.parametrize("received,passed", [(7, True), (7.01, False)])
@pytest.mark.parametrize("pilot", ["A", "C"])
def test_exact_two_second_lead(pilot, received, passed):
    row = candidate(fixture(pilot, receipt_s=received))
    assert row["normal"]["eligible"] is passed


def test_first_crossing_does_not_reopen_after_reversal():
    data = fixture(receipt_s=8)
    for t in data["events"]:
        if t["kind"] == "sim_tick":
            time = t["sim_time_s"]
            t["map_pose"][0] = 4 * time if time <= 5 else 20 - 2 * (time - 5)
    report = result(data)
    assert report["approach_crossing_sim_s"] == 4.5
    assert not report["candidates"][0]["eligible"]


@pytest.mark.parametrize("pilot", ["A", "C"])
def test_finite_safe_held_branch_without_invented_plane(pilot):
    data = fixture(pilot, held=True, receipt_s=3, cutoff_s=5)
    if pilot == "C":
        data["certificate"]["no_response_conflict_delay_after_onset_s"] = 1
    report = result(data)
    row = report["candidates"][0]
    assert report["approach_crossing_sim_s"] is None
    assert not row["normal"]["eligible"] and row["safe_held"]["eligible"]
    assert row["safe_held"]["observed_duration_s"] == 2
    assert row["safe_held"]["plane_lead_s"] is None
    if pilot == "C":
        assert report["counterfactual_conflict_sim_s"] == 1.1


@pytest.mark.parametrize(
    "change,reason",
    [
        ("short", "incomplete_predecision_held_horizon"),
        ("motion", "not_held_throughout_horizon"),
        ("gap", "insufficient_actual_held_gap"),
        ("contact", "unsafe_held_horizon"),
        ("missing_tick", "missing_native_samples_in_held_horizon"),
    ],
)
def test_safe_hold_requires_entire_observed_horizon(change, reason):
    data = fixture("C", held=True, receipt_s=3, cutoff_s=5)
    data["certificate"]["no_response_conflict_delay_after_onset_s"] = 1
    if change == "short":
        next(e for e in data["events"] if e["kind"] == "sim_finish_requested")["sim_time_s"] = 4.99
    if change == "gap":
        data["scene"]["scenario"]["colliders"][0]["circle"] = [10, 0, 0.32]
    tick = next(e for e in data["events"] if e["kind"] == "sim_tick" and e["sim_time_s"] == 4)
    if change == "motion":
        tick["speed_mps"] = 0.05
    if change == "contact":
        tick["scenario_contact"] = {"id": "hazard"}
    if change == "missing_tick":
        data["events"].remove(tick)
    row = candidate(data)
    assert not row["eligible"] and reason in row["safe_held"]["reasons"]


def test_dome_score_is_not_an_unsafe_ending_or_fictitious_D_conflict():
    data = d_fixture(held=True, receipt_s=6, cutoff_s=9)
    for e in data["events"]:
        if e["kind"] == "sim_tick":
            e["collision_ids"] = ["dome"]
    assert candidate(data)["normal"]["eligible"]
    assert not candidate(data)["safe_held"]["eligible"]
    assert result(data)["counterfactual_conflict_sim_s"] is None


def test_cleanup_braking_and_late_receipts_cannot_qualify():
    data = fixture(held=True, receipt_s=6, cutoff_s=5)
    row = candidate(data)
    assert not row["eligible"] and "receipt_not_bracketed_before_decision" in row["reasons"]
    data = fixture(held=True, receipt_s=4, cutoff_s=5)
    assert not candidate(data)["safe_held"]["eligible"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ("proof", "unverified_received_artifact"),
        ("frame_id", "missing_or_invalid_frame_identity"),
        ("digest", "invalid_frame_digest"),
        ("skew", "unsynchronized_camera_pair"),
        ("nan", "invalid_capture_time"),
        ("metrics", "narrow_not_whole_actor_clear"),
        ("source_metrics", "capture_metrics_mismatch"),
        ("source_absent", "missing_or_ambiguous_source_frame"),
    ],
)
def test_narrow_pair_and_source_proof_fail_closed(change, reason):
    data = fixture()
    n, w = data["receipts"][0]["frames"]
    if change == "proof":
        n["artifact_verified"] = False
    if change == "frame_id":
        n.pop("frame_id")
    if change == "digest":
        n["received_sha256"] = "wrong"
    if change == "skew":
        w["captured_at"] += 0.01
    if change == "nan":
        n["captured_at"] = float("nan")
    if change in ("metrics", "source_metrics"):
        n["billboards"]["hazard"]["whole_actor_visibility"]["region_survival"]["feet"] = 0.1
        if change == "metrics":
            sync_worker(data)
    if change == "source_absent":
        data["events"] = [e for e in data["events"] if e["kind"] != "sim_frame"]
    row = candidate(data)
    assert not row["eligible"] and reason in row["reasons"]
    assert w["billboards"]["hazard"]["whole_actor_visibility"]["region_survival"]["feet"] == 1


def test_guarded_child_clock_and_actual_yield_phase():
    data = fixture("C")
    for frame in data["receipts"][0]["frames"]:
        state = frame["actor_states"]["hazard"]
        state.update(paused=True, waiting_since_s=2)
    sync_worker(data)
    assert candidate(data)["actor_phase"] == "yielding"
    assert candidate(data)["normal"]["eligible"]
    for frame in data["receipts"][0]["frames"]:
        frame.pop("actor_states")
    sync_worker(data)
    assert "missing_or_inconsistent_capture_actor_clock" in candidate(data)["reasons"]


def test_before_onset_exited_and_invalid_clock_do_not_count():
    for label in ("before", "exited", "nan"):
        data = fixture("C")
        if label == "before":
            for frame in data["receipts"][0]["frames"]:
                frame["actor_onsets"]["hazard"] = 4
        if label == "exited":
            data["scene"]["scenario"]["trajectories"]["hazard"]["duration_s"] = 1
        if label == "nan":
            for frame in data["receipts"][0]["frames"]:
                frame["actor_states"]["hazard"]["elapsed_s"] = float("nan")
        sync_worker(data)
        assert not candidate(data)["eligible"]


def test_no_response_clock_is_not_extended_by_holds_and_rule_is_explicit():
    data = fixture("C", held=True, receipt_s=3, cutoff_s=5)
    data["certificate"]["no_response_conflict_delay_after_onset_s"] = 1
    report = result(data)
    assert report["counterfactual_conflict_sim_s"] == 1.1
    assert not report["candidates"][0]["normal"]["eligible"]
    for broken in ({}, dict(exposure.RULE, adult_height_px=50)):
        data["rule"] = broken
        with pytest.raises(ValueError):
            result(data)
    data["rule"] = exposure.RULE
    data["certificate"] = {}
    with pytest.raises(ValueError):
        result(data)


def jpeg(size=(1344, 760)):
    stream = io.BytesIO()
    Image.new("RGB", size, "#888888").save(stream, format="JPEG", quality=92)
    return stream.getvalue()


def artifact_fixture(tmp_path, *, transcode=False, pilot="C", **fixture_options):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = d_fixture(**fixture_options) if pilot == "D" else fixture(pilot, **fixture_options)
    observations = []
    frames = []
    delivery = next(e for e in data["events"] if e["kind"] == "sim_observe_delivery")
    for i, frame in enumerate(data["receipts"][0]["frames"]):
        producer = jpeg()
        received = jpeg((1024, 579)) if transcode else producer
        p = tmp_path / f"p{i}.jpg"
        r = tmp_path / f"r{i}.jpg"
        p.write_bytes(producer)
        r.write_bytes(received)
        match = exposure.image_correspondence(received, producer)
        digest = match["producer_sha256"]
        frame.update(match, sha256=digest, delivered_frame_id=frame["frame_id"] + "-" + digest[:12])
        delivery["frames"][i].update(frame_id=frame["delivered_frame_id"], sha256=digest)
        proof = dict(
            frame,
            received_path=str(r),
            producer_path=str(p),
            delivered_sha256=digest,
            delivered_timestamp=frame["timestamp"],
        )
        proof.pop("actor_states", None)
        frames.append(proof)
        frame.pop("artifact_verified")
        frame.pop("frame_id")
        frame.pop("delivered_frame_id")
    observations.append(
        dict(
            call_id=data["receipts"][0]["call_id"],
            error=None,
            status_timestamp=frames[0]["timestamp"],
            frames=frames,
        )
    )
    return data, {"receipt": {"observations": observations}}


@pytest.mark.parametrize("transcode", [False, True])
def test_adapter_recomputes_bytes_and_restores_identity_without_losing_clock(tmp_path, transcode):
    data, audit = artifact_fixture(tmp_path, transcode=transcode)
    original = copy.deepcopy(data["receipts"])
    data["receipts"] = exposure.verify_receipts(data["receipts"], audit, data["events"])
    assert candidate(data)["normal"]["eligible"]
    assert (
        data["receipts"][0]["frames"][0]["actor_states"] == original[0]["frames"][0]["actor_states"]
    )
    assert "frame_id" not in original[0]["frames"][0]
    assert data["receipts"][0]["frames"][0]["received_size"] == (
        [1024, 579] if transcode else [1344, 760]
    )


def test_adapter_rejects_changed_bytes_audit_or_worker_metadata(tmp_path):
    data, audit = artifact_fixture(tmp_path)
    source = copy.deepcopy(data)
    audit["receipt"]["observations"][0]["frames"][0]["received_path"] = str(tmp_path / "absent.jpg")
    checked = exposure.verify_receipts(data["receipts"], audit, data["events"])
    assert not checked[0]["frames"][0]["artifact_verified"]
    assert data == source


def test_manual_hash_gate_and_assigned_denominator_with_provider_failure():
    opportunity = result(fixture())
    review = dict(
        call_id="call-one", narrow_received_sha256="a" * 64, manual_pass=True, reviewer="root"
    )
    assert len(exposure.manual_matches(opportunity, [review])) == 1
    assert not exposure.manual_matches(opportunity, [dict(review, narrow_received_sha256="c" * 64)])
    cells = [
        dict(
            id=str(i),
            model=m,
            status="complete",
            opportunity=opportunity,
            reviews=[review],
            driving_infrastructure_problem=False,
        )
        for i, m in enumerate(("astra", "sol", "fable", "grok"))
    ]
    cells[3].update(status="failed_auth", opportunity={}, reviews=[])
    gate = exposure.checkpoint(cells)
    assert gate["denominator"] == 4 and gate["clear_narrow"] == 3 and gate["continue_automatically"]
    cells[3]["driving_infrastructure_problem"] = True
    assert not exposure.checkpoint(cells)["continue_automatically"]
    cells[0]["reviews"] = []
    assert exposure.checkpoint(cells)["clear_narrow"] == 2


@pytest.mark.parametrize("gap,passed", [(0.75, True), (0.749, False)])
def test_C_held_clearance_exact_boundary(gap, passed):
    data = fixture("C", held=True, receipt_s=3, cutoff_s=5)
    data["certificate"]["no_response_conflict_delay_after_onset_s"] = 1
    data["scene"]["scenario"]["colliders"][0]["circle"] = [10, 0.89 + 0.32 + gap, 0.32]
    row = candidate(data)
    assert row["safe_held"]["eligible"] is passed


@pytest.mark.parametrize("cutoff_kind", ["requested", "disabled"])
def test_equal_time_cleanup_receipt_is_excluded_when_native_tick_disabled(cutoff_kind):
    data = fixture(held=True, receipt_s=5, cutoff_s=5)
    for event in data["events"]:
        if event["kind"] == "sim_tick" and event["sim_time_s"] >= 5:
            event["enabled"] = False
    if cutoff_kind == "disabled":
        data["events"] = [e for e in data["events"] if e["kind"] != "sim_finish_requested"]
    assert not candidate(data)["eligible"]


@pytest.mark.parametrize(
    "target", ["audit_id", "audit_hash", "worker_clock", "delivery", "artifact_bytes"]
)
def test_adapter_rejects_inconsistent_proof(tmp_path, target):
    data, audit = artifact_fixture(tmp_path)
    proof = audit["receipt"]["observations"][0]["frames"][0]
    if target == "audit_id":
        proof["frame_id"] = "not-the-frame"
    if target == "audit_hash":
        proof["received_sha256"] = "c" * 64
    if target == "worker_clock":
        next(e for e in data["events"] if e["kind"] == "sim_frame")["actor_states"]["hazard"][
            "elapsed_s"
        ] = 1
    if target == "delivery":
        next(e for e in data["events"] if e["kind"] == "sim_observe_delivery")["frames"][0][
            "captured_at"
        ] += 1
    if target == "artifact_bytes":
        from pathlib import Path

        Path(proof["received_path"]).write_bytes(b"not an image")
    data["receipts"] = exposure.verify_receipts(data["receipts"], audit, data["events"])
    assert not data["receipts"][0]["frames"][0]["artifact_verified"]
    assert not candidate(data)["eligible"]


def test_missing_cutoff_nan_receipt_and_duplicate_call_are_negative():
    data = fixture()
    data["events"] = [e for e in data["events"] if e["kind"] != "sim_finish_requested"]
    assert not candidate(data)["eligible"]
    data = fixture()
    data["receipts"][0]["confirmed_monotonic_s"] = float("nan")
    assert not candidate(data)["eligible"]
    data = fixture()
    data["receipts"].append(copy.deepcopy(data["receipts"][0]))
    assert all(not r["eligible"] for r in result(data)["candidates"])


def test_capture_cannot_come_from_future_sim_time():
    data = fixture()
    for f in data["receipts"][0]["frames"]:
        f["sim_time_s"] = 4
    next(e for e in data["events"] if e["kind"] == "sim_frame")["sim_time_s"] = 4
    assert "capture_outside_receipt_native_bound" in candidate(data)["reasons"]


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("cutoff,passed", [(8, False), (8.01, True)])
def test_D_requires_actual_two_second_horizon_without_speed_or_held_gate(held, cutoff, passed):
    row = candidate(d_fixture(held=held, receipt_s=6, cutoff_s=cutoff))
    assert row["normal"]["eligible"] is passed
    assert not row["safe_held"]["eligible"]
    assert row["safe_held"]["reasons"] == ["D_has_no_speed_or_held_admission_branch"]
    if passed:
        assert row["normal"]["observed_completion_duration_s"] == 2
        assert row["normal"]["observed_duration_s"] == 2


@pytest.mark.parametrize("station,passed", [(10.99, False), (11, True), (18, True), (18.01, False)])
def test_D_exact_capture_front_station_window(station, passed):
    data = d_fixture()
    for event in data["events"]:
        if event["kind"] == "sim_tick":
            event["map_pose"][0] += station - 12
    row = candidate(data)
    assert row["normal"]["eligible"] is passed
    if not passed:
        assert "captured_body_outside_reviewed_approach_window" in row["reasons"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ("completion", "missing_or_invalid_predecision_native_timeline"),
        ("capture_pose", "captured_body_pose_not_joined_to_exact_native_sample"),
        ("missing_tick", "missing_native_samples_in_opportunity_horizon"),
        ("handback", "missing_or_invalid_actual_decision_bound"),
    ],
)
def test_D_requires_actual_completion_capture_and_handback_evidence(change, reason):
    data = d_fixture(cutoff_s=9)
    tick = next(e for e in data["events"] if e["kind"] == "sim_tick" and e["sim_time_s"] == 7)
    if change == "completion":
        tick.pop("native_completed_monotonic_s")
    elif change == "capture_pose":
        next(e for e in data["events"] if e["kind"] == "sim_frame")["pose"] = [999, 0, 0]
    elif change == "missing_tick":
        data["events"].remove(tick)
    else:
        data["events"] = [e for e in data["events"] if e["kind"] != "sim_client_handback_requested"]
    row = candidate(data)
    assert not row["eligible"]
    assert reason in row["normal"]["reasons"]


def test_D_rejects_old_plane_and_held_rule():
    data = d_fixture()
    data["rule"] = copy.deepcopy(exposure.RULE)
    with pytest.raises(ValueError, match="explicit accepted opportunity rule"):
        result(data)


@pytest.mark.parametrize("passage,passed", [(8, False), (8.01, True)])
def test_D_horizon_stays_before_first_forward_actor_passage(passage, passed):
    data = d_fixture()
    for event in data["events"]:
        if event["kind"] == "sim_tick" and event["sim_time_s"] >= passage:
            event["map_pose"][0] = 32
    row = candidate(data)
    assert row["normal"]["eligible"] is passed
    if not passed:
        assert "horizon_not_wholly_before_first_forward_actor_passage" in row["normal"]["reasons"]
