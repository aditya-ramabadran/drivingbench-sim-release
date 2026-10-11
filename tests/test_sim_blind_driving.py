import pytest
from drivingbench.sim.blind_driving import clock_bounds, exposure_at, receipt_cadence


def observation(identity, utc, capture=10):
    return dict(
        call_id=identity,
        received_utc_s=utc,
        frames=[
            dict(camera=camera, frame_id=str(capture), captured_monotonic_s=capture)
            for camera in ("narrow", "wide")
        ],
    )


def test_clock_registration_bounds_and_rejects_disjoint_anchors():
    rows = [dict(original_wall_timestamp="1970-01-01T00:01:50Z", arrived_at=9, delivered_at=11)]
    assert clock_bounds(rows)["offset_s"] == [99, 101]
    rows.append(dict(original_wall_timestamp="1970-01-01T00:02:00Z", arrived_at=9, delivered_at=11))
    assert clock_bounds(rows)["status"] == "inconsistent_clock"
    assert clock_bounds([])["status"] == "no_anchor"


def test_receipt_is_not_capture_or_future_receipt():
    rows = [observation("a", 115), observation("b", 130, 25)]
    result = exposure_at(20, rows, {"offset_s": [99.9, 100.1]})
    assert result["receipt_gap_s"] == pytest.approx([4.9, 5.1])
    assert result["capture_age_s"] == {"narrow": 10, "wide": 10}
    assert result["last_definite_call_id"] == "a"


def test_unknown_and_boundary_receipts_are_not_zero_latency():
    clock = {"offset_s": [99, 101]}
    assert exposure_at(20, [observation("a", 120)], clock)["status"] == "ambiguous_receipt"
    assert exposure_at(20, [observation("a", None)], clock)["receipt_gap_s"] is None
    assert exposure_at(20, [], clock)["status"] == "no_prior_receipt"
    assert exposure_at(20, [], {"offset_s": None})["status"] == "unknown_clock"


def test_pairs_repeats_and_replaced_command_window():
    rows = [observation("a", 110), observation("b", 112), observation("c", 116, 15)]
    commands = [
        dict(
            id="first",
            requested_duration_s=20,
            accepted_monotonic_s=9,
            end_monotonic_s=13,
            end_reason="replaced",
        ),
        dict(
            id="second",
            requested_duration_s=2,
            accepted_monotonic_s=13,
            end_monotonic_s=15,
            end_reason="expired",
        ),
    ]
    result = receipt_cadence(rows, commands, {"offset_s": [100, 100]})
    assert result["observations"] == 3
    assert result["fresh_observations"] == 2
    assert result["repeated_observations"] == 1
    assert result["median_receipt_interval_s"] == 3
    assert result["commands"][0]["received_while_active"] == ["a", "b"]
    assert result["commands"][1]["ended_without_fresh_receipt"] is True


def test_uncertain_command_boundary_stays_unknown():
    command = dict(id="a", accepted_monotonic_s=9, end_monotonic_s=10)
    result = receipt_cadence([observation("a", 110)], [command], {"offset_s": [99.9, 100.1]})
    assert result["commands"][0]["ended_without_fresh_receipt"] is None


def test_early_termination_telemetry_interval_is_not_exact():
    command = dict(
        id="a", accepted_monotonic_s=9, end_monotonic_s=12, early_clearance_interval_s=[11, 12]
    )
    result = receipt_cadence([observation("a", 111.5)], [command], {"offset_s": [100, 100]})
    assert result["commands"][0]["fresh_while_active"] == []
    assert result["commands"][0]["ambiguous_boundary_receipts"] == ["a"]
    assert result["commands"][0]["ended_without_fresh_receipt"] is None


def test_rerendered_key_does_not_make_same_capture_fresh():
    rows = [observation("a", 111), observation("b", 112)]
    for frame in rows[1]["frames"]:
        frame["frame_id"] = "new-render-key"
    result = receipt_cadence(rows, [], {"offset_s": [100, 100]})
    assert result["fresh_observations"] == 1
    assert result["repeated_observations"] == 1


def test_definite_fresh_receipt_proves_exposure_despite_other_uncertainty():
    command = dict(id="a", accepted_monotonic_s=9, end_monotonic_s=12)
    rows = [observation("fresh", 110), observation("boundary", 112, 11)]
    result = receipt_cadence(rows, [command], {"offset_s": [99.9, 100.1]})
    assert result["commands"][0]["fresh_while_active"] == ["fresh"]
    assert result["commands"][0]["ambiguous_boundary_receipts"] == ["boundary"]
    assert result["commands"][0]["ended_without_fresh_receipt"] is False
    rows.append(observation("unknown", None, 12))
    assert (
        receipt_cadence(rows, [command], {"offset_s": [99.9, 100.1]})["commands"][0][
            "ended_without_fresh_receipt"
        ]
        is False
    )
