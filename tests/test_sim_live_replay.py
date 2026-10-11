from copy import deepcopy

import pytest
from drivingbench.sim.live_replay import compare, measured_events


def test_actual_timing_replay_preserves_original_session_filter_and_source_records():
    events = [
        {"kind": "segment_start", "timestamp": "2026-09-17T00:00:00+00:00"},
        {"id": "old", "kind": "tool", "session": "other", "timestamp": "2026-09-17T00:00:01+00:00"},
        {
            "id": "move",
            "kind": "tool",
            "session": "selected",
            "timestamp": "2026-09-17T00:00:02+00:00",
        },
        {
            "id": "look",
            "kind": "tool",
            "session": "selected",
            "timestamp": "2026-09-17T00:00:03+00:00",
        },
    ]
    original = deepcopy(events)
    calls = [
        {"source": events[2], "request_started_t": 2.012, "native_processed_t": 2.03},
        {"source": events[3], "request_started_t": 3.001},
    ]
    replayed = measured_events(events, "selected", calls)
    assert events == original
    assert replayed[1] == original[1]
    assert replayed[2]["timestamp"] == "2026-09-17T00:00:02.030000+00:00"
    assert replayed[3]["timestamp"] == "2026-09-17T00:00:03.001000+00:00"


def test_comparison_matches_physics_ticks_without_crossing_missing_intervals():
    live = [
        {"sim_time_s": 0.019999999, "pose": [4, 0, 359]},
        {"sim_time_s": 0.04, "pose": [100, 0, 0]},
    ]
    reference = [{"t": 0.02, "local_pose": [3, 0, -1]}, {"t": 0.03, "local_pose": [0, 0, 0]}]
    result = compare(live, reference)
    assert result["ticks"] == 1
    assert result["position_rmse_m"] == 1
    assert result["heading_max_deg"] == 0
    with pytest.raises(ValueError, match="No common"):
        compare(live, [])
