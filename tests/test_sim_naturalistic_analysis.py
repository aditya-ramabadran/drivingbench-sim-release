import copy

import pytest
from drivingbench.sim.naturalistic_analysis import actor_phase, exposure_summary, physical_summary


def scene():
    return {
        "scenario": {
            "trajectories": {"hazard": {"duration_s": 2, "delta": [0, 4]}},
            "colliders": [{"id": "hazard", "circle": [0, -2, 0.32]}],
        }
    }


def tick(time, *, speed=0, x=0, end_reason=None):
    return dict(
        kind="sim_tick",
        sequence=int(time * 100),
        sim_time_s=time,
        speed_mps=speed,
        map_pose=[x, 0, 0],
        actor_onsets={"hazard": 1},
        end_reason=end_reason,
        scenario_contact=None,
    )


def test_phases_use_capture_time_and_delayed_onset():
    s = scene()
    assert actor_phase(s, "hazard", dict(sim_time_s=3, actor_onsets={})) == "before_motion"
    for time, phase in [(1, "before_motion"), (1.1, "moving"), (3, "after_motion")]:
        assert actor_phase(s, "hazard", tick(time)) == phase


def test_clearance_tracks_actor_and_preserves_overlap():
    result = physical_summary(scene(), [tick(0), tick(2), tick(4)])
    gap = result["minimum_sampled_clearance"]["hazard"]
    assert gap["tick"]["sim_time_s"] == 2
    assert gap["offset"] == [0, 2]
    assert gap["clearance_m"] == pytest.approx(-1.21)
    assert not result["contacts"]  # Geometry sampling never invents a runtime contact.


def test_stop_resume_excludes_initial_rest_and_terminal_freeze():
    events = [
        tick(0),
        tick(1, speed=1),
        tick(2),
        tick(3),
        tick(4, speed=1),
        tick(5, end_reason="terminal"),
    ]
    result = physical_summary(scene(), events)
    assert len(result["moving_to_stopped"]) == len(result["resumptions"]) == 1
    assert result["moving_to_stopped"][0]["duration_s"] == 2
    assert result["last_tick"]["sim_time_s"] == 4
    assert physical_summary(scene(), events, terminal_time_s=1)["moving_to_stopped"] == []


def test_visibility_requires_received_surviving_pixels_and_motion():
    receipt = {"observations": []}
    for i, time in enumerate([0, 2, 3]):
        receipt["observations"].append(
            dict(
                call_id=str(i),
                event_index=i + 1,
                client_timestamp=None,
                status_timestamp="2026-01-01T00:00:00Z",
                frames=[
                    dict(
                        camera=c,
                        frame_id=f"{i}-{c}",
                        sim_time_s=time,
                        actor_onsets={"hazard": 1},
                        accepted=True,
                        billboards={"hazard": {"visible_pixels": 2 if c == "narrow" else 0}},
                    )
                    for c in ["narrow", "wide"]
                ],
            )
        )
    original = copy.deepcopy(receipt)
    result = exposure_summary(scene(), receipt)["hazard"]
    assert [v["phase"] for v in result["visible_frames"]] == [
        "before_motion",
        "moving",
        "after_motion",
    ]
    assert result["timing"]["first_visible"]["narrow"]["frame"]["sim_time_s"] == 0
    assert result["timing"]["first_moving_visible"]["narrow"]["frame"]["sim_time_s"] == 2
    assert result["timing"]["first_visible"]["wide"] is None
    assert receipt == original


def test_ball_timing_retains_original_actor_identity():
    s = scene()
    s["scenario"]["trajectories"]["ball"] = {"duration_s": 2, "delta": [1, 0]}
    frame = dict(
        camera="narrow",
        frame_id="b",
        accepted=True,
        sim_time_s=1,
        actor_onsets={"ball": 0},
        billboards={"ball": {"visible_pixels": 3}},
    )
    receipt = {
        "observations": [
            dict(
                call_id="b",
                event_index=1,
                client_timestamp=None,
                status_timestamp="2026-01-01T00:00:00Z",
                frames=[frame],
            )
        ]
    }
    result = exposure_summary(s, receipt)["ball"]["timing"]["first_moving_visible"]["narrow"]
    assert result["actor_id"] == "ball"
    assert result["frame"] == frame
    assert "hazard" not in frame["billboards"]
