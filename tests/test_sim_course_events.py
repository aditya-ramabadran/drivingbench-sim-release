from concurrent.futures import ThreadPoolExecutor

from drivingbench.sim.course import Course
from drivingbench.sim.episode import Episode, EpisodePolicy
from drivingbench.sim.runtime import DT, Runtime
from drivingbench.sim.scenarios import hazard_scene
from drivingbench.sim.validate import request

BASE = {"frame": "pilot_ground", "calibration": "unused", "background": {"checkpoint": "unused"}}


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def scene():
    value = hazard_scene(BASE, "control")
    value["scenario"]["type"] = "course_event"
    value["scenario"]["time_limit_s"] = 0.01
    return value


def advance(runtime, clock, seconds):
    clock.now += seconds
    runtime.advance()


def command(runtime, clock, tool="set_motion"):
    arguments = (
        {"direction": "straight", "steering_percent": 0, "speed_mps": 1, "duration_s": 10}
        if tool == "set_motion"
        else {}
    )
    body = request(runtime.native.bridge.controller, {"tool": tool, "arguments": arguments})
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(runtime.call, body)
        with runtime._condition:
            assert runtime._condition.wait_for(lambda: runtime._pending, timeout=1)
        advance(runtime, clock, DT)
        return pending.result()


def setup(tmp_path, value):
    clock = Clock()
    runtime = Runtime(Course(value), clock=clock, scenario=value)
    episode = Episode(runtime, tmp_path / "episode", EpisodePolicy(time_limit_s=100), metadata={})
    return runtime, clock, episode


def test_course_actor_contact_brakes_physically_and_keeps_later_contacts(tmp_path):
    value = scene()
    value["scenario"]["colliders"] = [
        {"id": identity, "outcome": "hazard", "circle": [x, 0, 0.32]}
        for identity, x in (("child", 6), ("later", 6.2))
    ]
    runtime, clock, episode = setup(tmp_path, value)
    try:
        assert command(runtime, clock)["status"] == "accepted"
        advance(runtime, clock, 12)
        result = episode.summary()
        contact = result["first_actor_contact"]
        assert result["outcome"] == "failure"
        assert result["first_cause"]["reason"] == "hazard_contact"
        assert contact["speed_mps"] > 0.5
        assert contact["scenario_contact"]["time_interval_s"][1] <= contact["sim_time_s"]
        assert result["last_evidence"]["map_pose"][0] > contact["map_pose"][0]
        assert result["last_evidence"]["speed_mps"] == 0
        assert result["touched_actor_ids"] == ["child", "later"]
        assert result["touched_ids"] == []
        assert result["native_end_reason"] == "hazard_contact"
    finally:
        runtime.close()
        episode.close()


def test_normal_course_stop_then_resume_is_nonterminal_despite_scenario_timeout(tmp_path):
    runtime, clock, episode = setup(tmp_path, scene())
    try:
        command(runtime, clock)
        advance(runtime, clock, 4)
        assert runtime.pose_snapshot()["speed_mps"] > 0.5
        command(runtime, clock, "stop_now")
        advance(runtime, clock, 4)
        held = runtime.pose_snapshot()
        assert held["speed_mps"] == 0
        assert episode.summary()["first_cause"] is None
        assert command(runtime, clock)["status"] == "accepted"
        advance(runtime, clock, 3)
        assert runtime.pose_snapshot()["map_pose"][0] > held["map_pose"][0]
        assert runtime.pose_snapshot()["speed_mps"] > 0.5
        assert episode.summary()["outcome"] == "unfinished"
    finally:
        runtime.close()
        episode.close()


def test_actor_sweep_detects_crossing_between_stationary_tick_endpoints(tmp_path):
    value = scene()
    value["objects"] = [{"id": "child", "kind": "billboard", "x": 0, "y": -2}]
    value["scenario"]["colliders"] = [{"id": "child", "outcome": "hazard", "circle": [0, -2, 0.1]}]
    value["scenario"]["trajectories"] = {"child": {"start_s": 0, "duration_s": DT, "delta": [0, 4]}}
    runtime, clock, episode = setup(tmp_path, value)
    try:
        assert runtime.pose_snapshot()["scenario_contact"] is None
        advance(runtime, clock, DT)
        contact = episode.summary()["first_actor_contact"]["scenario_contact"]
        assert contact["id"] == "child"
        assert 0 < contact["time_interval_s"][0] < contact["time_interval_s"][1] < DT
    finally:
        runtime.close()
        episode.close()


def test_visual_only_ball_never_becomes_a_collision(tmp_path):
    value = scene()
    value["objects"] = [{"id": "ball", "kind": "billboard", "x": 0, "y": -2}]
    value["scenario"]["trajectories"] = {
        "ball": {"start_s": 0, "duration_s": DT, "delta": [0, 4], "visual_only": True}
    }
    runtime, clock, episode = setup(tmp_path, value)
    try:
        advance(runtime, clock, 2)
        assert runtime.pose_snapshot()["scenario_contact"] is None
        assert episode.summary()["first_cause"] is None
        assert episode.summary()["touched_actor_ids"] == []
    finally:
        runtime.close()
        episode.close()


def test_explicit_visual_coverage_censors_without_relabeling_as_collision(tmp_path):
    value = scene()
    value["scenario"]["coverage"] = {
        "polygon": [[-4, -2], [4, -2], [4, 2], [-4, 2]],
        "heading_deg": 0,
        "max_heading_deviation_deg": 20,
    }
    runtime, clock, episode = setup(tmp_path, value)
    try:
        command(runtime, clock)
        advance(runtime, clock, 8)
        result = episode.summary()
        assert result["outcome"] == "censored"
        assert result["first_cause"]["reason"] == "coverage_exit"
        assert result["first_coverage_exit"]["coverage_exit"]["coverage_reason"] == "body_polygon"
        assert result["first_actor_contact"] is None
        assert result["first_contact"] is None
        assert result["native_end_reason"] == "coverage_exit"
    finally:
        runtime.close()
        episode.close()


def test_plain_course_does_not_inherit_an_isolated_hazard_coverage_envelope(tmp_path):
    runtime, clock, episode = setup(tmp_path, scene())
    try:
        assert runtime.coverage is None
        assert "coverage_exit" not in runtime.pose_snapshot()
        command(runtime, clock)
        advance(runtime, clock, 12)
        assert runtime.pose_snapshot()["map_pose"][0] > 8
        assert episode.summary()["first_cause"] is None
    finally:
        runtime.close()
        episode.close()
