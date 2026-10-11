"""Optional actor waiting, with an authored motion clock separate from world time."""

from __future__ import annotations

import copy
import math

from drivingbench.sim.actor_animation import root_fraction
from drivingbench.sim.course import footprint
from shapely.geometry import LineString, Point, Polygon


def validate_policy(policy):
    if not isinstance(policy, dict) or policy.get("kind") != "wait_for_stopped_car":
        raise ValueError("Unknown actor yield policy")
    if set(policy) - {"kind", "max_car_speed_mps", "safety_margin_m"}:
        raise ValueError("Unknown actor yield setting")
    for key, default in (("max_car_speed_mps", 0.1), ("safety_margin_m", 0.75)):
        value = policy.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Actor yield settings must be finite numbers")
        if value < 0 or (key == "safety_margin_m" and value == 0):
            raise ValueError("Actor yield needs nonnegative speed and positive safety margin")


def validate_hold(hold):
    if not isinstance(hold, dict) or set(hold) != {"at_fraction", "duration_s"}:
        raise ValueError("Actor hold needs at_fraction and duration_s")
    for key in hold:
        if (
            isinstance(hold[key], bool)
            or not isinstance(hold[key], (int, float))
            or not math.isfinite(hold[key])
        ):
            raise ValueError("Actor hold values must be finite numbers")
    if not 0 <= hold["at_fraction"] <= 1 or hold["duration_s"] <= 0:
        raise ValueError("Actor hold needs normalized progress and positive duration")


def elapsed_at(states, identity, time, fallback):
    state = (states or {}).get(identity)
    if state is None:
        return fallback
    first = state["previous_time_s"]
    dt = max(0.0, time - first)
    if state.get("hold_started_s") is not None:
        dt -= max(0.0, min(time, state["hold_until_s"]) - max(first, state["hold_started_s"]))
    return min(state["elapsed_s"], state["previous_elapsed_s"] + dt)


class ActorMotion:
    """Advance source motion only while its declared path is clear of a slow car."""

    def __init__(self, scene):
        scenario = scene["scenario"]
        colliders = {row["id"]: row for row in scenario["colliders"]}
        self.actors = {
            identity: (motion, scenario["animations"][identity], colliders[identity]["circle"])
            for identity, motion in scenario["trajectories"].items()
            if "yield_policy" in motion or "hold" in motion
        }
        self.states = {}

    def update(self, pose, time, speed, onsets):
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Actor yield needs finite nonnegative car speed")
        for identity, (motion, animation, circle) in self.actors.items():
            start = motion.get("start_s", onsets.get(identity))
            if start is None or time < start:
                continue
            previous = self.states.get(identity)
            first_time = previous["time_s"] if previous else start
            first_elapsed = previous["elapsed_s"] if previous else 0.0
            dt = time - first_time
            hold = motion.get("hold")
            hold_started = previous.get("hold_started_s") if previous else None
            hold_until = previous.get("hold_until_s") if previous else None
            elapsed = first_elapsed + max(0.0, time - max(first_time, hold_until or first_time))
            hold_phase = self._hold_phase(motion, animation) if hold else None
            if hold and hold_started is None:
                elapsed = min(elapsed, hold_phase)
            elapsed, yielding = self._advance(
                pose, speed, motion, animation, circle, first_elapsed, elapsed, previous
            )
            if hold and hold_started is None and elapsed >= hold_phase:
                hold_started = first_time + elapsed - first_elapsed
                hold_until = hold_started + hold["duration_s"]
                if time > hold_until:
                    elapsed, yielding = self._advance(
                        pose,
                        speed,
                        motion,
                        animation,
                        circle,
                        elapsed,
                        elapsed + time - hold_until,
                        previous,
                    )
            holding = hold_started is not None and time < hold_until
            hold_spent = (
                max(0.0, min(time, hold_until) - max(first_time, hold_started))
                if hold_started is not None
                else 0.0
            )
            paused = yielding or holding
            waiting_since = (
                previous["waiting_since_s"]
                if paused and previous and previous["paused"]
                else hold_started
                if holding
                else first_time + elapsed - first_elapsed + hold_spent
                if yielding
                else None
            )
            self.states[identity] = dict(
                previous_time_s=first_time,
                time_s=time,
                previous_elapsed_s=first_elapsed,
                elapsed_s=elapsed,
                paused=paused,
                pause_reason="yield_to_car" if yielding else "authored_hold" if holding else None,
                waiting_since_s=waiting_since,
                waited_s=(previous["waited_s"] if previous else 0)
                + max(0.0, dt - (elapsed - first_elapsed)),
                hold_started_s=hold_started,
                hold_until_s=hold_until,
            )
        return copy.deepcopy(self.states)

    @staticmethod
    def _hold_phase(motion, animation):
        target = motion["hold"]["at_fraction"]
        lower, upper = 0.0, motion["duration_s"]
        for _ in range(40):
            middle = (lower + upper) / 2
            if root_fraction(animation, middle, motion["duration_s"]) < target:
                lower = middle
            else:
                upper = middle
        return upper

    def _advance(self, pose, speed, motion, animation, circle, first, last, previous):
        policy = motion.get("yield_policy")
        was_yielding = previous and previous.get("pause_reason") == "yield_to_car"
        if (
            policy is None
            or (not was_yielding and speed > policy.get("max_car_speed_mps", 0.1))
            or first >= motion["duration_s"]
        ):
            return last, False
        obstacle = Polygon(footprint(pose)).buffer(circle[2] + policy.get("safety_margin_m", 0.75))
        stop = self._first_blocked(
            motion, animation, circle, first, last + (0.5 if was_yielding else 0), obstacle
        )
        if stop is None:
            return last, False
        return (first if was_yielding else max(first, min(last, stop))), True

    @staticmethod
    def _first_blocked(motion, animation, circle, first, last, obstacle):
        duration = motion["duration_s"]
        last = min(last, duration)
        if last <= first:
            return None
        progress = animation.get("progress_fraction")
        knots = [first, last]
        if progress:
            knots.extend(
                i * duration / (len(progress) - 1)
                for i in range(len(progress))
                if first < i * duration / (len(progress) - 1) < last
            )
        knots.sort()

        def point(elapsed):
            fraction = root_fraction(animation, elapsed, duration)
            return tuple(circle[i] + fraction * motion["delta"][i] for i in (0, 1))

        for lower, upper in zip(knots, knots[1:]):
            a, b = point(lower), point(upper)
            path = LineString((a, b)) if a != b else Point(a)
            if not path.intersects(obstacle):
                continue
            if obstacle.intersects(Point(a)):
                return lower
            for _ in range(40):
                middle = (lower + upper) / 2
                if LineString((a, point(middle))).intersects(obstacle):
                    upper = middle
                else:
                    lower = middle
            return max(first, lower - 1e-6)
        return None
