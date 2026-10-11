"""Replay a complete published session, then render only its logged observe calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from drivingbench.sim.course import Course

DT = 0.01
SESSION = "2026-09-17-gpt-5-6-sol-codex-4fd054"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs/sim/visual/course-demo"


def _elapsed(timestamp, start):
    return (datetime.fromisoformat(timestamp) - datetime.fromisoformat(start)).total_seconds()


def replay_segment(events, session, course, *, native_factory=None, make_request=None):
    """Replay matching tools; session=None selects a curated whole segment."""
    if native_factory is None or make_request is None:
        from drivingbench.sim.car import SimNative
        from drivingbench.sim.validate import request

        native_factory = native_factory or SimNative
        make_request = make_request or request
    starts = [event for event in events if event["kind"] == "segment_start"]
    ends = [event for event in events if event["kind"] == "segment_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("Replay needs one published segment start and end")
    start, end = starts[0], ends[0]
    duration = _elapsed(end["timestamp"], start["timestamp"])
    if duration < 0:
        raise ValueError("Segment ends before it starts")
    selected = sorted(
        (
            event
            for event in events
            if event["kind"] == "tool" and (session is None or event.get("session") == session)
        ),
        key=lambda event: event["timestamp"],
    )
    native = native_factory(time=-DT)
    states, calls, observations, transitions = [], [], [], []
    previous = None

    def record():
        nonlocal previous
        pose = tuple(native.pose)
        map_pose = course.map_pose(pose)
        assessment = course.assess(map_pose)
        row = {
            "segment": start["segment"],
            "t": round(native.time, 9),
            "local_pose": pose,
            "map_pose": map_pose,
            "speed_mps": float(native.car.speed),
            **assessment,
        }
        states.append(row)
        if assessment != previous:
            transitions.append({"t": row["t"], **assessment})
            previous = assessment
        return row

    def advance(target):
        while native.time + DT / 2 < target:
            native.tick()
            record()

    record()
    for event in selected:
        t = _elapsed(event["timestamp"], start["timestamp"])
        if not 0 <= t <= duration:
            raise ValueError("Session tool is outside its published segment")
        advance(t)
        row = {
            "id": event["id"],
            "tool": event["tool"],
            "timestamp": event["timestamp"],
            "t": t,
            "arguments": event["arguments"],
            "recorded_outcome": event["outcome"],
            "recorded_session": event.get("session"),
        }
        if event["tool"] in ("set_motion", "stop_now"):
            row["processed_t"] = native.time
            row["outcome"] = native.call(make_request(native.bridge.controller, event))
            record()
        elif event["tool"] == "observe":
            row["outcome"] = native.status()
            observations.append(
                {
                    "id": event["id"],
                    "segment": start["segment"],
                    "timestamp": event["timestamp"],
                    "t": t,
                    "sampled_t": native.time,
                    "local_pose": list(native.pose),
                    "map_pose": list(course.map_pose(native.pose)),
                    "recorded_outcome": event["outcome"],
                    "recorded_images": event.get("images", []),
                }
            )
        else:
            raise ValueError(f"Unknown recorded tool: {event['tool']}")
        row["completed_t"] = native.time
        calls.append(row)
    advance(duration)
    motion_calls = [call for call in calls if call["tool"] != "observe"]
    summary = {
        "segment": start["segment"],
        "started_at": start["timestamp"],
        "ended_at": end["timestamp"],
        "end_reason": end.get("reason"),
        "recorded_duration_s": duration,
        "simulated_duration_s": native.time,
        "reset": "Standstill at authored start for each native engagement; operator repositioning is not interpolated.",
        "tool_calls": len(calls),
        "motion_calls": len(motion_calls),
        "observations": len(observations),
        "accepted_motion_calls": sum(
            call["outcome"].get("status") == "accepted" for call in motion_calls
        ),
        "ever_hit_ids": sorted({key for state in states for key in state["collision_ids"]}),
        "ever_outside_course": any(state["outside_course"] for state in states),
        "ever_finish_reached": any(state["finish_reached"] for state in states),
        "ever_valid_finish": any(state["valid_finish"] for state in states),
        "outside_duration_s": DT * sum(state["outside_course"] for state in states[1:]),
        "transitions": transitions,
        "final_state": states[-1],
    }
    return {"summary": summary, "states": states, "calls": calls, "observations": observations}


def replay(scene_path, session_path, out):
    course = Course(json.loads(scene_path.read_text()))
    session = json.loads(session_path.read_text())
    if not session.get("ended_at"):
        raise ValueError("Session must be published and ended")
    out.mkdir(parents=True, exist_ok=True)
    segments, observations = [], []
    with (out / "states.jsonl").open("w") as states:
        for segment in session["segments"]:
            source = ROOT / "traces/segments" / segment / "events.jsonl"
            events = [json.loads(line) for line in source.read_text().splitlines()]
            result = replay_segment(events, session["id"], course)
            for row in result.pop("states"):
                states.write(json.dumps(row) + "\n")
            observations.extend(result.pop("observations"))
            result["source"] = str(source.relative_to(ROOT))
            result["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            segments.append(result)
    report = {
        "session": session,
        "scene": str(scene_path.resolve()),
        "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "timing": "100 Hz nearest-tick scheduling; motion call itself advances one tick. Every tick assessed; no swept collision. Observes do not advance time.",
        "settings": "Recorded session speed ceiling 3.5 m/s; sim.validate.request clears unused reason only.",
        "segments": segments,
        "observations": observations,
    }
    (out / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def render(report, renderer, out, figure):
    """Render every observation in source order, preserving the recorded timestamps."""
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    observations = report["observations"]
    if not observations:
        raise ValueError("Session has no observe calls")
    width, height, columns = 448, 298, 3
    sheet = Image.new(
        "RGB", (width * columns, height * ((len(observations) + columns - 1) // columns)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(observations):
        path = out / f"{index:03d}-{row['id']}.jpg"
        path.write_bytes(renderer.render(tuple(row["local_pose"])))
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((width, 254))
            x, y = index % columns * width, index // columns * height
            sheet.paste(image, (x, y))
        draw.text((x + 5, y + 255), f"{row['segment']}  +{row['t']:.3f}s", fill="black")
        draw.text((x + 5, y + 270), row["timestamp"], fill="black")
    figure.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(figure, quality=92)
    return len(observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "render"))
    parser.add_argument("--scene", type=Path, default=ROOT / "sim/site1-course.json")
    parser.add_argument(
        "--session", type=Path, default=ROOT / "traces/sessions" / f"{SESSION}.json"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", type=Path, default=ROOT / "docs/sim/course-replay.jpg")
    args = parser.parse_args()
    if args.mode == "replay":
        report = replay(args.scene, args.session, args.out)
        print(json.dumps([segment["summary"] for segment in report["segments"]], indent=2))
    else:
        from drivingbench.sim.scene import Scene

        report = json.loads((args.out / "replay.json").read_text())
        if hashlib.sha256(args.scene.read_bytes()).hexdigest() != report["scene_sha256"]:
            raise ValueError("Scene changed since replay; rerun replay before rendering")
        print(
            {
                "rendered_observations": render(
                    report, Scene(args.scene), args.out / "frames", args.figure
                )
            }
        )


if __name__ == "__main__":
    main()
