"""Compare a complete recorded session over loopback HTTP with deterministic SimNative."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import time
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from drivingbench.gateway.app import create_app as gateway_app
from drivingbench.replay.run import http_server
from drivingbench.sim.cameras import SceneRenderer
from drivingbench.sim.course_demo import DT, ROOT, SESSION, _elapsed, replay_segment
from drivingbench.sim.episode import EpisodePolicy
from drivingbench.sim.producer import DEFAULT_SCENE, simulator_app


def wait_until(target):
    while (remaining := target - time.monotonic()) > 0:
        time.sleep(min(0.1, remaining))


def compare(states, reference):
    """Compare equal integration ticks, without interpolating between reset segments."""
    lookup = {round(row["t"] / DT): row for row in reference}
    distance, headings = [], []
    for row in states:
        other = lookup.get(round(row["sim_time_s"] / DT))
        if other is None:
            continue
        a, b = row["pose"], other["local_pose"]
        distance.append(math.dist(a[:2], b[:2]))
        headings.append(abs((a[2] - b[2] + 180) % 360 - 180))
    if not distance:
        raise ValueError("No common integration ticks")
    return {
        "ticks": len(distance),
        "position_rmse_m": math.sqrt(sum(v * v for v in distance) / len(distance)),
        "position_max_m": max(distance),
        "position_final_m": distance[-1],
        "heading_max_deg": max(headings),
    }


def measured_events(events, session, calls):
    start = next(event["timestamp"] for event in events if event["kind"] == "segment_start")
    timings = {call["source"]["id"]: call for call in calls}
    result = copy.deepcopy(events)
    for event in result:
        if event["kind"] != "tool" or event.get("session") != session:
            continue
        call = timings[event["id"]]
        elapsed = call.get("native_processed_t", call["request_started_t"])
        event["timestamp"] = (
            datetime.fromisoformat(start) + timedelta(seconds=elapsed)
        ).isoformat()
    return result


def run_segment(scene_path, source, session, output, *, renderer=None):
    events = [json.loads(line) for line in source.read_text().splitlines()]
    start = next(event for event in events if event["kind"] == "segment_start")
    end = next(event for event in events if event["kind"] == "segment_end")
    duration = _elapsed(end["timestamp"], start["timestamp"])
    selected = [
        event for event in events if event["kind"] == "tool" and event.get("session") == session
    ]
    renderer = renderer or SceneRenderer(scene_path, output / "renderer")
    with ExitStack() as cleanup:
        cleanup.callback(renderer.close)
        warming = time.monotonic()
        renderer.start()
        startup_s = time.monotonic() - warming
        app = simulator_app(
            output, scene_path, renderer=renderer, policy=EpisodePolicy(enabled=False)
        )
        app.state.settings.patch({"speed_limit_mps": 3.5})
        native = app.state.sim_native
        receipts, states, calls = [], [], []
        original_poll = native.native.inbox.poll

        def poll(handler):
            def receipt(body):
                result = handler(body)
                if body.get("operation") in ("motion", "stop"):
                    receipts.append(
                        {
                            "operation": body["operation"],
                            "request_id": body.get("request_id"),
                            "native_processed_t": native.native.time - native.started_at,
                            "outcome": copy.deepcopy(result),
                        }
                    )
                return result

            original_poll(receipt)

        native.native.inbox.poll = poll
        with http_server(app) as producer, http_server(gateway_app(producer)) as gateway:
            previous = native.on_evidence

            def evidence(kind, **data):
                if kind == "sim_tick":
                    states.append(copy.deepcopy(data))
                previous(kind, **data)

            native.on_evidence = evidence
            states.append(native.pose_snapshot())
            with httpx.Client(base_url=gateway, timeout=20, trust_env=False) as client:
                for event in selected:
                    target = _elapsed(event["timestamp"], start["timestamp"])
                    wait_until(native.started_at + target)
                    began = time.monotonic() - native.started_at
                    before = len(receipts)
                    if event["tool"] == "observe":
                        response = client.get("/api/observe")
                    else:
                        endpoint = "motion" if event["tool"] == "set_motion" else "stop"
                        response = client.post(f"/api/{endpoint}", json=event["arguments"])
                    packet = response.json()
                    for image in packet.get("images", []):
                        jpeg = base64.b64decode(image.pop("data"))
                        path = output / "observations" / f"{event['id']}-{image['camera']}.jpg"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(jpeg)
                        image.update(path=str(path), sha256=hashlib.sha256(jpeg).hexdigest())
                    row = {
                        "source": event,
                        "scheduled_t": target,
                        "request_started_t": began,
                        "response_completed_t": time.monotonic() - native.started_at,
                        "http_status": response.status_code,
                        "response": packet,
                    }
                    if event["tool"] != "observe":
                        if len(receipts) != before + 1:
                            raise RuntimeError("Motion did not produce exactly one native receipt")
                        row.update(receipts[-1])
                        row["lateness_s"] = row["native_processed_t"] - target
                    calls.append(row)
                wait_until(native.started_at + duration)
                native.close()
                native.on_evidence = previous
                final = native.pose_snapshot()
            ready = app.state.sim_cameras.ready
            camera_errors = dict(app.state.sim_cameras.errors)
        legacy = replay_segment(events, session, native.course)
        actual = replay_segment(measured_events(events, session, calls), session, native.course)
        summary = {
            "segment": start["segment"],
            "duration_s": duration,
            "first_recorded_sim_time_s": states[0]["sim_time_s"],
            "tool_calls": len(calls),
            "accepted_motion_calls": sum(
                row["response"].get("status") in ("accepted", "stopping") for row in calls
            ),
            "observations": sum(row["source"]["tool"] == "observe" for row in calls),
            "ever_hit_ids": sorted({marker for row in states for marker in row["collision_ids"]}),
            "ever_outside_course": any(row["outside_course"] for row in states),
            "outside_duration_s": DT * sum(row["outside_course"] for row in states),
            "ever_finish_reached": any(row["finish_reached"] for row in states),
            "final": final,
            "legacy_schedule": compare(states, legacy["states"]),
            "measured_schedule": compare(states, actual["states"]),
            "renderer_startup_s": startup_s,
            "renderer_ready": ready,
            "camera_errors": camera_errors,
            "episode_summary": json.loads((output / "episode/summary.json").read_text()),
        }
        (output / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in states))
        (output / "calls.json").write_text(json.dumps(calls, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary, calls


class RelativeNative:
    """Expose elapsed scheduler time while preserving the live controller clock epoch."""

    def __init__(self, epoch, time):
        from drivingbench.sim.car import SimNative

        self.epoch = epoch
        self.native = SimNative(time=epoch + time)

    @property
    def time(self):
        return self.native.time - self.epoch

    def __getattr__(self, name):
        return getattr(self.native, name)


def rescore(output, scene_path):
    from drivingbench.sim.course import Course

    report = json.loads((output / "summary.json").read_text())
    if hashlib.sha256(scene_path.read_bytes()).hexdigest() != report["scene_sha256"]:
        raise ValueError("Scene changed since live replay")
    course = Course(json.loads(scene_path.read_text()))
    for summary in report["segments"]:
        folder = output / summary["segment"]
        source = ROOT / "traces/segments" / summary["segment"] / "events.jsonl"
        events = [json.loads(line) for line in source.read_text().splitlines()]
        calls = json.loads((folder / "calls.json").read_text())
        states = [json.loads(line) for line in (folder / "states.jsonl").read_text().splitlines()]
        epoch = summary["final"]["monotonic_at"] - summary["final"]["sim_time_s"]
        reference = replay_segment(
            measured_events(events, report["session"]["id"], calls),
            report["session"]["id"],
            course,
            native_factory=lambda time: RelativeNative(epoch, time),
        )
        summary["native_clock_epoch"] = epoch
        summary["source_events_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        summary["measured_schedule_same_epoch"] = compare(states, reference["states"])
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--session", type=Path, default=ROOT / "traces/sessions" / f"{SESSION}.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "runs/sim/step4/recorded-replay")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Recompare saved poses with the exact live clock epoch",
    )
    args = parser.parse_args()
    if args.rescore:
        print(json.dumps(rescore(args.out, args.scene), indent=2))
        return
    session = json.loads(args.session.read_text())
    if not session.get("ended_at"):
        raise ValueError("Only ended published sessions may be replayed")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "session": session,
        "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "transport": "Actual loopback HTTP gateway to unchanged producer endpoints",
        "settings": {"speed_limit_mps": 3.5, "time_scale": 1.0},
        "policy": "Geometry stopping disabled only for step3 comparison; rendering/runtime faults remain infrastructure failures.",
        "segments": [],
    }
    latencies = []
    for segment in session["segments"]:
        print(f"Starting complete segment {segment}", flush=True)
        result, calls = run_segment(
            args.scene,
            ROOT / "traces/segments" / segment / "events.jsonl",
            session["id"],
            args.out / segment,
        )
        summary["segments"].append(result)
        latencies.extend(row["lateness_s"] for row in calls if "lateness_s" in row)
        summary["motion_lateness_s"] = latencies
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "segment",
                        "accepted_motion_calls",
                        "observations",
                        "ever_hit_ids",
                        "outside_duration_s",
                        "legacy_schedule",
                        "measured_schedule",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
