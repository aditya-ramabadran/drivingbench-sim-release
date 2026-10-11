"""Conservative live confirmation from completed public client image results."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
from drivingbench.replay.clients import read_events
from drivingbench.sim.event_analysis import frame_actor_state
from drivingbench.sim.hazard_evidence import (
    driving_stream_paths,
    image_correspondence,
    received_observations,
)


def selected_exposures(events):
    """Join immutable camera identity and capture time, never producer delivery alone."""
    frames, result = {}, []
    for row in events:
        if row.get("kind") != "sim_frame":
            continue
        for camera, identity in row.get("frame_ids", {}).items():
            worker = row.get("worker", {})
            metrics = worker if camera == "narrow" else worker.get("wide", {})
            frames[identity] = dict(
                camera=camera,
                captured_at=row["captured_at"],
                sim_time_s=row["sim_time_s"],
                actor_onsets=row.get("actor_onsets", {}),
                billboards=metrics.get("billboards", {}),
                **({"actor_states": row["actor_states"]} if "actor_states" in row else {}),
                **(
                    {"captured_coverage": worker["captured_coverage"]}
                    if "captured_coverage" in worker
                    else {}
                ),
            )
    for row in events:
        if row.get("kind") != "sim_observe_delivery":
            continue
        for selected in row["frames"]:
            suffix = "-" + selected["sha256"][:12]
            identity = selected["frame_id"].removesuffix(suffix)
            frame = frames.get(identity)
            if frame is None or any(
                frame[key] != selected[key] for key in ("camera", "captured_at")
            ):
                continue
            result.append(
                dict(frame, sha256=selected["sha256"], timestamp=row["delivered_timestamp"])
            )
    return result


class ReceiptWatcher:
    """Polling time is a conservative receipt upper bound, with exact/transcode proof."""

    def __init__(self, directory, scene, servers, producer, *, slowed=False, clock=time.monotonic):
        self.directory, self.scene = Path(directory), scene
        self.servers, self.producer, self.slowed = servers, producer, slowed
        self.clock = clock
        self.seen, self.records, self.size = set(), [], None
        self.observed, self.stream_size, self.exposures = [], None, []
        self.producers, self.indexed_paths = {}, {}
        self.last_poll = float("-inf")
        self.actor_id = "hazard"
        self.stream = self.directory / "client/stdout.jsonl"

    def poll(self, *, final=False):
        now = self.clock()
        if not final and now - self.last_poll < 0.25:
            return None
        self.last_poll = now
        path = self.stream
        evidence_path = self.directory / "episode/events.jsonl"
        size = (
            tuple(
                (str(p), p.stat().st_size if p.exists() else 0) for p in driving_stream_paths(path)
            ),
            evidence_path.stat().st_size if evidence_path.exists() else 0,
        )
        if size[0] != self.stream_size:
            self.stream_size = size[0]
            self.observed = received_observations(path, self.servers)
        pending = [row for row in self.observed if row["call_id"] not in self.seen]
        if pending:
            if size != self.size:
                self.size = size
                self.exposures = selected_exposures(
                    read_events(self.directory / "episode/events.jsonl")
                )
            for source in (self.directory / "producer/recordings").rglob("*.jpg"):
                stamp = (source.stat().st_size, source.stat().st_mtime_ns)
                if self.indexed_paths.get(source) != stamp:
                    self.producers[hashlib.sha256(source.read_bytes()).hexdigest()] = source
                    self.indexed_paths[source] = stamp
            for observed in pending:
                selected = [
                    e for e in self.exposures if e["timestamp"] == observed["status"]["timestamp"]
                ]
                selected.sort(key=lambda e: ("narrow", "wide").index(e["camera"]))
                if (
                    len(selected) != len(observed["images"])
                    or {f["camera"] for f in selected} != {"narrow", "wide"}
                    or len(selected) != 2
                ):
                    continue
                frames = []
                for frame, raw in zip(selected, observed["images"], strict=True):
                    source = self.producers.get(frame["sha256"])
                    if source is None:
                        break
                    match = image_correspondence(raw, source.read_bytes())
                    if not match["accepted"]:
                        self.seen.add(observed["call_id"])
                        rejection = dict(
                            call_id=observed["call_id"],
                            camera=frame["camera"],
                            confirmed_monotonic_s=self.clock(),
                            correspondence=match,
                        )
                        with (self.directory / "live-receipt-rejections.jsonl").open("a") as output:
                            output.write(json.dumps(rejection, allow_nan=False) + "\n")
                        break
                    frames.append(
                        dict(
                            frame,
                            **match,
                            actor=frame_actor_state(self.scene, self.actor_id, frame),
                        )
                    )
                if len(frames) != 2:
                    continue
                self.seen.add(observed["call_id"])
                received = self.clock()
                record = dict(
                    call_id=observed["call_id"],
                    **(
                        {k: observed[k] for k in ("raw_call_id", "raw_stream")}
                        if "raw_stream" in observed
                        else {}
                    ),
                    confirmed_monotonic_s=received,
                    timing_basis="postdrive_stream_drain_upper_bound"
                    if final
                    else "public_stream_poll_upper_bound",
                    frames=frames,
                )
                try:
                    if self.slowed and not final:
                        narrow = next(f for f in frames if f["camera"] == "narrow")
                        request = dict(
                            capture_sim_time_s=narrow["sim_time_s"],
                            captured_at=narrow["captured_at"],
                            received_at=received,
                            receipt_id=observed["call_id"],
                            correspondence="exact" if narrow["exact"] else "registered_transcode",
                            actor_bearing=narrow["actor"]["visible"],
                            narrow_verified=True,
                            path_qualified=narrow["actor"]["qualifies"],
                        )
                        record["supervisor_request"] = request
                        reply = httpx.post(
                            self.producer + "/sim/receipt", json=request, timeout=5, trust_env=False
                        )
                        reply.raise_for_status()
                        record["supervisor_reply"] = reply.json()
                except Exception as error:
                    record["supervisor_error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    self.records.append(record)
                    log = self.directory / "live-receipts.jsonl"
                    with log.open("a") as output:
                        output.write(json.dumps(record, allow_nan=False) + "\n")

        if self.slowed and not final:
            reply = httpx.get(self.producer + "/sim/clock", timeout=5, trust_env=False)
            reply.raise_for_status()
            return reply.json()
        return None
