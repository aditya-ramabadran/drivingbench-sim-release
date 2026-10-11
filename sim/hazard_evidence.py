"""Paired hazard image receipt and response timing, separate from physical outcomes."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from drivingbench.replay.clients import read_events
from drivingbench.replay.run import completed_stop
from drivingbench.replay.tool_activity import tool_activity
from PIL import Image, UnidentifiedImageError


def _contents(event, call_id):
    item = event.get("item", {})
    if item.get("type") == "mcp_tool_call" and item.get("status") == "completed":
        return (item.get("result") or {}).get("content", [])
    call = event.get("tool_call", {}).get("mcpToolCall", {})
    if event.get("subtype") == "completed":
        return call.get("result", {}).get("success", {}).get("content", [])
    for block in event.get("message", {}).get("content", []):
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_result"
            and block.get("tool_use_id") == call_id
            and not block.get("is_error")
        ):
            return block.get("content", [])
    return []


def _decode(content, *, status_keys=("timestamp",)):
    images, status = [], None
    for item in content if isinstance(content, list) else []:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, dict):
            text = text.get("text")
        if isinstance(text, str):
            try:
                row = json.loads(text)
            except ValueError:
                row = None
            if isinstance(row, dict) and all(key in row for key in status_keys):
                status = row
        block = item.get("image", item if item.get("type") == "image" else {})
        if not isinstance(block, dict):
            continue
        source = block.get("source", {})
        data = block.get("data") or (source.get("data") if isinstance(source, dict) else None)
        if isinstance(data, str):
            try:
                images.append(base64.b64decode(data, validate=True))
            except ValueError:
                pass
    return status, images


def driving_stream_paths(path):
    """An explicit append-only turn manifest joins evidence, never rewrites raw output."""
    path = Path(path)
    if path.name != "driving-streams.jsonl":
        return [path]
    rows = (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )
    result = []
    for row in rows:
        relative = row["stdout"]
        source = path.parent / relative
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or source.name != "stdout.jsonl"
            or source.resolve() in [p.resolve() for p in result]
        ):
            raise ValueError("Invalid or duplicate private driving stream")
        result.append(source)
    return result


def _turn_identity(row, stream, base):
    """Preserve provider identity alongside one collision-free private receipt identity."""
    if not row.get("call_id"):
        raise ValueError("Missing raw provider call identity")
    return dict(
        row,
        raw_call_id=row["call_id"],
        raw_stream=str(stream),
        call_id=json.dumps([str(stream.relative_to(base)), row["call_id"]], separators=(",", ":")),
    )


def received_observations(path, servers):
    """Keep one completed result per actual observe call; never count discovery images."""
    if Path(path).name == "driving-streams.jsonl":
        result, offset = [], 0
        for source in driving_stream_paths(path):
            for row in received_observations(source, servers):
                joined = _turn_identity(row, source, Path(path).parent)
                joined["raw_event_index"] = row["event_index"]
                joined["event_index"] += offset
                result.append(joined)
            offset += len(read_events(source))
        return result
    events = read_events(path)
    result = []
    for call in tool_activity(events, driving_servers=servers):
        if call["category"] != "driving" or call["tool"] != "observe":
            continue
        for index in reversed(call["event_indices"]):
            status, images = _decode(_contents(events[index - 1], call["call_id"]))
            if status is not None and images:
                result.append(
                    dict(
                        call_id=call["call_id"],
                        event_index=index,
                        client_timestamp=events[index - 1].get("timestamp"),
                        status=status,
                        images=images,
                    )
                )
                break
    return sorted(result, key=lambda row: row["event_index"])


def client_actions(path, servers):
    """Keep rejected schema calls that never reached the producer, in stream order."""
    if Path(path).name == "driving-streams.jsonl":
        result, offset = [], 0
        for source in driving_stream_paths(path):
            for row in client_actions(source, servers):
                joined = _turn_identity(row, source, Path(path).parent)
                joined["first_event_index"] += offset
                joined["event_indices"] = [i + offset for i in row["event_indices"]]
                result.append(joined)
            offset += len(read_events(source))
        return result
    events = read_events(path)
    return [
        dict(
            call,
            first_event_index=min(call["event_indices"]),
            client_timestamp=events[min(call["event_indices"]) - 1].get("timestamp"),
        )
        for call in tool_activity(events, driving_servers=servers)
        if (call["category"] == "driving" and call["tool"] in {"set_motion", "stop_now"})
        or call["category"] == "unresolved"
    ]


def image_correspondence(received, producer, *, presentation=None):
    if presentation not in {None, "video_1024x576"}:
        raise ValueError("Unknown private image presentation")
    digest = hashlib.sha256(received).hexdigest()
    source_digest = hashlib.sha256(producer).hexdigest()
    try:
        with Image.open(io.BytesIO(received)) as picture:
            target = np.asarray(picture.convert("RGB"), dtype=float)
            size = picture.size
        with Image.open(io.BytesIO(producer)) as picture:
            original_size = picture.size
            reference = np.asarray(
                picture.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=float
            )
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return dict(
            received_sha256=digest,
            producer_sha256=source_digest,
            received_size=None,
            producer_size=None,
            exact=digest == source_digest,
            reference_resize_psnr_db=None,
            accepted=False,
            error=f"image_decode_{type(error).__name__}",
        )
    mse = float(np.mean((target - reference) ** 2))
    psnr = float(10 * np.log10(255**2 / mse)) if mse else None
    exact = digest == source_digest
    resize_size = (1024, 579) if presentation is None else (1024, 576)
    aspect_ok = presentation is None or original_size[0] * 576 == original_size[1] * 1024
    return dict(
        received_sha256=digest,
        producer_sha256=source_digest,
        received_size=list(size),
        producer_size=list(original_size),
        exact=exact,
        reference_resize_psnr_db=psnr,
        accepted=aspect_ok and (exact or (size == resize_size and (psnr is None or psnr >= 35))),
    )


def video_receipt_matches(events, exposures, observations, producer_tools, servers):
    """Closed CLI content attribution only; no live receipt instant or age adjustment."""
    matches = []
    for call in tool_activity(events, driving_servers=servers):
        if (
            call["category"] != "driving"
            or call["tool"] != "observe"
            or call["status"] != "completed"
        ):
            continue
        for index in reversed(call["event_indices"]):
            event = events[index - 1]
            result = event.get("item", {}).get("result") or event.get("tool_call", {}).get(
                "mcpToolCall", {}
            ).get("result", {}).get("success", {})
            if result.get("isError"):
                continue
            status, images = _decode(
                _contents(event, call["call_id"]), status_keys=("vehicle", "fork", "fault")
            )
            if status is None or len(images) != 1:
                continue
            raw = images[0]
            received_sha = hashlib.sha256(raw).hexdigest()
            # Provider wall stamps are a cross-recording-clock diagnostic, not a
            # containment gate or a reconstruction of the actual receipt instant.
            interval = None
            tool = event.get("tool_call", {})
            if (
                event.get("call_id")
                and tool.get("toolCallId")
                and event["call_id"] != tool["toolCallId"]
            ):
                continue
            try:
                if event.get("call_id") == tool.get("toolCallId") == call["call_id"]:
                    if isinstance(tool["startedAtMs"], bool) or isinstance(
                        tool["completedAtMs"], bool
                    ):
                        raise ValueError("Invalid recorded transaction stamps")
                    start = float(tool["startedAtMs"]) / 1000
                    end = float(tool["completedAtMs"]) / 1000
                    if math.isfinite(start) and math.isfinite(end) and start < end:
                        interval = (start, end)
            except (KeyError, TypeError, ValueError):
                pass
            candidates = []
            for observed in observations:
                if observed["response"] != status or len(observed["images"]) != 1:
                    continue
                timestamp = observed["producer_metadata"].get("timestamp")
                selected = [
                    e
                    for e in exposures
                    if e["delivered_timestamp"] == timestamp and e["camera"] == "narrow"
                ]
                if len(selected) != 1:
                    continue
                selection = selected[0]
                path = Path(observed["images"][0])
                if not path.is_file():
                    continue
                producer = path.read_bytes()
                if hashlib.sha256(producer).hexdigest() != selection["delivered_sha256"]:
                    continue
                comparison = image_correspondence(raw, producer, presentation="video_1024x576")
                if not comparison["accepted"]:
                    continue
                # These private fields bind the observed image to its actual producer
                # tool record and selected exposure; the CLI status does not carry them.
                metadata = observed["producer_metadata"]
                recorded = [
                    t
                    for t in producer_tools
                    if t.get("tool") == "observe"
                    and t.get("outcome", {}).get("timestamp") == timestamp
                ]
                if len(recorded) != 1:
                    continue
                try:
                    selected_wall = datetime.fromisoformat(timestamp).timestamp()
                    record_wall = datetime.fromisoformat(recorded[0]["timestamp"]).timestamp()
                    age = metadata["image_age_s"]
                    if not (
                        selected_wall <= record_wall
                        and type(age) in (int, float)
                        and math.isfinite(age)
                        and age >= 0
                        and recorded[0]["outcome"].get("image_age_s") == age
                        and selection.get("image_age_s") == age
                    ):
                        continue
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                candidates.append(
                    dict(
                        selection,
                        received_sha256=received_sha,
                        receipt_call_id=call["call_id"],
                        receipt_event_index=index,
                        image_correspondence=comparison,
                        receipt_basis="exact_completed_cli_bytes"
                        if comparison["exact"]
                        else "video_1024x576_unique_status_producer_and_content",
                        recorded_cli_wall_interval_s=list(interval)
                        if interval is not None
                        else None,
                        receipt_timing_qualification="Recorded CLI wall bounds are an unqualified cross-recording-clock diagnostic only; content attribution supplies no live receipt instant, upstream model-read, attention or age adjustment",
                    )
                )
            if len(candidates) == 1:
                matches.append(candidates[0])
            break
    return matches


def audit_receipt(exposures, stream, recording, output, servers):
    """Join timestamp and camera order; preserve failed matches instead of imputing exposure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    observations = received_observations(stream, servers)
    producers = {
        hashlib.sha256(p.read_bytes()).hexdigest(): p for p in Path(recording).rglob("*.jpg")
    }
    results = []
    for index, observed in enumerate(observations):
        matched = [
            e for e in exposures if e.get("delivered_timestamp") == observed["status"]["timestamp"]
        ]
        matched.sort(key=lambda e: ("narrow", "wide").index(e["camera"]))
        item = dict(
            call_id=observed["call_id"],
            **(
                {k: observed[k] for k in ("raw_call_id", "raw_stream", "raw_event_index")}
                if "raw_stream" in observed
                else {}
            ),
            event_index=observed["event_index"],
            client_timestamp=observed["client_timestamp"],
            status_timestamp=observed["status"]["timestamp"],
            frames=[],
            error=None,
        )
        if len(matched) != len(observed["images"]) or len({e["camera"] for e in matched}) != len(
            matched
        ):
            item["error"] = "ambiguous_timestamp_or_camera_count"
        else:
            for exposure, raw in zip(matched, observed["images"], strict=True):
                source = producers.get(exposure["delivered_sha256"])
                if source is None:
                    item["frames"].append(
                        dict(
                            camera=exposure["camera"],
                            accepted=False,
                            error="producer_bytes_unavailable",
                        )
                    )
                    continue
                audit = image_correspondence(raw, source.read_bytes())
                path = output / f"{index:03d}-{exposure['camera']}-{audit['received_sha256']}.jpg"
                with path.open("xb") as f:
                    f.write(raw)
                item["frames"].append(
                    dict(exposure, **audit, received_path=str(path), producer_path=str(source))
                )
        results.append(item)
    return dict(
        observations=results,
        selected_frames=len(exposures),
        received_calls=len(observations),
        accepted_frames=sum(f["accepted"] for r in results for f in r["frames"]),
        note="Timestamp/camera/image correspondence proves recorded receipt, not attention. Resized bytes remain nonidentical.",
    )


def _epoch(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def response_timing(receipt, calls, actions_from_client=()):
    """Separate available client timestamps from the producer-response latency proxy."""
    result = {}
    actions = [c for c in calls if c.get("tool") in {"set_motion", "stop_now"}]
    for camera in ("narrow", "wide"):
        first = next(
            (
                (r, f)
                for r in receipt["observations"]
                for f in r["frames"]
                if f.get("camera") == camera
                and f.get("accepted")
                and f.get("billboards", {}).get("hazard", {}).get("visible_pixels", 0) > 0
            ),
            None,
        )
        if first is None:
            result[camera] = None
            continue
        row, frame = first
        selected = _epoch(row["status_timestamp"])
        received = _epoch(row["client_timestamp"])
        client_command = next(
            (
                c
                for c in actions_from_client
                if c["first_event_index"] > row["event_index"] and c["category"] == "driving"
            ),
            None,
        )
        client_command_time = (
            _epoch(client_command.get("client_timestamp")) if client_command else None
        )
        command = next(
            (
                c
                for c in actions
                if _epoch(c.get("timestamp")) is not None
                and selected is not None
                and _epoch(c["timestamp"]) >= (received if received is not None else selected)
            ),
            None,
        )
        target = _epoch(command["timestamp"]) if command else None
        result[camera] = dict(
            frame=frame,
            call_id=row["call_id"],
            first_client_command=client_command,
            client_receipt_to_first_client_command_s=client_command_time - received
            if client_command_time is not None and received is not None
            else None,
            subsequent_client_actions=[
                c for c in actions_from_client if c["first_event_index"] > row["event_index"]
            ],
            first_server_recorded_command=command,
            client_receipt_to_first_server_record_s=target - received
            if target is not None and received is not None
            else None,
            producer_response_to_server_command_proxy_s=target - selected
            if target is not None and selected is not None
            else None,
            latency_note="Client action includes schema rejection. Server-recorded action may be later: rejected MCP calls never reach it. Unresolved tools need manual review. Missing stream timestamps prevent exact receipt-to-command latency; proxy is not attention time.",
        )
    return result


def stop_receipt_audit(terminal, calls, stream):
    assessment = (terminal or {}).get("assessment", {})
    identity = assessment.get("stop_tool_id")
    stop = next((c for c in calls if c["id"] == identity), None)
    return dict(
        server_stop_id=identity,
        completed_client_evidence=completed_stop(stream, stop) if stop else None,
        post_checkpoint_resume="not tested; episode ended at approved checkpoint",
    )
