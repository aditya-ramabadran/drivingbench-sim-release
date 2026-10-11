"""Offline review videos from recorded poses; never an observation-exposure record."""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import io
import json
import math
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

from drivingbench.device.cameras import adjust
from drivingbench.shared.contracts import ImageSettings
from drivingbench.sim.blind_driving import clock_bounds
from drivingbench.sim.cameras import SceneRenderer
from drivingbench.sim.event_study_v2_analysis import WorldClock
from drivingbench.sim.logs import REPO
from drivingbench.sim.video_camera import VideoSnapshotRenderer
from PIL import Image, ImageDraw, ImageFont

SIM_TIME_NOTICE = "SIM-TIME REPLAY: clock slowdown and thinking pauses omitted"


def read_events(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class Trajectory:
    """Interpolate continuous tick fields; branch and actor activation remain discrete."""

    def __init__(self, events):
        self.rows = []
        for row in events:
            if row.get("kind") not in {"sim_tick", "sim_terminal_state"}:
                continue
            value = copy.deepcopy(row)
            if not all(
                math.isfinite(float(value[key]))
                for key in ("monotonic_at", "sim_time_s", "speed_mps")
            ) or not all(math.isfinite(float(v)) for v in value["pose"]):
                raise ValueError("Trajectory values must be finite")
            if self.rows and value["monotonic_at"] < self.rows[-1]["monotonic_at"]:
                raise ValueError("Trajectory wall time moved backwards")
            if self.rows and value["sim_time_s"] < self.rows[-1]["sim_time_s"]:
                raise ValueError("Trajectory simulation time moved backwards")
            if self.rows and value["monotonic_at"] == self.rows[-1]["monotonic_at"]:
                self.rows[-1] = value
            else:
                self.rows.append(value)
        if len(self.rows) < 2:
            raise ValueError("A review video requires at least two distinct physics samples")
        self.times = [row["monotonic_at"] for row in self.rows]
        self.sim_times = [row["sim_time_s"] for row in self.rows]
        world = WorldClock(events)
        transitions = any(row.get("kind") == "sim_clock_rate" for row in events)
        live_pause = any(
            row.get("kind") in {"sim_pause", "sim_resume"}
            and self.times[0] <= world.origin + row["wall_time_s"] <= self.times[-1]
            for row in events
        )
        self.world = world if transitions or live_pause else None

    def at(self, stamp):
        if not math.isfinite(stamp) or not self.times[0] <= stamp <= self.times[-1]:
            raise ValueError("Review sample is outside the recorded trajectory")
        if self.world is not None:
            simulated = self.world.sim(stamp)
            if simulated is None:
                raise ValueError("Review sample lacks a recorded world-clock mapping")
            result = self._interpolate(simulated, self.sim_times)
            result["monotonic_at"] = stamp
            return result
        return self._interpolate(stamp, self.times)

    def at_sim(self, stamp):
        if not math.isfinite(stamp) or not self.sim_times[0] <= stamp <= self.sim_times[-1]:
            raise ValueError("Review sample is outside the recorded simulation trajectory")
        result = self._interpolate(stamp, self.sim_times)
        if self.world is not None:
            interval = self.world.wall_interval(stamp)
            if interval is None:
                raise ValueError("Review sample lacks a recorded inverse world-clock mapping")
            result["monotonic_at"] = interval[-1]
        return result

    def _interpolate(self, stamp, times):
        if stamp < times[0] - 1e-6 or stamp > times[-1] + 1e-6:
            raise ValueError("Recorded world clock exceeds the physics trajectory")
        stamp = min(times[-1], max(times[0], stamp))
        index = bisect.bisect_right(times, stamp) - 1
        left = self.rows[index]
        if index == len(self.rows) - 1:
            return copy.deepcopy(left)
        right = self.rows[index + 1]
        fraction = (stamp - times[index]) / (times[index + 1] - times[index])
        result = copy.deepcopy(left)
        for key in ("monotonic_at", "sim_time_s", "speed_mps"):
            result[key] = left[key] + fraction * (right[key] - left[key])
        result["pose"] = [a + fraction * (b - a) for a, b in zip(left["pose"], right["pose"])]
        angle = (right["pose"][2] - left["pose"][2] + 180) % 360 - 180
        result["pose"][2] = left["pose"][2] + fraction * angle
        if "actor_states" in right:
            result["actor_states"] = copy.deepcopy(right["actor_states"])
        if "video_fork" in left:
            result["video_fork"]["progress_m"] = left["video_fork"]["progress_m"] + fraction * (
                right["video_fork"]["progress_m"] - left["video_fork"]["progress_m"]
            )
        return result


def sample_times(start, end, fps=10):
    if not all(math.isfinite(v) for v in (start, end, fps)) or fps <= 0 or end <= start:
        raise ValueError("Expected finite increasing bounds and a positive frame rate")
    return [start + index / fps for index in range(math.floor((end - start) * fps) + 1)]


def operator_messages(attempt):
    """Follow-up messages sent into the same conversation, on the native monotonic clock."""
    messages = []
    for path in sorted(Path(attempt).glob("pressure-*/dispatch.json")):
        row = json.loads(path.read_text())
        messages.append(
            dict(
                monotonic_at=row["dispatch_monotonic_s"],
                text=f'Operator: "{row["prompt"]}"',
                source=str(path),
            )
        )
    return messages


def caption_calls(events, tools, *, approximate=False, reasons_only=False):
    """Map producer completion timestamps, keeping their clock uncertainty in evidence."""
    registration = clock_bounds([r for r in events if r.get("kind") == "sim_observe_delivery"])
    if approximate and registration["status"] == "inconsistent_clock":
        intervals = [
            [
                datetime.fromisoformat(
                    row["original_wall_timestamp"].replace("Z", "+00:00")
                ).timestamp()
                - row["delivered_at"],
                datetime.fromisoformat(
                    row["original_wall_timestamp"].replace("Z", "+00:00")
                ).timestamp()
                - row["arrived_at"],
            ]
            for row in events
            if row.get("kind") == "sim_observe_delivery" and row.get("original_wall_timestamp")
        ]
        registration = dict(
            status="caption_only_approximation",
            strict_registration=registration,
            anchor_offset_intervals_s=intervals,
            offset_s=[min(row[0] for row in intervals), max(row[1] for row in intervals)],
            interpretation="Envelope of observed clock anchors; not an absolute accuracy guarantee or exposure metric",
        )
    if registration["offset_s"] is None:
        raise ValueError(f"Tool-caption clock registration failed: {registration['status']}")
    lower, upper = registration["offset_s"]
    offset = (lower + upper) / 2
    captions = []
    for row in tools:
        if row.get("kind") != "tool":
            continue
        utc = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp()
        arguments = dict(row.get("arguments", {}))
        reason = arguments.pop("reason", "")
        if reasons_only:
            if reason:
                captions.append(dict(monotonic_at=utc - offset, text=reason, source=row.get("id")))
            continue
        parameters = ", ".join(f"{key}={value}" for key, value in arguments.items())
        outcome = row.get("outcome", {})
        status = outcome.get("status", outcome.get("state", ""))
        text = f"{row['tool']}({parameters}) → {status}"
        if reason:
            text += f" | {reason}"
        captions.append(dict(monotonic_at=utc - offset, text=text, source=row.get("id")))
    return sorted(captions, key=lambda row: row["monotonic_at"]), registration


def _font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def compose(
    images,
    snapshot,
    *,
    label,
    elapsed_s,
    caption="",
    panel_width=640,
    time_basis="wall",
    sim_elapsed_s=None,
):
    """Side-by-side cameras with a labelled missing panel, never an invented wide view."""
    height = round(panel_width * 760 / 1344)
    height += height % 2
    canvas = Image.new("RGB", (panel_width * 2, height + 116), (20, 20, 20))
    draw, font = ImageDraw.Draw(canvas), _font(19)
    for index, camera in enumerate(("narrow", "wide")):
        if camera in images:
            with Image.open(io.BytesIO(images[camera])) as image:
                fitted = image.convert("RGB").resize(
                    (panel_width, height), Image.Resampling.LANCZOS
                )
            canvas.paste(fitted, (index * panel_width, 0))
        draw.rectangle((index * panel_width, 0, index * panel_width + 230, 27), fill="black")
        draw.text(
            (index * panel_width + 8, 3),
            camera if camera in images else f"{camera}: unavailable",
            font=font,
            fill="white",
        )
    simulation_label = (
        f"sim +{sim_elapsed_s:.2f} s"
        if sim_elapsed_s is not None
        else f"sim {snapshot['sim_time_s']:.2f} s"
    )
    header = (
        f"{label} | wall +{elapsed_s:.1f} s | {simulation_label} | "
        f"{snapshot['speed_mps']:.2f} m/s | offline trajectory replay"
    )
    draw.text((8, height + 5), header, font=font, fill="white")
    caption_y, caption_lines = height + 31, 3
    if time_basis == "sim":
        draw.text((8, height + 29), SIM_TIME_NOTICE, font=_font(14), fill=(255, 200, 145))
        caption_y, caption_lines = height + 51, 2
    if "video_fork" in snapshot:
        state = snapshot["video_fork"]
        branch = state.get("locked_branch") or state.get("requested_branch")
        note = (
            f"Physical progress {state['progress_m']:.2f} m | selected branch {branch} | "
            f"{state.get('phase', 'approach')} | command selects branch, physics controls progress"
        )
        offset = 23 if time_basis == "sim" else 0
        draw.text((8, height + 29 + offset), note, font=font, fill=(170, 210, 245))
        caption_y, caption_lines = height + 54 + offset, 1 if offset else 2
    for index, line in enumerate(textwrap.wrap(caption, width=122)[:caption_lines]):
        draw.text((8, caption_y + index * 23), line, font=font, fill=(255, 227, 155))
    return canvas


CLEAN_NOTICE = "DrivingBench simulator · re-rendered from the recorded drive"


def compose_clean(images, *, label, caption=""):
    """Narrow camera only, with the title, the model's stated reason and a permanent notice."""
    with Image.open(io.BytesIO(images["narrow"])) as image:
        frame = image.convert("RGB")
    width, height = frame.size
    canvas = Image.new("RGB", (width, height + 150), (8, 9, 12))
    canvas.paste(frame, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title, small, body = _font(30), _font(19), _font(26)
    draw.rounded_rectangle(
        (18, 16, 30 + draw.textlength(label, font=title), 62), 10, fill=(8, 9, 12)
    )
    draw.text((24, 20), label, font=title, fill="white")
    notice = draw.textlength(CLEAN_NOTICE, font=small)
    draw.text(
        (width - 22 - notice, height + 150 - 28), CLEAN_NOTICE, font=small, fill=(150, 154, 160)
    )
    for index, line in enumerate(textwrap.wrap(caption, width=92)[:3]):
        color = (255, 214, 120) if caption.startswith("Operator:") else (236, 231, 223)
        draw.text((28, height + 12 + index * 34), line, font=body, fill=color)
    return canvas


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_review(
    attempt,
    output,
    *,
    label,
    scene=None,
    start_s=0,
    end_s=None,
    fps=10,
    panel_width=640,
    renderer_factory=None,
    approximate_captions=False,
    time_basis="wall",
    style="review",
):
    """Create a new review directory, MP4 and source manifest from one closed attempt."""
    if style not in {"review", "clean"}:
        raise ValueError("Review style must be review or clean")
    attempt, output = Path(attempt).resolve(), Path(output).resolve()
    metadata_path = attempt / "episode/metadata.json"
    event_path = attempt / "episode/events.jsonl"
    metadata = json.loads(metadata_path.read_text())["metadata"]
    scene_path = Path(scene or metadata["scene"]).resolve()
    if _sha(scene_path) != metadata["scene_sha256"]:
        raise ValueError("Replay scene differs from the scene recorded for this episode")
    scene_data = json.loads(scene_path.read_text())
    assets = dict(metadata.get("assets", {}))
    for item in (metadata.get("wide_camera") or {}).values():
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            assets[item["path"]] = item["sha256"]
    calibration = scene_data.get("calibration")
    if calibration and metadata.get("calibration_sha256"):
        assets[calibration] = metadata["calibration_sha256"]
    for path, expected in assets.items():
        source = Path(path)
        source = source if source.is_absolute() else REPO / source
        if _sha(source) != expected:
            raise ValueError(f"Recorded render asset changed: {source}")
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("calibration", "calibration_sha256"),
    ):
        path, expected = metadata.get(path_key), metadata.get(hash_key)
        if path and expected and _sha(path) != expected:
            raise ValueError(f"Recorded {path_key} changed: {path}")
    events = read_events(event_path)
    trajectory = Trajectory(events)
    tool_paths = sorted((attempt / "producer/recordings/segments").glob("*/events.jsonl"))
    tools = [row for path in tool_paths for row in read_events(path) if row.get("kind") == "tool"]
    captions, registration = caption_calls(
        events, tools, approximate=approximate_captions, reasons_only=style == "clean"
    )
    captions = sorted(captions + operator_messages(attempt), key=lambda row: row["monotonic_at"])
    first = trajectory.times[0]
    if time_basis not in {"wall", "sim"}:
        raise ValueError("Review time_basis must be wall or sim")
    basis = trajectory.times if time_basis == "wall" else trajectory.sim_times
    start, end = basis[0] + start_s, basis[-1] if end_s is None else basis[0] + end_s
    if start < basis[0] or end > basis[-1]:
        raise ValueError("Review interval exceeds recorded physics")
    samples = sample_times(start, end, fps)
    sample = trajectory.at if time_basis == "wall" else trajectory.at_sim
    first_snapshot, last_snapshot = sample(start), sample(end)
    if panel_width < 320 or panel_width % 2:
        raise ValueError("Panel width must be even and at least 320 pixels")
    settings_path = attempt / "producer/settings.json"
    settings_record = json.loads(settings_path.read_text())
    revisions = {
        row["settings_revision"] for row in tools if row.get("settings_revision") is not None
    }
    if revisions and revisions != {settings_record.get("revision")}:
        raise ValueError("Replay requires the recorded image settings for every tool revision")
    settings = settings_record["settings"]
    adjustments = ImageSettings(**settings["image_adjustments"])
    is_video = "video_camera" in scene_data
    factory = renderer_factory or (VideoSnapshotRenderer if is_video else SceneRenderer)
    output.mkdir(parents=True, exist_ok=False)
    renderer = factory(scene_path, output / "renderer")
    frames_path = output / "frames.jsonl"
    encoder = None
    calls_at = [row["monotonic_at"] for row in captions]
    try:
        renderer.start()
        with (output / "encoder.log").open("wb") as errors, frames_path.open("x") as frames:
            for index, stamp in enumerate(samples):
                snapshot = sample(stamp)
                wall_stamp = snapshot["monotonic_at"]
                images = (
                    {"narrow": renderer.render_snapshot(snapshot)}
                    if is_video
                    else renderer.render_frames(snapshot)
                )
                expected = {"narrow"} if is_video else {"narrow", "wide"}
                if set(images) != expected:
                    raise RuntimeError(
                        f"Review renderer returned {set(images)}, expected {expected}"
                    )
                images = {name: adjust(jpeg, adjustments) for name, jpeg in images.items()}
                call_index = bisect.bisect_right(calls_at, wall_stamp) - 1
                active = captions[call_index] if call_index >= 0 else None
                caption = (
                    active["text"] if active and wall_stamp - active["monotonic_at"] <= 8 else ""
                )
                if caption and registration["status"] == "caption_only_approximation":
                    caption = "[Approximate caption timing] " + caption
                image = (
                    compose_clean(images, label=label, caption=caption)
                    if style == "clean"
                    else compose(
                        images,
                        snapshot,
                        label=label,
                        elapsed_s=wall_stamp - first,
                        caption=caption,
                        panel_width=panel_width,
                        time_basis=time_basis,
                        sim_elapsed_s=(snapshot["sim_time_s"] - trajectory.sim_times[0])
                        if time_basis == "sim"
                        else None,
                    )
                )
                if encoder is None:
                    encoder = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-threads",
                            "1",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            f"{image.width}x{image.height}",
                            "-framerate",
                            str(fps),
                            "-i",
                            "pipe:0",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "fast",
                            "-crf",
                            "27",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            "-threads",
                            "2",
                            "-n",
                            str(output / "replay.mp4"),
                        ],
                        stdin=subprocess.PIPE,
                        stderr=errors,
                    )
                encoder.stdin.write(image.tobytes())
                frames.write(
                    json.dumps(
                        dict(
                            frame=index,
                            wall_elapsed_s=wall_stamp - first,
                            sim_time_s=snapshot["sim_time_s"],
                            playback_elapsed_s=stamp - start,
                            time_basis=time_basis,
                            pose=snapshot["pose"],
                            actor_onsets=snapshot.get("actor_onsets", {}),
                            **(
                                {"actor_states": snapshot["actor_states"]}
                                if "actor_states" in snapshot
                                else {}
                            ),
                            tool_caption_source=active["source"] if caption else None,
                            cameras=sorted(images),
                        )
                    )
                    + "\n"
                )
            encoder.stdin.close()
            if encoder.wait(timeout=60):
                raise RuntimeError("Video encoder failed; see encoder.log")
    finally:
        renderer.close()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    sources = [metadata_path, event_path, scene_path, settings_path, *tool_paths]
    report = dict(
        source_attempt=str(attempt),
        sources={str(path): _sha(path) for path in sources},
        video=str(output / "replay.mp4"),
        video_sha256=_sha(output / "replay.mp4"),
        fps=fps,
        frames=len(samples),
        source_wall_interval_s=[
            first_snapshot["monotonic_at"] - first,
            last_snapshot["monotonic_at"] - first,
        ],
        source_sim_interval_s=[first_snapshot["sim_time_s"], last_snapshot["sim_time_s"]],
        time_basis=time_basis,
        style=style,
        operator_messages=[
            row["source"] for row in captions if row["text"].startswith("Operator:")
        ],
        treatment_notice=SIM_TIME_NOTICE if time_basis == "sim" else None,
        world_clock_mapping="Recorded rate/pause/resume anchors"
        if trajectory.world is not None
        else "Physics tick interpolation; no in-drive clock transition",
        playback_duration_s=len(samples) / fps,
        settings=settings,
        clock_registration=registration,
        caption_timestamp="Producer tool completion UTC; midpoint of "
        + (
            "observed anchor envelope (approximate)"
            if registration["status"] == "caption_only_approximation"
            else "bounded clock registration"
        ),
        maximum_caption_clock_uncertainty_s=(
            registration["offset_s"][1] - registration["offset_s"][0]
        )
        / 2,
        maximum_tick_gap_s=max(b - a for a, b in zip(trajectory.times, trajectory.times[1:])),
        recorded_renderer_sources=metadata.get("renderer", {}),
        replay_renderer_sources={
            name: _sha(Path(__file__).with_name(name))
            for name in metadata.get("renderer", {})
            if Path(__file__).with_name(name).exists()
        },
        replay_utility_sha256=_sha(__file__),
        interpretation="Offline render from interpolated recorded physics, not images received by the model; no physics rerun.",
        cameras=["narrow"] if is_video else ["narrow", "wide"],
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--start-s", type=float, default=0)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--approximate-captions", action="store_true")
    parser.add_argument(
        "--time-basis",
        choices=("wall", "sim"),
        default="wall",
        help="Sampling and trim units; sim omits slowed time and pauses, with an explicit treatment banner",
    )
    parser.add_argument(
        "--style",
        choices=("review", "clean"),
        default="review",
        help="clean: narrow camera, the model's stated reasons and operator messages, for presentation",
    )
    args = vars(parser.parse_args())
    print(json.dumps(render_review(**args), indent=2))


if __name__ == "__main__":
    main()
