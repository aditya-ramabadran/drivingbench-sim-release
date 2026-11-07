"""Render the attempts declared in traces/collections.json into a release folder.

For every attempt (a "part"): openpilot's route video cut to the attempt as H.264 clips
(`road.mp4`, `wide.mp4`), a `timeline.json` keyed to video frames (tool calls with the
model's reasons, 5 Hz telemetry, segment bounds), the frames the model saw, an annotated
clip with the calls burned in, and an SRT of the calls. For every group (one model's
trial): the chat transcript as markdown. Reads traces/ and the staged artifacts under
runs/ and writes only under the output folder; nothing recorded is touched.
"""

import bisect
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from drivingbench.gateway.artifacts import segment_facts
from drivingbench.gateway.traces import read_events

FPS = 20  # openpilot encodes both road cameras at 20 fps
MARGIN_BEFORE_S = 3  # video before the model's first tool call
MARGIN_AFTER_S = 2  # and after the last: enough to see the stop, not the operator taking over
CAMERAS = {
    "narrow": ("fcamera.hevc", "narrowRoadEncodeIdx"),
    "wide": ("ecamera.hevc", "wideRoadEncodeIdx"),
}
BITRATE = {"narrow": "6M", "wide": "4M"}
DEFAULT_SCALE_DEG = 180  # steering scale during these recordings


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="milliseconds")


# --- route index: wall-clock time of every recorded video frame --------------------------------


def read_route(repo: Path, route: Path) -> dict:
    """Per camera: wall time of frame 0, mean frame period, frame count; from the rlog."""
    import zstandard

    sys.path[:0] = [str(repo / "vendor/openpilot"), str(repo / "vendor/openpilot/opendbc_repo")]
    from openpilot.cereal import log

    data = zstandard.ZstdDecompressor().stream_reader((route / "rlog.zst").open("rb")).read()
    offsets, frames = [], {camera: {} for camera in CAMERAS}
    for message in log.Event.read_multiple_bytes(data):
        kind = message.which()
        if kind == "clocks":
            offsets.append(message.clocks.wallTimeNanos - message.logMonoTime)
        for camera, (_, service) in CAMERAS.items():
            if kind == service:
                index = getattr(message, service)
                frames[camera][index.segmentId] = index.timestampEof
    result = {"offset_ns": statistics.median(offsets) if offsets else None}
    for camera, stamps in frames.items():
        ids = sorted(stamps)
        if not ids or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError(f"{route.name}: {camera} frames are not contiguous in the log")
        first, last = stamps[ids[0]], stamps[ids[-1]]
        result[camera] = {
            "first_id": ids[0],
            "n": len(ids),
            "mono_first_ns": first,
            "dt": (last - first) / 1e9 / max(len(ids) - 1, 1),
        }
    return result


def route_index(repo: Path, routes: Path, cache: Path) -> dict:
    """All staged route minutes, cached as JSON; missing clock offsets borrow from the same boot."""
    index = json.loads(cache.read_text()) if cache.is_file() else {}
    for route in sorted(routes.iterdir()):
        if route.name not in index and (route / "rlog.zst").is_file():
            index[route.name] = read_route(repo, route)
            cache.write_text(json.dumps(index, indent=1))
    for name, entry in index.items():
        if entry["offset_ns"] is None:
            boot = name.rsplit("--", 1)[0]
            peers = [
                v["offset_ns"] for k, v in index.items() if k.startswith(boot) and v["offset_ns"]
            ]
            entry["offset_ns"] = statistics.median(peers)
    for entry in index.values():
        for camera in CAMERAS:
            if camera in entry:
                entry[camera]["t0"] = (entry[camera]["mono_first_ns"] + entry["offset_ns"]) / 1e9
    return index


def frames_between(index: dict, camera: str, start: float, end: float) -> list[tuple]:
    """(wall, route, frame position in its file) for every frame of one camera in the window."""
    frames = []
    for name, entry in index.items():
        if camera not in entry:
            continue
        cam = entry[camera]
        for k in range(cam["n"]):
            wall = cam["t0"] + k * cam["dt"]
            if start <= wall <= end:
                frames.append((wall, name, cam["first_id"] + k))
    frames.sort()
    return frames


# --- video ---------------------------------------------------------------------------------------


def encoder(bitrate: str) -> list[str]:
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    if "h264_videotoolbox" in probe.stdout:
        return ["-c:v", "h264_videotoolbox", "-b:v", bitrate, "-allow_sw", "1"]
    return ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]


def cut(routes: Path, index: dict, camera: str, frames: list[tuple], output: Path) -> None:
    """Concatenate the raw HEVC minutes the frames span and trim to exactly those frames."""
    names = list(dict.fromkeys(name for _, name, _ in frames))
    offsets, total = {}, 0
    for name in names:
        offsets[name] = total
        total += index[name][camera]["n"]
    first = offsets[frames[0][1]] + frames[0][2] - index[frames[0][1]][camera]["first_id"]
    last = offsets[frames[-1][1]] + frames[-1][2] - index[frames[-1][1]][camera]["first_id"]
    with tempfile.NamedTemporaryFile(suffix=".hevc", delete=False) as raw:
        for name in names:
            raw.write((routes / name / CAMERAS[camera][0]).read_bytes())
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "hevc",
                "-r",
                str(FPS),
                "-i",
                raw.name,
                "-vf",
                f"trim=start_frame={first}:end_frame={last + 1},setpts=PTS-STARTPTS",
                *encoder(BITRATE[camera]),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    finally:
        Path(raw.name).unlink(missing_ok=True)


# --- timeline ------------------------------------------------------------------------------------


def frame_of(walls: list[float], t: float) -> int:
    """Index of the video frame nearest to a wall-clock time."""
    i = bisect.bisect_left(walls, t)
    if i == 0:
        return 0
    if i >= len(walls):
        return len(walls) - 1
    return i if walls[i] - t < t - walls[i - 1] else i - 1


def timeline(traces: Path, segments: list[str], walls: list[float], meta: dict) -> dict:
    calls, telemetry, bounds = [], [], []
    for segment in segments:
        events = read_events(traces / "segments" / segment / "events.jsonl")
        bounds.append(
            {
                "segment": segment,
                "t_start": frame_of(walls, epoch(events[0]["timestamp"])) / FPS,
                "t_end": frame_of(walls, epoch(events[-1]["timestamp"])) / FPS,
                "ended_by": next(
                    (e.get("reason") for e in reversed(events) if e["kind"] == "segment_end"), None
                ),
            }
        )
        samples = 0
        for event in events:
            t = frame_of(walls, epoch(event["timestamp"])) / FPS
            if event["kind"] == "tool":
                calls.append(
                    {
                        "t": round(t, 2),
                        "timestamp": event["timestamp"],
                        "tool": event["tool"],
                        "client": event.get("client"),
                        "arguments": event.get("arguments") or {},
                        "outcome": event.get("outcome") or {},
                        "frames": [
                            f"{frame_of(walls, epoch(event['timestamp'])):06d}-{image['camera']}-{image['url'].rsplit('/', 1)[-1][:8]}.jpg"
                            for image in event.get("images") or []
                        ],
                        "segment": segment,
                    }
                )
            elif event["kind"] == "telemetry":
                samples += 1
                if samples % 2:  # 10 Hz recorded, 5 Hz released
                    telemetry.append(
                        {
                            "t": round(t, 2),
                            "speed_mps": event.get("speed_mps"),
                            "steering_deg": event.get("steering_deg"),
                            "state": event.get("state"),
                            "reason": event.get("reason"),
                            "remaining_s": event.get("remaining_s"),
                        }
                    )
    return {
        **meta,
        "video": {
            "road": "road.mp4",
            "wide": "wide.mp4",
            "annotated": "road-annotated.mp4",
            "fps": FPS,
            "frames": len(walls),
            "duration_s": round(len(walls) / FPS, 2),
            "t0": iso(walls[0]),
            "steering_scale_deg": DEFAULT_SCALE_DEG,
        },
        "segments": bounds,
        "calls": calls,
        "telemetry": telemetry,
    }


def srt(tl: dict, output: Path) -> None:
    def stamp(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    lines = []
    calls = tl["calls"]
    for n, call in enumerate(calls, 1):
        end = calls[n]["t"] if n < len(calls) else tl["video"]["duration_s"]
        lines += [
            str(n),
            f"{stamp(call['t'])} --> {stamp(max(end, call['t'] + 0.5))}",
            describe(call),
            "",
        ]
    output.write_text("\n".join(lines))


def describe(call: dict) -> str:
    a, tool = call["arguments"], call["tool"]
    if tool == "set_motion":
        head = f"set_motion {a.get('direction')} {a.get('steering_percent', 0):.0f}% · {a.get('speed_mps')} m/s · {a.get('duration_s'):.0f} s"
    else:
        head = tool
    if call["outcome"].get("status") not in (None, "accepted"):
        head += f" → {call['outcome'].get('status')}" + (
            f" ({call['outcome']['reason']})" if call["outcome"].get("reason") else ""
        )
    reason = a.get("reason")
    return head + (f'\n"{reason}"' if reason else "")


# --- annotated video -----------------------------------------------------------------------------


def annotate(road: Path, tl: dict, output: Path) -> None:
    """Burn a bottom band into road.mp4: time, speed, measured steering bar, the active call."""
    from PIL import Image

    width, height = probe_size(road)
    band = 168
    fonts = load_fonts()
    calls, telemetry = tl["calls"], tl["telemetry"]
    tele_times = [s["t"] for s in telemetry]
    scale = tl["video"]["steering_scale_deg"]
    decode = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(road),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    encode = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "-",
            *encoder(BITRATE["narrow"]),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    cache = {}
    size = width * height * 3
    commands = [c for c in calls if c["tool"] != "observe"]
    command_times = [c["t"] for c in commands]
    observes = [c["t"] for c in calls if c["tool"] == "observe"]
    for k in range(tl["video"]["frames"]):
        raw = decode.stdout.read(size)
        if len(raw) < size:
            break
        t = k / FPS
        ci = bisect.bisect_right(command_times, t) - 1
        oi = bisect.bisect_right(observes, t) - 1
        observing = oi >= 0 and t - observes[oi] < 1.5
        ti = max(bisect.bisect_right(tele_times, t) - 1, 0)
        sample = telemetry[ti] if telemetry else {}
        speed = round(sample.get("speed_mps") or 0, 1)
        steer = round((sample.get("steering_deg") or 0) / scale * 100)
        key = (int(t), ci, speed, steer, observing, sample.get("state"), sample.get("reason"))
        if key not in cache:
            cache.clear()
            cache[key] = draw_band(
                width,
                band,
                fonts,
                t,
                speed,
                steer,
                commands[ci] if ci >= 0 else None,
                sample,
                observing,
            )
        frame = Image.frombuffer("RGB", (width, height), raw, "raw", "RGB", 0, 1).copy()
        frame.paste(cache[key], (0, height - band), cache[key])
        encode.stdin.write(frame.tobytes())
    decode.stdout.close()
    encode.stdin.close()
    decode.wait()
    encode.wait()
    if encode.returncode:
        raise RuntimeError(f"annotation encode failed for {road}")


def frame_count(video: Path) -> int:
    """From the container when it knows (mp4), else by decoding."""
    for query in (
        ["-show_entries", "stream=nb_frames"],
        ["-count_frames", "-show_entries", "stream=nb_read_frames"],
    ):
        out = (
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    *query,
                    "-of",
                    "csv=p=0",
                    str(video),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .split(",")[0]
        )
        if out.isdigit():
            return int(out)
    raise ValueError(f"cannot count frames in {video}")


def probe_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def load_fonts():
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return {
                "big": ImageFont.truetype(path, 34),
                "mid": ImageFont.truetype(path, 26),
                "small": ImageFont.truetype(path, 22),
            }
    default = ImageFont.load_default()
    return {"big": default, "mid": default, "small": default}


def draw_band(width, band, fonts, t, speed, steer, call, sample, observing):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, band), (0, 0, 0, 165))
    draw = ImageDraw.Draw(image)
    # Left column: clock, speed, steering bar (measured, positive left).
    draw.text((24, 14), f"{int(t // 60):02d}:{int(t % 60):02d}", font=fonts["big"], fill="white")
    draw.text((24, 60), f"{speed:.1f} m/s", font=fonts["mid"], fill="white")
    state = sample.get("state") or ""
    draw.text(
        (24, 96),
        state + (f" · {sample['reason']}" if sample.get("reason") else ""),
        font=fonts["small"],
        fill=(200, 200, 200, 255),
    )
    bar_x, bar_y, bar_w, bar_h = 24, 130, 300, 18
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline="white", width=2)
    mid = bar_x + bar_w // 2
    fill = int(max(-100, min(100, steer)) / 100 * (bar_w // 2))
    if fill:  # positive left: fill from the middle toward the left
        draw.rectangle(
            [min(mid, mid - fill), bar_y + 3, max(mid, mid - fill), bar_y + bar_h - 3], fill="white"
        )
    draw.text(
        (bar_x + bar_w + 12, bar_y - 4),
        f"steering {'L' if steer > 0 else 'R' if steer < 0 else ''} {abs(steer)}%",
        font=fonts["small"],
        fill="white",
    )
    # Right block: the last command; an observe is flashed briefly in the corner.
    x0 = 420
    if observing:
        label = "observe"
        x = width - 24 - draw.textlength(label, font=fonts["small"])
        draw.ellipse([x - 36, 18, x - 10, 44], fill=(90, 160, 240, 255))
        draw.text((x, 16), label, font=fonts["small"], fill="white")
    if call:
        colors = {"set_motion": (60, 200, 90, 255), "stop_now": (230, 70, 70, 255)}
        draw.ellipse([x0, 18, x0 + 26, 44], fill=colors.get(call["tool"], "white"))
        age = t - call["t"]
        head = describe(call).split("\n")[0]
        draw.text((x0 + 40, 14), f"{head}   +{age:.1f} s", font=fonts["mid"], fill="white")
        reason = call["arguments"].get("reason") or ""
        for i, line in enumerate(wrap(reason, fonts["small"], width - x0 - 60, draw)[:3]):
            draw.text((x0 + 40, 56 + i * 30), line, font=fonts["small"], fill=(235, 235, 235, 255))
        if call["outcome"].get("status") not in (None, "accepted"):
            draw.text(
                (x0 + 40, band - 32),
                f"outcome: {call['outcome'].get('status')} {call['outcome'].get('reason') or ''}",
                font=fonts["small"],
                fill=(255, 190, 120, 255),
            )
    return image


def wrap(text, font, max_width, draw):
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


# --- conversations: the driving chat, and nothing else -------------------------------------------
#
# Chat transcripts carry far more than the conversation (client state, paths, account details).
# Only an allowlist of record types and fields is read into turns; everything else never leaves
# the raw file. The turn schema is what the public dataset and website consume:
#   {"t": UTC iso | null, "role": "user"|"assistant"|"tool", "kind": "text"|"call"|"result"|"thinking",
#    "tool": name | null, "arguments": dict | null, "text": str | null}

MCP_TOOLS = ("observe", "set_motion", "stop_now")
TOOL_PREFIX = re.compile(r"^(mcp__drivingbench_sandbox__|drivingbench_sandbox[._])")
EXEC_CALL = re.compile(r"tools\.mcp__drivingbench_sandbox__(\w+)\((\{.*\})?\)", re.S)
PROMPT_START = "Use only the `drivingbench_sandbox` MCP"
IMAGE_REF = re.compile(r"\[image [^\]]+\]")


def turn(role, kind, *, t=None, tool=None, arguments=None, text=None) -> dict:
    return {"t": t, "role": role, "kind": kind, "tool": tool, "arguments": arguments, "text": text}


def utc(stamp: str | None) -> str | None:
    if not stamp:
        return None
    return (
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        .astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
    )


def clean(text: str) -> str:
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = re.sub(r"<timestamp>.*?</timestamp>", "", text, flags=re.S)
    text = re.sub(r"</?user_query>", "", text)
    return text.strip()


def call_turn(name: str, arguments, t=None) -> dict:
    """An MCP driving tool keeps its arguments; any other tool is named only."""
    name = TOOL_PREFIX.sub("", name)
    if name not in MCP_TOOLS:
        return turn("assistant", "call", t=t, tool=name)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = None
    return turn(
        "assistant",
        "call",
        t=t,
        tool=name,
        arguments=arguments if isinstance(arguments, dict) else {},
    )


def result_turn(text: str, t=None) -> dict | None:
    """Tool output: the JSON the tool returned, images counted rather than embedded."""
    text = text.strip()
    if not text:
        return None
    images = len(IMAGE_REF.findall(text))
    body = unwrap_result(IMAGE_REF.sub("", text).strip())
    return turn("tool", "result", t=t, text=body + (f" [{images} image(s)]" if images else ""))


def unwrap_result(body: str) -> str:
    """Some clients log the MCP envelope {"content": [{"type": "text", "text": ...}]} around the
    tool's JSON; keep the tool's own text."""
    try:
        envelope = json.loads(body)
    except ValueError:
        return body
    if isinstance(envelope, dict) and isinstance(envelope.get("content"), list):
        return "\n".join(
            i.get("text", "")
            for i in envelope["content"]
            if isinstance(i, dict) and i.get("type") == "text"
        ).strip()
    return body


def conversation(path: Path) -> list[dict]:
    """One chat file → allowlisted turns, starting at the driving prompt."""
    client = path.name.split("-", 1)[0]
    reader = {"codex": codex_turns, "claude": claude_turns, "cursor": cursor_turns}.get(client)
    if reader is None:
        return []  # harness logs and unknown clients are not conversations
    turns = []
    for entry in path.read_text(errors="replace").splitlines():
        try:
            turns += reader(json.loads(entry))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return keep_driving(turns)


def chapter(text: str | None) -> str | None:
    """The operator's prompts that structure a trial: the objective, then after every attempt a
    reflection prompt and a continuation prompt (a forked thread may be told just "Continue")."""
    if not text:
        return None
    if text.startswith(PROMPT_START):
        return "objective"
    if "previous attempt has now ended" in text:
        return "reflection"
    if "granted another attempt" in text or text.strip() == "Continue":
        return "continuation"
    return None


def keep_driving(turns: list[dict]) -> list[dict]:
    """Drop everything before the first operator prompt (the objective, or the continuation a
    forked thread starts with), and tool results that do not follow an MCP call."""
    started, mcp_call, kept = False, False, []
    for item in turns:
        if not started:
            started = (
                item["kind"] == "text"
                and item["role"] == "user"
                and chapter(item["text"]) is not None
            )
            if not started:
                continue
        if item["kind"] == "call":
            mcp_call = item["tool"] in MCP_TOOLS
        elif item["kind"] == "result":
            if not mcp_call:
                continue
            mcp_call = False
        kept.append(item)
    return kept


def text_turns(role: str, text: str, t=None) -> list[dict]:
    text = clean(text)
    if not text or text.startswith("<"):
        return []
    return [turn(role, "text", t=t, text=text)]


def codex_turns(entry: dict) -> list[dict]:
    if entry.get("type") != "response_item":
        return []
    payload, t = entry["payload"], utc(entry.get("timestamp"))
    kind = payload.get("type")
    if kind == "message" and payload.get("role") in ("user", "assistant"):
        text = "\n".join(
            c.get("text", "") for c in payload.get("content", []) if isinstance(c, dict)
        )
        return text_turns(payload["role"], text, t)
    if kind in ("custom_tool_call", "function_call"):
        name, arguments = payload.get("name", ""), payload.get("input") or payload.get("arguments")
        if name == "exec":  # Codex calls the MCP from a script; keep the tool, never the source
            match = EXEC_CALL.search(str(arguments))
            if not match:
                return [turn("assistant", "call", t=t, tool="exec")]
            name, body = match.groups()
            try:
                arguments = json.loads(re.sub(r"(\w+)\s*:", r'"\1":', body or "{}"))
            except ValueError:
                arguments = {}
        return [call_turn(name, arguments, t)]
    if kind in ("custom_tool_call_output", "function_call_output"):
        output = payload.get("output")
        if isinstance(output, str) and output.startswith("["):
            try:
                output = json.loads(output)
            except ValueError:
                pass
        if isinstance(output, list):  # a script result: its text items, images counted
            texts = [i.get("text", "") for i in output if i.get("type") == "input_text"]
            images = sum(i.get("type") == "input_image" for i in output)
            output = "\n".join(x for x in texts if not x.startswith("Script completed")).strip()
            output += f" [{images} image(s)]" if images else ""
        result = result_turn(output if isinstance(output, str) else json.dumps(output), t)
        return [result] if result else []
    if kind == "reasoning":
        summary = "\n".join(
            x.get("text", "") for x in payload.get("summary", []) if isinstance(x, dict)
        )
        return [turn("assistant", "thinking", t=t, text=summary.strip())] if summary.strip() else []
    return []


def content_turns(role: str, content, t=None) -> list[dict]:
    """Claude / Cursor message content: text, tool_use, tool_result, thinking."""
    if isinstance(content, str):
        return text_turns(role, content, t)
    out = []
    for item in content or []:
        kind = item.get("type")
        if kind == "text":
            out += text_turns(role, item.get("text", ""), t)
        elif kind == "tool_use":
            name, arguments = item.get("name", ""), item.get("input") or {}
            if name == "CallDynamicTool":  # Cursor wraps MCP calls
                name, arguments = arguments.get("toolName", ""), arguments.get("arguments")
            out.append(call_turn(name, arguments, t))
        elif kind == "tool_result":
            parts = item.get("content")
            text = (
                parts
                if isinstance(parts, str)
                else "\n".join(p.get("text", "") for p in parts or [] if isinstance(p, dict))
            )
            if result := result_turn(text, t):
                out.append(result)
        elif kind == "thinking" and item.get("thinking", "").strip():
            out.append(turn("assistant", "thinking", t=t, text=item["thinking"].strip()))
    return out


def claude_turns(entry: dict) -> list[dict]:
    if entry.get("type") not in ("user", "assistant"):
        return []
    return content_turns(
        entry["message"]["role"], entry["message"].get("content"), utc(entry.get("timestamp"))
    )


def cursor_turns(entry: dict) -> list[dict]:
    if entry.get("role") not in ("user", "assistant"):
        return []
    return content_turns(entry["role"], (entry.get("message") or {}).get("content"))


def transcript_markdown(title: str, chats: dict[str, list[dict]]) -> str:
    lines = [
        f"# {title} — chat transcript",
        "",
        "Only the driving conversation: prompts, the model's text and thinking summaries, its tool calls "
        "with their reasons, and the tools' replies (images counted, the frames themselves are in each "
        "attempt's `frames/`). Client state and everything outside the conversation is not included.",
        "",
    ]
    for name, turns in chats.items():
        lines += [f"## {name}", ""]
        for item in turns:
            when = f" · {item['t'][11:19]} UTC" if item["t"] else ""
            if item["kind"] == "text":
                lines += [f"**{item['role']}**{when}", "", item["text"], ""]
            elif item["kind"] == "thinking":
                lines += [f"*thinking: {item['text']}*", ""]
            elif item["kind"] == "call":
                args = (
                    json.dumps(item["arguments"], ensure_ascii=False) if item["arguments"] else ""
                )
                lines += [f"→ `{item['tool']}` {args}{when}".rstrip(), ""]
            elif item["kind"] == "result":
                text = item["text"]
                lines += [f"← `{text[:400]}{'…' if len(text) > 400 else ''}`", ""]
    return "\n".join(lines) + "\n"


def group_chats(traces: Path, segments: list[str], client: str | None = None) -> list[Path]:
    """The chat transcripts attached to these segments, one per transcript: a chat that drove
    several segments is attached to each of them as the same file."""
    by_name: dict[str, Path] = {}
    for segment in segments:
        for path in (traces / "segments" / segment / "chat").glob(f"{client or '*'}-*"):
            by_name.setdefault(path.name, path)
    return [by_name[name] for name in sorted(by_name)]


def numbered(clients: list[str]) -> list[str]:
    """Public chat names, in order: the client and, when a client has several chats, a number."""
    total, seen = Counter(clients), Counter()
    names = []
    for client in clients:
        seen[client] += 1
        names.append(f"{client} chat" + (f" {seen[client]}" if total[client] > 1 else ""))
    return names


def chat_labels(chat_files: list[Path]) -> dict[str, str]:
    """Public names by transcript filename. Transcript filenames carry the clients' thread ids
    and never leave the private repo."""
    names = sorted({p.name for p in chat_files})
    return dict(zip(names, numbered([name.split("-", 1)[0] for name in names])))


def write_conversation(folder: Path, title: str, chat_files: list[Path]) -> dict[str, list[dict]]:
    """conversation.json and transcript.md for one group, from the allowlisted turns only."""
    labels = chat_labels(chat_files)
    chats = {labels[p.name]: conversation(p) for p in chat_files}
    chats = {name: turns for name, turns in chats.items() if turns}
    (folder / "conversation.json").write_text(
        json.dumps(chats, indent=1, ensure_ascii=False) + "\n"
    )
    (folder / "transcript.md").write_text(transcript_markdown(title, chats))
    return chats


def cut_attempt(turns: list[dict], first_call: float, last_call: float) -> list[dict]:
    """The chat without the attempt whose tool calls ran first_call..last_call (epoch seconds):
    from the operator prompt that started it up to the prompt that started the next one, plus
    the reflection prompt that led into it when the chat carries one with no driving in between
    (it promises an attempt that is not here). A chat with neither is returned unchanged."""
    times, t = [], None
    for item in turns:  # untimed turns (Cursor) inherit the previous time
        t = epoch(item["t"]) if item["t"] else t
        times.append(t)

    def prompts(*kinds):
        return [
            i
            for i, item in enumerate(turns)
            if item["kind"] == "text"
            and item["role"] == "user"
            and times[i] is not None
            and chapter(item["text"]) in kinds
        ]

    drove = any(
        item["kind"] == "call" and t is not None and first_call - 1 <= t <= last_call + 1
        for item, t in zip(turns, times)
    )
    start = stop = len(turns)
    if drove:
        starts = prompts("objective", "continuation")
        start = max((i for i in starts if times[i] <= first_call + 1), default=0)
        stop = next((i for i in starts if i > start and times[i] > last_call), len(turns))
    lead = max(
        (i for i in prompts("reflection") if i < start and times[i] <= first_call), default=None
    )
    if lead is not None and not any(item["kind"] == "call" for item in turns[lead:start]):
        start = lead
    return turns[:start] + turns[stop:]


def cut_attempts(turns: list[dict], timelines: list[dict]) -> list[dict]:
    """The chat with every attempt in `timelines` cut out."""
    for tl in timelines:
        stamps = [epoch(c["timestamp"]) for c in tl["calls"]]
        if stamps:
            turns = cut_attempt(turns, min(stamps), max(stamps))
    return turns


def without_attempts(chats: dict[str, list[dict]], timelines: list[dict]) -> dict[str, list[dict]]:
    """The group's chats with every excluded attempt cut out; a chat left with nothing from the
    model (an abandoned operator prompt, say) disappears, and the rest are numbered afresh so the
    public names do not hint at what was left out."""
    kept = [
        (name, turns)
        for name, turns in ((n, cut_attempts(t, timelines)) for n, t in chats.items())
        if any(item["role"] == "assistant" for item in turns)
    ]
    return dict(zip(numbered([name.split(" chat")[0] for name, _ in kept]), (t for _, t in kept)))


# --- tripwire: nothing private in any public text file -------------------------------------------

LEAKS = re.compile(
    r"/Users/|"
    r"Library/Application Support|\.codex|\.cursor|\.claude|America/Los_Angeles|"
    r"codex-rollout-|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|"
    r"[\w.+-]+@[\w-]+\.[\w.]+|\bsk-[\w-]{8,}|\bxai-[\w-]{8,}|\bghp_\w{8,}|\bAIza[\w-]{8,}|Bearer\s+\S+"
)


class PrivacyError(RuntimeError):
    pass


def checked(text: str, name: str) -> str:
    """Return the text if it is clean; otherwise refuse with the file and the match."""
    if match := LEAKS.search(text):
        raise PrivacyError(f"{name}: private text would be published: {match.group(0)!r}")
    return text


# --- blurring for public release -----------------------------------------------------------------

BLUR_CLASSES = {"person": 0.15, "face": 0.15, "license plate": 0.1, "car": 0.3}
BLUR_HOLD = 6  # frames a detection keeps blurring after it was last seen (flicker guard)
PLATE_ZONE = (0.25, 0.40, 0.75, 0.85)  # where a plate sits in a car's box, as fractions


class Blurrer:
    """YOLO-World finds people, faces and plates; boxes are widened, held briefly, and blurred.

    Plates are too small for the detector at this resolution, so every car wide enough for a
    plate to be legible also gets its plate zone blurred. Our own hood, which reaches the
    bottom edge of the wide camera, is skipped.
    """

    def __init__(self, weights: Path = Path.home() / ".cache/drivingbench/weights"):
        import os

        from ultralytics import YOLO, settings

        settings.update({"weights_dir": str(weights)})
        os.chdir(weights)  # ultralytics downloads into the working directory
        self.model = YOLO("yolov8s-worldv2.pt")
        self.model.set_classes(list(BLUR_CLASSES))
        self.held = []  # (frames_left, box)

    def boxes(self, image, width, height) -> list[tuple]:
        result = self.model.predict(
            image, imgsz=960, conf=min(BLUR_CLASSES.values()), device="mps", verbose=False
        )[0]
        fresh = []
        for cls, conf, (x0, y0, x1, y1) in zip(
            result.boxes.cls, result.boxes.conf, result.boxes.xyxy.tolist()
        ):
            name = self.model.names[int(cls)]
            if float(conf) < BLUR_CLASSES[name]:
                continue
            if name == "car":
                if x1 - x0 < 60 or y1 > height - 5:
                    continue
                w, h = x1 - x0, y1 - y0
                x0, y0, x1, y1 = (
                    x0 + w * PLATE_ZONE[0],
                    y0 + h * PLATE_ZONE[1],
                    x0 + w * PLATE_ZONE[2],
                    y0 + h * PLATE_ZONE[3],
                )
                pad_x = pad_y = 4
            else:
                pad_x, pad_y = (x1 - x0) * 0.25 + 4, (y1 - y0) * 0.25 + 4
            fresh.append(
                (
                    max(0, int(x0 - pad_x)),
                    max(0, int(y0 - pad_y)),
                    min(width, int(x1 + pad_x)),
                    min(height, int(y1 + pad_y)),
                )
            )
        self.held = [(n - 1, box) for n, box in self.held if n > 1] + [
            (BLUR_HOLD, box) for box in fresh
        ]
        return [box for _, box in self.held]

    def blur(self, image):
        from PIL import ImageFilter

        for x0, y0, x1, y1 in self.boxes(image, *image.size):
            region = image.crop((x0, y0, x1, y1))
            radius = max(6, (x1 - x0) // 6)
            image.paste(region.filter(ImageFilter.GaussianBlur(radius)), (x0, y0))
        return image


def blur_video(blurrer: Blurrer, source: Path, output: Path, bitrate: str) -> None:
    from PIL import Image

    width, height = probe_size(source)
    blurrer.held = []
    decode = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    encode = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "-",
            *encoder(bitrate),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    size = width * height * 3
    while True:
        raw = decode.stdout.read(size)
        if len(raw) < size:
            break
        frame = Image.frombuffer("RGB", (width, height), raw, "raw", "RGB", 0, 1).copy()
        encode.stdin.write(blurrer.blur(frame).tobytes())
    decode.stdout.close()
    encode.stdin.close()
    decode.wait()
    encode.wait()
    if encode.returncode:
        raise RuntimeError(f"blur encode failed for {source}")


PUBLIC_REPO = "https://github.com/aditya-ramabadran/drivingbench_harness_code"
PUBLIC_ABOUT = (
    "Language models driving a real car. Each attempt below is one model's try at a mini-cone "
    "course in an empty parking lot, at walking pace, with a human in the driver's seat ready to "
    "brake. The model saw the road through two cameras and drove with three tools (observe, "
    "set_motion, stop_now); every command, the reason the model gave for it, the car's telemetry "
    "and the video are here, aligned to the frame. People and license plates are blurred. "
    f"The harness that recorded this is open source: {PUBLIC_REPO}."
)
PUBLIC_TEXT_FILES = ("timeline.json", "calls.srt", "README.md")
CREATED_BY = (
    "Created by: [Simon Mahns](https://x.com/nautsimon_), [Tobias Gessler](https://x.com/tobiges), "
    "[Aditya Ramabadran](https://x.com/a_ramabadran)."
)
DATASET_CARD = """---
license: cc-by-4.0
pretty_name: DrivingBench v1
tags:
  - robotics
  - autonomous-driving
  - agents
  - benchmark
size_categories:
  - n<1K
---

"""


def public_texts(source: Path, only: str | None) -> tuple[dict[Path, str], dict]:
    """Every text file the public copy will contain, checked for private content first.

    Groups and attempts marked `public: false` in the manifest are left out, including their
    stretch of the group's conversation, and the top-level README and manifest are rewritten
    from the public-facing text.
    """
    manifest = json.loads((source / "manifest.json").read_text())
    texts, public = {}, {"about": PUBLIC_ABOUT, "collections": {}}
    for cname, col in manifest["collections"].items():
        if only and cname != only:
            continue
        groups = {
            g: {**v, "parts": {n: i for n, i in v["parts"].items() if i.get("public", True)}}
            for g, v in col["groups"].items()
            if v.get("public", True)
        }
        public["collections"][cname] = {
            "title": col["title"],
            "about": col["about"],
            "groups": groups,
        }
        texts[Path(cname) / "README.md"] = collection_readme(col, groups)
        for gname, group in groups.items():
            folder = source / cname / gname
            excluded = [
                json.loads((folder / part / "timeline.json").read_text())
                for part in col["groups"][gname]["parts"]
                if part not in group["parts"]
            ]
            chats = without_attempts(
                json.loads((folder / "conversation.json").read_text()), excluded
            )
            texts[Path(cname) / gname / "conversation.json"] = (
                json.dumps(chats, indent=1, ensure_ascii=False) + "\n"
            )
            texts[Path(cname) / gname / "transcript.md"] = transcript_markdown(
                group["title"], chats
            )
            texts[Path(cname) / gname / "README.md"] = group_readme(group, group["parts"], chats)
            for part in group["parts"]:
                for name in PUBLIC_TEXT_FILES:
                    texts[Path(cname) / gname / part / name] = (
                        source / cname / gname / part / name
                    ).read_text()
    texts[Path("README.md")] = DATASET_CARD + release_readme(public, CREATED_BY)
    texts[Path("manifest.json")] = json.dumps(public, indent=1) + "\n"
    for relative, text in texts.items():
        checked(text, str(relative))
    return texts, public


def blur_release(source: Path, output: Path, only: str | None = None) -> list[str]:
    """The public copy: text files checked and written first, then every video and frame blurred
    and the annotated clip redrawn from the blurred road video. Nothing is written if any text
    file would publish private content."""
    from PIL import Image

    texts, public = public_texts(source, only)
    for relative, text in texts.items():
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        (output / relative).write_text(text)
    blurrer = Blurrer()
    done = []
    for cname, col in public["collections"].items():
        for gname, group in col["groups"].items():
            for part_name in group["parts"]:
                part, target = (
                    source / cname / gname / part_name,
                    output / cname / gname / part_name,
                )
                if (target / "road-annotated.mp4").is_file():
                    continue
                print(f"{cname}/{gname}/{part_name}", flush=True)
                (target / "frames").mkdir(parents=True, exist_ok=True)
                for frame in sorted((part / "frames").glob("*.jpg")):
                    blurrer.held = []
                    blurrer.blur(Image.open(frame).convert("RGB")).save(
                        target / "frames" / frame.name, quality=92
                    )
                blur_video(blurrer, part / "road.mp4", target / "road.mp4", BITRATE["narrow"])
                if (part / "wide.mp4").is_file():
                    blur_video(blurrer, part / "wide.mp4", target / "wide.mp4", BITRATE["wide"])
                annotate(
                    target / "road.mp4",
                    json.loads((target / "timeline.json").read_text()),
                    target / "road-annotated.mp4",
                )
                done.append(f"{cname}/{gname}/{part_name}")
    return done


# --- trimming an existing clip ------------------------------------------------------------------


def trim_window(tl: dict) -> tuple[int, int]:
    """The frames a clip should span: MARGIN_BEFORE_S before the first tool call to
    MARGIN_AFTER_S after the last engagement, within the frames it has."""
    frames = tl["video"]["frames"]
    first = min((c["t"] for c in tl["calls"]), default=tl["segments"][0]["t_start"])
    last = max(s["t_end"] for s in tl["segments"])
    return max(0, round((first - MARGIN_BEFORE_S) * FPS)), min(
        frames, round((last + MARGIN_AFTER_S) * FPS)
    )


def trim_video(source: Path, first: int, last: int, bitrate: str) -> None:
    """Re-cut an encoded clip to frames [first, last) in place, frame-accurately."""
    output = source.with_suffix(".trim.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS",
            *encoder(bitrate),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    output.replace(source)


def trim_part(part: Path) -> dict | None:
    """Shift a rendered attempt onto a shorter clip: videos re-cut, timeline times and frame
    names moved by the offset, derived files removed so the next stages regenerate them.
    Blurred copies stay blurred; nothing is re-detected."""
    tl = json.loads((part / "timeline.json").read_text())
    first, last = trim_window(tl)
    if first == 0 and last == tl["video"]["frames"]:
        return None
    offset = first / FPS
    for name, bitrate in (("road.mp4", BITRATE["narrow"]), ("wide.mp4", BITRATE["wide"])):
        if (part / name).is_file():
            trim_video(part / name, first, last, bitrate)
    for call in tl["calls"]:
        call["t"] = round(call["t"] - offset, 2)
        renamed = []
        for frame in call["frames"]:
            index, rest = frame.split("-", 1)
            new = f"{int(index) - first:06d}-{rest}"
            if (part / "frames" / frame).is_file():
                (part / "frames" / frame).rename(part / "frames" / new)
            renamed.append(new)
        call["frames"] = renamed
    tl["telemetry"] = [
        {**s, "t": round(s["t"] - offset, 2)}
        for s in tl["telemetry"]
        if first <= s["t"] * FPS <= last
    ]
    for segment in tl["segments"]:
        segment["t_start"] = round(segment["t_start"] - offset, 2)
        segment["t_end"] = round(segment["t_end"] - offset, 2)
    tl["video"].update(
        {
            "frames": last - first,
            "duration_s": round((last - first) / FPS, 2),
            "t0": iso(epoch(tl["video"]["t0"]) + offset),
        }
    )
    (part / "timeline.json").write_text(json.dumps(tl, indent=1) + "\n")
    srt(tl, part / "calls.srt")
    for stale in ("road-annotated.mp4", "road-720p.mp4", "poster.jpg", "end.jpg"):
        (part / stale).unlink(missing_ok=True)
    annotate(part / "road.mp4", tl, part / "road-annotated.mp4")
    return {"first": first, "last": last, "offset_s": round(offset, 2)}


def trim_release(release: Path, traces: Path, only: str | None = None) -> dict[str, dict]:
    """Trim every rendered attempt in a release folder and refresh its README."""
    done = {}
    for timeline_file in sorted(release.rglob("timeline.json")):
        part = timeline_file.parent
        relative = part.relative_to(release)
        if only and relative.parts[0] != only:
            continue
        print(relative, flush=True)
        result = trim_part(part)
        if result:
            tl = json.loads(timeline_file.read_text())
            facts = [segment_facts(traces, s["segment"]) for s in tl["segments"]]
            (part / "README.md").write_text(
                part_readme(tl, tl, facts)
            )  # the timeline carries the meta
            done[str(relative)] = result
    return done


# --- assembly ------------------------------------------------------------------------------------


def render_part(
    repo, traces, mirror, routes, index, out: Path, meta: dict, segments: list[str]
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    events = [e for s in segments for e in read_events(traces / "segments" / s / "events.jsonl")]
    calls = [e for e in events if e["kind"] == "tool"]
    start = min(epoch(e["timestamp"]) for e in calls or events) - MARGIN_BEFORE_S
    end = max(epoch(e["timestamp"]) for e in events) + MARGIN_AFTER_S
    road = frames_between(index, "narrow", start, end)
    if not road:
        raise ValueError(f"{meta['part']}: no route video covers {iso(start)} – {iso(end)}")
    walls = [w for w, _, _ in road]
    if (out / "road.mp4").is_file():
        # An existing clip is the reference: a trimmed clip may round one frame differently
        # from the wall-time window; anything more means the margins changed under it.
        have = frame_count(out / "road.mp4")
        if abs(have - len(walls)) > 1:
            raise ValueError(
                f"{meta['part']}: road.mp4 has {have} frames, the current window {len(walls)}; "
                "run `drivingbench trim` on the release or delete the clip to re-cut it"
            )
        if have < len(walls):
            road = road[:have]
        elif have > len(walls):
            wall, name, k = road[-1]
            road = road + [(wall + index[name]["narrow"]["dt"], name, k + 1)]
        walls = [w for w, _, _ in road]
    else:
        cut(routes, index, "narrow", road, out / "road.mp4")
    wide = frames_between(index, "wide", walls[0] - 0.026, walls[-1] + 0.026)
    if wide and not (out / "wide.mp4").is_file():
        cut(routes, index, "wide", wide, out / "wide.mp4")
    tl = timeline(traces, segments, walls, meta)
    if not wide:
        tl["video"]["wide"] = None
    (out / "frames").mkdir(exist_ok=True)
    for call in tl["calls"]:
        for name in call["frames"]:
            digest8 = name.rsplit("-", 1)[-1][:8]
            source = next(iter((mirror / "images").glob(f"{digest8}*.jpg")), None)
            if source and not (out / "frames" / name).is_file():
                shutil.copyfile(source, out / "frames" / name)
    (out / "timeline.json").write_text(json.dumps(tl, indent=1) + "\n")
    srt(tl, out / "calls.srt")
    if not (out / "road-annotated.mp4").is_file():
        annotate(out / "road.mp4", tl, out / "road-annotated.mp4")
    facts = [segment_facts(traces, s) for s in segments]
    (out / "README.md").write_text(part_readme(meta, tl, facts))
    return {
        "duration_s": tl["video"]["duration_s"],
        "calls": len(tl["calls"]),
        "frames": sum(len(c["frames"]) for c in tl["calls"]),
    }


def part_readme(meta: dict, tl: dict, facts: list[dict]) -> str:
    lines = [
        f"# {meta['group_title']} · {meta['part'].split('--')[0].replace('-', ' ')}",
        "",
        meta["about"],
        "",
        f"Collection `{meta['collection']}`, group `{meta['group']}`, result `{meta['result']}`. "
        f"Video starts {tl['video']['t0']} and runs {tl['video']['duration_s']} s at {FPS} fps.",
        "",
        "| file | what |",
        "| --- | --- |",
        "| `road.mp4` | narrow road camera, continuous, from 3 s before the model's first tool call to 2 s after its last engagement |",
        "| `wide.mp4` | wide road camera, same window (frame-aligned with road.mp4) |"
        if tl["video"]["wide"]
        else "| `wide.mp4` | not recorded for this window |",
        "| `road-annotated.mp4` | road.mp4 with the active tool call, its reason, speed and measured steering burned in |",
        "| `calls.srt` | the tool calls as subtitles, same timing as timeline.json |",
        "| `timeline.json` | `t` in seconds of video: every tool call (arguments, reason, outcome, the frames it saw), 5 Hz telemetry, segment bounds |",
        "| `frames/` | every image `observe` returned to the model, full resolution, named `<video frame>-<camera>-<hash>.jpg` |",
        "",
        "Segments (traces/segments/<id>/ in Git: 10 Hz telemetry, every command with the model's reason, thumbnails, chat/):",
        "",
    ]
    for f in facts:
        lines.append(
            f"- `{f['segment']}` — {f['start']}→{f['end']} UTC, {f['duration_s']} s, {f['commands']} set_motion"
            f"{' + ' + str(f['stops']) + ' stop_now' if f['stops'] else ''}, peak {f['peak_mps']} m/s, ended by {f['ended_by']}, "
            f"{len(f['frames'])} frames"
        )
    return "\n".join(lines) + "\n"


def render(repo: Path, out: Path, only: str | None = None) -> dict:
    traces, mirror, routes = (
        repo / "traces",
        repo / "runs/recordings",
        repo / "runs/artifacts/routes",
    )
    out.mkdir(parents=True, exist_ok=True)
    index = route_index(repo, routes, out / "routes-index.json")
    spec = json.loads((traces / "collections.json").read_text())
    manifest = {"about": spec["dataset_about"], "collections": {}}
    for cname, col in spec["collections"].items():
        if only and cname != only:
            continue
        groups = {}
        for gname, group in col["groups"].items():
            folder = out / cname / gname
            parts = {}
            for part in group["parts"]:
                if not part["segments"]:
                    continue
                meta = {
                    "collection": cname,
                    "group": gname,
                    "group_title": group["title"],
                    "result": group["result"],
                    "part": part["folder"],
                    "about": part["about"],
                    "segments": part["segments"],
                }
                print(f"{cname}/{gname}/{part['folder']}", flush=True)
                parts[part["folder"]] = {
                    **render_part(
                        repo,
                        traces,
                        mirror,
                        routes,
                        index,
                        folder / part["folder"],
                        meta,
                        part["segments"],
                    ),
                    "public": part.get("public", True),
                }
            chats = group_chats(traces, [s for part in group["parts"] for s in part["segments"]])
            written = write_conversation(folder, group["title"], chats)
            (folder / "README.md").write_text(group_readme(group, parts, written))
            groups[gname] = {
                "title": group["title"],
                "about": group["about"],
                "result": group["result"],
                "public": group.get("public", True),
                "parts": parts,
            }
        (out / cname / "README.md").write_text(collection_readme(col, groups))
        manifest["collections"][cname] = {
            "title": col["title"],
            "about": col["about"],
            "groups": groups,
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "README.md").write_text(release_readme(manifest))
    return manifest


def collection_readme(col: dict, groups: dict) -> str:
    return (
        f"# {col['title']}\n\n{col['about']}\n\n"
        + "\n".join(f"- `{g}/` — {v['title']}: {v['result']}" for g, v in groups.items())
        + "\n"
    )


def group_readme(group: dict, parts: dict, chats: dict) -> str:
    lines = [f"# {group['title']}", "", group["about"], "", f"Result: {group['result']}", ""]
    for name, info in parts.items():
        lines.append(
            f"- `{name}/` — {info['duration_s']} s of video, {info['calls']} tool calls, {info['frames']} observed frames"
        )
    lines += [
        "",
        f"`conversation.json` / `transcript.md` — the driving conversation from the chat(s) that drove this trial ({len(chats)} file{'s' if len(chats) != 1 else ''}).",
        "",
    ]
    return "\n".join(lines) + "\n"


def release_readme(manifest: dict, created_by: str = "") -> str:
    lines = [
        "# DrivingBench release",
        "",
        manifest["about"],
        "",
        *([created_by, ""] if created_by else []),
        "Layout: `<collection>/<model group>/<attempt>/` with `road.mp4`, `wide.mp4`, `road-annotated.mp4`, `calls.srt`, "
        "`timeline.json`, `frames/`, `README.md`; each group has the trial's `transcript.md`. `manifest.json` indexes everything.",
        "",
    ]
    for cname, col in manifest["collections"].items():
        lines += [f"## `{cname}/` — {col['title']}", "", col["about"], ""]
        for gname, group in col["groups"].items():
            lines.append(
                f"- `{gname}/` — {group['title']}: {group['result']} · "
                + ", ".join(f"`{p}`" for p in group["parts"])
            )
        lines.append("")
    return "\n".join(lines) + "\n"
