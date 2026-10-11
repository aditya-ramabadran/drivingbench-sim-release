"""Checks and helpers for the generated fork (trolley) videos; see docs/replay-video-generation.md.

python -m drivingbench.replay.fork_video decision docs/replay-video/<pairing>
python -m drivingbench.replay.fork_video check docs/replay-video/<pairing> [...]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

CLIPS = ("approach", "keep", "switch")
SIZE = (1280, 720)
JOIN_LIMIT = (
    8.0  # mean absolute pixel difference (0-255) between a clip's first frame and its start still
)


def probe(video):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=width,height,nb_read_frames:format=duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(out.stdout)
    stream = info["streams"][0]
    return (
        (stream["width"], stream["height"]),
        int(stream["nb_read_frames"]),
        float(info["format"]["duration"]),
    )


def frame(video, index, out):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )
    return Path(out)


def difference(a, b):
    a, b = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    return sum(ImageStat.Stat(ImageChops.difference(a, b.resize(a.size))).mean) / 3


def decision(folder):
    """Write decision.png: the exact last frame of approach.mp4, the start still for keep and switch."""
    folder = Path(folder)
    _, frames, _ = probe(folder / "approach.mp4")
    return frame(folder / "approach.mp4", frames - 1, folder / "decision.png")


def check(folder):
    folder = Path(folder)
    problems = []
    for name in ("start.png", "decision.png", *(f"{c}.mp4" for c in CLIPS)):
        if not (folder / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems
    for name in ("start.png", "decision.png"):
        if Image.open(folder / name).size != SIZE:
            problems.append(f"{name} is {Image.open(folder / name).size}, expected {SIZE}")
    for clip, start in (
        ("approach", "start.png"),
        ("keep", "decision.png"),
        ("switch", "decision.png"),
    ):
        size, frames, duration = probe(folder / f"{clip}.mp4")
        if size != SIZE:
            problems.append(f"{clip}.mp4 is {size}, expected {SIZE}")
        if duration < 3:
            problems.append(f"{clip}.mp4 is only {duration:.1f} s")
        first = frame(folder / f"{clip}.mp4", 0, folder / f".{clip}-first.png")
        diff = difference(folder / start, first)
        first.unlink()
        if diff > JOIN_LIMIT:
            problems.append(
                f"{clip}.mp4 does not start from {start} (difference {diff:.1f} > {JOIN_LIMIT})"
            )
    _, frames, _ = probe(folder / "approach.mp4")
    last = frame(folder / "approach.mp4", frames - 1, folder / ".approach-last.png")
    # Re-encoding a cut clip shifts pixels by about 1.3 on average; the previous frame differs by 2 or more.
    if difference(folder / "decision.png", last) > 1.6:
        problems.append("decision.png is not the last frame of approach.mp4 (re-run `decision`)")
    last.unlink()
    return problems


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("command", choices=("decision", "check"))
    p.add_argument("folders", nargs="+", type=Path)
    a = p.parse_args()
    failed = False
    for folder in a.folders:
        if a.command == "decision":
            print(decision(folder))
            continue
        problems = check(folder)
        failed |= bool(problems)
        print(f"{folder}: " + ("ok" if not problems else "; ".join(problems)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
