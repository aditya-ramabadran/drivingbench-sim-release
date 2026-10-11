"""Private, distance-indexed source clips; originals are only ever read."""

import argparse
import bisect
import hashlib
import json
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from drivingbench.gateway.release import CAMERAS, frames_between, iso, route_index
from PIL import Image


def read_telemetry(repo: Path, route: Path, offset_ns: float) -> list[dict]:
    import zstandard

    sys.path[:0] = [str(repo / "vendor/openpilot"), str(repo / "vendor/openpilot/opendbc_repo")]
    from openpilot.cereal import log

    with (route / "rlog.zst").open("rb") as source:
        data = zstandard.ZstdDecompressor().stream_reader(source).read()
    rows = []
    for message in log.Event.read_multiple_bytes(data):
        if message.which() == "carState":
            state = message.carState
            rows.append(
                {
                    "t": (message.logMonoTime + offset_ns) / 1e9,
                    "speed_mps": float(state.vEgo),
                    "steering_deg": float(state.steeringAngleDeg),
                    "gear": str(state.gearShifter),
                }
            )
    return rows


def distance_table(samples: list[dict], times: list[float], *, allow_stops=False) -> list[dict]:
    """Trapezoidal vEgo integration, linearly interpolated to actual frame timestamps."""
    if not times or len(samples) < 2:
        raise ValueError("frames and bracketing telemetry required")
    if any(
        not math.isfinite(v)
        for row in samples
        for v in (row["t"], row["speed_mps"], row["steering_deg"])
    ) or any(not math.isfinite(t) for t in times):
        raise ValueError("finite telemetry and frame times required")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("frame times must increase")
    stamps = [s["t"] for s in samples]
    if stamps[0] > times[0] or stamps[-1] < times[-1]:
        raise ValueError("telemetry does not cover video")
    distances = [0.0]
    for a, b in zip(samples, samples[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0 or dt > 0.15:
            raise ValueError("nonmonotonic telemetry or gap exceeding 150ms")
        if a["gear"] == "reverse" or b["gear"] == "reverse":
            raise ValueError("reverse gear is invalid for a forward stimulus")
        if min(a["speed_mps"], b["speed_mps"]) < -0.15:
            raise ValueError("negative speed exceeds standstill sensor noise")
        distances.append(distances[-1] + dt * (max(0, a["speed_mps"]) + max(0, b["speed_mps"])) / 2)
    result = []
    for t in times:
        i = min(len(samples) - 2, max(0, bisect.bisect_right(stamps, t) - 1))
        a, b = samples[i : i + 2]
        fraction = (t - a["t"]) / (b["t"] - a["t"])
        speed = (
            max(0, a["speed_mps"]) + (max(0, b["speed_mps"]) - max(0, a["speed_mps"])) * fraction
        )
        d = distances[i] + (t - a["t"]) * (max(0, a["speed_mps"]) + speed) / 2
        result.append(
            {
                "source_time": t,
                "distance_m": d,
                "speed_mps": speed,
                "steering_deg": a["steering_deg"]
                + (b["steering_deg"] - a["steering_deg"]) * fraction,
            }
        )
    origin = result[0]["distance_m"]
    for row in result:
        row["distance_m"] -= origin
    if not allow_stops and any(r["speed_mps"] < 0.1 for r in result):
        raise ValueError("stationary imagery is ambiguous in distance; trim or explicitly allow")
    if result[-1]["distance_m"] <= 0:
        raise ValueError("clip has no forward travel")
    return result


def prepare(
    repo: Path, start: float, end: float, output: Path, *, step_m=0.25, allow_stops=False
) -> Path:
    if output.exists():
        raise ValueError("output exists; source preparations are immutable")
    if not math.isfinite(step_m) or step_m <= 0 or end <= start:
        raise ValueError("positive step and ordered window required")
    routes = repo / "runs/artifacts/routes"
    index = route_index(repo, routes, repo / "runs/artifacts/cache/routes-index.json")
    raw = frames_between(index, "narrow", start, end)
    if not raw or any(b[0] - a[0] > 0.08 for a, b in zip(raw, raw[1:])):
        raise ValueError("missing or discontinuous video")
    names = list(dict.fromkeys(r[1] for r in raw))
    samples = sorted(
        (
            r
            for name in names
            for r in read_telemetry(repo, routes / name, index[name]["offset_ns"])
            if start - 0.15 <= r["t"] <= end + 0.15
        ),
        key=lambda r: r["t"],
    )
    table = distance_table(samples, [f[0] for f in raw], allow_stops=allow_stops)
    distances = [r["distance_m"] for r in table]
    chosen = sorted(
        set(
            [
                min(len(table) - 1, bisect.bisect_left(distances, k * step_m))
                for k in range(math.floor(distances[-1] / step_m) + 1)
            ]
            + [len(table) - 1]
        )
    )
    rows = []
    for i in chosen:
        row = table[i] | {"route": raw[i][1], "frame_index": raw[i][2]}
        if not rows or row["distance_m"] > rows[-1]["distance_m"]:
            rows.append(row)
    output.mkdir(parents=True)
    sources = {}
    for camera, (filename, _) in CAMERAS.items():
        requested = {}
        camera_frames = frames_between(index, camera, start - 0.1, end + 0.1)
        stamps = [f[0] for f in camera_frames]
        for row in rows:
            j = bisect.bisect_left(stamps, row["source_time"])
            choices = camera_frames[max(0, j - 1) : j + 1]
            if not choices:
                raise ValueError(f"no {camera} frame")
            match = min(choices, key=lambda f: abs(f[0] - row["source_time"]))
            if abs(match[0] - row["source_time"]) > 0.03:
                raise ValueError("camera synchronization exceeds 30ms")
            _, name, frame_id = match
            frame_index = frame_id - index[name][camera]["first_id"]
            relative = f"{camera}/{name}/{frame_index:06d}.png"
            row[camera] = relative
            row[f"{camera}_source_time"] = match[0]
            requested.setdefault(name, set()).add(frame_index)
        for name, ids in requested.items():
            dest = output / camera / name
            dest.mkdir(parents=True)
            source = routes / name / filename
            selection = "+".join(f"eq(n\\,{i})" for i in sorted(ids))
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "hevc",
                    "-r",
                    "20",
                    "-i",
                    str(source),
                    "-vf",
                    f"select='{selection}'",
                    "-fps_mode",
                    "passthrough",
                    "-start_number",
                    "0",
                    str(dest / "decoded-%06d.png"),
                ],
                check=True,
            )
            decoded = sorted(dest.glob("decoded-*.png"))
            if len(decoded) != len(ids):
                raise ValueError("decoded frame count mismatch")
            for path, i in zip(decoded, sorted(ids)):
                path.rename(dest / f"{i:06d}.png")
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            sources[str(source.relative_to(repo))] = {
                "sha256": digest,
                "size_bytes": source.stat().st_size,
            }
    manifest = {
        "schema_version": 1,
        "length_m": distances[-1],
        "quantization_m": step_m,
        "frames": rows,
        "telemetry": samples,
        "provenance": {
            "start": iso(raw[0][0]),
            "end": iso(raw[-1][0]),
            "sources": sources,
            "distance": "trapezoidal carState.vEgo; interpolated at camera timestamps",
            "frame_clock": "rlog encode timestamps affine approximation from release.route_index",
        },
        "validity": {
            "stationary_allowed": allow_stops,
            "moving_subjects": "Scene activity follows ego distance and freezes at stops; not a causal traffic simulation.",
            "steering": "Source trajectory is fixed; commanded steering cannot alter imagery.",
            "max_abs_steering_deg": max(abs(r["steering_deg"]) for r in table),
        },
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


class ReplayClip:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text())
        self.length_m = float(self.manifest["length_m"])
        self.frames = self.manifest["frames"]
        self.distances = [float(r["distance_m"]) for r in self.frames]
        if (
            not self.frames
            or any(not math.isfinite(d) for d in self.distances)
            or self.distances[0] != 0
            or any(b <= a for a, b in zip(self.distances, self.distances[1:]))
        ):
            raise ValueError("frame distances must strictly increase from zero")
        if not math.isfinite(self.length_m) or self.length_m != self.distances[-1]:
            raise ValueError("length must match final source frame")

    def sample_at(self, distance_m):
        if not math.isfinite(distance_m):
            raise ValueError("finite distance required")
        # Hold the last reached sample; clamp outside the source, never extrapolate scenery.
        return self.frames[
            max(0, bisect.bisect_right(self.distances, min(distance_m, self.length_m)) - 1)
        ].copy()

    @lru_cache(maxsize=8)
    def _image(self, relative):
        path = (self.path.parent / relative).resolve()
        if not path.is_relative_to(self.path.parent):
            raise ValueError("frame escapes clip directory")
        with Image.open(path) as im:
            return im.convert("RGB")

    def frame_at(self, distance_m, camera="narrow"):
        if camera not in CAMERAS:
            raise ValueError("unknown camera")
        return self._image(self.sample_at(distance_m)[camera]).copy()


def main():
    from datetime import datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-stops", action="store_true")
    args = parser.parse_args()
    print(
        prepare(
            Path(__file__).resolve().parents[1],
            datetime.fromisoformat(args.start).timestamp(),
            datetime.fromisoformat(args.end).timestamp(),
            args.output,
            allow_stops=args.allow_stops,
        )
    )


if __name__ == "__main__":
    main()
