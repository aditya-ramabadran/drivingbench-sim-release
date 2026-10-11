"""Index site-1 video on the log clock and extract a reproducible motion sample."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from drivingbench.sim.logs import REPO, events, use_openpilot
from drivingbench.sim.series import load, route_segments

ROUTES = ("00000041--fdf9cede4b", "00000042--544757d3f5")
WIDTH, HEIGHT = 1344, 760
TEST = (340.0, 480.0)
GUARD_S = 10.0


def split(route, t):
    if route != ROUTES[1]:
        return "train"
    if TEST[0] <= t <= TEST[1]:
        return "test"
    if TEST[0] - GUARD_S <= t <= TEST[1] + GUARD_S:
        return "guard"
    return "train"


def read_segment(path):
    frames, calibration, odometry, sensors = [], [], [], set()
    for mono, kind, msg in events(
        path,
        {"narrowRoadEncodeIdx", "extrinsicsCalibration", "cameraOdometry", "narrowRoadCameraState"},
    ):
        if kind == "narrowRoadEncodeIdx":
            if str(msg.type) != "fullHEVC":
                continue
            frames.append(
                {
                    "segment": int(msg.segmentNum),
                    "index": int(msg.segmentId),
                    "frame_id": int(msg.frameId),
                    "sof_ns": int(msg.timestampSof),
                    "eof_ns": int(msg.timestampEof),
                }
            )
        elif kind == "extrinsicsCalibration":
            calibration.append({"t": mono * 1e-9, **msg.to_dict()})
        elif kind == "cameraOdometry":
            odometry.append({"t": msg.timestampEof * 1e-9, **msg.to_dict()})
        else:
            sensors.add(str(msg.sensor))
    return frames, calibration, odometry, sorted(sensors)


def validate_index(rows):
    """Require a one-to-one presentation-order mapping, never infer time from 20 fps."""
    rows = sorted(rows, key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(len(rows))):
        raise ValueError("missing or duplicate video presentation index")
    times = [r["eof_ns"] for r in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("non-increasing exposure timestamps")
    if any(not 0 < r["sof_ns"] <= r["eof_ns"] for r in rows):
        raise ValueError("invalid exposure timestamps")
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def select_expression(indices):
    """Balance additions to stay within FFmpeg's expression depth limit."""
    if not indices:
        raise ValueError("select requires at least one frame index")
    if len(indices) == 1:
        return f"eq(n\\,{indices[0]})"
    middle = len(indices) // 2
    return f"({select_expression(indices[:middle])}+{select_expression(indices[middle:])})"


def decode(job):
    """Decode selected YUV frames through the producer's exact RGB/JPEG conversion."""
    from drivingbench.device.cameras import encode_nv12

    video, rows = job
    pending = [r for r in rows if not Path(r["file_path"]).exists()]
    if not pending:
        return 0
    expression = select_expression([r["index"] for r in pending])
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "2",
        "-i",
        str(video),
        "-vf",
        f"select={expression}",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "nv12",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    size = WIDTH * HEIGHT * 3 // 2
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        for row in pending:
            raw = process.stdout.read(size)
            if len(raw) != size:
                raise ValueError(f"truncated decode: {video}, index {row['index']}")
            image = encode_nv12(
                SimpleNamespace(
                    width=WIDTH, height=HEIGHT, stride=WIDTH, uv_offset=WIDTH * HEIGHT, data=raw
                )
            )
            path = Path(row["file_path"])
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(image)
            temporary.replace(path)
        if process.stdout.read(1) or process.wait():
            raise ValueError(f"unexpected decode output: {video}")
    return len(pending)


def prepare(root, series, out, *, stride=20, workers=6, pilot=False):
    from drivingbench.sim.visual_pose import build_pose_seed, camera_to_world, sample_pose

    if stride < 1:
        raise ValueError("stride must be positive")
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    paths = route_segments(root)
    # GPS origin of the site's local frame; withheld for anonymity, set DRIVINGBENCH_SITE_ORIGIN="lat,lon".
    origin = tuple(float(v) for v in os.environ.get("DRIVINGBENCH_SITE_ORIGIN", "0,0").split(","))
    manifest, full_index, jobs, reports = [], [], [], {}
    use_openpilot()
    from openpilot.common.transformations.camera import DEVICE_CAMERAS

    config = DEVICE_CAMERAS[("mici", "os04c10")].narrow_road
    assert config.size == (WIDTH, HEIGHT)
    with ProcessPoolExecutor(workers) as pool:
        for route in ROUTES:
            groups, calibration, odometry, sensors = defaultdict(list), [], [], set()
            for frames, cal, odo, found in pool.map(read_segment, paths[route]):
                for frame in frames:
                    groups[frame["segment"]].append(frame)
                calibration.extend(cal)
                odometry.extend(odo)
                sensors.update(found)
            if sensors != {"os04c10"}:
                raise ValueError(f"unexpected sensors: {sensors}")
            calibration.sort(key=lambda r: r["t"])
            odometry.sort(key=lambda r: r["t"])
            write_json(out / f"{route}-calibration.json", calibration)
            write_rows(out / f"{route}-odometry.jsonl", odometry)
            _, _, columns = load(series / f"{route}.npz")
            seed, report = build_pose_seed(columns, origin=origin, odometry=odometry)
            np.savez_compressed(out / f"{route}-seed.npz", **seed)
            reports[route] = report
            cal_t = np.array([r["t"] for r in calibration])
            for segment, rows in sorted(groups.items()):
                rows = validate_index(rows)
                video = root / f"{route}--{segment}" / "fcamera.hevc"
                if not video.exists():
                    raise FileNotFoundError(video)
                selected = []
                for row in rows:
                    t = row["eof_ns"] * 1e-9
                    name = f"{route}--{segment:03d}--{row['index']:04d}.jpg"
                    row = dict(row, route=route, t=t, split=split(route, t), name=name)
                    full_index.append(row)
                    if row["index"] % stride or row["split"] == "guard":
                        continue
                    if pilot and (route != ROUTES[1] or not (100 <= t <= 510 or 690 <= t <= 800)):
                        continue
                    k = np.searchsorted(columns["t"], t)
                    if k >= len(columns["t"]) or columns["v"][k] < 0.15:
                        continue
                    pose = sample_pose(seed, np.array([t]))[0]
                    if not np.isfinite(pose).all():
                        continue
                    j = np.searchsorted(cal_t, t, side="right") - 1
                    if j < 0 or calibration[j]["calStatus"] != "calibrated":
                        continue
                    x, y, heading = map(float, pose)
                    rpy = calibration[j]["rpyCalib"]
                    matrix = camera_to_world(x, y, heading, rpy, height=1.21)
                    row.update(
                        x=x,
                        y=y,
                        heading=heading,
                        rpy=rpy,
                        transform_matrix=matrix.tolist(),
                        file_path=str((out / "images" / name).resolve()),
                    )
                    selected.append(row)
                manifest.extend(selected)
                if selected:
                    jobs.append((video, selected))
        write_rows(out / "index.jsonl", full_index)
        write_rows(out / "frames.jsonl", manifest)
        write_json(
            out / "metadata.json",
            {
                "camera": {
                    "w": WIDTH,
                    "h": HEIGHT,
                    "fl_x": config.focal_length,
                    "fl_y": config.focal_length,
                    "cx": WIDTH / 2,
                    "cy": HEIGHT / 2,
                },
                "origin_lat_lon": origin,
                "height_m": 1.21,
                "stride": stride,
                "test_route": ROUTES[1],
                "test_time_s": TEST,
                "guard_s": GUARD_S,
                "indexed": len(full_index),
                "selected": len(manifest),
                "poses": reports,
            },
        )
        for count in pool.map(decode, jobs):
            print(f"decoded {count} frames", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/artifacts/routes")
    parser.add_argument("--series", type=Path, default=REPO / "runs/sim/series")
    parser.add_argument("--out", type=Path, default=REPO / "runs/sim/visual/site1")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    rows = prepare(
        args.root, args.series, args.out, stride=args.stride, workers=args.workers, pilot=args.pilot
    )
    print(f"selected {len(rows)} frames in {args.out}")


if __name__ == "__main__":
    main()
