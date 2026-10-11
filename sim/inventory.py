"""Step 1 of the simulator plan: what our openpilot route logs cover.

    python -m drivingbench.sim.inventory [--root runs/artifacts] [--out runs/sim/inventory]

Reads every rlog under the root once (segments appear under several session folders; they are
deduplicated by segment name) and writes per-segment summaries, per-site totals and a coverage
map per site (a site is where the car was driven while engaged).
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from statistics import median

from drivingbench.sim.logs import REPO, events

SPEED_BINS = [0, 0.2, 0.5, 1, 1.5, 2, 3, 5, 10]  # m/s, lower edges
STEER_BINS = [0, 5, 15, 30, 60, 90, 120, 180]  # |steering-wheel deg| while moving, lower edges
MOVING = 0.2  # m/s
SITE_RADIUS = 500  # m: segments whose median GPS fix is this close share a site
PACIFIC = timezone(timedelta(hours=-7))
TYPES = {
    "carState", "carControl", "selfdriveState", "gpsLocationExternal", "narrowRoadEncodeIdx",
    "wideRoadEncodeIdx", "vehicleParameters", "lateralDelay", "extrinsicsCalibration", "carParams",
    "initData",
}  # fmt: skip


def histogram(values_dt, bins):
    """Seconds spent in each bin, keyed by the bin's lower edge."""
    out = [0.0] * len(bins)
    for value, dt in values_dt:
        out[max(0, bisect_right(bins, value) - 1)] += dt  # carState reports tiny negative speeds
    return {str(edge): round(s, 1) for edge, s in zip(bins, out)}


def summarize(path: Path) -> dict:
    speed, steer, gps, enabled, lat_active, long_active = [], [], [], [], [], []
    frames = {"narrowRoadEncodeIdx": 0, "wideRoadEncodeIdx": 0}
    params, gps_messages = {}, 0
    for mono, kind, msg in events(path, TYPES):
        if kind == "carState":
            speed.append((mono, msg.vEgo))
            steer.append((mono, msg.steeringAngleDeg))
        elif kind == "selfdriveState":
            enabled.append((mono, msg.enabled))
        elif kind == "carControl":
            lat_active.append((mono, msg.latActive))
            long_active.append((mono, msg.longActive))
        elif kind == "gpsLocationExternal":
            gps_messages += 1
            if msg.hasFix:
                fix = (msg.latitude, msg.longitude, msg.horizontalAccuracy, msg.unixTimestampMillis)
                gps.append((mono, *fix))
        elif kind in frames:
            frames[kind] += 1
        elif kind == "vehicleParameters":
            params["steer_ratio"] = msg.steerRatio
            params["stiffness"] = msg.stiffnessFactor
            params["angle_offset_deg"] = msg.angleOffsetDeg
        elif kind == "lateralDelay":
            params["lateral_delay_s"] = msg.lateralDelayEstimate
        elif kind == "extrinsicsCalibration":
            params["camera_height_m"] = list(msg.height)[0] if len(msg.height) else None
            params["rpy_calib"] = list(msg.rpyCalib)
        elif kind == "carParams":
            params["fingerprint"] = msg.carFingerprint
            params["wheelbase_m"] = msg.wheelbase
            params["steer_ratio_default"] = msg.steerRatio
            params["mass_kg"] = msg.mass
        elif kind == "initData":
            params["device"] = str(msg.deviceType)
            params["openpilot"] = msg.version

    def dts(series):
        return [(v, (series[i + 1][0] - t) / 1e9) for i, (t, v) in enumerate(series[:-1])]

    enabled_times = [t for t, _ in enabled]

    def engaged_at(t):
        i = bisect_right(enabled_times, t) - 1
        return i >= 0 and enabled[i][1]

    def speed_at(t):
        return speed[min(len(speed) - 1, bisect_right(speed_times, t))][1] if speed else 0.0

    first, last = (speed[0][0], speed[-1][0]) if speed else (0, 0)  # other types carry stale times
    speed_times = [t for t, _ in speed]
    speed_dt = dts(speed)
    engaged = [engaged_at(t) for t, _ in speed[:-1]]
    steer_moving = [
        (abs(steer[i][1]), dt, engaged[i]) for i, (v, dt) in enumerate(speed_dt) if v > MOVING
    ]
    wall = None
    if gps:
        mono, *_, unix_ms = gps[0]
        instant = datetime.fromtimestamp((unix_ms * 1e6 - mono + first) / 1e9, UTC)
        wall = instant.astimezone(PACIFIC).isoformat()
    step = max(1, len(gps) // max(1, round((last - first) / 1e9)))  # about 1 Hz
    track = [
        (round(lat, 7), round(lon, 7), round(acc, 1), speed_at(t) > MOVING, engaged_at(t))
        for t, lat, lon, acc, _ in gps[::step]
    ]
    return {
        "segment": path.parent.name,
        "start_local": wall,
        "duration_s": round((last - first) / 1e9, 1),
        "moving_s": round(sum(dt for v, dt in speed_dt if v > MOVING), 1),
        "engaged_s": round(sum(dt for v, dt in dts(enabled) if v), 1),
        "lat_active_s": round(sum(dt for v, dt in dts(lat_active) if v), 1),
        "long_active_s": round(sum(dt for v, dt in dts(long_active) if v), 1),
        "max_speed_mps": round(max((v for _, v in speed), default=0), 2),
        "speed_hist_engaged_s": histogram([x for x, e in zip(speed_dt, engaged) if e], SPEED_BINS),
        "speed_hist_human_s": histogram(
            [x for x, e in zip(speed_dt, engaged) if not e], SPEED_BINS
        ),
        "steer_hist_engaged_moving_s": histogram(
            [(a, dt) for a, dt, e in steer_moving if e], STEER_BINS
        ),
        "steer_hist_human_moving_s": histogram(
            [(a, dt) for a, dt, e in steer_moving if not e], STEER_BINS
        ),
        "gps_fix_fraction": round(len(gps) / max(1, gps_messages), 2),
        "gps_accuracy_median_m": round(median(g[3] for g in gps), 2) if gps else None,
        "gps_track": track,  # lat, lon, accuracy m, moving, engaged at about 1 Hz
        "frames": frames,
        "params": params,
        "video_mb": {
            name: round((path.parent / name).stat().st_size / 1e6, 1)
            for name in ("fcamera.hevc", "ecamera.hevc", "qcamera.ts")
            if (path.parent / name).exists()
        },
    }


def local_xy(lat, lon, lat0, lon0):
    """Metres east and north of (lat0, lon0)."""
    r = 6371000.0
    return math.radians(lon - lon0) * r * math.cos(math.radians(lat0)), math.radians(lat - lat0) * r


def sites(segments):
    """Group segments by where they were recorded; sites are ordered by engaged time."""
    groups = []  # [center lat, center lon, segments]
    for s in segments:
        if not s["gps_track"]:
            continue
        lat, lon = median(p[0] for p in s["gps_track"]), median(p[1] for p in s["gps_track"])
        near = [g for g in groups if math.hypot(*local_xy(lat, lon, g[0], g[1])) < SITE_RADIUS]
        (near[0][2] if near else groups.append([lat, lon, []]) or groups[-1][2]).append(s)
    return sorted((g[2] for g in groups), key=lambda ss: -sum(s["engaged_s"] for s in ss))


def coverage_map(segments, out: Path, size=1400, margin=70):
    from PIL import Image, ImageDraw

    engaged = [(p[0], p[1]) for s in segments for p in s["gps_track"] if p[4]]
    anchor = engaged or [(p[0], p[1]) for s in segments for p in s["gps_track"]]
    lat0, lon0 = median(p[0] for p in anchor), median(p[1] for p in anchor)
    near = sorted(math.hypot(*local_xy(lat, lon, lat0, lon0)) for lat, lon in anchor)
    radius = max(40.0, near[int(0.98 * (len(near) - 1))] * 1.4)
    scale = (size - 2 * margin) / (2 * radius)
    img = Image.new("RGB", (size, size), (14, 16, 20))
    d = ImageDraw.Draw(img)
    for s in segments:
        for lat, lon, _, moving, on in s["gps_track"]:
            x, y = local_xy(lat, lon, lat0, lon0)
            if max(abs(x), abs(y)) > radius:
                continue
            X, Y = margin + (x + radius) * scale, size - margin - (y + radius) * scale
            c = (242, 140, 40) if on and moving else (61, 139, 255) if moving else (70, 74, 82)
            d.ellipse([X - 2, Y - 2, X + 2, Y + 2], fill=c)
    light = (236, 231, 223)
    d.line(
        [(margin, size - margin / 2), (margin + 10 * scale, size - margin / 2)], fill=light, width=3
    )
    d.text((margin, size - margin / 2 + 6), "10 m", fill=light)
    legend = "GPS fixes, about 1 Hz. Orange: moving, engaged. Blue: moving, human. Grey: stopped."
    d.text((margin, 20), legend, fill=light)
    img.save(out)
    return {"center": [round(lat0, 7), round(lon0, 7)], "map_radius_m": round(radius, 1)}


def totals(segments):
    hist = {
        key: {k: round(sum(s[key][k] for s in segments), 1) for k in map(str, bins)}
        for key, bins in (
            ("speed_hist_engaged_s", SPEED_BINS),
            ("speed_hist_human_s", SPEED_BINS),
            ("steer_hist_engaged_moving_s", STEER_BINS),
            ("steer_hist_human_moving_s", STEER_BINS),
        )
    }
    starts = sorted(s["start_local"] for s in segments if s["start_local"])
    accuracy = [s["gps_accuracy_median_m"] for s in segments if s["gps_accuracy_median_m"]]
    return {
        "segments": len(segments),
        "hours": round(sum(s["duration_s"] for s in segments) / 3600, 2),
        "moving_h": round(sum(s["moving_s"] for s in segments) / 3600, 2),
        "engaged_h": round(sum(s["engaged_s"] for s in segments) / 3600, 2),
        "engaged_moving_h": round(sum(sum(s["speed_hist_engaged_s"].values()) - s["speed_hist_engaged_s"]["0"] for s in segments) / 3600, 2),
        "local_time_range": [starts[0], starts[-1]] if starts else None,
        "days": sorted({t[:10] for t in starts}),
        "gps_accuracy_median_m": median(accuracy) if accuracy else None,
        "road_frames": sum(s["frames"]["narrowRoadEncodeIdx"] for s in segments),
        "wide_frames": sum(s["frames"]["wideRoadEncodeIdx"] for s in segments),
        "video_gb": round(sum(sum(s["video_mb"].values()) for s in segments) / 1000, 1),
        "sessions": sorted({x for s in segments for x in s["sessions"]}),
        **hist,
    }  # fmt: skip


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=REPO / "runs/artifacts")
    p.add_argument("--out", type=Path, default=REPO / "runs/sim/inventory")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    found = defaultdict(list)
    for rlog in sorted(a.root.rglob("rlog.zst")):
        found[rlog.parent.name].append(rlog)
    with ProcessPoolExecutor(a.workers) as pool:
        segments = list(pool.map(summarize, [paths[0] for paths in found.values()], chunksize=4))
    for s in segments:
        folders = {r.relative_to(a.root).parts[0] for r in found[s["segment"]]}
        s["sessions"] = sorted(folders - {"routes"})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "segments.json").write_text(json.dumps(segments, indent=1))
    report = {
        "all": totals(segments),
        "copies_found": sum(len(v) for v in found.values()),
        "segments_without_gps_fix": sum(not s["gps_track"] for s in segments),
        "car": next((s["params"] for s in sorted(segments, key=lambda s: s["start_local"] or "", reverse=True) if s["params"].get("steer_ratio")), {}),
        "sites": [],
    }  # fmt: skip
    for i, group in enumerate(sites(segments)):
        site = totals(group) | coverage_map(group, a.out / f"coverage-site{i + 1}.png")
        report["sites"].append(
            {k: v for k, v in site.items() if not k.startswith(("speed", "steer"))}
        )
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
