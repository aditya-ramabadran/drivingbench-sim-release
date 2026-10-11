"""Per-pass marker consistency and renderer-independent provisional physics layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from drivingbench.sim.visual_cones import K, triangulate

PASSES = {
    "early_out": (100, 195),
    "early_return": (195, 250),
    "mid_return": (490, 510),
    "late_out": (690, 775),
    "late_return": (775, 800),
}
PILOT_SESSION = "2026-09-17-gpt-6-astra-codex-664abe"
LAYOUT_ID = "pilot_664abe_100_800"


def fit_pass(observations):
    """Refit one pass without borrowing observations or positions from another pass."""
    observations = list(
        {o.get("name", tuple(np.round(o["center"], 6))): o for o in observations}.values()
    )
    if len(observations) < 5 or len({tuple(np.round(o["center"], 6)) for o in observations}) < 5:
        return None
    centers = np.array([o["center"] for o in observations])
    directions = np.array([o["direction"] for o in observations])
    try:
        point, residual, condition = triangulate(centers, directions)
    except np.linalg.LinAlgError:
        return None
    baseline = np.degrees(np.arccos(np.clip(np.min(directions @ directions.T), -1, 1)))
    depth = np.sum((point - centers) * directions, axis=1)
    if baseline < 4 or condition > 1e5 or np.any(depth <= 0):
        return None
    pixel_residual = residual / depth * K[0, 0]
    if np.median(pixel_residual) > 4 or np.percentile(pixel_residual, 95) > 8:
        return None
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    information = np.sum(projectors * ((K[0, 0] / depth / 2) ** 2)[:, None, None], axis=0)
    covariance = np.linalg.inv(information)
    return {
        "position_enu_m": point.tolist(),
        "views": len(observations),
        "baseline_deg": float(baseline),
        "reprojection_median_px": float(np.median(residual / depth * K[0, 0])),
        "conditional_sigma_xy_m": float(np.sqrt(np.linalg.eigvalsh(covariance[:2, :2]).max())),
    }


def audit(markers):
    rows = []
    for marker in markers:
        fits = {}
        for name, (start, end) in PASSES.items():
            selected = [o for o in marker["observations"] if start <= o["t"] < end]
            if fitted := fit_pass(selected):
                fits[name] = fitted
        row = {"id": marker["id"], "fits": fits}
        if "early_out" in fits and "late_out" in fits:
            a, b = (fits[k] for k in ("early_out", "late_out"))
            delta = np.array(b["position_enu_m"]) - a["position_enu_m"]
            row["early_late_delta_xy_m"] = float(np.linalg.norm(delta[:2]))
            row["combined_conditional_sigma_xy_m"] = float(
                np.hypot(a["conditional_sigma_xy_m"], b["conditional_sigma_xy_m"])
            )
        rows.append(row)
    return rows


def footprint(marker):
    """Nominal physical-base circles; colored silhouettes do not survey outer bases."""
    if marker["kind"] == "dome_marker":
        radius = 0.10
        source = "Rounded above crop-derived colored radius upper quartile 0.0934m; outer-base allowance is an assumption"
    elif marker["kind"] == "tall_cone":
        radius = float(np.ceil(marker["apparent_color_width_m"] / 2 * 100) / 100)
        source = "Per-marker training-view apparent colored base-width/2, rounded upward to 0.01m"
    else:
        raise ValueError(f"Unsupported physical marker class: {marker['kind']}")
    return {
        "shape": "circle",
        "radius_m": radius,
        "status": "nominal physical outer-base estimate, not a surveyed footprint",
        "source": source,
        "uncertainty": "Physical outer edge, camera scale and segmentation bias are unmeasured; no validated confidence interval",
    }


def digest(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def export_layout(markers, fits, calibration):
    base = json.loads(Path("docs/sim/pilot-course.json").read_text())
    by_id = {r["id"]: r for r in fits}
    objects = []
    for marker in markers:
        fit = by_id[marker["id"]]
        objects.append(
            {
                "id": marker["id"],
                "class": marker["kind"],
                "color": marker["color"],
                "position_enu_m": marker["position_enu_m"],
                "conditional_sigma_xy_m": marker["pixel_sigma_xy_m"],
                "fitted_passes": list(fit["fits"]),
                "observed_time_range_s": marker["time_range_s"],
                "collision_footprint": footprint(marker),
            }
        )
    layout = {
        "id": LAYOUT_ID,
        "session_id": PILOT_SESSION,
        "route": "00000042--544757d3f5",
        "time_range_monotonic_s": [100, 800],
        "held_out_range_s": [340, 480],
        "status": "Provisional shared-layout hypothesis; repeated observations support consistency, not proof of immobility throughout gaps",
        "frame": base["frame"],
        "registration": base["provenance"],
        "markers": objects,
        "boundary_order": None,
        "course_exit_rule": None,
    }
    return {
        "purpose": "Renderer-independent physics layout evidence; circles are nominal physical-base estimates",
        "provenance": {
            "course": digest(Path("docs/sim/pilot-course.json")),
            "calibration": digest(calibration),
            "observations": digest(Path("runs/sim/visual/cones/landmarks.json")),
            "broad_audit": digest(Path("runs/sim/visual/marker-layouts/broad/audit.json")),
        },
        "layouts": [layout],
        "session_index": digest(Path("docs/sim/marker-session-layouts.json")),
        "absence_semantics": "Sampled unmarked parking patches do not imply an empty entire site; null coordinate layout means unknown",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/sim/visual/marker-layouts"))
    parser.add_argument("--layout", type=Path, default=Path("docs/sim/marker-layouts.json"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    markers = json.loads(Path("runs/sim/visual/cones/landmarks.json").read_text())["landmarks"]
    rows = audit(markers)
    (args.out / "pass-fits.json").write_text(json.dumps(rows, indent=2))
    layout = export_layout(
        markers,
        rows,
        Path("runs/sim/visual/marker-eval/calibration.json"),
    )
    args.layout.write_text(json.dumps(layout, indent=2) + "\n")


if __name__ == "__main__":
    main()
