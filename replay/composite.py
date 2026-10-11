"""Private deterministic sprite propagation; generated assets require separate human QA.

The source image is never resampled. Only the RGBA subject is projected onto it.
This is a rigid upright billboard approximation, not a novel-view renderer.
"""

import argparse
import hashlib
import html
import json
import math
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from drivingbench.replay.clip import ReplayClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@dataclass(frozen=True)
class Appearance:
    exposure_ev: float = 0.0
    contrast: float = 1.0
    blur_px: float = 0.0
    shadow_opacity: float = 0.0
    shadow_width_m: float = 0.45
    shadow_length_m: float = 0.25

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("finite appearance values required")
        if not (
            -2 <= self.exposure_ev <= 2
            and 0 < self.contrast <= 2
            and 0 <= self.blur_px <= 3
            and 0 <= self.shadow_opacity <= 1
            and 0 < self.shadow_width_m <= 3
            and 0 < self.shadow_length_m <= 3
        ):
            raise ValueError("appearance values outside supported range")


class CompositeClip:
    def __init__(
        self,
        source,
        asset_path,
        *,
        height_m,
        hazard_distance_m=20.0,
        lateral_m=0.0,
        min_depth_m=1.5,
        geometry_path=None,
        foot_xy=None,
        body_height_px=None,
        appearance=None,
    ):
        self.appearance = appearance or Appearance()
        self.source = source
        self.path = source.path
        self.asset_path = Path(asset_path).resolve()
        self.height_m = float(height_m)
        self.hazard_distance_m = float(hazard_distance_m)
        self.lateral_m = float(lateral_m)
        self.min_depth_m = float(min_depth_m)
        if not all(
            math.isfinite(v)
            for v in (self.height_m, self.hazard_distance_m, self.lateral_m, self.min_depth_m)
        ):
            raise ValueError("finite geometry required")
        if (
            self.height_m <= 0
            or self.min_depth_m <= 0
            or self.hazard_distance_m <= self.min_depth_m
        ):
            raise ValueError("positive height and hazard distance beyond minimum depth required")
        self.length_m = min(source.length_m, self.hazard_distance_m - self.min_depth_m)
        with Image.open(self.asset_path) as asset:
            if "A" not in asset.getbands():
                raise ValueError("asset needs a real alpha channel")
            self.sprite = asset.convert("RGBA")
        alpha = self.sprite.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None or alpha.getextrema()[0] > 0:
            raise ValueError("asset must contain both subject and fully transparent background")
        self.alpha_diagnostics = {
            "nonzero_bbox": bbox,
            "strong_bbox_gt10": alpha.point(lambda value: 255 if value > 10 else 0).getbbox(),
            "extrema": alpha.getextrema(),
            "note": "Diagnostics only; original alpha and soft edges are preserved.",
        }
        self.foot_xy = tuple(foot_xy or ((bbox[0] + bbox[2]) / 2, bbox[3]))
        self.body_height_px = float(body_height_px or (bbox[3] - bbox[1]))
        if len(self.foot_xy) != 2 or not all(math.isfinite(v) for v in self.foot_xy):
            raise ValueError("finite foot coordinates required")
        if not math.isfinite(self.body_height_px) or self.body_height_px <= 0:
            raise ValueError("positive body height required")
        self.geometry_path = Path(geometry_path).resolve() if geometry_path else None
        self.geometry = json.loads(self.geometry_path.read_text()) if self.geometry_path else {}
        self.focal_px = 1141.5
        self.center_x = 672.0
        self.horizon_y = 416.2
        self.camera_height_m = 1.064
        # Optional sequence contract: reference-to-current ground image homographies.
        # A lone H in geometry-estimate.json is not a per-frame registration sequence.
        self.homographies = self.geometry.get("ground_homographies", [])
        self.geometry_mode = (
            "per_frame_ground_registration"
            if self.homographies
            else "rigid_flat_straight_approximation"
        )
        self.manifest = source.manifest | {"length_m": self.length_m}

    def sample_at(self, distance_m):
        if not math.isfinite(distance_m):
            raise ValueError("finite distance required")
        return self.source.sample_at(max(0.0, min(distance_m, self.length_m)))

    def projection(self, distance_m):
        row = self.sample_at(distance_m)
        distance = row["distance_m"]
        depth = self.hazard_distance_m - distance
        if depth < self.min_depth_m - 1e-8:
            raise ValueError("hazard projection passed minimum depth")
        x = self.center_x + self.focal_px * self.lateral_m / depth
        y = self.horizon_y + self.focal_px * self.camera_height_m / depth
        height = self.focal_px * self.height_m / depth
        if self.homographies:
            matches = [r for r in self.homographies if abs(r["distance_m"] - distance) < 1e-5]
            if len(matches) != 1:
                raise ValueError("per-frame ground homography missing or duplicated")
            matrix = matches[0]["H"]
            reference = (
                self.center_x + self.focal_px * self.lateral_m / self.hazard_distance_m,
                self.horizon_y + self.focal_px * self.camera_height_m / self.hazard_distance_m,
                1,
            )
            if len(matrix) != 3 or any(len(r) != 3 for r in matrix):
                raise ValueError("homography must be 3x3")
            mapped = [sum(a * b for a, b in zip(r, reference)) for r in matrix]
            if abs(mapped[2]) < 1e-8:
                raise ValueError("singular footprint projection")
            x, y = mapped[0] / mapped[2], mapped[1] / mapped[2]
            # Ground registration updates the footprint. Upright body remains a
            # billboard whose scale follows nominal depth, not the road homography.
        if not all(math.isfinite(v) for v in (x, y, height)) or height <= 0:
            raise ValueError("invalid projected footprint")
        return x, y, height

    def frame_at(self, distance_m, camera="narrow"):
        if camera != "narrow":
            raise ValueError("composite has no registered wide-camera asset")
        frame = self.source.frame_at(min(max(distance_m, 0), self.length_m), camera)
        # Calibration is for the raw 1344x760 narrow road image only.
        if frame.size != (1344, 760):
            raise ValueError("narrow source dimensions do not match projection calibration")
        x, y, height = self.projection(distance_m)
        scale = height / self.body_height_px
        size = tuple(max(1, round(v * scale)) for v in self.sprite.size)
        if max(size) > 8 * max(frame.size):
            raise ValueError("projected sprite exceeds allocation limit")
        sprite = self.sprite.resize(size, Image.Resampling.LANCZOS)
        appearance = self.appearance
        if appearance.contrast != 1 or appearance.exposure_ev != 0:
            alpha = sprite.getchannel("A")
            rgb = ImageEnhance.Contrast(sprite.convert("RGB")).enhance(appearance.contrast)
            rgb = ImageEnhance.Brightness(rgb).enhance(2**appearance.exposure_ev)
            sprite = rgb.convert("RGBA")
            sprite.putalpha(alpha)
        if appearance.blur_px:
            # Premultiplication prevents transparent RGB colors making an edge halo.
            sprite = (
                sprite.convert("RGBa")
                .filter(ImageFilter.GaussianBlur(appearance.blur_px))
                .convert("RGBA")
            )
        if appearance.shadow_opacity:
            depth = self.hazard_distance_m - self.sample_at(distance_m)["distance_m"]
            width = self.focal_px * appearance.shadow_width_m / depth
            length = self.focal_px * self.camera_height_m * appearance.shadow_length_m / depth**2
            blur = max(0.7, width * 0.09)
            margin = math.ceil(4 * blur)
            shadow_size = (math.ceil(width) + 2 * margin, math.ceil(max(1, length)) + 2 * margin)
            if max(shadow_size) > 8 * max(frame.size):
                raise ValueError("projected shadow exceeds allocation limit")
            alpha = Image.new("L", shadow_size)
            ImageDraw.Draw(alpha).ellipse(
                (margin, margin, margin + width, margin + max(1, length)),
                fill=round(255 * appearance.shadow_opacity),
            )
            alpha = alpha.filter(ImageFilter.GaussianBlur(blur))
            location = (round(x - width / 2 - margin), round(y - length / 2 - margin))
            frame.paste(Image.new("RGB", shadow_size), location, alpha)
        position = (round(x - self.foot_xy[0] * scale), round(y - self.foot_xy[1] * scale))
        # Pillow clips the sprite at frame boundaries; zero-alpha source pixels
        # remain bit-for-bit identical to the original RGB frame.
        frame.paste(sprite, position, sprite.getchannel("A"))
        return frame


def export(clip, output, provenance=None, *, fps=10, seed=0):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output exists; composite exports are immutable")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("positive integer fps required")
    output.mkdir(parents=True)
    (output / "narrow").mkdir()
    distances = [r["distance_m"] for r in clip.source.frames if r["distance_m"] <= clip.length_m]
    if distances[-1] != clip.length_m:
        distances.append(clip.length_m)
    rows = []
    hashes = {}
    for i, distance in enumerate(distances):
        relative = f"narrow/{i:05d}.png"
        clip.frame_at(distance).save(output / relative)
        row = clip.sample_at(distance)
        row = {k: v for k, v in row.items() if not k.startswith("wide")}
        row.update(distance_m=distance, narrow=relative)
        rows.append(row)
        hashes[relative] = sha256(output / relative)
    manifest = clip.source.manifest | {
        "length_m": clip.length_m,
        "frames": rows,
        "provenance": {
            "source_manifest": str(clip.source.path),
            "source_manifest_sha256": sha256(clip.source.path),
            "asset": str(clip.asset_path),
            "asset_sha256": sha256(clip.asset_path),
            "asset_alpha": clip.alpha_diagnostics,
            "geometry_sha256": sha256(clip.geometry_path) if clip.geometry_path else None,
            "generation": provenance or {},
            "appearance": asdict(clip.appearance),
            "frame_sha256": hashes,
            "projection": {
                "mode": clip.geometry_mode,
                "height_m": clip.height_m,
                "hazard_distance_m": clip.hazard_distance_m,
                "lateral_m": clip.lateral_m,
                "min_depth_m": clip.min_depth_m,
                "foot_xy": clip.foot_xy,
                "body_height_px": clip.body_height_px,
                "focal_px": clip.focal_px,
                "center_x": clip.center_x,
                "horizon_y": clip.horizon_y,
                "camera_height_m": clip.camera_height_m,
            },
        },
        "validity": clip.source.manifest.get("validity", {})
        | {
            "composite_validation": "pending",
            "cameras": ["narrow"],
            "geometry": "Approximate upright billboard on flat ground; no occlusion, pose animation or relighting.",
            "identity": "One fixed RGBA sprite with deterministic scale and recorded appearance adjustments.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "validation.json").write_text(
        json.dumps(
            {
                "status": "pending",
                "blinded_human_check": {
                    "status": "pending",
                    "reviewers_required": 3,
                    "accuracy_threshold": 0.6,
                    "design_caveat": "This board pairs empty originals with hazard composites. Hazard presence predicts the label, so classification accuracy is not a clean test of realism. A formal blinded check requires natural hazard-containing controls and no category disclosure; passing this board alone cannot validate realism.",
                },
                "sequence_3x_check": {"status": "pending"},
                "far_frame_perceivability": {"status": "pending"},
                "interpretation": "Exploratory only until all specified checks pass; software export is not human validation.",
            },
            indent=2,
        )
        + "\n"
    )
    # Render at constant 1m/s in replay distance, independent of source driving speed.
    preview = output / ".preview-frames"
    preview.mkdir()
    try:
        for i in range(math.ceil(clip.length_m * fps) + 1):
            clip.frame_at(min(clip.length_m, i / fps)).save(preview / f"{i:05d}.png")
        for multiplier, name in ((1, "preview-1x.mp4"), (3, "preview-3x.mp4")):
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-framerate",
                    str(fps * multiplier),
                    "-i",
                    str(preview / "%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output / name),
                ],
                check=True,
            )
    finally:
        shutil.rmtree(preview)
    make_contact_sheet(clip, output)
    assessment = output / "assessment"
    assessment.mkdir()
    selected = sorted({round(i * (len(distances) - 1) / 5) for i in range(6)})
    conditions = [(i, edited) for i in selected for edited in (False, True)]
    random.Random(seed).shuffle(conditions)
    answers = []
    cards = []
    for number, (i, edited) in enumerate(conditions, 1):
        filename = f"frame-{number:02d}.jpg"
        image = clip.frame_at(distances[i]) if edited else clip.source.frame_at(distances[i])
        image.save(assessment / filename, quality=92)
        answers.append({"frame": number, "edited": edited, "distance_m": distances[i]})
        cards.append(
            f'<article><h2>Frame {number}</h2><img src="{html.escape(filename)}" alt="Frame {number}"><p>Real / Edited — confidence: ____</p></article>'
        )
    (assessment / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Frame assessment</title><style>body{font-family:system-ui;max-width:1000px;margin:auto}img{max-width:100%}article{margin:2rem 0}</style><h1>Independent frame assessment</h1><p>Classify each frame as real or edited. Record answers before viewing videos or discussing with others.</p>'
        + "\n".join(cards)
    )
    (output / "assessment-answer-key.json").write_text(
        json.dumps({"seed": seed, "answers": answers}, indent=2) + "\n"
    )
    return output / "manifest.json"


def make_contact_sheet(clip, output):
    """Review the actual model JPEG encoding, retaining full-resolution keyframes."""
    output = Path(output)
    review = output / "review"
    review.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (3 * 672, 420), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, travelled) in enumerate((("far", 0), ("mid", 8), ("near", 15))):
        distance = min(travelled, clip.length_m)
        path = review / f"{label}.jpg"
        clip.frame_at(distance).save(path, "JPEG", quality=92)
        # Reopen the encoded bytes: the sheet must not show unencoded PNGs.
        with Image.open(path) as decoded:
            thumbnail = decoded.resize((672, 380), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, (column * 672, 40))
        draw.text(
            (column * 672 + 12, 12),
            f"{label}: travelled {distance:g}m; depth {clip.hazard_distance_m - clip.sample_at(distance)['distance_m']:.2f}m",
            fill="black",
        )
    sheet.save(review / "contact-sheet.jpg", quality=95)
    return review / "contact-sheet.jpg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height-m", required=True, type=float)
    parser.add_argument("--hazard-distance-m", type=float, default=20)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    clip = CompositeClip(
        ReplayClip(args.manifest),
        args.asset,
        height_m=args.height_m,
        hazard_distance_m=args.hazard_distance_m,
        geometry_path=args.geometry,
    )
    print(
        export(
            clip, args.output, json.loads(args.provenance.read_text()) if args.provenance else None
        )
    )


if __name__ == "__main__":
    main()
