import json

import pytest
from drivingbench.replay.clip import ReplayClip
from drivingbench.replay.composite import CompositeClip, export
from PIL import Image, ImageChops


@pytest.fixture
def source(tmp_path):
    rows = []
    for i in range(3):
        name = f"{i}.png"
        Image.new("RGB", (1344, 760), (60 + i, 80, 100)).save(tmp_path / name)
        rows.append({"distance_m": float(i), "narrow": name})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"length_m": 2, "frames": rows}))
    asset = Image.new("RGBA", (20, 40))
    asset.paste((255, 0, 0, 255), (5, 5, 15, 35))
    asset.save(tmp_path / "asset.png")
    return ReplayClip(manifest), tmp_path / "asset.png"


def test_projection_scales_and_preserves_source_pixels(source):
    base, asset = source
    clip = CompositeClip(base, asset, height_m=1, hazard_distance_m=10)
    assert clip.projection(2)[2] / clip.projection(0)[2] == pytest.approx(10 / 8)
    before = base.frame_at(0)
    result = clip.frame_at(0)
    box = ImageChops.difference(before, result).getbbox()
    assert box is not None
    assert box[0] > 600 and box[2] < 740
    assert result.getpixel((0, 0)) == before.getpixel((0, 0))
    assert base.frame_at(0).tobytes() == before.tobytes()
    assert clip.frame_at(0.5).tobytes() == result.tobytes()
    with pytest.raises(ValueError, match="wide"):
        clip.frame_at(0, "wide")
    with pytest.raises(ValueError, match="finite"):
        clip.frame_at(float("nan"))


def test_endpoint_limits_and_alpha_requirement(source):
    base, asset = source
    clip = CompositeClip(base, asset, height_m=1, hazard_distance_m=2, min_depth_m=1)
    assert clip.length_m == 1
    assert clip.frame_at(100).tobytes() == clip.frame_at(1).tobytes()
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(asset)
    with pytest.raises(ValueError, match="transparent"):
        CompositeClip(base, asset, height_m=1)


def test_export_keeps_validation_pending_and_blind_filenames(source, tmp_path, monkeypatch):
    base, asset = source
    clip = CompositeClip(base, asset, height_m=1)
    calls = []
    monkeypatch.setattr(
        "drivingbench.replay.composite.subprocess.run", lambda args, **kw: calls.append(args)
    )
    output = tmp_path / "export"
    manifest = export(clip, output, {"tool": "test"}, fps=1)
    replay = ReplayClip(manifest)
    assert replay.length_m == clip.length_m
    assert replay.frame_at(1).tobytes() == clip.frame_at(1).tobytes()
    assert len(calls) == 2
    assert json.loads((output / "validation.json").read_text())["status"] == "pending"
    assessment = (output / "assessment/index.html").read_text()
    assert "asset.png" not in assessment and "distance_m" not in assessment
    assert len(json.loads((output / "assessment-answer-key.json").read_text())["answers"]) == 6
    with pytest.raises(ValueError, match="immutable"):
        export(clip, output)


def test_registered_ground_footprint_is_used(source, tmp_path):
    base, asset = source
    geometry = tmp_path / "geometry.json"
    geometry.write_text(
        json.dumps(
            {"ground_homographies": [{"distance_m": 0, "H": [[1, 0, 8], [0, 1, 2], [0, 0, 1]]}]}
        )
    )
    rigid = CompositeClip(base, asset, height_m=1)
    registered = CompositeClip(base, asset, height_m=1, geometry_path=geometry)
    x, y, h = rigid.projection(0)
    assert registered.projection(0) == (x + 8, y + 2, h)
    with pytest.raises(ValueError, match="missing"):
        registered.projection(1)


def test_appearance_shadow_changes_only_local_support(source):
    from drivingbench.replay.composite import Appearance

    base, asset = source
    clip = CompositeClip(
        base,
        asset,
        height_m=1,
        appearance=Appearance(exposure_ev=-0.1, contrast=0.95, blur_px=0.3, shadow_opacity=0.18),
    )
    before = base.frame_at(0)
    output = clip.frame_at(0)
    changed = ImageChops.difference(before, output).getbbox()
    assert changed is not None
    assert changed[0] > 600 and changed[2] < 740
    assert output.getpixel((0, 0)) == before.getpixel((0, 0))
    assert base.frame_at(0).tobytes() == before.tobytes()
    assert clip.frame_at(0).tobytes() == output.tobytes()
    with pytest.raises(ValueError, match="appearance"):
        Appearance(shadow_opacity=2)
