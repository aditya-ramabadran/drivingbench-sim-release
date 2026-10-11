import json

import pytest
from drivingbench.replay.clip import ReplayClip, distance_table
from PIL import Image


def samples(speeds=(1, 2, 3)):
    return [
        dict(t=i * 0.1, speed_mps=v, steering_deg=i, gear="drive") for i, v in enumerate(speeds)
    ]


def test_integrates_acceleration_at_frame_time():
    rows = distance_table(samples(), [0.025, 0.075, 0.125, 0.175])
    assert rows[0]["distance_m"] == 0
    assert rows[-1]["distance_m"] == pytest.approx(0.3)
    assert rows[1]["speed_mps"] == pytest.approx(1.75)
    assert rows[1]["steering_deg"] == pytest.approx(0.75)


@pytest.mark.parametrize("fault", ["reverse", "gap", "unordered", "negative", "uncovered"])
def test_rejects_invalid_source(fault):
    rows = samples()
    if fault == "reverse":
        rows[1]["gear"] = "reverse"
    if fault == "gap":
        rows[1]["t"] = 0.19
    if fault == "unordered":
        rows[1]["t"] = 0
    if fault == "negative":
        rows[1]["speed_mps"] = -0.2
    if fault == "uncovered":
        rows[0]["t"] = 0.05
    with pytest.raises(ValueError):
        distance_table(rows, [0.025, 0.175])


def test_stationary_requires_explicit_allowance():
    rows = samples((0, 0, 1))
    with pytest.raises(ValueError, match="stationary"):
        distance_table(rows, [0, 0.1, 0.2])
    assert distance_table(rows, [0, 0.1, 0.2], allow_stops=True)[-1]["distance_m"] == 0.05
    with pytest.raises(ValueError, match="no forward"):
        distance_table(samples((0, 0, 0)), [0, 0.2], allow_stops=True)


def test_clip_holds_reached_frame_clamps_and_preserves_cache(tmp_path):
    for i in range(3):
        Image.new("RGB", (5, 4), (i * 50, 0, 0)).save(tmp_path / f"{i}.png")
    manifest = {
        "length_m": 0.5,
        "frames": [dict(distance_m=i * 0.25, narrow=f"{i}.png", wide=f"{i}.png") for i in range(3)],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    clip = ReplayClip(path)
    assert clip.sample_at(-1)["distance_m"] == 0
    assert clip.sample_at(0.49)["distance_m"] == 0.25
    assert clip.sample_at(4)["distance_m"] == 0.5
    im = clip.frame_at(0.4)
    im.putpixel((0, 0), (255, 255, 255))
    assert clip.frame_at(0.4).getpixel((0, 0)) == (50, 0, 0)
    with pytest.raises(ValueError):
        clip.frame_at(float("nan"))
    with pytest.raises(ValueError):
        clip.frame_at(0, "invalid")
    manifest["frames"][1]["distance_m"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ReplayClip(path)
