import json
import shutil
import subprocess

import pytest
from drivingbench.sim.visual_data import ROUTES, decode, select_expression, split, validate_index
from PIL import Image


def frame(i, t):
    return {"index": i, "sof_ns": t - 1, "eof_ns": t}


def test_frame_index_uses_presentation_order_and_rejects_ambiguous_mapping():
    assert validate_index([frame(1, 20), frame(0, 10)])[0]["eof_ns"] == 10
    for rows in (
        [frame(0, 10), frame(2, 20)],
        [frame(0, 10), frame(0, 20)],
        [frame(0, 10), frame(1, 9)],
        [frame(0, 0)],
    ):
        with pytest.raises(ValueError):
            validate_index(rows)


def test_split_holds_out_a_time_range_with_guard_on_both_sides():
    assert [split(ROUTES[1], t) for t in (329, 330, 339, 340, 480, 481, 490, 491)] == [
        "train",
        "guard",
        "guard",
        "test",
        "test",
        "guard",
        "guard",
        "train",
    ]
    assert split(ROUTES[0], 400) == "train"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is an optional visual tool")
def test_decode_selects_exact_indices_and_resumes_without_renumbering(tmp_path):
    source = tmp_path / "source.hevc"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "nullsrc=s=1344x760:r=20,geq=lum='40+40*N':cb=128:cr=128",
        "-frames:v",
        "4",
        "-c:v",
        "libx265",
        "-x265-params",
        "log-level=error",
        str(source),
    ]
    subprocess.run(command, check=True, capture_output=True)
    rows = [{"index": k, "file_path": str(tmp_path / f"{k}.jpg")} for k in (1, 3)]
    assert decode((source, rows)) == 2
    with Image.open(rows[0]["file_path"]) as first, Image.open(rows[1]["file_path"]) as last:
        assert first.size == last.size == (1344, 760)
        assert first.getpixel((600, 400))[0] == pytest.approx(80, abs=3)
        assert last.getpixel((600, 400))[0] == pytest.approx(160, abs=3)
    assert decode((source, json.loads(json.dumps(rows)))) == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is an optional visual tool")
def test_dense_frame_selection_stays_within_ffmpeg_parser_limit():
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=2x2:r=1200:d=1",
            "-vf", f"select={select_expression(list(range(0, 1200, 4)))}",
            "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
        ],
        capture_output=True,
        check=True,
    )  # fmt: skip
    assert len(result.stdout) == 300 * 2 * 2
