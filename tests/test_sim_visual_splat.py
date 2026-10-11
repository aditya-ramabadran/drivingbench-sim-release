"""Split/camera invariants and optional native checks using synthetic geometry."""

import json

import numpy as np
import pytest
from drivingbench.sim.visual_splat import camera_manifest, train_command


def cameras():
    return dict(
        w=1344,
        h=760,
        fl_x=1000,
        fl_y=1000,
        cx=672,
        cy=380,
        frames=[dict(file_path="images/frame.jpg", transform_matrix=np.eye(4).tolist())],
    )


def test_camera_manifest_preserves_opengl_pose():
    data = cameras()
    data["frames"][0]["transform_matrix"][0][3] = 3
    result = camera_manifest(data)
    assert result["frames"][0]["name"] == "frame"
    assert result["frames"][0]["transform_matrix"][0][3] == 3
    assert result["w"] == 1344 and result["h"] == 760


@pytest.mark.parametrize(
    "matrix", [np.eye(3), np.diag([1, 1, -1, 1]), np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)]
)
def test_reject_nonrigid_camera(matrix):
    data = cameras()
    data["frames"][0]["transform_matrix"] = matrix.tolist()
    with pytest.raises(ValueError):
        camera_manifest(data)


@pytest.mark.parametrize("name", ["../outside", "", ".."])
def test_reject_unsafe_render_name(name):
    data = cameras()
    data["frames"][0]["name"] = name
    with pytest.raises(ValueError):
        camera_manifest(data)


def test_duplicate_render_names():
    data = cameras()
    data["frames"] *= 2
    with pytest.raises(ValueError):
        camera_manifest(data)


def test_split_rejects_path_alias_leakage(tmp_path):
    for split, name in [("train", "images/a.jpg"), ("val", "images/../images/a.jpg")]:
        (tmp_path / f"transforms_{split}.json").write_text(
            json.dumps({"frames": [{"file_path": name}]})
        )
    with pytest.raises(ValueError, match="disjoint"):
        train_command(tmp_path, tmp_path / "out")


def test_explicit_split_and_training_budget(tmp_path):
    for split in ("train", "val"):
        (tmp_path / f"transforms_{split}.json").write_text(
            json.dumps({"frames": [{"file_path": f"images/{split}.jpg"}]})
        )
    command = train_command(tmp_path, tmp_path / "out", steps=30, max_splats=5000)
    assert "--eval-split-every" not in command
    assert command[command.index("--total-steps") + 1] == "30"
    (tmp_path / "transforms.json").write_text("{}")
    with pytest.raises(ValueError, match="prefer"):
        train_command(tmp_path, tmp_path / "out")


def test_prepare_excludes_colmap_and_preserves_split(tmp_path):
    from drivingbench.sim.visual_splat import prepare_dataset

    source, out = tmp_path / "sfm", tmp_path / "brush"
    source.mkdir()
    (source / "cameras.bin").write_bytes(b"excluded")
    (source / "init.ply").write_text("points")
    for split in ("train", "val"):
        (source / f"{split}.jpg").write_bytes(b"image")
        (source / f"transforms_{split}.json").write_text(
            json.dumps({"frames": [{"file_path": f"{split}.jpg"}]})
        )
    prepare_dataset(source, out)
    assert not (out / "cameras.bin").exists()
    assert (out / "images/train.jpg").samefile(source / "train.jpg")
    assert (
        json.loads((out / "transforms_val.json").read_text())["frames"][0]["file_path"]
        == "images/val.jpg"
    )


def test_native_expected_depth_and_rgb_unchanged(tmp_path):
    import subprocess

    from drivingbench.sim.visual_splat import RENDER, render_frames, serve_command, stream_request
    from PIL import Image

    if not RENDER.is_file():
        pytest.skip("Optional native Brush renderer is not installed")
    fields = (
        "x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
    )
    rows = [
        [0, 0, -2, 1.77, -1.77, -1.77, 0, *([np.log(0.15)] * 3), 1, 0, 0, 0],
        [0, 0, -6, -1.77, -1.77, 1.77, np.log(3), *([np.log(0.45)] * 3), 1, 0, 0, 0],
        [1, 0, -4, -1.77, 1.77, -1.77, 3, *([np.log(0.02)] * 3), 1, 0, 0, 0],
    ]
    checkpoint = tmp_path / "three.ply"
    checkpoint.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        + "".join(f"property float {field}\n" for field in fields)
        + "end_header\n"
        + "\n".join(" ".join(map(str, row)) for row in rows)
        + "\n"
    )
    pose = np.eye(4)
    forward = pose.copy()
    forward[2, 3] = -1
    data = dict(
        w=65,
        h=49,
        fl_x=40,
        fl_y=40,
        cx=32.5,
        cy=24.5,
        frames=[
            dict(name="start", transform_matrix=pose.tolist()),
            dict(name="forward", transform_matrix=forward.tolist()),
        ],
    )
    render_frames(checkpoint, data, tmp_path / "rgb")
    render_frames(checkpoint, data, tmp_path / "depth", depth=True)
    alpha = np.load(tmp_path / "depth/start_depth_alpha.npy")
    depth = np.load(tmp_path / "depth/start_depth.npy")
    translated = np.load(tmp_path / "depth/forward_depth.npy")
    assert depth.dtype == np.float32 and depth.shape == (49, 65)
    assert alpha[24, 32] == pytest.approx(0.875, abs=2e-5)
    assert depth[24, 32] == pytest.approx((0.5 * 2 + 0.5 * 0.75 * 6) / 0.875, abs=2e-5)
    assert translated[24, 32] == pytest.approx(depth[24, 32] - 1, abs=2e-5)
    assert np.isnan(depth[0, 0]) and alpha[0, 0] < 0.5
    for name in ("start", "forward"):
        assert np.array_equal(
            np.asarray(Image.open(tmp_path / f"rgb/{name}_color.png")),
            np.asarray(Image.open(tmp_path / f"depth/{name}_color.png")),
        )

    camera = {k: v for k, v in data.items() if k != "frames"}
    requests = [
        dict(
            id="invalid",
            camera=camera,
            transform_matrix=np.diag([2, 1, 1, 1]).tolist(),
            output=str(tmp_path / "invalid"),
        ),
        *[
            stream_request(
                name, matrix @ np.diag([1, -1, -1, 1]), camera, tmp_path / f"stream-{name}"
            )
            for name, matrix in (("start", pose), ("forward", forward))
        ],
    ]
    result = subprocess.run(
        serve_command(checkpoint),
        input="{\n" + "\n".join(json.dumps(r) for r in requests) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    responses = [
        json.loads(line.removeprefix("DRIVINGBENCH "))
        for line in result.stdout.splitlines()
        if line.startswith("DRIVINGBENCH ")
    ]
    assert responses[0]["ready"]
    assert responses[1]["id"] is None and not responses[1]["ok"]
    assert responses[2]["id"] == "invalid" and not responses[2]["ok"]
    assert [r["id"] for r in responses[3:]] == ["start", "forward"]
    assert all(r["ok"] for r in responses[3:])
    for name in ("start", "forward"):
        folder = tmp_path / f"stream-{name}"
        assert {p.name for p in folder.iterdir()} == {"rgb.npy", "depth.npy", "alpha.npy"}
        rgb = np.load(folder / "rgb.npy")
        assert rgb.dtype == np.uint8 and rgb.shape == (49, 65, 3)
        assert np.array_equal(rgb, np.asarray(Image.open(tmp_path / f"rgb/{name}_color.png")))
        assert np.allclose(
            np.load(folder / "depth.npy"),
            np.load(tmp_path / f"depth/{name}_depth.npy"),
            atol=2e-5,
            equal_nan=True,
        )
        assert np.allclose(
            np.load(folder / "alpha.npy"),
            np.load(tmp_path / f"depth/{name}_depth_alpha.npy"),
            atol=2e-5,
        )


def test_stream_request_is_opencv_and_absolute_without_starting_worker(tmp_path):
    from drivingbench.sim.visual_splat import serve_command, stream_request

    data = cameras()
    camera = {k: v for k, v in data.items() if k != "frames"}
    pose = np.eye(4)
    pose[0, 3] = 3
    request = stream_request(12, pose, camera, tmp_path / "frame")
    assert request["id"] == 12
    assert request["transform_matrix"] == pose.tolist()
    assert request["camera"] == camera
    assert request["output"] == str(tmp_path / "frame")
    assert serve_command(tmp_path / "scene.ply")[1:] == [str(tmp_path / "scene.ply"), "--serve"]
    with pytest.raises(ValueError):
        stream_request(1, np.diag([1, 1, -1, 1]), camera, tmp_path)


def test_explicit_training_experiments(tmp_path):
    for split in ("train", "val"):
        (tmp_path / f"transforms_{split}.json").write_text(
            json.dumps({"frames": [{"file_path": f"{split}.jpg"}]})
        )
    default = train_command(tmp_path, tmp_path / "out")
    assert "--start-iter" not in default and "--growth-grad-threshold" not in default
    command = train_command(
        tmp_path,
        tmp_path / "out",
        steps=40000,
        start_iter=30000,
        growth_stop_iter=15000,
        growth_grad_threshold=2e-5,
        refine_every=100,
    )
    assert command[command.index("--start-iter") + 1] == "30000"
    assert command[command.index("--growth-grad-threshold") + 1] == "2e-05"
    for options in [
        dict(start_iter=2000),
        dict(start_iter=-1),
        dict(growth_stop_iter=-1),
        dict(growth_grad_threshold=0),
        dict(growth_grad_threshold=np.nan),
        dict(refine_every=0),
    ]:
        with pytest.raises(ValueError):
            train_command(tmp_path, tmp_path / "out", **options)
