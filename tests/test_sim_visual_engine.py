import json

import numpy as np
import pytest
from drivingbench.sim.visual_engine import (
    background_projection,
    ground_projection,
    nearest_training,
    render_batch,
    splat_engine_image,
)


def camera(x=0, y=0):
    return np.array([[0, 0, 1, x], [-1, 0, 0, y], [0, -1, 0, 1.21], [0, 0, 0, 1.0]])


def test_ground_projection_identity_masks_sky_and_hood():
    u, v, visible = ground_projection(camera(), camera(), width=100, height=60, focal=80)
    yy, xx = np.mgrid[:60, :100]
    assert visible[45, 50]
    assert not visible[:31].any()
    assert not visible[54:].any()
    np.testing.assert_allclose(u[visible], xx[visible], atol=1e-10)
    np.testing.assert_allclose(v[visible], yy[visible], atol=1e-10)


def test_ground_projection_translation_has_expected_parallax():
    u, v, visible = ground_projection(camera(), camera(y=1), width=100, height=60, focal=80)
    assert visible[45, 50]
    assert u[45, 50] > 50
    assert v[45, 50] == pytest.approx(45)


def test_texture_selection_rejects_any_held_out_source():
    row = {"split": "train", "transform_matrix": camera().tolist()}
    assert nearest_training(row, [row]) is row
    with pytest.raises(ValueError, match="only training"):
        nearest_training(row, [row, dict(row, split="test")])
    with pytest.raises(ValueError):
        nearest_training(row, [])


def test_texture_selection_uses_pose_only():
    query = {"transform_matrix": camera().tolist()}
    near = {"split": "train", "transform_matrix": camera(x=0.1).tolist()}
    far = {"split": "train", "transform_matrix": camera(x=10).tolist()}
    assert nearest_training(query, [far, near]) is near


def test_infinite_background_ignores_translation_and_masks_near_ground():
    before = background_projection(camera(), camera(), width=100, height=60, focal=80)
    after = background_projection(camera(x=3, y=-2), camera(), width=100, height=60, focal=80)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b)
    assert before[2][10, 50]
    assert not before[2][50, 50]


def test_opposite_camera_has_no_projected_pixels():
    backward = camera()
    backward[:3, :3] = backward[:3, :3] @ np.diag([-1, 1, -1])
    for projection in (background_projection, ground_projection):
        _, _, visible = projection(camera(), backward, width=100, height=60, focal=80)
        assert not visible.any()


def test_splat_engine_composite_respects_premultiplied_color():
    engine = np.full((2, 2, 3), 100, dtype=np.uint8)
    empty, full = np.zeros((2, 2)), np.ones((2, 2))
    np.testing.assert_array_equal(splat_engine_image(engine * 0, empty, engine), engine)
    np.testing.assert_array_equal(splat_engine_image(engine * 2, full, engine), engine * 2)
    result = splat_engine_image(engine, full * 0.5, engine)
    np.testing.assert_array_equal(result, np.full_like(engine, 150))
    with pytest.raises(ValueError, match="Opacity"):
        splat_engine_image(engine, full * 255, engine)


def test_blender_failure_cannot_accept_stale_render(tmp_path, monkeypatch):
    from PIL import Image

    source = tmp_path / "source.jpg"
    Image.new("RGB", (1344, 760), (100, 100, 100)).save(source)
    row = {
        "split": "train",
        "file_path": str(source),
        "name": "frame.jpg",
        "transform_matrix": camera().tolist(),
    }
    output = tmp_path / "output"
    output.mkdir()
    (output / "frame.jpg").write_bytes(source.read_bytes())
    (output / "timings.json").write_text(
        json.dumps(
            {
                "run_id": "stale",
                "renders": [{"name": "frame.jpg"}],
            }
        )
    )

    def failed_script(command, *, check):
        assert command[command.index("--python-exit-code") + 1] == "1"
        assert command.index("--python-exit-code") < command.index("--python")
        assert check is True

    monkeypatch.setattr("drivingbench.sim.visual_engine.subprocess.run", failed_script)
    with pytest.raises(RuntimeError, match="stale images"):
        render_batch([row], [row], output, tmp_path / "Blender")
