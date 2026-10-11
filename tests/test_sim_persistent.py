from pathlib import Path

import numpy as np
import pytest
from drivingbench.sim.persistent import PersistentRenderer


class Worker:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False
        self.requests = []

    def request(self, **payload):
        self.requests.append(payload)
        if self.fail:
            raise RuntimeError("shader abort")
        return {"ok": True}

    def close(self):
        self.closed = True


def renderer():
    value = PersistentRenderer.__new__(PersistentRenderer)
    value.objects = Worker(fail=True)
    value.restarts = 0
    value.device = "METAL"
    return value


def test_shader_crash_and_failed_warmup_retry_exact_pose_on_cpu():
    value = renderer()
    attempts = []

    def start(matrix):
        attempts.append((value.device, matrix.copy()))
        value.objects = Worker()
        if len(attempts) == 1:
            raise RuntimeError("warmup abort")

    value._objects_start = start
    matrix = np.eye(4)
    matrix[0, 3] = 7
    assert value._request("objects", matrix, Path("/tmp/one-frame"))["ok"]
    assert [row[0] for row in attempts] == ["METAL", "CPU"]
    assert value.restarts == 2
    assert value.objects.requests[0]["transform_matrix"] == matrix.tolist()


def test_retry_budget_is_bounded():
    value = renderer()

    def start(matrix):
        value.objects = Worker(fail=True)

    value._objects_start = start
    with pytest.raises(RuntimeError, match="shader abort"):
        value._request("objects", np.eye(4), Path("/tmp/failed-frame"))
    assert value.restarts == 2
    assert value.objects.closed


def test_optional_support_file_does_not_change_jpeg_or_replace_existing_artifact(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from PIL import Image

    value = PersistentRenderer.__new__(PersistentRenderer)
    value.output, value.response, value.restarts, value.device = tmp_path, None, 0, "CPU"
    value.scene = SimpleNamespace(
        objects_at=lambda *a, **k: [],
        ground=SimpleNamespace(normal=(0, 0, 1)),
        data={"markers": [], "objects": []},
    )
    value.billboards = SimpleNamespace(layers=lambda *a: [])
    shape = (8, 12)
    value.camera = dict(w=12, h=8, fl_x=10, fl_y=10, cx=6, cy=4)
    rgb = np.full((*shape, 3), 77, np.uint8)
    support = np.zeros(shape, bool)
    support[2:4, 5:7] = True

    def request(kind, matrix, output, *, camera=None):
        output.mkdir()
        if kind == "splat":
            for name, array in (("rgb", rgb), ("depth", np.ones(shape)), ("alpha", np.ones(shape))):
                np.save(output / f"{name}.npy", array)
            return {}
        paths = {}
        for name, array in (("rgba", np.zeros((*shape, 4), np.uint8)), ("depth", np.ones(shape))):
            path = output / f"{name}.npy"
            np.save(path, array)
            paths[name] = path
        return {"paths": paths}

    value._request = request
    monkeypatch.setattr(
        "drivingbench.sim.persistent.compose",
        lambda *a, **kw: (rgb, {}, support) if kw["return_support"] else (rgb, {}),
    )
    ordinary = value.render_matrix(np.eye(4))
    path = tmp_path / "frame-mask.png"
    diagnostic = value.render_matrix(np.eye(4), support_output=path)
    assert ordinary == diagnostic
    np.testing.assert_array_equal(np.array(Image.open(path)), support.astype(np.uint8) * 255)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        value.render_matrix(np.eye(4), support_output=path)
    assert path.read_bytes() == before


def test_tile_pipeline_overlaps_draw_and_compose_without_reordering():
    from threading import Event

    value = PersistentRenderer.__new__(PersistentRenderer)
    composing, release = Event(), Event()
    calls = []

    def draw(matrix, *, camera):
        index = int(matrix[0, 3])
        calls.append(index)
        if index == 1:
            assert composing.wait(1), "First composition must start before the next draw"
            release.set()
        return index

    def compose(frame, **kwargs):
        assert kwargs == {"sim_time_s": 7, "actor_onsets": {"child": 2}}
        if frame == 0:
            composing.set()
            assert release.wait(1), "Composition must overlap the next GPU draw"
        return np.full((2, 3, 3), frame, np.uint8), {"index": frame}

    value._draw_matrix, value._compose_frame = draw, compose
    views = []
    for index in range(3):
        matrix = np.eye(4)
        matrix[0, 3] = index
        views.append((matrix, {"w": 3, "h": 2}))
    results = value.render_matrices_rgb(views, sim_time_s=7, actor_onsets={"child": 2})
    assert calls == [0, 1, 2]
    assert [metrics["index"] for image, metrics in results] == calls
    for index, (image, _) in enumerate(results):
        assert np.all(image == index)


def test_tile_pipeline_propagates_composition_failure_and_finishes_workers():
    from threading import Event

    value = PersistentRenderer.__new__(PersistentRenderer)
    completed = Event()
    value._draw_matrix = lambda matrix, **kwargs: matrix

    def compose(frame, **kwargs):
        if frame == 0:
            raise RuntimeError("bad tile")
        completed.set()
        return frame

    value._compose_frame = compose
    with pytest.raises(RuntimeError, match="bad tile"):
        value.render_matrices_rgb([(0, {}), (1, {})])
    assert completed.is_set()
