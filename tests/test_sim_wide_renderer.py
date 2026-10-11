"""CPU-only contracts for native wide composition and paired worker ownership."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from drivingbench.sim import render_worker, wide_renderer
from drivingbench.sim.visual_wide import INTRINSICS
from PIL import Image


@pytest.fixture
def wide_scene(tmp_path):
    calibration = dict(
        intrinsics=dict(INTRINSICS, fl_x=1000, fl_y=1000),
        distortion=[0, 0, 0, 0],
        rotation_wide_from_narrow=np.eye(3).tolist(),
        translation_wide_from_narrow=[0.1, 0, 0],
    )
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    body = np.zeros((760, 1344, 4), np.uint8)
    body[680:] = [130, 140, 150, 255]
    Image.fromarray(body).save(tmp_path / "body.png")
    tiles = []
    for name, angle in (("left", -0.15), ("right", 0.15)):
        c, s = np.cos(angle), np.sin(angle)
        tiles.append(
            dict(
                name=name,
                w=80,
                h=80,
                fl_x=32,
                fl_y=32,
                cx=40,
                cy=40,
                rotation_wide_from_tile=[[c, 0, s], [0, 1, 0], [-s, 0, c]],
            )
        )
    config = dict(
        checkpoint=str(tmp_path / "wide.ply"),
        calibration=str(tmp_path / "calibration.json"),
        vehicle_layer=str(tmp_path / "body.png"),
        tiles=tiles,
    )
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"wide_camera": config}))
    return path, config


def test_wide_raw_tiles_share_pose_time_then_warp_and_single_encode(
    wide_scene, monkeypatch, tmp_path
):
    pytest.importorskip("cv2")
    path, config = wide_scene
    requests, constructors, encoded = [], [], []
    matrix = np.eye(4)
    matrix[:3, 3] = [4, 5, 6]

    class Backend:
        def __init__(self, *args, **kwargs):
            constructors.append(kwargs)
            self.scene = SimpleNamespace(camera_matrix=lambda pose: matrix)
            self.last_metrics = {}

        def render_matrices_rgb(self, views, **kwargs):
            results = []
            for camera_matrix, camera in views:
                requests.append((camera_matrix.copy(), dict(kwargs, camera=camera)))
                results.append(
                    (np.full((80, 80, 3), [20, 40, 60], np.uint8), {"count": len(requests)}, {})
                )
            return results

    monkeypatch.setattr(wide_renderer, "PersistentRenderer", Backend)
    renderer = wide_renderer.WideRenderer(path, output=tmp_path / "output")
    save = Image.Image.save

    def capture(image, *args, **kwargs):
        encoded.append((np.asarray(image).copy(), args, kwargs))
        return save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", capture)
    result = renderer.render((7, 8, 9), sim_time_s=12, actor_onsets={"dog": 3})
    assert constructors[0]["checkpoint_override"] == Path(config["checkpoint"])
    assert len(requests) == 2 and len(encoded) == 1
    wide = renderer.calibration.camera_pose(matrix)
    for (actual, kwargs), tile in zip(requests, config["tiles"], strict=True):
        np.testing.assert_allclose(actual[:3, :3], wide[:3, :3] @ tile["rotation_wide_from_tile"])
        np.testing.assert_allclose(actual[:3, 3], wide[:3, 3])
        assert kwargs["sim_time_s"] == 12 and kwargs["actor_onsets"] == {"dog": 3}
    image, args, kwargs = encoded[0]
    np.testing.assert_array_equal(image[:680], np.broadcast_to([20, 40, 60], image[:680].shape))
    np.testing.assert_array_equal(image[680:], np.broadcast_to([130, 140, 150], image[680:].shape))
    assert kwargs["quality"] == 92 and args[1] == "JPEG"
    with Image.open(io.BytesIO(result)) as jpeg:
        assert jpeg.size == (1344, 760)
    assert renderer.last_metrics["tiles"]["left"] == {"count": 1}
    assert renderer.last_metrics["tiles"]["right"] == {"count": 2}


@pytest.mark.parametrize("bad", ["alpha", "dimensions", "uncovered", "rotation"])
def test_wide_rejects_invalid_layers_or_uncovered_exterior(wide_scene, tmp_path, bad):
    path, config = wide_scene
    if bad in {"alpha", "dimensions"}:
        size = (10, 10) if bad == "dimensions" else (1344, 760)
        Image.new("RGBA", size, (1, 2, 3, 128 if bad == "alpha" else 0)).save(
            config["vehicle_layer"]
        )
    elif bad == "uncovered":
        for tile in config["tiles"]:
            tile["fl_x"] = tile["fl_y"] = 10000
    else:
        config["tiles"][0]["rotation_wide_from_tile"] = np.zeros((3, 3)).tolist()
    path.write_text(json.dumps({"wide_camera": config}))
    with pytest.raises(ValueError, match="binary alpha|1344x760|uncovered|rotation"):
        wide_renderer.WideRenderer(path, output=tmp_path / "output")


@pytest.mark.parametrize("failure", [None, "startup", "render"])
def test_worker_pair_protocol_and_teardown(wide_scene, monkeypatch, tmp_path, failure):
    path, _ = wide_scene
    instances, replies, ready = {}, [], []

    def factory(camera):
        class Backend:
            def __init__(self, *args, **kwargs):
                self.closed = 0
                self.last_metrics = {"billboards": {camera: 10}, "elapsed_s": 1}
                instances[camera] = self

            def start(self):
                if camera == "wide" and failure == "startup":
                    raise RuntimeError("wide startup failed")
                return {"ready": True, "device": "fake"}

            def render(self, pose, **kwargs):
                self.request = (pose, kwargs)
                if camera == "wide" and failure == "render":
                    raise RuntimeError("wide render failed")
                return camera.encode()

            def close(self):
                self.closed += 1

        return Backend

    request = dict(
        pose=[1, 2, 3], sim_time_s=8, actor_onsets={"child": 4}, output=str(tmp_path / "frame")
    )
    monkeypatch.setattr(render_worker, "PersistentRenderer", factory("narrow"))
    monkeypatch.setattr(render_worker, "WideRenderer", factory("wide"))
    monkeypatch.setattr(render_worker, "emit", ready.append)
    monkeypatch.setattr(
        render_worker, "serve_requests", lambda stream, render: replies.append(render(request))
    )
    if failure == "startup":
        with pytest.raises(RuntimeError, match="startup"):
            render_worker.run(path, tmp_path / "worker", io.StringIO())
        assert not ready and not replies
    else:
        render_worker.run(path, tmp_path / "worker", io.StringIO())
        response = replies[0]
        assert set(ready[0]["cameras"]) == {"narrow", "wide"}
        assert instances["narrow"].request == instances["wide"].request
        assert instances["wide"].request[1] == {"sim_time_s": 8, "actor_onsets": {"child": 4}}
        assert response["metrics"]["billboards"] == {"narrow": 10}
        assert Path(response["paths"]["narrow"]).read_bytes() == b"narrow"
        if failure == "render":
            assert set(response["paths"]) == {"narrow"}
            assert "wide render failed" in response["errors"]["wide"]
            assert "wide" not in response["metrics"]
        else:
            assert Path(response["paths"]["wide"]).read_bytes() == b"wide"
            assert response["metrics"]["wide"]["billboards"] == {"wide": 10}
            assert not response["errors"]
    assert all(renderer.closed == 1 for renderer in instances.values())


def test_worker_narrow_failure_keeps_wide_and_recovers_next_pair(wide_scene, monkeypatch, tmp_path):
    path, _ = wide_scene
    instances, replies = {}, []

    def factory(camera):
        class Backend:
            def __init__(self, *args, **kwargs):
                self.requests = []
                self.closed = 0
                self.last_metrics = {}
                instances[camera] = self

            def start(self):
                return {"ready": True}

            def render(self, pose, **kwargs):
                self.requests.append((pose, kwargs))
                # Mutate metrics in place to catch accidental aliasing between replies.
                self.last_metrics["capture"] = {"number": len(self.requests)}
                if camera == "narrow" and len(self.requests) == 1:
                    raise RuntimeError("injected narrow failure")
                return f"{camera}-{len(self.requests)}".encode()

            def close(self):
                self.closed += 1

        return Backend

    requests = [
        dict(
            pose=[number, 2, 3],
            sim_time_s=number,
            actor_onsets={"child": number - 0.5},
            output=str(tmp_path / f"frame-{number}"),
        )
        for number in (1, 2)
    ]
    monkeypatch.setattr(render_worker, "PersistentRenderer", factory("narrow"))
    monkeypatch.setattr(render_worker, "WideRenderer", factory("wide"))
    monkeypatch.setattr(render_worker, "emit", lambda _: None)

    def serve(stream, render):
        for request in requests:
            replies.append(render(request))

    monkeypatch.setattr(render_worker, "serve_requests", serve)
    render_worker.run(path, tmp_path / "worker", io.StringIO())
    failed, recovered = replies
    assert set(failed["paths"]) == {"wide"}
    assert failed["errors"] == {"narrow": "RuntimeError: injected narrow failure"}
    assert Path(failed["paths"]["wide"]).read_bytes() == b"wide-1"
    assert "capture" not in failed["metrics"]
    assert failed["metrics"]["wide"]["capture"] == {"number": 1}
    assert not recovered["errors"]
    assert set(recovered["paths"]) == {"narrow", "wide"}
    for camera, backend in instances.items():
        assert Path(recovered["paths"][camera]).read_bytes() == f"{camera}-2".encode()
        assert backend.closed == 1
        assert backend.requests == [
            (request["pose"], {key: request[key] for key in ("sim_time_s", "actor_onsets")})
            for request in requests
        ]
    assert recovered["metrics"]["capture"] == {"number": 2}
    assert recovered["metrics"]["wide"]["capture"] == {"number": 2}
