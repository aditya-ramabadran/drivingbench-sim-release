import io
import json

import numpy as np
import pytest
from drivingbench.sim.object_worker import PREFIX, emit, serve_requests
from drivingbench.sim.visual_objects import (
    PlacedObject,
    build_object_job,
    empty_layer,
    projected_bounds,
)


def test_object_job_shares_serialization_and_plane_basis():
    normal = np.array([0.01, -0.02, 1.0])
    normal /= np.linalg.norm(normal)
    obj = PlacedObject(
        "dome_marker", 1, 2, yaw=0.3, diameter_m=0.2, height_m=0.06, color_srgb=(0.8, 0.2, 0.1)
    )
    job = build_object_job([obj], normal)
    assert json.loads(json.dumps(job))["objects"][0]["diameter_m"] == 0.2
    rotation = np.array(job["rotations"][0])
    np.testing.assert_allclose(rotation[:, 2], normal)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    assert "transform_matrix" not in job and "output" not in job
    with pytest.raises(ValueError):
        build_object_job([obj], [0, 0, 0])


def test_protocol_error_does_not_desynchronize_next_request_or_shutdown():
    replies = []
    calls = []

    def render(request):
        calls.append(request["id"])
        if request["id"] == "bad":
            raise ValueError("invalid camera")
        return {"elapsed_s": 0.1}

    lines = io.StringIO(
        "invalidjson\n"
        + json.dumps({"id": "bad"})
        + "\n"
        + json.dumps({"id": "good"})
        + "\n"
        + json.dumps({"id": "stop", "shutdown": True})
        + "\n"
        + json.dumps({"id": "not reached"})
        + "\n"
    )
    serve_requests(lines, render, replies.append)
    assert [r["id"] for r in replies] == [None, "bad", "good", "stop"]
    assert [r["ok"] for r in replies] == [False, False, True, True]
    assert calls == ["bad", "good"]


def test_protocol_prefix_and_json_line(capsys):
    emit({"id": 4, "ok": True})
    line = capsys.readouterr().out
    assert line.startswith(PREFIX) and line.endswith("\n")
    assert json.loads(line[len(PREFIX) :]) == {"id": 4, "ok": True}


def test_render_bounds_include_padding_and_reject_offscreen_near_plane_boxes():
    camera = dict(w=100, h=80, fl_x=50, fl_y=50, cx=50, cy=40)
    box = np.array([[-1, -1, 5], [1, 1, 6]])
    assert projected_bounds([box], np.eye(4), camera, padding=2) == (38, 28, 62, 52)
    offscreen = np.array([[9, -1, -0.1], [11, 1, 0.1]])
    assert projected_bounds([offscreen], np.eye(4), camera) is None
    crossing = np.array([[-1, -1, -0.1], [1, 1, 1]])
    assert projected_bounds([crossing], np.eye(4), camera) == (0, 0, 100, 80)
    assert projected_bounds([box + np.array([0, 0, -10])], np.eye(4), camera) is None


def test_empty_layer_is_transparent_with_unsupported_depth_and_no_instances():
    layer = empty_layer(dict(w=8, h=5))
    assert layer.rgba.shape == (5, 8, 4) and layer.rgba.dtype == np.uint8
    assert not layer.rgba.any() and not layer.instance.any() and not layer.repaired.any()
    assert layer.depth.dtype == np.float32 and np.isnan(layer.depth).all()
