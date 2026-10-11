import base64
import copy
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from drivingbench.sim.hazard_evidence import audit_receipt
from drivingbench.sim.live_receipt import ReceiptWatcher, selected_exposures
from PIL import Image
from test_sim_event_study_v2 import live_fixture


def test_cached_selected_frame_keeps_private_captured_coverage_not_newer_pose(tmp_path):
    _, _, frame, delivery = live_fixture(tmp_path)
    frame["worker"]["captured_coverage"] = {
        "map_pose": [1, 2, 179],
        "measurement": {"supported": False, "distance_to_any_paired_pass_parent_m": 0.01},
    }
    newer = copy.deepcopy(frame)
    newer["frame_ids"] = {camera: key + "-newer" for camera, key in frame["frame_ids"].items()}
    newer["worker"]["captured_coverage"] = {
        "map_pose": [3, 4, 179],
        "measurement": {"supported": True, "distance_to_any_paired_pass_parent_m": 0.0},
    }
    exposures = selected_exposures([frame, newer, delivery])
    assert {row["camera"] for row in exposures} == {"narrow", "wide"}
    assert all(
        row["captured_coverage"] == frame["worker"]["captured_coverage"] for row in exposures
    )


@pytest.mark.parametrize("resized", [False, True])
def test_private_coverage_metadata_reaches_verified_live_and_closed_receipts(tmp_path, resized):
    directory, scene, frame, delivery = live_fixture(tmp_path)
    frame["worker"]["captured_coverage"] = {
        "map_pose": [1, 2, 179],
        "measurement": {"supported": False, "distance_to_any_paired_pass_parent_m": 0.01},
    }
    (directory / "episode/events.jsonl").write_text(
        "\n".join(map(json.dumps, [frame, delivery])) + "\n"
    )
    stream = directory / "client/stdout.jsonl"
    if resized:
        event = json.loads(stream.read_text())
        for block in event["item"]["result"]["content"]:
            if block["type"] == "image":
                with Image.open(io.BytesIO(base64.b64decode(block["data"]))) as image:
                    output = io.BytesIO()
                    image.resize((1024, 579)).save(output, "JPEG", quality=92)
                block["data"] = base64.b64encode(output.getvalue()).decode()
        stream.write_text(json.dumps(event) + "\n")
    watcher = ReceiptWatcher(
        directory, scene, {"drivingbench_sandbox"}, "http://private", clock=lambda: 20
    )
    watcher.poll()
    evidence = json.loads((directory / "live-receipts.jsonl").read_text())
    assert all(
        item["captured_coverage"] == frame["worker"]["captured_coverage"]
        for item in evidence["frames"]
    )
    exposures = [
        dict(row, delivered_timestamp=row["timestamp"], delivered_sha256=row["sha256"])
        for row in selected_exposures([frame, delivery])
    ]
    closed = audit_receipt(
        exposures,
        stream,
        directory / "producer/recordings",
        directory / "received",
        {"drivingbench_sandbox"},
    )
    assert closed["accepted_frames"] == 2
    for item in [*evidence["frames"], *closed["observations"][0]["frames"]]:
        assert item["captured_coverage"] == frame["worker"]["captured_coverage"]
        assert item["exact"] is not resized
        assert (item["producer_sha256"] != item["received_sha256"]) is resized


def test_verified_off_actor_narrow_is_forwarded_with_truthful_flags(tmp_path, monkeypatch):
    directory, scene, frame, delivery = live_fixture(tmp_path)
    frame["worker"]["billboards"]["hazard"]["visible_pixels"] = 0
    (directory / "episode/events.jsonl").write_text(
        "\n".join(map(json.dumps, [frame, delivery])) + "\n"
    )
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs["json"]))
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"status": "confirmed", "reason": None}
        )

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"censored": False}
        ),
    )
    watcher = ReceiptWatcher(
        directory, scene, {"drivingbench_sandbox"}, "http://private", slowed=True, clock=lambda: 20
    )
    assert watcher.poll() == {"censored": False}
    assert len(requests) == 1
    endpoint, request = requests[0]
    assert endpoint == "http://private/sim/receipt"
    assert request["narrow_verified"] is True
    assert request["actor_bearing"] is False
    assert request["path_qualified"] is False
    assert request["correspondence"] == "exact"
    evidence = json.loads((directory / "live-receipts.jsonl").read_text())
    assert evidence["supervisor_request"] == request
    assert evidence["supervisor_reply"]["status"] == "confirmed"
    assert evidence["frames"][0]["actor"]["visible"] is False
    assert evidence["frames"][1]["actor"]["visible"] is True


def test_supervisor_failure_retains_confirmed_public_receipt_evidence(tmp_path, monkeypatch):
    directory, scene, frame, delivery = live_fixture(tmp_path)
    (directory / "episode/events.jsonl").write_text(
        "\n".join(map(json.dumps, [frame, delivery])) + "\n"
    )

    def broken(*_args, **_kwargs):
        raise httpx.ConnectError("private clock endpoint unavailable")

    monkeypatch.setattr(httpx, "post", broken)
    watcher = ReceiptWatcher(
        directory, scene, {"drivingbench_sandbox"}, "http://private", slowed=True, clock=lambda: 20
    )
    with pytest.raises(httpx.ConnectError):
        watcher.poll()
    evidence = json.loads((directory / "live-receipts.jsonl").read_text())
    assert evidence["supervisor_error"].startswith("ConnectError:")
    assert evidence["supervisor_request"]["narrow_verified"] is True
    assert len(evidence["frames"]) == 2


@pytest.mark.parametrize("slowed", [False, True])
def test_final_stream_drain_bypasses_throttle_without_clock_renewal(tmp_path, monkeypatch, slowed):
    directory, scene, frame, delivery = live_fixture(tmp_path)
    (directory / "episode/events.jsonl").write_text(
        "\n".join(map(json.dumps, [frame, delivery])) + "\n"
    )
    stream = directory / "client/stdout.jsonl"
    final_bytes = stream.read_bytes()
    stream.write_bytes(b"")
    now = [20.0]
    watcher = ReceiptWatcher(
        directory,
        scene,
        {"drivingbench_sandbox"},
        "http://private",
        slowed=slowed,
        clock=lambda: now[0],
    )
    watcher.last_poll = now[0]
    stream.write_bytes(final_bytes)
    now[0] += 0.01
    assert watcher.poll() is None
    assert not watcher.records

    def no_clock(*_args, **_kwargs):
        pytest.fail("Terminal collection cannot renew or query the slowed clock")

    monkeypatch.setattr(httpx, "post", no_clock)
    monkeypatch.setattr(httpx, "get", no_clock)
    assert watcher.poll(final=True) is None
    assert len(watcher.records) == 1
    record = watcher.records[0]
    assert record["confirmed_monotonic_s"] == 20.01
    assert record["timing_basis"] == "postdrive_stream_drain_upper_bound"
    assert "supervisor_request" not in record
    assert {(f["captured_at"], f["sim_time_s"]) for f in record["frames"]} == {(11, 2)}
    for selected in selected_exposures([frame, delivery]):
        received = next(f for f in record["frames"] if f["camera"] == selected["camera"])
        assert all(received[key] == value for key, value in selected.items())
    watcher.poll(final=True)
    assert len(watcher.records) == 1


def test_final_stream_drain_still_rejects_nonmatching_images(tmp_path):
    directory, scene, frame, delivery = live_fixture(tmp_path, mismatch=True)
    (directory / "episode/events.jsonl").write_text(
        "\n".join(map(json.dumps, [frame, delivery])) + "\n"
    )
    watcher = ReceiptWatcher(
        directory, scene, {"drivingbench_sandbox"}, "http://private", clock=lambda: 20
    )
    watcher.last_poll = 20
    watcher.poll(final=True)
    assert not watcher.records
    rejection = json.loads((directory / "live-receipt-rejections.jsonl").read_text())
    assert rejection["correspondence"]["accepted"] is False
