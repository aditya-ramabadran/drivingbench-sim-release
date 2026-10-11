"""Frozen perception probes use exact received bytes without a live producer."""

import json
import sys
from pathlib import Path

import pytest
from drivingbench.replay.clients import is_offline_replay
from drivingbench.sim.perception_probe import freeze, parse_answer, verify_images
from PIL import Image


def test_freeze_preserves_bytes_and_separate_camera_blocks(tmp_path):
    paths = {}
    for camera, color in (("narrow", "red"), ("wide", "blue")):
        path = tmp_path / f"{camera}.jpg"
        Image.new("RGB", (16, 9), color).save(path)
        paths[camera] = str(path)
    manifest = freeze([{**paths, "source_id": "received-pair"}], tmp_path / "frozen")
    assert len(manifest["items"]) == 3
    pair = manifest["items"][0]
    assert [row["camera"] for row in pair["images"]] == ["narrow", "wide"]
    assert all(
        Path(row["path"]).read_bytes() == Path(paths[row["camera"]]).read_bytes()
        for row in pair["images"]
    )
    assert all(
        sorted(order) == sorted(item["id"] for item in manifest["items"])
        for order in manifest["orders"].values()
    )
    with pytest.raises(FileExistsError):
        freeze([paths], tmp_path / "frozen")
    Path(paths["wide"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_images(pair["images"])


def test_neutral_answers_keep_literal_counts_and_uncertainty():
    answer = {
        "actors": [
            {"kind": "dog", "count": "one", "location": "left", "road_relation": "beside road"}
        ],
        "uncertainty": "motion unknown",
    }
    assert parse_answer("```json\n" + json.dumps(answer) + "\n```") == answer
    assert parse_answer('{"actors": []}') == {"actors": []}
    assert parse_answer('{"actors": [{"kind": "person"}]}') is None
    assert parse_answer("none visible") is None


def test_static_server_is_allowlisted_without_public_transport(tmp_path):
    command = [
        sys.executable,
        "-m",
        "drivingbench.sim.perception_probe",
        "serve",
        "--images",
        str(tmp_path / "images.json"),
    ]
    assert is_offline_replay(command)
    assert not is_offline_replay([*command[:-1], "relative.json"])
    assert not is_offline_replay([*command, "--gateway-url", "http://example.com"])


def test_prose_before_json_does_not_discard_literal_answer():
    assert parse_answer(
        'Nothing is visible.\n```json\n{"actors": [], "uncertainty": "blurred"}\n```'
    ) == {"actors": [], "uncertainty": "blurred"}
    assert parse_answer('No actors. {"actors": [], "uncertainty": ""} Additional text.') == {
        "actors": [],
        "uncertainty": "",
    }


def test_delivery_accepts_registered_resize_but_not_different_picture(tmp_path):
    import base64
    import io

    from drivingbench.sim.perception_probe import receipt_matches
    from drivingbench.sim.realism import delivered_hashes, delivered_images, digest

    source = tmp_path / "source.jpg"
    Image.new("RGB", (1344, 760), (120, 130, 140)).save(source, quality=92)
    sources = [{"camera": "wide", "path": str(source), "sha256": digest(source)}]
    stream = io.BytesIO()
    Image.open(source).resize((1024, 579)).save(stream, "JPEG", quality=92)
    payload = stream.getvalue()
    events = [
        {"result": {"content": [{"type": "image", "data": base64.b64encode(payload).decode()}]}}
    ]
    decoded = delivered_images(events)
    assert set(decoded) == delivered_hashes(events)
    evidence = receipt_matches(decoded, sources)
    assert evidence[0]["match"]["accepted"]
    assert not evidence[0]["match"]["exact"]
    assert evidence[0]["match"]["received_size"] == [1024, 579]
    assert receipt_matches({"original": source.read_bytes()}, sources)[0]["match"]["exact"]
    wrong = io.BytesIO()
    Image.new("RGB", (1024, 579), "red").save(wrong, "JPEG")
    assert (
        receipt_matches({"wrong": wrong.getvalue(), "junk": b"not image"}, sources)[0]["match"]
        is None
    )
