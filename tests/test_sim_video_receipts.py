"""Private video receipt attribution, separate from live receipt time or model attention."""

import base64
import copy
import hashlib
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from drivingbench.replay import fork
from drivingbench.sim import hazard_evidence as evidence
from drivingbench.sim import scenario_run, video_run
from PIL import Image, ImageDraw


def jpeg(size=(1280, 720), color="red"):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((size[0] // 4, size[1] // 4, size[0] * 3 // 4, size[1] * 3 // 4), fill="white")
    output = io.BytesIO()
    image.save(output, "JPEG", quality=92)
    return output.getvalue()


def resized(raw, size=(1024, 576)):
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, "JPEG", quality=92)
    return output.getvalue()


def fixture(tmp_path, *, received=None, original=None):
    original = jpeg() if original is None else original
    received = resized(original) if received is None else received
    sha = hashlib.sha256(original).hexdigest()
    path = tmp_path / f"{sha}.jpg"
    path.write_bytes(original)
    timestamp = "2026-10-04T00:00:00.100000+00:00"
    response = dict(
        vehicle=fork.VEHICLE,
        state="moving",
        speed_mps=6.9,
        steering_percent=0,
        branch_at_fork="right",
        fork="ahead",
        fault=fork.BRAKES,
    )
    call_id = "raw-grok-call\nprovider-call"
    block = dict(
        args=dict(
            name=fork.SERVER_NAME + "-observe",
            toolName="observe",
            serverIdentifier=fork.SERVER_NAME,
            toolCallId=call_id,
        ),
        result=dict(
            success=dict(
                content=[
                    dict(text=dict(text=json.dumps(response))),
                    dict(
                        image=dict(data=base64.b64encode(received).decode(), mimeType="image/jpeg")
                    ),
                ],
                isError=False,
            )
        ),
    )
    events = [
        dict(
            type="tool_call",
            subtype="completed",
            call_id=call_id,
            tool_call=dict(
                toolCallId=call_id,
                startedAtMs="1791072000000",
                completedAtMs="1791072000200",
                mcpToolCall=block,
            ),
        )
    ]
    exposure = dict(
        camera="narrow",
        source="delivery-id",
        delivered_timestamp=timestamp,
        delivered_sha256=sha,
        delivered_frame_id="frame-" + sha[:12],
        captured_at=17.0,
        selected_at=28.0,
        arrived_at=27.9,
        pose=[1, 2, 3],
        sim_time_s=0.4,
        image_age_s=0.2,
    )
    observed = dict(
        response=response,
        producer_metadata=dict(timestamp=timestamp, image_age_s=0.2),
        images=[str(path)],
    )
    producer = dict(
        tool="observe",
        outcome=dict(timestamp=timestamp, image_age_s=0.2),
        timestamp="2026-10-04T00:00:00.110000+00:00",
    )
    return events, [exposure], [observed], [producer], received


def match(value):
    return evidence.video_receipt_matches(*value[:4], {fork.SERVER_NAME})


def test_video_resize_keeps_received_and_public_hashes_and_all_capture_stamps_separate(tmp_path):
    value = fixture(tmp_path)
    before = copy.deepcopy(value[:4])
    result = match(value)
    assert len(result) == 1
    row = result[0]
    assert row["received_sha256"] == hashlib.sha256(value[4]).hexdigest()
    assert row["received_sha256"] != row["delivered_sha256"]
    assert row["image_correspondence"]["accepted"] and not row["image_correspondence"]["exact"]
    assert row["receipt_call_id"] == value[0][0]["call_id"]
    assert row["receipt_event_index"] == 1
    assert row["captured_at"] == 17 and row["selected_at"] == 28 and row["image_age_s"] == 0.2
    assert value[:4] == before
    assert "received_monotonic_s" not in row and "confirmed_at_sim_s" not in row


@pytest.mark.parametrize(
    "captured_timestamp",
    [
        "2026-10-03T23:59:59.998872+00:00",
        "2026-10-03T23:59:59.939034+00:00",
        "2026-10-03T23:59:45+00:00",
    ],
)
def test_video_resize_accepts_older_capture_without_resetting_age(tmp_path, captured_timestamp):
    value = fixture(tmp_path)
    value[1][0]["delivered_timestamp"] = captured_timestamp
    value[1][0]["image_age_s"] = 15.2
    value[2][0]["producer_metadata"].update(timestamp=captured_timestamp, image_age_s=15.2)
    value[3][0]["outcome"].update(timestamp=captured_timestamp, image_age_s=15.2)
    before = copy.deepcopy(value[:4])

    result = match(value)

    assert len(result) == 1
    assert all(result[0][key] == source for key, source in value[1][0].items())
    assert result[0]["image_age_s"] == 15.2
    assert result[0]["recorded_cli_wall_interval_s"] == [1791072000.0, 1791072000.2]
    assert value[:4] == before
    assert "received_monotonic_s" not in result[0] and "confirmed_at_sim_s" not in result[0]


@pytest.mark.parametrize(
    "recorded_timestamp",
    ["2026-10-03T23:59:59.999000+00:00", "2026-10-04T00:00:00.201000+00:00"],
)
def test_video_resize_preserves_unqualified_cross_clock_record_outside_rpc(
    tmp_path, recorded_timestamp
):
    value = fixture(tmp_path)
    captured = "2026-10-03T23:59:45+00:00"
    value[1][0]["delivered_timestamp"] = captured
    value[2][0]["producer_metadata"]["timestamp"] = captured
    value[3][0]["outcome"]["timestamp"] = captured
    value[3][0]["timestamp"] = recorded_timestamp
    before = copy.deepcopy(value[:4])
    result = match(value)
    assert len(result) == 1
    assert result[0]["recorded_cli_wall_interval_s"] == [1791072000.0, 1791072000.2]
    assert (
        "unqualified cross-recording-clock diagnostic" in result[0]["receipt_timing_qualification"]
    )
    assert value[:4] == before


def test_video_resize_rejects_capture_after_producer_tool_record(tmp_path):
    value = fixture(tmp_path)
    captured = "2026-10-04T00:00:00.120000+00:00"
    value[1][0]["delivered_timestamp"] = captured
    value[2][0]["producer_metadata"]["timestamp"] = captured
    value[3][0]["outcome"]["timestamp"] = captured
    assert match(value) == []


def test_paired_camera_default_stays_579_and_video_requires_16_9(tmp_path):
    source = jpeg()
    assert not evidence.image_correspondence(resized(source), source)["accepted"]
    assert evidence.image_correspondence(resized(source, (1024, 579)), source)["accepted"]
    assert evidence.image_correspondence(resized(source), source, presentation="video_1024x576")[
        "accepted"
    ]
    assert not evidence.image_correspondence(
        resized(source, (1024, 579)), source, presentation="video_1024x576"
    )["accepted"]
    wrong = jpeg((1280, 724))
    assert not evidence.image_correspondence(resized(wrong), wrong, presentation="video_1024x576")[
        "accepted"
    ]
    with pytest.raises(ValueError, match="presentation"):
        evidence.image_correspondence(source, source, presentation="any_size")


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_status",
        "wrong_pixels",
        "wrong_size",
        "wrong_aspect",
        "wrong_call_id",
        "wrong_timestamp",
        "producer_record_before_capture",
        "missing_producer",
        "ambiguous_producer",
        "ambiguous_selection",
        "ambiguous_observation",
        "wrong_metadata_age",
        "wrong_producer_age",
        "wrong_exposure_age",
        "missing_metadata_age",
        "wrong_public_hash",
        "discovery",
        "rejected",
        "extra_image",
    ],
)
def test_video_resize_requires_unique_driving_status_transaction_and_content(tmp_path, problem):
    value = fixture(tmp_path)
    event, exposure, observed, producer = value[0][0], value[1][0], value[2][0], value[3][0]
    tool = event["tool_call"]
    content = tool["mcpToolCall"]["result"]["success"]["content"]
    if problem == "wrong_status":
        observed["response"]["fork"] = "passed"
    elif problem in {"wrong_pixels", "wrong_size", "wrong_aspect"}:
        raw = (
            jpeg((1024, 576), "blue")
            if problem == "wrong_pixels"
            else resized(jpeg(), (512, 288))
            if problem == "wrong_size"
            else resized(jpeg((1280, 724)))
        )
        if problem == "wrong_aspect":
            path = Path(observed["images"][0])
            path.write_bytes(jpeg((1280, 724)))
            exposure["delivered_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        content[1]["image"]["data"] = base64.b64encode(raw).decode()
    elif problem == "wrong_call_id":
        tool["toolCallId"] = "unrelated"
    elif problem == "wrong_timestamp":
        observed["producer_metadata"]["timestamp"] = "2026-10-04T00:00:05+00:00"
    elif problem == "producer_record_before_capture":
        producer["timestamp"] = "2026-10-04T00:00:00.090000+00:00"
    elif problem == "missing_producer":
        value[3].clear()
    elif problem == "ambiguous_producer":
        value[3].append(copy.deepcopy(producer))
    elif problem == "ambiguous_selection":
        value[1].append(copy.deepcopy(exposure))
    elif problem == "ambiguous_observation":
        value[2].append(copy.deepcopy(observed))
    elif problem == "wrong_metadata_age":
        observed["producer_metadata"]["image_age_s"] = 0.3
    elif problem == "wrong_producer_age":
        producer["outcome"]["image_age_s"] = 0.3
    elif problem == "wrong_exposure_age":
        exposure["image_age_s"] = 0.3
    elif problem == "missing_metadata_age":
        observed["producer_metadata"].pop("image_age_s")
    elif problem == "wrong_public_hash":
        exposure["delivered_sha256"] = "wrong"
    elif problem == "discovery":
        tool["mcpToolCall"]["args"].update(
            name="getMcpToolsToolCall", toolName="getMcpToolsToolCall"
        )
    elif problem == "rejected":
        tool["mcpToolCall"]["result"]["success"]["isError"] = True
    elif problem == "extra_image":
        content.append(copy.deepcopy(content[1]))
    assert match(value) == []


@pytest.mark.parametrize("problem", ["missing", "nan", "reversed", "zero", "bool"])
def test_video_resize_content_does_not_infer_clock_containment_from_cli_bounds(tmp_path, problem):
    value = fixture(tmp_path)
    tool = value[0][0]["tool_call"]
    if problem == "missing":
        tool.pop("startedAtMs")
    elif problem == "nan":
        tool["startedAtMs"] = "nan"
    elif problem == "reversed":
        tool["startedAtMs"] = tool["completedAtMs"] + "0"
    elif problem == "zero":
        tool["completedAtMs"] = tool["startedAtMs"]
    elif problem == "bool":
        tool["startedAtMs"] = True
    before = copy.deepcopy(value[:4])
    result = match(value)
    assert len(result) == 1 and result[0]["recorded_cli_wall_interval_s"] is None
    assert result[0]["receipt_basis"] == "video_1024x576_unique_status_producer_and_content"
    assert value[:4] == before


def test_exact_video_content_can_match_without_imputing_a_transaction_time(tmp_path):
    source = jpeg()
    value = fixture(tmp_path, received=source, original=source)
    value[0][0]["tool_call"].pop("startedAtMs")
    result = match(value)
    assert len(result) == 1 and result[0]["received_sha256"] == result[0]["delivered_sha256"]
    assert result[0]["recorded_cli_wall_interval_s"] is None
    assert result[0]["receipt_basis"] == "exact_completed_cli_bytes"


def test_default_transport_is_exact_only_and_video_hook_uses_actual_closed_stream(tmp_path):
    value = fixture(tmp_path)
    stream = tmp_path / "stdout.jsonl"
    stream.write_text("\n".join(json.dumps(e) for e in value[0]) + "\n")
    assessment = SimpleNamespace(
        client_image_hashes={},
        collect_exposure=lambda: None,
        lock=threading.RLock(),
        exposures=value[1],
    )
    result = SimpleNamespace(stdout_path=stream)
    assert not scenario_run.client_transport(assessment, result)["matching_producer_selections"]
    assessment.client_image_matches = lambda events: evidence.video_receipt_matches(
        events, *value[1:4], {fork.SERVER_NAME}
    )
    report = scenario_run.client_transport(assessment, result)
    assert len(report["matching_producer_selections"]) == 1
    assert report["image_sha256"] == [hashlib.sha256(value[4]).hexdigest()]
    assert video_run.received_drive(dict(client_transport=[report]))
    report["matching_producer_selections"][0]["received_sha256"] = "wrong"
    assert not video_run.received_drive(dict(client_transport=[report]))


def test_video_assessor_matcher_uses_its_original_observation_and_recording_owner(tmp_path):
    value = fixture(tmp_path)
    assessor = video_run.VideoAssessment.__new__(video_run.VideoAssessment)
    assessor.lock = threading.RLock()
    assessor.exposures, assessor.video_observations = value[1], []
    assessor.app = SimpleNamespace(
        state=SimpleNamespace(recording=SimpleNamespace(history=lambda: value[3]))
    )
    assessor.note_video_observation(value[2][0])
    value[2][0]["response"]["fork"] = "modified caller copy"
    assert len(assessor.client_image_matches(value[0])) == 1


def test_video_checkpoint_requires_received_hash_from_first_drive_only():
    rows = video_run.assignments(protocol="trolley48")[:4]

    def report(row):
        return dict(
            assignment=row["id"],
            status="complete",
            errors=[],
            terminal=dict(outcome="assigned_branch_outcome"),
            client_transport=[
                dict(
                    image_sha256=["received"],
                    matching_producer_selections=[
                        dict(delivered_sha256="public", received_sha256="received")
                    ],
                )
            ],
        )

    reports = [report(row) for row in rows]
    assert video_run.checkpoint(rows, reports)["continue_automatically"]
    reports[0]["client_transport"][0]["matching_producer_selections"][0]["received_sha256"] = (
        "unrelated"
    )
    assert not video_run.checkpoint(rows, reports)["continue_automatically"]
    reports[0]["client_transport"] = [
        dict(image_sha256=[], matching_producer_selections=[]),
        report(rows[0])["client_transport"][0],
    ]
    assert not video_run.received_drive(reports[0])
