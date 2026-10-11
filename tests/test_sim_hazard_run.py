import base64
import io
import json
from pathlib import Path

import pytest
from drivingbench.replay.fork_analysis import RUBRIC
from drivingbench.sim import hazard_evidence as evidence
from drivingbench.sim import scenario_run as run
from drivingbench.sim.scenarios import hazard_scene
from PIL import Image


def jpeg(color, size=(1344, 760)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, "JPEG", quality=92)
    return output.getvalue()


def content(images, timestamp="2026-10-01T00:00:00+00:00"):
    return [dict(type="text", text=json.dumps(dict(timestamp=timestamp)))] + [
        dict(type="image", data=base64.b64encode(image).decode(), mimeType="image/jpeg")
        for image in images
    ]


def stream(path, images, family="codex"):
    blocks = content(images)
    if family == "codex":
        events = [
            dict(
                type="item.completed",
                item=dict(
                    type="mcp_tool_call",
                    id="one",
                    server="drivingbench_sandbox",
                    tool="observe",
                    arguments={},
                    status="completed",
                    result=dict(content=blocks),
                ),
            )
        ]
    elif family == "claude":
        events = [
            dict(
                type="assistant",
                message=dict(
                    content=[
                        dict(
                            type="tool_use",
                            id="one",
                            name="mcp__drivingbench_sandbox__observe",
                            input={},
                        )
                    ]
                ),
            ),
            dict(
                type="user",
                message=dict(content=[dict(type="tool_result", tool_use_id="one", content=blocks)]),
            ),
        ]
    else:
        events = [
            dict(
                type="tool_call",
                subtype="started",
                call_id="one",
                tool_call=dict(
                    mcpToolCall=dict(
                        args=dict(
                            serverIdentifier="drivingbench_sandbox", toolName="observe", args={}
                        )
                    )
                ),
            ),
            dict(
                type="tool_call",
                subtype="completed",
                call_id="one",
                tool_call=dict(mcpToolCall=dict(result=dict(success=dict(content=blocks)))),
            ),
        ]
    path.write_text("\n".join(map(json.dumps, events)))
    return path


@pytest.mark.parametrize("family", ["codex", "claude", "cursor"])
def test_receipt_uses_call_identity_and_one_based_events(tmp_path, family):
    images = [jpeg("red"), jpeg("blue")]
    rows = evidence.received_observations(
        stream(tmp_path / "raw", images, family), {"drivingbench_sandbox"}
    )
    assert len(rows) == 1 and rows[0]["images"] == images
    assert rows[0]["event_index"] == (1 if family == "codex" else 2)


def test_paired_receipt_preserves_actual_bytes_and_missingness(tmp_path):
    recording = tmp_path / "recording"
    recording.mkdir()
    images = [jpeg("red"), jpeg("blue")]
    exposures = []
    for camera, raw in zip(("narrow", "wide"), images, strict=True):
        (recording / f"{camera}.jpg").write_bytes(raw)
        exposures.append(
            dict(
                camera=camera,
                delivered_timestamp="2026-10-01T00:00:00+00:00",
                delivered_sha256=evidence.hashlib.sha256(raw).hexdigest(),
                billboards={"hazard": {"visible_pixels": 12}},
            )
        )
    raw = stream(tmp_path / "raw", images)
    audit = evidence.audit_receipt(
        exposures, raw, recording, tmp_path / "receipt", {"drivingbench_sandbox"}
    )
    assert audit["accepted_frames"] == 2
    assert [
        Path(row["received_path"]).read_bytes() for row in audit["observations"][0]["frames"]
    ] == images
    ambiguous = evidence.audit_receipt(
        exposures + exposures, raw, recording, tmp_path / "ambiguous", {"drivingbench_sandbox"}
    )
    assert ambiguous["observations"][0]["error"] == "ambiguous_timestamp_or_camera_count"
    assert ambiguous["accepted_frames"] == 0


def test_resize_correspondence_is_not_exact_and_wrong_frame_is_rejected():
    source = jpeg("red")
    audit = evidence.image_correspondence(jpeg("red", (1024, 579)), source)
    assert audit["accepted"] and not audit["exact"]
    assert not evidence.image_correspondence(jpeg("blue", (1024, 579)), source)["accepted"]
    assert not evidence.image_correspondence(jpeg("red", (512, 289)), source)["accepted"]


def test_latency_uses_received_visible_camera_not_produced_or_invisible_frame():
    receipt = dict(
        observations=[
            dict(
                call_id="first",
                event_index=2,
                status_timestamp="2026-10-01T00:00:00+00:00",
                client_timestamp="2026-10-01T00:00:02+00:00",
                frames=[
                    dict(
                        camera="narrow", accepted=True, billboards={"hazard": {"visible_pixels": 0}}
                    ),
                    dict(
                        camera="wide", accepted=True, billboards={"hazard": {"visible_pixels": 10}}
                    ),
                ],
            )
        ]
    )
    calls = [
        dict(
            tool="set_motion",
            timestamp=f"2026-10-01T00:00:0{i}+00:00",
            outcome={"status": "accepted"},
        )
        for i in (1, 3)
    ]
    report = evidence.response_timing(receipt, calls)
    assert report["narrow"] is None
    assert report["wide"]["client_receipt_to_first_server_record_s"] == 1
    assert report["wide"]["producer_response_to_server_command_proxy_s"] == 3
    receipt["observations"][0]["client_timestamp"] = None
    assert (
        evidence.response_timing(receipt, calls)["wide"]["client_receipt_to_first_server_record_s"]
        is None
    )


def test_hazard_twenty_cells_freeze_exact_debrief_and_no_pressure(tmp_path, monkeypatch):
    base = {
        "frame": "pilot_ground",
        "calibration": "unused",
        "background": {"checkpoint": "unused"},
        "wide_camera": {"checkpoint": "wide", "calibration": "widecal", "vehicle_layer": "body"},
    }
    basepath = tmp_path / "base.json"
    basepath.write_text(json.dumps(base))
    scenes = {}
    for condition in run.HAZARD_CONDITIONS:
        scene = hazard_scene(base, condition, assets={})
        scene["wide_camera"] = base["wide_camera"]
        scene["scenario"]["coverage"] = dict(
            polygon=[[-10, -10], [60, -10], [60, 10], [-10, 10]],
            heading_deg=0,
            max_heading_deviation_deg=20,
        )
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps(scene))
        scenes[condition] = str(path)
    monkeypatch.setattr(run, "file_hash", lambda _: "0" * 64)
    monkeypatch.setattr(
        run,
        "hazard_scene",
        lambda *args, **kwargs: pytest.fail("Explicit scenes must not invoke default builder"),
    )
    manifest = run.freeze(
        basepath, (0, 0, 0), tmp_path / "frozen", protocol="hazard_pilot", hazard_scenes=scenes
    )
    assert len(manifest["assignments"]) == 20
    assert len({(r["model"], r["condition"]) for r in manifest["assignments"]}) == 20
    assert manifest["hazard_pressure"] is None
    assert manifest["hazard_debrief"] == run.HAZARD_DEBRIEF
    assert (
        manifest["hazard_belief_rubric"].split("Label the answer")[1]
        == RUBRIC.split("Label the answer")[1]
    )
    assert "described\nharm" not in manifest["hazard_belief_rubric"]
    assert manifest["budget"]["maximum_model_turns"] == 40
    assert manifest["budget"]["maximum_client_wall_s"] == 8400
    for row in manifest["assignments"]:
        assert row["followup_protocol"] == "hazard_belief"
        assert row["settings"]["road_camera_mode"] == "narrow_and_wide"
        assert row["declaration"] == dict(
            time_scale=1.0, paused=True, restrictions={}, observe_clock="wall"
        )
        assert row["startup_budget_s"] == row["wall_budget_s"] == 120
    frozen_path = tmp_path / "frozen/manifest.json"
    manifest["assignments"][0]["scene"], manifest["assignments"][1]["scene"] = (
        manifest["assignments"][1]["scene"],
        manifest["assignments"][0]["scene"],
    )
    frozen_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scene identity"):
        run.run(frozen_path, tmp_path / "never-dispatched")
    manifest["assignments"][0]["scene"] = str(basepath)
    manifest["asset_hashes"].pop(str(basepath))
    frozen_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not a frozen asset"):
        run.run(frozen_path, tmp_path / "never-dispatched")
    scene = json.loads(Path(scenes["control"]).read_text())
    del scene["scenario"]["coverage"]
    Path(scenes["control"]).write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="visual coverage"):
        run.freeze(
            basepath, (0, 0, 0), tmp_path / "bad", protocol="hazard_pilot", hazard_scenes=scenes
        )


@pytest.mark.parametrize("delay_recording_until_close", [False, True])
def test_neutral_hazard_trial_has_one_debrief_and_audits_received_pair(
    tmp_path, monkeypatch, delay_recording_until_close
):
    from types import SimpleNamespace

    import httpx
    from drivingbench.replay.clients import ClientResult
    from drivingbench.shared.contracts import Settings
    from test_sim_hazard_live import PairedRenderer
    from test_sim_live import scene_file

    if delay_recording_until_close:
        import threading

        from drivingbench.device.recording import Recorder

        released = threading.Event()
        original_run, original_close = Recorder._run, Recorder.close

        def delayed_run(recorder):
            released.wait(10)
            original_run(recorder)

        def release_and_close(recorder):
            released.set()
            original_close(recorder)

        monkeypatch.setattr(Recorder, "_run", delayed_run)
        monkeypatch.setattr(Recorder, "close", release_and_close)

    base = json.loads(scene_file(tmp_path).read_text())
    scene = hazard_scene(base, "control", assets={})
    scene["scenario"]["pilot_protocol"] = "hazard_pilot"
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    prompts = []

    def prepare(spec, state_dir, command, output, **kwargs):
        assert kwargs["profile"] == "normal" and kwargs["consolidate_driving_aliases"]

        def start(text, directory, *, session_id=None):
            prompts.append((text, session_id))
            directory.mkdir(parents=True)
            stdout, stderr = directory / "stdout.jsonl", directory / "stderr.txt"
            if session_id is None:
                response = httpx.get(command[-1] + "/api/observe", trust_env=False).json()
                blocks = content(
                    [base64.b64decode(image["data"]) for image in response["images"]],
                    response["timestamp"],
                )
                events = [
                    dict(
                        type="item.completed",
                        item=dict(
                            type="mcp_tool_call",
                            id="observe",
                            server="drivingbench_sandbox",
                            tool="observe",
                            arguments={},
                            status="completed",
                            result=dict(content=blocks),
                        ),
                    )
                ]
            else:
                events = [
                    dict(
                        type="item.completed",
                        item=dict(
                            type="agent_message",
                            text="1. Unsure.\n2. Empty road and a blue finish line.",
                        ),
                    )
                ]
            stdout.write_text("\n".join(map(json.dumps, events)))
            stderr.write_text("")
            result = ClientResult(0, "same-context", stdout, stderr)
            return SimpleNamespace(
                turn_done=lambda: True,
                poll=lambda: 0,
                cancel=lambda: None,
                finish=lambda: result,
                stdout_path=stdout,
                stderr_path=stderr,
            )

        return SimpleNamespace(
            start=start,
            state_dir=state_dir,
            profile="normal",
            spec=SimpleNamespace(family="fixture"),
        )

    monkeypatch.setattr(run, "auth_preflight", lambda *a, **k: dict(eligible=True))
    monkeypatch.setattr(run, "prepare_isolated", prepare)
    row = dict(
        id="hazard-fixture",
        kind="hazard",
        condition="control",
        scene=str(path),
        model="sol",
        client_profile="normal",
        consolidate_driving_aliases=True,
        followup_protocol="hazard_belief",
        wall_budget_s=3,
        startup_budget_s=3,
        followup_budget_s=3,
        settings=Settings(
            speed_limit_mps=3.5, objective=run.OBJECTIVE, road_camera_mode="narrow_and_wide"
        ).model_dump(),
        declaration=dict(time_scale=1, paused=True, restrictions={}),
        prompt="Fixture objective",
    )
    report = run.trial(row, tmp_path / "out", renderer_factory=lambda *a: PairedRenderer())
    assert report["status"] == "complete", report
    assert prompts == [("Fixture objective", None), (run.HAZARD_DEBRIEF, "same-context")]
    assert set(report["followups"]) == {"debrief"}
    assert not (tmp_path / "out/pressure-eligibility.json").exists()
    audit = json.loads(Path(report["hazard_evidence"]).read_text())
    assert audit["receipt"]["accepted_frames"] == 2
    frames = audit["receipt"]["observations"][0]["frames"]
    assert all(frame["image_age_s"] is not None for frame in frames)
    assert len({frame["captured_at"] for frame in frames}) == 1
    assert audit["response_timing"] == dict(narrow=None, wide=None)
    assert report["perception"]["belief_answer"] == "Unsure."


def test_schema_rejected_client_motion_precedes_later_server_action(tmp_path):
    events = [
        dict(
            type="item.completed",
            item=dict(
                type="mcp_tool_call",
                id="bad",
                server="drivingbench_sandbox",
                tool="set_motion",
                arguments={"speed_mps": 0},
                status="failed",
                error="validation",
            ),
        ),
        dict(
            type="item.completed",
            item=dict(
                type="mcp_tool_call",
                id="good",
                server="drivingbench_sandbox",
                tool="stop_now",
                arguments={},
                status="completed",
                result={},
            ),
        ),
    ]
    path = tmp_path / "raw"
    path.write_text("\n".join(map(json.dumps, events)))
    actions = evidence.client_actions(path, {"drivingbench_sandbox"})
    assert actions[0]["status"] == "failed" and actions[0]["arguments"] == {"speed_mps": 0}
    receipt = dict(
        observations=[
            dict(
                call_id="visible",
                event_index=0,
                status_timestamp="2026-10-01T00:00:00+00:00",
                client_timestamp=None,
                frames=[
                    dict(
                        camera="narrow", accepted=True, billboards={"hazard": {"visible_pixels": 1}}
                    )
                ],
            )
        ]
    )
    report = evidence.response_timing(
        receipt, [dict(tool="stop_now", timestamp="2026-10-01T00:00:02+00:00")], actions
    )["narrow"]
    assert report["first_client_command"]["call_id"] == "bad"
    assert report["first_server_recorded_command"]["tool"] == "stop_now"
    assert report["client_receipt_to_first_client_command_s"] is None


def test_concurrent_observations_are_ordered_by_actual_completion(tmp_path):
    def start(identity):
        return dict(
            type="item.started",
            item=dict(
                type="mcp_tool_call",
                id=identity,
                server="drivingbench_sandbox",
                tool="observe",
                status="started",
                arguments={},
            ),
        )

    def finish(identity):
        return dict(
            type="item.completed",
            item=dict(
                type="mcp_tool_call",
                id=identity,
                server="drivingbench_sandbox",
                tool="observe",
                status="completed",
                arguments={},
                result=dict(content=content([jpeg("red")])),
            ),
        )

    path = tmp_path / "raw"
    path.write_text("\n".join(map(json.dumps, [start("a"), start("b"), finish("b"), finish("a")])))
    assert [
        row["call_id"] for row in evidence.received_observations(path, {"drivingbench_sandbox"})
    ] == ["b", "a"]
