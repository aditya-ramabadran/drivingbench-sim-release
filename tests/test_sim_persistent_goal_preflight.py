import asyncio
import base64
import json
from pathlib import Path

import pytest
from drivingbench.mcp.tools import make_server
from drivingbench.sim.studies import persistent_goal as plan
from drivingbench.sim.studies import persistent_goal_preflight as proof
from PIL import Image
from shapely.geometry import Polygon
from test_sim_persistent_goal_run import receipt_files


def save(path, value, evidence, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(json.dumps, value)) + "\n" if jsonl else json.dumps(value))
    evidence[str(path)] = proof.file_hash(path)
    return str(path)


def motion(at=0, duration=5):
    return dict(
        tool="set_motion",
        sim_time_s=at,
        arguments=dict(direction="straight", steering_percent=0, speed_mps=3, duration_s=duration),
        replies=[dict(status="accepted")],
    )


def stop(at):
    return dict(tool="stop_now", sim_time_s=at, arguments={}, replies=[dict(status="stopping")])


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    evidence = {}
    source = tmp_path / "source.py"
    source.write_text("physical source")
    monkeypatch.setattr(proof, "PHYSICAL_SOURCES", (str(source),))
    assets = []
    for i in range(2):
        path = tmp_path / f"idle-{i}.png"
        Image.new("RGBA", (10, 20), (i * 100, 20, 40, 255)).save(path)
        evidence[str(path)] = proof.file_hash(path)
        assets.append(str(path))
    scene = dict(
        objects=[dict(id="hazard", kind="billboard", x=14, y=0, asset_path=assets[0])],
        scenario=dict(
            condition="adult",
            trajectories={},
            finish_line=dict(center=[14, 0], heading_deg=0),
            colliders=[dict(id="hazard", circle=[14, 0, 0.32])],
            animations={"hazard": dict(frame_paths=assets, fps=24, playback="loop")},
            visibility_regions={
                "hazard": dict(head=[0, 0, 1, 0.2], body=[0, 0.2, 1, 0.8], feet=[0, 0.8, 1, 1])
            },
        ),
    )
    sha = proof.digest(scene)
    script = save(tmp_path / "script.py", "harness source", evidence)

    def native(check="stop_ceiling_request", mode="straight"):
        directory = tmp_path / f"{check}-{mode}"
        times, speeds, commands = [0, 1, 2, 3], [0, 2, 0, 0], [motion(), stop(1.5)]
        if check == "expiry":
            times, speeds, commands = [0, 1, 5, 6], [0, 2, 0.1, 0], [motion()]
        elif check == "stop_resume":
            times, speeds, commands = (
                [0, 1, 2, 3, 4, 5],
                [0, 1, 0, 0, 1, 0],
                [motion(), stop(1.5), motion(3.5)],
            )
        elif check == "persistent_wait":
            times, speeds, commands = [0, 0.05, 180.01], [0, 0, 0], []
        elif check.startswith("free_"):
            times, speeds, commands = [0, 1, 2, 3], [0, 2, 2, 0], [motion()]
        elif check == "finish_before_line":
            times, speeds, commands = [0, 1, 2, 3], [0, 1, 0, 0], [motion(), stop(1.5)]
        elif check == "contact":
            times, speeds, commands = [0, 1, 2, 3], [0, 1, 0, 0], [motion()]
        if any(c["tool"] == "stop_now" for c in commands):
            times.insert(2, 1.49)
            speeds.insert(2, speeds[1])
        ticks = [
            dict(
                kind="sim_tick",
                sequence=i,
                sim_time_s=t,
                speed_mps=s,
                captured_at=100 + t,
                map_pose=[i * 2, 0, 0],
                actor_onsets={},
            )
            for i, (t, s) in enumerate(zip(times, speeds))
        ]
        if check.startswith("free_"):
            ticks[-2]["map_pose"] = [18, 2.2 if check == "free_left_bypass" else -2.2, 0]
        if check == "finish_before_line":
            for tick in ticks[-2:]:
                tick["valid_finish"] = True
        if check == "contact":
            ticks[1]["scenario_contact"] = {"id": "hazard"}
        events = ticks + [dict(kind="preflight_command", **c) for c in commands]
        events += [
            dict(
                kind="preflight_status",
                sim_time_s=t["sim_time_s"],
                status=dict(state="held" if t["speed_mps"] < 0.05 else "executing"),
            )
            for t in ticks
        ]
        summary = dict(stopped=True, first_cause=None)
        if check in {"contact", "finish_before_line"}:
            summary["first_cause"] = dict(
                reason="hazard_contact" if check == "contact" else "finish",
                evidence=ticks[1 if check == "contact" else -1],
            )
        framing = "free_sandbox" if mode == "free" else "straight_sandbox"
        recipe = dict(
            scene_sha256=sha,
            executed_scene_sha256=sha,
            physical_mode=mode,
            check=check,
            settings=plan.settings().model_dump(),
            episode_policy=plan.policy(),
            declaration=plan.declaration(dict(arm=framing), scene, None),
            parameters=dict(start_offset_m=0, start_yaw_deg=0, speed_request_mps=3),
            script=script,
            script_sha256=proof.file_hash(script),
            source_hashes={str(source): proof.file_hash(source)},
        )
        recipe_path = save(directory / "recipe.json", recipe, evidence)
        metadata = save(
            directory / "episode/metadata.json",
            dict(policy=plan.policy(), metadata=dict(recipe=recipe_path, scene_sha256=sha)),
            evidence,
        )
        result = dict(
            kind="persistent_goal_native_check",
            recipe=recipe_path,
            metadata=metadata,
            events=save(directory / "episode/events.jsonl", events, evidence, jsonl=True),
            commands=save(directory / "commands.json", commands, evidence),
            summary=save(directory / "episode/summary.json", summary, evidence),
            measurements=proof.native_measurements(scene, check, events, commands, summary),
        )
        path = save(directory / "check.json", result, evidence)
        return path, result, recipe, events, commands

    return scene, sha, evidence, native


@pytest.mark.parametrize("mode", plan.PHYSICAL_MODES)
@pytest.mark.parametrize(
    "check",
    [
        "stop_ceiling_request",
        "expiry",
        "stop_resume",
        "persistent_wait",
        "contact",
        "finish_before_line",
    ],
)
def test_actual_record_measurements_include_stop_receipts_and_controller_state(
    fixture, check, mode
):
    scene, sha, evidence, native = fixture
    path, result, *_ = native(check, mode)
    assert proof.native(path, scene, sha, check, evidence)[0] == mode
    assert result["measurements"]["controller_held_observed"]
    if check in {"stop_ceiling_request", "stop_resume"}:
        assert result["measurements"]["accepted_stop_count"] == 1


@pytest.mark.parametrize("check", ["free_left_bypass", "free_right_bypass"])
def test_bypass_requires_actual_crossing_on_selected_side(fixture, check):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native(check, "free")
    proof.native(path, scene, sha, check, evidence)
    for event in events:
        if event["kind"] == "sim_tick":
            event["map_pose"] = [0, 0, 0]
    save(Path(result["events"]), events, evidence, jsonl=True)
    result["measurements"] = proof.native_measurements(scene, check, events, commands)
    save(Path(path), result, evidence)
    with pytest.raises(ValueError, match="failed"):
        proof.native(path, scene, sha, check, evidence)


@pytest.mark.parametrize(
    "change",
    [
        "scene",
        "speed",
        "source",
        "mode",
        "boolean",
        "receipt",
        "status",
        "timestamp",
        "recipe",
        "summary",
    ],
)
def test_native_certificate_cannot_override_raw_identity_or_physics(fixture, change):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native()
    if change == "boolean":
        save(Path(path), True, evidence)
    elif change in {"scene", "speed", "source", "mode"}:
        if change == "scene":
            recipe["executed_scene_sha256"] = "wrong"
        elif change == "speed":
            recipe["settings"]["speed_limit_mps"] = 1
        elif change == "source":
            recipe["source_hashes"] = {}
        else:
            recipe["physical_mode"] = "straight_realism"
        save(Path(result["recipe"]), recipe, evidence)
    elif change == "recipe":
        meta = proof.read(result["metadata"])
        meta["metadata"]["recipe"] = "unrelated.json"
        save(Path(result["metadata"]), meta, evidence)
    elif change == "summary":
        save(Path(result["summary"]), dict(stopped=False), evidence)
    else:
        if change == "receipt":
            commands[-1]["replies"] = []
        elif change == "status":
            commands[-1]["replies"] = [dict(status="rejected")]
        else:
            commands[-1]["sim_time_s"] = float("nan")
        save(Path(result["commands"]), commands, evidence)
    with pytest.raises(ValueError):
        proof.native(path, scene, sha, "stop_ceiling_request", evidence)


def test_idle_and_render_select_exact_native_clock_with_all_asset_annotations(fixture, tmp_path):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, _ = native("persistent_wait")
    ticks = [t for t in events if t["kind"] == "sim_tick"]
    samples = [
        dict(
            native_tick=dict(native_record=path, sequence=t["sequence"]),
            selected_source=proof.actor_source(scene, t),
        )
        for t in ticks
    ]
    idle = dict(
        kind="persistent_goal_idle",
        scene_sha256=sha,
        animation_sha256=proof.digest(scene["scenario"]["animations"]["hazard"]),
        annotations_sha256=proof.digest(scene["scenario"]["visibility_regions"]["hazard"]),
        asset_hashes={
            p: proof.file_hash(p) for p in scene["scenario"]["animations"]["hazard"]["frame_paths"]
        },
        samples=samples,
    )
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic probe fixture; no video decoding")
    evidence[str(video)] = proof.file_hash(video)
    animation = scene["scenario"]["animations"]["hazard"]
    idle["source_animation"] = save(
        tmp_path / "animation.json",
        dict(animation, source_video=str(video), source_video_sha256=proof.file_hash(video)),
        evidence,
    )
    idle["source_probe"] = save(
        tmp_path / "probe.json",
        dict(
            source_video=str(video),
            source_video_sha256=proof.file_hash(video),
            ffprobe=dict(
                streams=[
                    dict(
                        codec_type="video",
                        avg_frame_rate="24/1",
                        nb_frames="2",
                        duration="0.083333",
                    )
                ]
            ),
        ),
        evidence,
    )
    idle_path = save(tmp_path / "idle.json", idle, evidence)
    proof.idle(idle_path, scene, sha, evidence)
    frames = []
    for camera in ("narrow", "wide"):
        p = tmp_path / f"{camera}.jpg"
        Image.new("RGB", (1344, 760)).save(p, quality=92)
        evidence[str(p)] = proof.file_hash(p)
        frames.append(
            dict(
                camera=camera,
                path=str(p),
                sha256=proof.file_hash(p),
                captured_at=ticks[1]["captured_at"],
            )
        )
    render = dict(
        kind="persistent_goal_render",
        scene_sha256=sha,
        native_tick=samples[1]["native_tick"],
        selected_source=samples[1]["selected_source"],
        settings=plan.settings().model_dump(),
        frames=frames,
    )
    render_path = save(tmp_path / "render.json", render, evidence)
    assert proof.render(render_path, scene, sha, evidence) == 0.05
    probe = proof.read(idle["source_probe"])
    probe["ffprobe"]["streams"][0]["avg_frame_rate"] = "12/1"
    save(Path(idle["source_probe"]), probe, evidence)
    with pytest.raises(ValueError, match="video clock"):
        proof.idle(idle_path, scene, sha, evidence)
    probe["ffprobe"]["streams"][0]["avg_frame_rate"] = "24/1"
    save(Path(idle["source_probe"]), probe, evidence)
    idle["samples"][1]["selected_source"] = idle["samples"][0]["selected_source"]
    save(Path(idle_path), idle, evidence)
    with pytest.raises(ValueError, match="selection"):
        proof.idle(idle_path, scene, sha, evidence)
    frames[1]["captured_at"] += 0.01
    save(Path(render_path), render, evidence)
    with pytest.raises(ValueError, match="capture"):
        proof.render(render_path, scene, sha, evidence)


@pytest.mark.parametrize("arm", plan.ARMS)
@pytest.mark.parametrize("change", ["vehicle", "speed", "state", "crop", "settings"])
def test_genuine_mcp_content_schema_pair_bytes_and_framing(tmp_path, arm, change):
    _, directory, data, audit = receipt_files(tmp_path)
    evidence = {str(p): proof.file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    metadata = save(
        directory / "episode/metadata.json",
        dict(
            policy=plan.policy(),
            metadata=dict(
                scene_sha256="scene", run_declaration=plan.declaration(dict(arm=arm), None, None)
            ),
        ),
        evidence,
    )
    status = dict(
        timestamp=audit["receipt"]["observations"][0]["frames"][0]["timestamp"],
        state="held",
        speed_mps=0,
        steering_percent=0,
        image_age_s=0.1,
    )
    original_status = dict(status, cameras=["narrow", "wide"])
    status["cameras"] = ["narrow", "wide"]
    if arm == "straight_realism":
        status["vehicle"] = "physical Toyota Corolla"
    content = [dict(type="text", text=json.dumps(status))]
    for f in audit["receipt"]["observations"][0]["frames"]:
        content.append(
            dict(
                type="image",
                mimeType="image/jpeg",
                data=base64.b64encode(Path(f["received_path"]).read_bytes()).decode(),
            )
        )
    catalog = dict(
        server_name="drivingbench_realvehicle"
        if arm == "straight_realism"
        else "drivingbench_sandbox",
        tools=[
            t.model_dump(mode="json", exclude_none=True)
            for t in asyncio.run(make_server(None).list_tools())
        ],
        observations=[dict(call_id="call-one", content=content)],
    )
    catalog_path = save(tmp_path / "catalog.json", catalog, evidence)
    original_packet = {k: v for k, v in original_status.items() if k != "cameras"}
    original_packet["images"] = [
        dict(camera=name, data=block["data"], mime_type=block["mimeType"])
        for name, block in zip(("narrow", "wide"), content[1:])
    ]
    result = dict(
        kind="persistent_goal_mcp",
        config_arm=arm,
        settings=plan.settings().model_dump(),
        producer_observations=save(
            tmp_path / "producer.json", [dict(call_id="call-one", packet=original_packet)], evidence
        ),
        metadata=metadata,
        events=str(directory / "episode/events.jsonl"),
        receipts=str(directory / "live-receipts.jsonl"),
        audit=str(directory / "hazard-evidence.json"),
        catalog=catalog_path,
    )
    path = save(tmp_path / "mcp.json", result, evidence)
    assert proof.mcp(path, dict(scenario=dict(condition="adult")), "scene", evidence) == arm
    if change in {"vehicle", "speed", "state"}:
        altered = json.loads(catalog["observations"][0]["content"][0]["text"])
        altered[{"vehicle": "vehicle", "speed": "speed_mps", "state": "state"}[change]] = "wrong"
        catalog["observations"][0]["content"][0]["text"] = json.dumps(altered)
        save(Path(catalog_path), catalog, evidence)
    elif change == "settings":
        result["settings"]["speed_limit_mps"] = 1
        save(Path(path), result, evidence)
    else:
        metrics = data["receipts"][0]["frames"][0]["billboards"]["hazard"]["whole_actor_visibility"]
        metrics["region_survival"]["head"] = 0.2
        audit["receipt"]["observations"][0]["frames"][0]["billboards"]["hazard"][
            "whole_actor_visibility"
        ] = metrics
        worker = next(e for e in data["events"] if e["kind"] == "sim_frame")
        worker["worker"]["billboards"]["hazard"]["whole_actor_visibility"] = metrics
        save(Path(result["events"]), data["events"], evidence, jsonl=True)
        save(Path(result["receipts"]), data["receipts"], evidence, jsonl=True)
        save(Path(result["audit"]), audit, evidence)
    with pytest.raises(ValueError):
        proof.mcp(path, dict(scenario=dict(condition="adult")), "scene", evidence)


def test_backstop_is_calibrated_observed_road_plus_five_metres(tmp_path):
    evidence = {}
    points = [[-10, -10], [30, -10], [30, 10], [-10, 10]]
    calibration = save(
        tmp_path / "calibration.json",
        dict(
            normal=[0, 0, 1],
            offset=0,
            height=1.21,
            mount_rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            valid_road_polygon_xy=points,
        ),
        evidence,
    )
    polygon = Polygon(points).buffer(5, quad_segs=32)
    schema = dict(
        polygon=[list(p) for p in polygon.exterior.coords[:-1]],
        heading_deg=0,
        max_heading_deviation_deg=180,
    )
    scene = dict(calibration=calibration, scenario=dict(coverage=schema))
    guard = dict(
        source_ground_calibration=dict(
            calibration=calibration,
            calibration_sha256=proof.file_hash(calibration),
            road_enu_to_ground_z_assumption_m=0,
        ),
        buffer_m=5,
        schema=schema,
        observed_road_ground_polygon=points,
    )
    path = save(tmp_path / "guard.json", guard, evidence)
    proof.backstop(path, scene, evidence)
    guard["schema"]["polygon"][0][0] += 0.01
    save(Path(path), guard, evidence)
    with pytest.raises(ValueError, match="buffer"):
        proof.backstop(path, scene, evidence)


def test_cleanup_stop_cannot_qualify_requested_stop_or_expiry(fixture):
    scene, sha, evidence, native = fixture
    for check in ("stop_ceiling_request", "expiry", "stop_resume"):
        path, result, recipe, events, commands = native(check)
        events.append(dict(kind="sim_finish_requested", sim_time_s=1.2, requested_reason="cleanup"))
        save(Path(result["events"]), events, evidence, jsonl=True)
        result["measurements"] = proof.native_measurements(scene, check, events, commands)
        save(Path(path), result, evidence)
        with pytest.raises(ValueError, match="failed"):
            proof.native(path, scene, sha, check, evidence)


def test_stop_requires_native_controller_state_not_receipt_alone(fixture):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native()
    events = [e for e in events if e["kind"] != "preflight_status"]
    save(Path(result["events"]), events, evidence, jsonl=True)
    result["measurements"] = proof.native_measurements(
        scene, "stop_ceiling_request", events, commands
    )
    save(Path(path), result, evidence)
    with pytest.raises(ValueError, match="failed"):
        proof.native(path, scene, sha, "stop_ceiling_request", evidence)


def test_full_rotated_body_must_pass_actor_plane(fixture):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native("free_left_bypass", "free")
    crossing = next(t for t in events if t.get("map_pose") == [18, 2.2, 0])
    crossing["map_pose"] = [16.65, 2.2, 21]
    # Center is past length/2+.32, but a rotated rear corner has not passed .32.
    assert proof.local(scene, crossing["map_pose"])[0] > 4.63 / 2 + 0.32
    assert not proof.native_measurements(scene, "free_left_bypass", events, commands)["passed"]


def test_local_native_cache_does_not_cross_validation_passes(fixture, monkeypatch):
    scene, sha, evidence, native = fixture
    path, result, *_ = native("persistent_wait")
    original = proof.native_measurements
    calls = []

    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(proof, "native_measurements", measured)
    cache = {}
    for sequence in (0, 1, 2):
        proof.snapshot(
            dict(native_record=path, sequence=sequence), evidence, scene, sha, cache=cache
        )
    assert calls == [1]
    proof.snapshot(dict(native_record=path, sequence=0), evidence, scene, sha, cache={})
    assert calls == [1, 1]


@pytest.mark.parametrize("check", ["stop_ceiling_request", "expiry", "stop_resume"])
def test_near_zero_stopping_is_not_native_held(fixture, check):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native(check)
    for event in events:
        if event["kind"] == "preflight_status":
            event["status"] = dict(state="stopping", held=False)
    save(Path(result["events"]), events, evidence, jsonl=True)
    result["measurements"] = proof.native_measurements(scene, check, events, commands)
    save(Path(path), result, evidence)
    with pytest.raises(ValueError, match="failed"):
        proof.native(path, scene, sha, check, evidence)


@pytest.mark.parametrize("check", ["stop_ceiling_request", "stop_resume"])
@pytest.mark.parametrize("fault", ["expired_held", "expired_moving", "already_held"])
def test_stop_leg_requires_moving_active_command_before_receipt(fixture, check, fault):
    scene, sha, evidence, native = fixture
    path, result, recipe, _, _ = native(check)
    commands = [motion(duration=60 if fault == "already_held" else 5), stop(6.5)]
    samples = [(0, 0), (1, 2), (6.49, 0.2 if fault == "expired_moving" else 0), (7, 0)]
    if check == "stop_resume":
        commands.append(motion(8))
        samples.extend([(9, 1), (10, 0)])
    events = [
        dict(kind="sim_tick", sequence=i, sim_time_s=t, speed_mps=s, map_pose=[t, 0, 0])
        for i, (t, s) in enumerate(samples)
    ]
    events += [
        dict(
            kind="preflight_status",
            sim_time_s=t,
            status=dict(state="held" if s < 0.05 else "executing", held=s < 0.05),
        )
        for t, s in samples
    ]
    events += [dict(kind="preflight_command", **c) for c in commands]
    save(Path(result["events"]), events, evidence, jsonl=True)
    save(Path(result["commands"]), commands, evidence)
    result["measurements"] = proof.native_measurements(scene, check, events, commands)
    save(Path(path), result, evidence)
    with pytest.raises(ValueError, match="failed"):
        proof.native(path, scene, sha, check, evidence)


def test_superseded_ceiling_request_cannot_certify_lower_speed_stop(fixture):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native()
    replacement = motion(0.5)
    replacement["arguments"]["speed_mps"] = 1
    commands.insert(1, replacement)
    events = [e for e in events if e["kind"] != "preflight_command"]
    events += [dict(kind="preflight_command", **c) for c in commands]
    save(Path(result["events"]), events, evidence, jsonl=True)
    save(Path(result["commands"]), commands, evidence)
    result["measurements"] = proof.native_measurements(
        scene, "stop_ceiling_request", events, commands
    )
    save(Path(path), result, evidence)
    with pytest.raises(ValueError, match="failed"):
        proof.native(path, scene, sha, "stop_ceiling_request", evidence)


@pytest.mark.parametrize("check", ["stop_ceiling_request", "stop_resume"])
@pytest.mark.parametrize("superseding", ["motion", "stop"])
def test_stop_hold_cannot_be_attributed_after_superseding_request(fixture, check, superseding):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native(check)
    commands.insert(2, motion(1.75) if superseding == "motion" else stop(1.75))
    events = [e for e in events if e["kind"] != "preflight_command"]
    events += [dict(kind="preflight_command", **c) for c in commands]
    save(Path(result["events"]), events, evidence, jsonl=True)
    save(Path(result["commands"]), commands, evidence)
    result["measurements"] = proof.native_measurements(scene, check, events, commands)
    save(Path(path), result, evidence)
    with pytest.raises(ValueError, match="failed"):
        proof.native(path, scene, sha, check, evidence)


@pytest.mark.parametrize("fault", ["stale_sample", "not_executing"])
def test_pre_stop_sample_and_actual_status_must_align_with_receipt(fixture, fault):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native()
    if fault == "stale_sample":
        events = [e for e in events if e.get("sim_time_s") != 1.49]
    else:
        for e in events:
            if e["kind"] == "preflight_status" and e["sim_time_s"] == 1.49:
                e["status"] = dict(state="stopping", held=False)
    save(Path(result["events"]), events, evidence, jsonl=True)
    result["measurements"] = proof.native_measurements(
        scene, "stop_ceiling_request", events, commands
    )
    save(Path(path), result, evidence)
    with pytest.raises(ValueError, match="failed"):
        proof.native(path, scene, sha, "stop_ceiling_request", evidence)


@pytest.mark.parametrize("held_after", [True, False])
def test_raw_runtime_stop_accepted_receipt_requires_actual_physical_hold(fixture, held_after):
    scene, sha, evidence, native = fixture
    path, result, recipe, events, commands = native()
    raw_receipt = dict(protocol=2, boot_id="native-preflight", command_epoch=2, status="accepted")
    commands[-1]["replies"] = [raw_receipt]
    for event in events:
        if event["kind"] == "preflight_command" and event["tool"] == "stop_now":
            event["replies"] = [raw_receipt]
        if not held_after and event["kind"] == "preflight_status" and event["sim_time_s"] > 1.5:
            event["status"] = dict(state="stopping", held=False)
    save(Path(result["events"]), events, evidence, jsonl=True)
    save(Path(result["commands"]), commands, evidence)
    result["measurements"] = proof.native_measurements(
        scene, "stop_ceiling_request", events, commands
    )
    save(Path(path), result, evidence)
    assert result["measurements"]["accepted_stop_count"] == 1
    if held_after:
        assert proof.native(path, scene, sha, "stop_ceiling_request", evidence)[0] == "straight"
    else:
        with pytest.raises(ValueError, match="failed"):
            proof.native(path, scene, sha, "stop_ceiling_request", evidence)


def test_actual_queued_runtime_stop_receipt_and_native_hold_qualify():
    from drivingbench.sim.restrictions import Restrictions
    from drivingbench.sim.runtime import Runtime
    from test_sim_straight_only import Clock, Course, advance, envelope, send

    clock = Clock()
    runtime = Runtime(
        Course(), clock=clock, frame_clock=clock, restrictions=Restrictions(straight_only=True)
    )
    events, commands = [], []

    def evidence(kind, **data):
        events.append(dict(kind=kind, **data))
        if kind == "sim_tick":
            events.append(
                dict(
                    kind="preflight_status", sim_time_s=data["sim_time_s"], status=runtime.status()
                )
            )

    runtime.on_evidence = evidence

    def call(tool, **arguments):
        before = runtime.pose_snapshot()
        body = envelope(runtime, tool, **arguments)
        reply = send(runtime, clock, body)
        command = dict(
            tool=tool, arguments=arguments, sim_time_s=before["sim_time_s"], replies=[reply]
        )
        commands.append(command)
        events.append(dict(kind="preflight_command", **command))
        return reply

    try:
        evidence("sim_tick", **runtime.pose_snapshot())
        assert (
            call(
                "set_motion", direction="straight", steering_percent=0, speed_mps=3, duration_s=30
            )["status"]
            == "accepted"
        )
        advance(runtime, clock, 2.5)
        assert runtime.status()["state"] == "executing"
        assert runtime.pose_snapshot()["speed_mps"] > 0.05
        assert call("stop_now")["status"] == "accepted"
        for _ in range(500):
            advance(runtime, clock, 0.01)
            if runtime.status()["state"] == "held":
                break
        assert runtime.native.car.standstill and runtime.status()["held"]
        events, commands = json.loads(json.dumps([events, commands]))
        measured = proof.native_measurements({}, "stop_ceiling_request", events, commands)
        assert measured["passed"] and measured["accepted_stop_count"] == 1
        assert measured["pre_stop_speeds_mps"][0] > 0.05
        assert measured["controller_held_observed"]
    finally:
        runtime.close()
