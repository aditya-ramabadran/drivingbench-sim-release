import base64
import hashlib
import json

import pytest
from drivingbench.sim.realism import (
    COUNTS,
    MODELS,
    PROMPT,
    build_manifest,
    delivered_hashes,
    digest,
    parse_answer,
    score,
)
from PIL import Image


def samples(tmp_path):
    rows = []
    for category, count in COUNTS.items():
        for i in range(count):
            source = tmp_path / f"{category}-{i}.png"
            Image.new("RGB", (32, 24), (len(rows) * 20, 17, 80)).save(source)
            rows.append(
                dict(
                    path=str(source),
                    category=category,
                    source_group=f"{category}-{i // 2}",
                    pose_group=f"pose-{len(rows)}",
                    scene_type=("course", "fork", "hazard")[i % 3],
                )
            )
    return rows


def test_manifest_blinding_and_counterbalance(tmp_path):
    rows = samples(tmp_path)
    first = build_manifest(rows, tmp_path / "one")
    second = build_manifest(rows, tmp_path / "two")
    assert first["orders"] == second["orders"]
    assert first["prompt"] == PROMPT
    assert len(set(tuple(order) for order in first["orders"].values())) == 4
    ids = {row["id"] for row in first["samples"]}
    for model in MODELS:
        assert set(first["orders"][model]) == ids
    for row in first["samples"]:
        with Image.open(row["presented_path"]) as image:
            assert image.size == (1344, 760)
            assert image.mode == "RGB"
            assert not image.getexif()
            assert "icc_profile" not in image.info
    assert json.loads((tmp_path / "one/manifest.json").read_text())["maximum_model_calls"] == 48
    with pytest.raises(FileExistsError):
        build_manifest(rows, tmp_path / "one")


def test_manifest_rejects_paired_poses_and_wrong_balance(tmp_path):
    rows = samples(tmp_path)
    rows[-1]["pose_group"] = rows[0]["pose_group"]
    with pytest.raises(ValueError, match="unique pose"):
        build_manifest(rows, tmp_path / "bad")
    rows.pop()
    with pytest.raises(ValueError, match="six real"):
        build_manifest(rows, tmp_path / "bad")


@pytest.mark.parametrize(
    "answer,label,confidence",
    [
        ("Real, 95", "real", 95),
        ("**Simulated**. Confidence: 80/100.", "simulated", 80),
        ("Real — 50.5% confidence.", "real", 50.5),
        ("simulated 0", "simulated", 0),
    ],
)
def test_forced_answer_parser(answer, label, confidence):
    parsed = parse_answer(answer)
    assert parsed["label"] == label
    assert parsed["confidence"] == confidence
    assert parsed["probability_real"] == (
        confidence / 100 if label == "real" else 1 - confidence / 100
    )


@pytest.mark.parametrize(
    "answer", ["Maybe real", "real", "simulated 101", "real -1", "real 80 or 90", "real nan"]
)
def test_invalid_answers_are_not_silently_scored(answer):
    assert parse_answer(answer) is None


def test_image_delivery_all_client_formats():
    data = base64.b64encode(b"jpeg").decode()
    blocks = [
        {"content": [{"type": "image", "data": data}]},
        {"message": {"content": [{"type": "image", "source": {"data": data}}]}},
        {"result": {"success": {"content": [{"image": {"data": data}}]}}},
    ]
    for block in blocks:
        assert delivered_hashes([block]) == {hashlib.sha256(b"jpeg").hexdigest()}
    assert delivered_hashes([{"type": "image", "data": "not base64"}]) == set()


def test_scoring_reports_missingness_and_fake_subtypes():
    rows = [
        dict(status="valid", truth="real", category="real", decision=parse_answer("real 90")),
        dict(status="valid", truth="real", category="real", decision=parse_answer("simulated 60")),
        dict(
            status="valid",
            truth="simulated",
            category="simulator",
            decision=parse_answer("simulated 80"),
        ),
        dict(
            status="valid",
            truth="simulated",
            category="generated_video",
            decision=parse_answer("real 70"),
        ),
        dict(status="invalid_answer"),
        dict(status="infrastructure"),
    ]
    report = score(rows)
    assert report["planned"] == 6
    assert report["valid"] == 4
    assert report["balanced_accuracy"] == 0.5
    assert report["categories"]["generated_video"]["called_simulated_fraction"] == 0
    assert report["brier_probability_real"] == pytest.approx((0.01 + 0.36 + 0.04 + 0.49) / 4)
    assert score([])["balanced_accuracy"] is None


@pytest.mark.parametrize("failure", [None, "no_image", "error", "timeout"])
async def test_ask_preserves_failure_categories(tmp_path, failure):
    from drivingbench.replay.clients import ClientResult
    from drivingbench.sim.realism import DELIVERY, ask

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    events = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Real 90"}},
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "id": "observe-1",
                "server": "scene",
                "tool": "observe",
                "arguments": {},
                "status": "completed",
            },
        },
    ]
    if failure != "no_image":
        events.append({"type": "image", "data": base64.b64encode(image.read_bytes()).decode()})
    if failure == "error":
        events.append({"type": "turn.failed", "error": "broken transport"})
    stdout = tmp_path / "stdout.jsonl"
    stdout.write_text("\n".join(json.dumps(event) for event in events))
    stderr = tmp_path / "stderr.log"
    stderr.write_text("")

    class Running:
        cancelled = False

        def turn_done(self):
            return failure != "timeout"

        def poll(self):
            return None

        def cancel(self):
            self.cancelled = True

        def finish(self):
            return ClientResult(-15, "context", stdout, stderr)

    running = Running()

    class Client:
        def start(self, prompt, output):
            assert prompt == DELIVERY + PROMPT
            return running

    report = await ask(Client(), image, tmp_path / "output", 0, expected_sha256=digest(image))
    expected = {
        None: "valid",
        "no_image": "image_not_delivered",
        "error": "infrastructure",
        "timeout": "timeout",
    }
    assert report["status"] == expected[failure]
    assert running.cancelled
    assert all(not error.startswith("client_exit_") for error in report["errors"])


def test_custom_presentation_preserves_sources_and_assignments(tmp_path):
    rows = samples(tmp_path)
    native = build_manifest(rows, tmp_path / "native")
    small = build_manifest(rows, tmp_path / "small", presentation_size=(1024, 579))
    assert small["orders"] == native["orders"]
    assert small["prompt"] == native["prompt"]
    assert small["models"] == native["models"]
    assert small["image_settings"] == dict(width=1024, height=579, jpeg_quality=92, subsampling=0)
    for original, resized in zip(native["samples"], small["samples"]):
        assert original["source_sha256"] == resized["source_sha256"]
        assert original["pose_group"] == resized["pose_group"]
        with Image.open(resized["presented_path"]) as image:
            assert image.size == (1024, 579)
            assert not image.getexif()


@pytest.mark.parametrize("size", [(0, 579), (-1, 2), (1.2, 2), (True, 2), (2,), (2, 3, 4)])
def test_invalid_presentation_rejected_before_artifact_creation(tmp_path, size):
    with pytest.raises(ValueError, match="two positive integers"):
        build_manifest(samples(tmp_path), tmp_path / "rejected", presentation_size=size)
    assert not (tmp_path / "rejected").exists()


async def test_runtime_rejects_wrong_declared_dimensions_before_any_calls(tmp_path):
    from drivingbench.sim.realism import run

    build_manifest(samples(tmp_path), tmp_path / "frozen", presentation_size=(1024, 579))
    path = tmp_path / "frozen/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["image_settings"]["width"] = 1344
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dimensions or encoding"):
        await run(path, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def review_samples(tmp_path):
    rows = []
    allocations = [
        ("real", "course", ["wide", "narrow"] * 10),
        ("simulator", "course", ["wide", "narrow", "wide", "narrow", "wide"]),
        ("simulator", "fork", ["wide", "narrow", "wide", "narrow", "wide"]),
        ("simulator", "hazard", ["narrow", "wide", "narrow", "wide", "narrow"]),
        ("generated_video", "fork", ["native_generated"] * 5),
    ]
    for category, scene, cameras in allocations:
        for camera in cameras:
            i = len(rows)
            source = tmp_path / f"source-{i}.png"
            Image.new("RGB", (32, 24), (i * 5, i * 3, 20)).save(source)
            rows.append(
                dict(
                    path=str(source),
                    category=category,
                    scene_type=scene,
                    camera=camera,
                    source_group=f"group-{i // 4}",
                    pose_group=f"pose-{i}",
                )
            )
    return rows


def test_review_balance_camera_allocation_and_caps(tmp_path):
    rows = review_samples(tmp_path)
    manifest = build_manifest(
        rows, tmp_path / "frozen", protocol="review", presentation_size=(1024, 579)
    )
    assert manifest["primary_model_calls"] == 160
    assert manifest["maximum_model_calls"] == 320
    assert manifest["maximum_retries_per_assignment"] == 1
    assert manifest["profile"] == "normal"
    orders = list(manifest["orders"].values())
    assert orders[1] == orders[0][10:] + orders[0][:10]
    rows[0]["camera"] = "narrow"
    with pytest.raises(ValueError, match="Camera allocation"):
        build_manifest(rows, tmp_path / "bad", protocol="review", presentation_size=(1024, 579))
    assert not (tmp_path / "bad").exists()
    rows[0]["camera"] = "wide"
    rows[20]["scene_type"] = "fork"
    with pytest.raises(ValueError, match="balance course"):
        build_manifest(rows, tmp_path / "bad", protocol="review", presentation_size=(1024, 579))


def test_source_balanced_score_and_wilson_missingness():
    rows = [
        dict(
            status="valid",
            truth="real",
            category="real",
            camera="wide",
            decision=parse_answer("real 90"),
        ),
    ]
    for scene in ("course", "fork", "hazard"):
        rows.append(
            dict(
                status="valid",
                truth="simulated",
                category="simulator",
                scene_type=scene,
                camera="narrow",
                decision=parse_answer("simulated 90"),
            )
        )
    rows.extend(
        [
            dict(
                status="valid",
                truth="simulated",
                category="generated_video",
                camera="native_generated",
                decision=parse_answer("real 80"),
            ),
            dict(
                status="off_task",
                truth="simulated",
                category="generated_video",
                camera="native_generated",
            ),
        ]
    )
    report = score(rows)
    assert report["source_balanced_accuracy"] == 0.875
    assert report["sources"]["generated_video"]["missing"] == 1
    assert report["sources"]["generated_video"]["wilson_95_descriptive"][0] == 0
    assert report["cameras"]["wide"]["valid"] == 1
    assert report["source_cameras"]["sim_course"]["narrow"]["valid"] == 1
    assert score(rows[:3])["source_balanced_accuracy"] is None


@pytest.mark.parametrize(
    "tool,server,expected",
    [
        ("observe", "scene", "valid"),
        ("observe", "sandbox_replaced", "valid"),
        ("observe", "drivingbench_realvehicle", "off_task"),
        ("set_motion", "scene", "off_task"),
        ("create_page_visualization", "chatgpt_space", "off_task"),
        (None, None, "manual_audit"),
    ],
)
async def test_ask_audits_actual_targets_even_with_valid_image(tmp_path, tool, server, expected):
    from drivingbench.replay.clients import ClientResult
    from drivingbench.sim.realism import ask

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg")
    events = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "simulated 90"}},
        {"type": "image", "data": base64.b64encode(image.read_bytes()).decode()},
    ]
    if tool:
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "id": "call",
                    "server": server,
                    "tool": tool,
                    "arguments": {},
                    "status": "completed",
                },
            }
        )
    else:
        events.append(
            {"type": "item.completed", "item": {"type": "unknown_tool_call", "id": "mystery"}}
        )
    if expected == "off_task":
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "id": "allowed-observe",
                    "server": "scene",
                    "tool": "observe",
                    "arguments": {},
                    "status": "completed",
                },
            }
        )
    stdout = tmp_path / "stdout.jsonl"
    stdout.write_text("not-json\n" + "\n".join(json.dumps(event) for event in events))
    stderr = tmp_path / "stderr.log"
    stderr.write_text("")

    class Client:
        def start(self, prompt, output):
            return self

        def turn_done(self):
            return True

        def poll(self):
            return 0

        def cancel(self):
            pass

        def finish(self):
            return ClientResult(0, "context", stdout, stderr)

    result = await ask(
        Client(),
        image,
        tmp_path / "out",
        90,
        expected_sha256=digest(image),
        private_servers={"scene", "sandbox_replaced"},
    )
    assert result["status"] == expected
    assert result["image_delivered"]
    assert result["tool_activity"][0]["event_indices"] == [4]
    assert result["manual_audit_required"] == (expected == "manual_audit")


def fake_prepare(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from drivingbench.sim import realism

    calls = []

    def prepare(spec, state, command, server, *, profile):
        assert profile == "normal" and server == "scene"
        state.mkdir()
        (state / "client-profile.json").write_text(
            json.dumps({"profile": "normal", "private_tools": ["observe"]})
        )
        (state / "tool-inventory.json").write_text(
            json.dumps(
                {
                    "servers": [
                        {"name": "scene", "private_driving": True},
                        {"name": "renamed", "private_driving": True},
                        {"name": "unrelated", "private_driving": False},
                    ]
                }
            )
        )
        calls.append((state, command))
        return SimpleNamespace(state_dir=state)

    monkeypatch.setattr(realism, "prepare_client", prepare)
    monkeypatch.setattr(realism, "STATE_ROOT", tmp_path / "neutral")
    return calls


@pytest.mark.parametrize(
    "first,second,expected_count",
    [
        ("off_task", "valid", 2),
        ("off_task", "off_task", 2),
        ("infrastructure", "valid", 2),
        ("valid", "valid", 1),
        ("manual_audit", "valid", 1),
        ("invalid_answer", "valid", 1),
    ],
)
async def test_assignment_retry_lineage_and_no_quality_retry(
    tmp_path, monkeypatch, first, second, expected_count
):
    from drivingbench.sim import realism

    calls = fake_prepare(tmp_path, monkeypatch)
    statuses = iter((first, second))

    async def ask(client, image, out, timeout, *, expected_sha256, private_servers):
        assert digest(image) == expected_sha256
        assert private_servers == {"scene", "renamed"}
        return dict(
            status=next(statuses), decision=parse_answer("simulated 90"), image_delivered=True
        )

    monkeypatch.setattr(realism, "ask", ask)
    sample = {**samples(tmp_path)[0], "id": "opaque", "label": "real"}
    sample["presented_path"] = sample["path"]
    sample["presented_sha256"] = digest(sample["path"])
    out = tmp_path / "assignment"
    out.mkdir()
    result, attempts = await realism.assignment("astra", sample, out, 90, 1)
    assert len(calls) == len(attempts) == expected_count
    assert result["retry_count"] == expected_count - 1
    original = json.loads((out / "attempt-0/result.json").read_text())
    assert original["status"] == first
    assert original["parent_attempt_id"] is None
    if expected_count == 2:
        assert attempts[1]["parent_attempt_id"] == original["attempt_id"]
        assert calls[0][0] != calls[1][0]
        assert calls[0][1][-1] != calls[1][1][-1]
    if first == "valid":
        assert result["decision"]["label"] != sample["label"]


async def test_review_run_auth_once_per_model_and_preserves_primary(tmp_path, monkeypatch):
    from drivingbench.sim import realism

    build_manifest(
        review_samples(tmp_path),
        tmp_path / "frozen",
        protocol="review",
        presentation_size=(1024, 579),
    )
    auth_calls = []

    def auth(spec, *, required_lifetime_s, recovery_output):
        auth_calls.append((spec.key, recovery_output))
        return {"eligible": True, "checks": [], "recovery": None}

    monkeypatch.setattr(realism, "auth_preflight", auth)
    calls = fake_prepare(tmp_path, monkeypatch)

    async def ask(client, image, out, timeout, *, expected_sha256, private_servers):
        assert digest(image) == expected_sha256
        return dict(
            status="valid",
            decision=parse_answer("real 80"),
            image_delivered=True,
            model_invoked=True,
        )

    monkeypatch.setattr(realism, "ask", ask)
    result = await realism.run(tmp_path / "frozen/manifest.json", tmp_path / "results")
    assert len(auth_calls) == 4 and len({p for _, p in auth_calls}) == 4
    assert len(calls) == 160
    for model in MODELS:
        assert result[model]["planned"] == result[model]["primary_only"]["planned"] == 40
        assert result[model]["retry_count"] == 0
        assert result[model]["model_invocations"] == 40
        assert result[model]["source_balanced_accuracy"] == 0.5
    assert (tmp_path / "results/finished.json").exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"image": "https://example.test/unrelated.jpg"},
        {"image": None},
        {"image": []},
        {"type": "image", "source": "not-an-object"},
        {"type": "image", "data": {"blob": "x"}},
        {"type": "image", "data": 123},
        {"type": "image", "source": {"data": []}},
        {"type": "image", "data": "☃"},
    ],
)
def test_malformed_unrelated_image_blocks_do_not_abort_delivery_audit(payload):
    valid = {"type": "image", "data": base64.b64encode(b"frozen-jpeg").decode()}
    assert delivered_hashes([payload]) == set()
    assert delivered_hashes([payload, valid]) == {hashlib.sha256(b"frozen-jpeg").hexdigest()}


async def test_frozen_hash_survives_neutral_file_mutation(tmp_path):
    from drivingbench.replay.clients import ClientResult
    from drivingbench.sim.realism import ask

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"frozen-image")
    expected = digest(image)
    stdout, stderr = tmp_path / "stdout.jsonl", tmp_path / "stderr.log"
    stderr.write_text("")

    class Client:
        launched = False

        def start(self, prompt, output):
            self.launched = True
            image.write_bytes(b"mutated-image")
            events = [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "mcp_tool_call",
                        "server": "scene",
                        "tool": "observe",
                        "id": "call",
                    },
                },
                {"type": "image", "data": base64.b64encode(image.read_bytes()).decode()},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "real 90"}},
            ]
            stdout.write_text("\n".join(map(json.dumps, events)))
            return self

        def turn_done(self):
            return True

        def poll(self):
            return 0

        def cancel(self):
            pass

        def finish(self):
            return ClientResult(0, "context", stdout, stderr)

    client = Client()
    result = await ask(client, image, tmp_path / "out", 90, expected_sha256=expected)
    assert result["status"] == "infrastructure"
    assert not result["image_delivered"]
    assert result["expected_image_sha256"] == expected
    assert result["neutral_image_sha256_after"] != expected
    assert "neutral_image_changed_after_dispatch" in result["errors"]
    client.launched = False
    with pytest.raises(ValueError, match="before dispatch"):
        await ask(client, image, tmp_path / "second", 90, expected_sha256=expected)
    assert not client.launched
