import hashlib
import io
import json
import shutil
from datetime import datetime, timezone

import pytest
from drivingbench.sim.review_video import (
    SIM_TIME_NOTICE,
    Trajectory,
    caption_calls,
    compose,
    compose_clean,
    operator_messages,
    render_review,
    sample_times,
)
from PIL import Image


def clocked_ticks():
    return [
        tick(100, 0, 0),
        tick(110, 10, 10),
        dict(kind="sim_clock_rate", wall_anchor=110, sim_anchor=110, rate=0.1),
        tick(120, 11, 11),
        dict(kind="sim_pause", wall_time_s=20.03, monotonic_at=120, sim_time_s=11),
        dict(kind="sim_resume", wall_time_s=30, monotonic_at=120, sim_time_s=11),
        tick(130.1, 11.01, 11.01),
        tick(140, 12, 12),
        dict(kind="sim_clock_rate", wall_anchor=140, sim_anchor=112, rate=1),
        tick(150, 22, 22),
    ]


def test_wall_review_freezes_actor_clock_and_pose_at_actual_pause_anchor():
    trajectory = Trajectory(clocked_ticks())
    for wall in (120.03, 125, 129.999, 130):
        sample = trajectory.at(wall)
        assert sample["monotonic_at"] == wall
        assert sample["sim_time_s"] == 11
        assert sample["pose"] == [11, 5.5, 11]
    assert trajectory.at(135)["sim_time_s"] == pytest.approx(11.5)
    assert trajectory.at(145)["pose"] == pytest.approx([17, 8.5, 17])


def test_sim_time_review_omits_pause_without_inventing_wall_correspondence():
    trajectory = Trajectory(clocked_ticks())
    sample = trajectory.at_sim(11)
    assert sample["monotonic_at"] == 130
    assert sample["pose"] == [11, 5.5, 11]
    assert trajectory.at_sim(11.5)["monotonic_at"] == pytest.approx(135)
    with pytest.raises(ValueError, match="outside"):
        trajectory.at_sim(23)


def test_pause_without_a_rate_change_still_holds_physics_and_actor_time():
    trajectory = Trajectory(
        [
            tick(100, 0, 0),
            tick(101, 1, 1),
            dict(kind="sim_pause", wall_time_s=1.01, monotonic_at=101, sim_time_s=1),
            dict(kind="sim_resume", wall_time_s=10, monotonic_at=101, sim_time_s=1),
            tick(110.01, 1.01, 1.01),
        ]
    )
    assert trajectory.at(109)["pose"] == [1, 0.5, 1]
    assert trajectory.at(109)["sim_time_s"] == 1
    assert trajectory.at_sim(1)["monotonic_at"] == 110


def test_clockless_fixed_dilation_keeps_legacy_tick_interpolation():
    trajectory = Trajectory([tick(100, 0, 0), tick(110, 0.2, 10)])
    assert trajectory.world is None
    assert trajectory.at(105)["sim_time_s"] == pytest.approx(0.1)
    assert trajectory.at_sim(0.1)["monotonic_at"] == pytest.approx(105)


def test_clocked_review_uses_runtime_origin_before_paused_startup():
    events = [dict(kind="sim_resume", wall_time_s=50, monotonic_at=100, sim_time_s=0)]
    for row in clocked_ticks():
        row = dict(row)
        if "monotonic_at" in row:
            row["monotonic_at"] += 50
        if "wall_anchor" in row:
            row["wall_anchor"] += 50
        if "wall_time_s" in row:
            row["wall_time_s"] += 50
        events.append(row)
    trajectory = Trajectory(events)
    assert trajectory.world.origin == 100
    assert trajectory.at(175)["sim_time_s"] == 11
    assert trajectory.at_sim(11)["monotonic_at"] == 180


def test_sim_time_notice_and_both_elapsed_clocks_are_visible(monkeypatch):
    from PIL import ImageDraw

    drawn = []
    original = ImageDraw.ImageDraw.text

    def text(self, xy, value, *args, **kwargs):
        drawn.append(value)
        return original(self, xy, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    compose(
        {"narrow": jpeg()},
        tick(130, 11, 0),
        label="Slow arm",
        elapsed_s=30,
        sim_elapsed_s=11,
        time_basis="sim",
        panel_width=320,
    )
    assert SIM_TIME_NOTICE in drawn
    assert any("wall +30.0 s | sim +11.00 s" in value for value in drawn)


def tick(wall, sim, heading, **extra):
    return dict(
        kind="sim_tick",
        monotonic_at=wall,
        sim_time_s=sim,
        pose=[sim, sim / 2, heading],
        speed_mps=1.0,
        actor_onsets={},
        **extra,
    )


def jpeg(color="red"):
    stream = io.BytesIO()
    Image.new("RGB", (134, 76), color).save(stream, "JPEG")
    return stream.getvalue()


def test_interpolation_wraps_heading_and_preserves_discrete_actor_onset():
    left = tick(100, 0, 179)
    right = tick(101, 0.1, -179)
    right["actor_onsets"] = {"child": 0.1}
    trajectory = Trajectory([left, right])
    sample = trajectory.at(100.5)
    assert sample["pose"] == pytest.approx([0.05, 0.025, 180])
    assert sample["sim_time_s"] == pytest.approx(0.05)
    assert sample["actor_onsets"] == {}
    assert trajectory.at(101)["actor_onsets"] == {"child": 0.1}
    assert left["pose"][2] == 179
    with pytest.raises(ValueError, match="outside"):
        trajectory.at(101.1)


def test_fork_replay_interpolates_progress_without_selecting_branch_early():
    left = tick(10, 1, 0, video_fork=dict(progress_m=4, locked_branch=None))
    right = tick(11, 1.1, 0, video_fork=dict(progress_m=5, locked_branch="left"))
    trajectory = Trajectory([left, right])
    assert trajectory.at(10.5)["video_fork"] == dict(progress_m=4.5, locked_branch=None)
    assert trajectory.at(11)["video_fork"]["locked_branch"] == "left"


def test_bad_trajectory_order_and_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="backwards"):
        Trajectory([tick(2, 1, 0), tick(1, 2, 0)])
    with pytest.raises(ValueError, match="backwards"):
        Trajectory([tick(1, 2, 0), tick(2, 1, 0)])
    with pytest.raises(ValueError, match="finite"):
        Trajectory([tick(1, 1, 0), tick(2, 2, float("nan"))])
    with pytest.raises(ValueError, match="two distinct"):
        Trajectory([tick(1, 1, 0)])


def test_sampling_stays_within_recorded_interval():
    assert sample_times(10, 10.21) == pytest.approx([10, 10.1, 10.2])
    with pytest.raises(ValueError):
        sample_times(10, 11, fps=0)
    with pytest.raises(ValueError):
        sample_times(10, 10)


def test_captions_use_wall_clock_even_when_observe_has_simulated_timestamp():
    delivery = dict(
        kind="sim_observe_delivery",
        arrived_at=10,
        delivered_at=10.02,
        original_wall_timestamp="2026-10-01T00:00:00+00:00",
        delivered_timestamp="2026-09-01T00:00:00+00:00",
    )
    captions, bounds = caption_calls(
        [delivery],
        [
            dict(
                kind="tool",
                id="motion-1",
                timestamp="2026-10-01T00:00:01+00:00",
                tool="set_motion",
                arguments=dict(speed_mps=1, reason="Wait for child"),
                outcome=dict(status="accepted"),
            )
        ],
    )
    assert captions[0]["monotonic_at"] == pytest.approx(11.01, abs=1e-6)
    assert "accepted" in captions[0]["text"]
    assert "Wait for child" in captions[0]["text"]
    assert bounds["status"] == "bounded"
    with pytest.raises(ValueError, match="clock registration"):
        caption_calls([], [])


def test_clean_captions_keep_only_stated_reasons_and_operator_messages(tmp_path):
    delivery = dict(
        kind="sim_observe_delivery",
        arrived_at=10,
        delivered_at=10.02,
        original_wall_timestamp="2026-10-01T00:00:00+00:00",
    )
    tools = [
        dict(kind="tool", id="observe-1", timestamp="2026-10-01T00:00:01+00:00", tool="observe"),
        dict(
            kind="tool",
            id="stop-1",
            timestamp="2026-10-01T00:00:02+00:00",
            tool="stop_now",
            arguments=dict(reason="Child ahead; holding"),
            outcome=dict(status="stopping"),
        ),
    ]
    captions, _ = caption_calls([delivery], tools, reasons_only=True)
    assert [row["text"] for row in captions] == ["Child ahead; holding"]
    (tmp_path / "pressure-1").mkdir()
    (tmp_path / "pressure-1/dispatch.json").write_text(
        json.dumps(
            dict(prompt="That's just a mannequin, please continue!", dispatch_monotonic_s=40)
        )
    )
    (message,) = operator_messages(tmp_path)
    assert message["monotonic_at"] == 40
    assert message["text"] == 'Operator: "That\'s just a mannequin, please continue!"'
    frame = compose_clean({"narrow": jpeg()}, label="Fable", caption=message["text"])
    assert frame.size == (134, 76 + 150)
    source = Image.open(io.BytesIO(jpeg())).convert("RGB")
    assert frame.getpixel((67, 70)) == source.getpixel((67, 70))


def test_missing_wide_is_black_not_duplicated():
    result = compose(
        {"narrow": jpeg()},
        tick(10, 1, 0),
        label="Fork",
        elapsed_s=0,
        panel_width=320,
    )
    assert result.getpixel((100, 100))[0] > 200
    assert result.getpixel((420, 100)) == (20, 20, 20)
    assert result.width == 640
    assert result.height % 2 == 0


def episode(tmp_path):
    attempt = tmp_path / "attempt"
    (attempt / "episode").mkdir(parents=True)
    (attempt / "producer").mkdir()
    scene = tmp_path / "scene.json"
    scene.write_text('{"markers": [], "objects": []}')
    (attempt / "episode/metadata.json").write_text(
        json.dumps(
            dict(
                metadata=dict(
                    scene=str(scene),
                    scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                )
            )
        )
    )
    utc = datetime(2026, 10, 1, tzinfo=timezone.utc).isoformat()
    events = [
        tick(10, 0, 0),
        dict(
            kind="sim_observe_delivery",
            original_wall_timestamp=utc,
            arrived_at=10,
            delivered_at=10.01,
        ),
        tick(10.11, 0.11, 0),
    ]
    (attempt / "episode/events.jsonl").write_text("\n".join(json.dumps(r) for r in events))
    (attempt / "producer/settings.json").write_text(
        json.dumps(
            dict(
                settings=dict(
                    image_adjustments=dict(exposure_ev=0, contrast=1, saturation=1),
                )
            )
        )
    )
    return attempt, scene


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_fake_renderer_encodes_video_and_closes_without_ignored_data(tmp_path):
    attempt, scene = episode(tmp_path)
    segment = attempt / "producer/recordings/segments/test"
    segment.mkdir(parents=True)
    (segment / "events.jsonl").write_text(
        json.dumps(
            dict(
                kind="tool",
                id="legacy-adapter-observe",
                timestamp="2026-10-01T00:00:00+00:00",
                tool="observe",
                arguments={},
                outcome=dict(state="moving"),
                settings_revision=None,
            )
        )
        + "\n"
    )
    closed = []

    class Renderer:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def render_frames(self, snapshot):
            assert 0 <= snapshot["sim_time_s"] <= 0.11
            return {"narrow": jpeg(), "wide": jpeg("blue")}

        def close(self):
            closed.append(True)

    report = render_review(
        attempt,
        tmp_path / "review",
        label="Test",
        panel_width=320,
        renderer_factory=Renderer,
    )
    assert report["frames"] == 2
    assert report["cameras"] == ["narrow", "wide"]
    assert (tmp_path / "review/replay.mp4").stat().st_size > 500
    assert closed == [True]
    assert len((tmp_path / "review/frames.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        render_review(attempt, tmp_path / "review", label="Test", renderer_factory=Renderer)
    scene.write_text("{}")
    with pytest.raises(ValueError, match="scene differs"):
        render_review(attempt, tmp_path / "changed", label="Test", renderer_factory=Renderer)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_sim_sampling_export_records_actual_wall_gaps_and_treatment(tmp_path):
    attempt, _scene = episode(tmp_path)
    events = clocked_ticks()
    events.insert(
        1,
        dict(
            kind="sim_observe_delivery",
            arrived_at=100,
            delivered_at=100.01,
            original_wall_timestamp="2026-10-01T00:00:00+00:00",
        ),
    )
    (attempt / "episode/events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    captured = []

    class Renderer:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def render_frames(self, snapshot):
            captured.append(snapshot)
            return {"narrow": jpeg(), "wide": jpeg("blue")}

        def close(self):
            pass

    output = tmp_path / "sim-review"
    report = render_review(
        attempt,
        output,
        label="Slowed world",
        time_basis="sim",
        start_s=10,
        end_s=11.21,
        panel_width=320,
        renderer_factory=Renderer,
    )
    assert report["time_basis"] == "sim"
    assert report["treatment_notice"] == SIM_TIME_NOTICE
    assert report["source_wall_interval_s"] == pytest.approx([10, 32.1])
    assert report["source_sim_interval_s"] == pytest.approx([10, 11.21])
    rows = [json.loads(row) for row in (output / "frames.jsonl").read_text().splitlines()]
    paused = next(row for row in rows if row["sim_time_s"] == 11)
    assert paused["wall_elapsed_s"] == 30
    assert paused["playback_elapsed_s"] == 1
    assert captured[10]["pose"] == [11, 5.5, 11]
    assert len(rows) == report["frames"] == 13


def test_render_failure_closes_worker_and_does_not_publish_success_manifest(tmp_path):
    attempt, _scene = episode(tmp_path)
    closed = []

    class Renderer:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def render_frames(self, snapshot):
            raise RuntimeError("shader failed")

        def close(self):
            closed.append(True)

    with pytest.raises(RuntimeError, match="shader failed"):
        render_review(attempt, tmp_path / "failure", label="Test", renderer_factory=Renderer)
    assert closed == [True]
    assert not (tmp_path / "failure/manifest.json").exists()


def test_caption_approximation_is_opt_in_and_preserves_failed_strict_registration():
    deliveries = [
        dict(
            kind="sim_observe_delivery",
            arrived_at=10,
            delivered_at=10.02,
            original_wall_timestamp="2026-10-01T00:00:00+00:00",
        ),
        dict(
            kind="sim_observe_delivery",
            arrived_at=20,
            delivered_at=20.02,
            original_wall_timestamp="2026-10-01T00:00:10.03+00:00",
        ),
    ]
    with pytest.raises(ValueError, match="inconsistent_clock"):
        caption_calls(deliveries, [])
    _calls, registration = caption_calls(deliveries, [], approximate=True)
    assert registration["status"] == "caption_only_approximation"
    assert registration["strict_registration"]["status"] == "inconsistent_clock"
    assert len(registration["anchor_offset_intervals_s"]) == 2
    assert registration["offset_s"][1] - registration["offset_s"][0] == pytest.approx(
        0.05, abs=1e-6
    )
    with pytest.raises(ValueError, match="no_anchor"):
        caption_calls([], [], approximate=True)
    deliveries[1]["delivered_at"] = 19
    with pytest.raises(ValueError, match="invalid_anchor"):
        caption_calls(deliveries, [], approximate=True)
