import json

import pytest
from drivingbench.replay.review import build_review, build_validation, safe_path
from PIL import Image


def fixture_run(study, name="C-sol-01"):
    run = study / "runs" / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "model": "sol",
                "experiment": "C",
                "condition": "<script>bad()</script>",
                "status": "complete",
                "end_reason": "accepted_motion",
            }
        )
    )
    (run / "prompt.txt").write_text('<script>alert("prompt")</script>')
    (run / "transcript.normalized.json").write_text(
        json.dumps([{"role": "assistant", "text": "<img src=x onerror=bad()>"}])
    )
    image = run / "recordings/recordings/images/abc.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image)
    (run / "events.jsonl").write_text(
        json.dumps(
            {
                "kind": "tool",
                "tool": "observe",
                "timestamp": "now",
                "images": [{"url": "/api/recordings/abc.jpg", "camera": "narrow"}],
                "arguments": {"reason": "<svg onload=bad()>"},
            }
        )
        + "\n"
        + "{incomplete"
    )
    return run


def test_browser_escapes_and_links_frames_without_copying(tmp_path):
    study = tmp_path / "study"
    run = fixture_run(study)
    smoke = fixture_run(study, "smoke-sol-01")
    fixture_run(study, "B-smoke-A-sol-child-0-imperative")
    page = build_review(study)
    result = page.read_text()
    detail = (run / "review.html").read_text()
    assert "<script>bad" not in result
    assert "&lt;script&gt;bad" in result
    assert "<script>alert" not in detail and "&lt;script&gt;alert" in detail
    assert "<img src=x" not in detail and "&lt;img src=x" in detail
    assert 'src="recordings/recordings/images/abc.jpg"' in detail
    assert (smoke / "review.html").exists()
    assert result.index("smoke-sol-01") > result.index("<details>")
    assert result.index("B-smoke-A-sol-child-0-imperative") > result.index("<details>")
    assert len(list(study.rglob("*.jpg"))) == 3


def test_auth_and_symlink_escape_never_render(tmp_path):
    study = tmp_path / "study"
    run = fixture_run(study)
    private = run / "private-client-state"
    private.mkdir()
    (private / "auth.json").write_text("SECRET-AUTH")
    (study / "runs" / "escape").symlink_to(private, target_is_directory=True)
    outside = tmp_path / "outside"
    outside.write_text("SECRET-OUTSIDE")
    with pytest.raises(ValueError):
        safe_path(private / "auth.json", study)
    with pytest.raises(ValueError):
        safe_path(outside, study)
    (run / "prompt.txt").unlink()
    (run / "prompt.txt").symlink_to(outside)
    page = build_review(study)
    assert "SECRET" not in page.read_text()
    assert not (run / "review.html").exists()


def test_validation_balanced_neutral_and_immutable(tmp_path):
    for source in ("empty-aisle", "natural-person", "composite-child-v2", "composite-dog-v2"):
        folder = tmp_path / source
        folder.mkdir()
        Image.new("RGB", (4, 4)).save(folder / "image.png")
        (folder / "manifest.json").write_text(
            json.dumps({"frames": [{"narrow": "image.png", "distance_m": d} for d in range(4)]})
        )
    board = build_validation(tmp_path)
    text = board.read_text()
    key = json.loads((board.parent / "answer-key.json").read_text())
    assert len(key["cards"]) == 12
    assert sum(card["edited"] for card in key["cards"]) == 6
    for name in ("composite-child", "composite-dog", "natural-person", "answer-key.json"):
        assert name not in text
    assert "pending_three_independent_humans" == key["status"]
    assert text.count('type="radio"') == 24
    original = board.read_bytes()
    assert build_validation(tmp_path, seed=2).read_bytes() == original
    assert len(list(board.parent.glob("frame-*.jpg"))) == 12


def test_pressure_links_parent_image_and_exact_followup(tmp_path):
    study = tmp_path / "study"
    parent = fixture_run(study, "A-sol-01")
    child = fixture_run(study, "B-A-sol-01")
    metadata = json.loads((child / "run.json").read_text())
    metadata["pressure"] = {"parent_run_id": parent.name}
    (child / "run.json").write_text(json.dumps(metadata))
    (child / "prompt.txt").unlink()
    (child / "client").mkdir()
    (child / "client/invocation.json").write_text(json.dumps({"prompt": "Continue <safely>"}))
    (child / "recordings/recordings/images/abc.jpg").unlink()
    build_review(study)
    text = (child / "review.html").read_text()
    assert "../A-sol-01/recordings/recordings/images/abc.jpg" in text
    assert "Continue &lt;safely&gt;" in text
