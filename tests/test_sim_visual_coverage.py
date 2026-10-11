"""Coverage is geometric evidence; validation excludes whole intervals and their guards."""

import numpy as np
import pytest
from drivingbench.sim.visual_coverage import split_rows, support


def test_split_uses_half_open_intervals_and_guards_without_mutation():
    rows = [
        dict(route="a", t=t, split="train", transform_matrix=[[t]])
        for t in (99, 100, 109, 110, 119, 120, 129, 130)
    ]
    result = split_rows(rows, [("a", 110, 120)], guard_s=10)
    assert [r["split"] for r in result] == [
        "train",
        "guard",
        "guard",
        "test",
        "test",
        "guard",
        "guard",
        "train",
    ]
    assert all(r["split"] == "train" for r in rows)
    assert result[3]["transform_matrix"] == rows[3]["transform_matrix"]
    assert split_rows([dict(route="b", t=115)], [("a", 110, 120)])[0]["split"] == "train"


def test_support_requires_matching_heading_and_never_uses_heldout_cameras():
    train = [dict(split="train", x=0, y=0, heading=0)]
    poses = [[1, 0, 2 * np.pi], [3, 0, 0], [0, 0, np.pi], [10, 0, 0]]
    rows = support(poses, train)
    assert [r["state"] for r in rows] == ["supported", "marginal", "unsupported", "unsupported"]
    assert rows[2]["nearest_m"] == 0
    assert rows[2]["nearest_aligned_m"] is None
    with pytest.raises(ValueError, match="training cameras"):
        support(poses, [dict(train[0], split="test")])
    with pytest.raises(ValueError, match="finite"):
        support([[0, np.nan, 0]], train)


def test_grid_preserves_heading_support_and_does_not_fill_turns():
    from drivingbench.sim.visual_coverage import support_grid

    training = [dict(split="train", x=0, y=0, heading=0)]
    x, y, heading, states = support_grid(training, spacing=1, directions=4)
    ix, iy = int(np.flatnonzero(x == 0)[0]), int(np.flatnonzero(y == 0)[0])
    assert states[:, iy, ix].tolist() == [2, 0, 0, 0]
    assert heading.tolist() == pytest.approx([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    assert not np.any(np.all(states == 2, axis=0))


def test_quality_overlay_rejects_training_and_duplicate_scores():
    from drivingbench.sim.visual_coverage import measured_frames

    rows = [
        dict(name=n, split=s, x=0, y=0, heading=0)
        for n, s in (("train", "train"), ("test", "test"))
    ]
    record = dict(name="test", scores={"Gaussian": {"psnr_db": 25}})
    assert measured_frames(rows, {"frames": [record]})[0]["score"]["psnr_db"] == 25
    for bad in ([dict(record, name="train")], [record, record]):
        with pytest.raises(ValueError, match="held-out"):
            measured_frames(rows, {"frames": bad})


def test_validation_rejects_fitted_pose_source(tmp_path):
    import json

    from drivingbench.sim.visual_coverage import prepare_validation

    manifest = tmp_path / "frames.jsonl"
    manifest.write_text(json.dumps(dict(seed_pose={}, route="a", t=0)) + "\n")
    with pytest.raises(ValueError, match="original pose seeds"):
        prepare_validation(manifest, tmp_path / "metadata.json", tmp_path / "out")


def test_interval_report_preserves_missing_denominators_and_new_split():
    from drivingbench.sim.visual_coverage import summarize_intervals

    rows = [
        dict(name="old", split="test", route="00000042--544757d3f5", t=400),
        dict(name="new", split="test", route="a", t=15),
        dict(name="missing", split="test", route="a", t=16),
    ]
    scores = dict(psnr_db=25, ssim=0.8, lpips_alex=0.3, identical=False)
    metrics = {"frames": [dict(name=n, scores={"Gaussian": scores}) for n in ("old", "new")]}
    localization = [dict(name="new", localized=True, inliers=31)]
    result = summarize_intervals(rows, metrics, localization, ranges=[("a", 10, 20)])
    assert result["all_test"]["requested"] == 3
    assert result["all_test"]["scored"] == 2
    assert result["development"]["requested"] == 1
    assert result["new_intervals"]["requested"] == 2
    assert result["new_intervals"]["scored"] == 1
    assert result["low_inlier_test_poses"][0]["name"] == "new"
