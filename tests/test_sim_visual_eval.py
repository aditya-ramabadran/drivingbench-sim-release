import numpy as np
import pytest
from drivingbench.sim.visual_eval import coverage, evaluation_training, pixel_scores, summarize


def test_coverage_reports_lateral_motion_and_wraps_heading():
    train = [{"name": "a", "x": 0, "y": 0, "heading": np.pi - 0.01}]
    result = coverage({"x": 0, "y": 2, "heading": -np.pi + 0.01}, train)
    assert result["distance_m"] == 2
    assert result["lateral_m"] == pytest.approx(2, abs=0.001)
    assert result["heading_deg"] == pytest.approx(np.degrees(0.02))


def test_pixel_metrics_do_not_silently_resize():
    pytest.importorskip("skimage")
    with pytest.raises(ValueError, match="matching RGB"):
        pixel_scores(np.zeros((20, 20, 3)), np.zeros((10, 10, 3)))
    real = np.full((20, 20, 3), 0.5, dtype=np.float32)
    assert pixel_scores(real, real)["identical"]
    assert pixel_scores(real, real + 0.1)["psnr_db"] == pytest.approx(20, abs=0.001)


def test_summary_keeps_perfect_predictions_and_failure_denominators():
    def row(psnr, exact):
        return {
            "scores": {"sim": {"psnr_db": psnr, "identical": exact, "ssim": 1.0, "lpips_alex": 0.0}}
        }

    result = summarize(
        [row(None, True), row(20.0, False), {"scores": {"sim": {"error": "missing"}}}], ["sim"]
    )["sim"]
    assert result["scored"] == 2 and result["requested"] == 3
    assert result["psnr_db"] is None and result["psnr_infinite"]
    assert result["identical"] == 1


def test_synthetic_offsets_cannot_be_scored_against_the_unshifted_photo():
    train = {"split": "train", "name": "a.jpg", "file_path": "a.jpg"}
    real = {"split": "test", "name": "b.jpg", "file_path": "b.jpg"}
    assert evaluation_training([train, real], [real]) == [train]
    for bad in (dict(real, perturbation={"side_m": 2}), dict(real, file_path="a.jpg")):
        with pytest.raises(ValueError, match="observed, disjoint"):
            evaluation_training([train, real], [bad])
