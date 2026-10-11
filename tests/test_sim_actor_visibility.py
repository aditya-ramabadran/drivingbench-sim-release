"""Full source-alpha denominators prevent cropped fragments from qualifying as whole actors."""

import numpy as np
import pytest
from drivingbench.sim.actor_visibility import region_rectangles, whole_actor_visibility
from drivingbench.sim.billboards import BillboardCache
from drivingbench.sim.visual_objects import PlacedObject
from PIL import Image

REGIONS = {"head": [0, 0, 1, 0.3], "body": [0, 0.3, 1, 0.8], "feet": [0, 0.8, 1, 1]}
CAMERA = dict(w=100, h=100, fl_x=100, fl_y=100, cx=50, cy=50)
MATRIX = np.eye(4)
MATRIX[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]


def fixture(tmp_path, x=0):
    pixels = np.full((20, 20, 4), 255, np.uint8)
    pixels[0, 0, 3] = 0
    path = tmp_path / "actor.png"
    Image.fromarray(pixels).save(path)
    obj = PlacedObject(kind="billboard", x=x, y=5, asset_path=str(path), width_m=1, height_m=2)
    cache = BillboardCache()
    layer = cache.project(obj, MATRIX, CAMERA)
    return cache, obj, layer.rgba[..., 3] > 0


def test_whole_actor_all_regions_visible(tmp_path):
    cache, obj, visible = fixture(tmp_path)
    result = whole_actor_visibility(cache, obj, MATRIX, CAMERA, visible, REGIONS)
    assert result["valid"] and result["in_frame_fraction"] == 1
    assert result["survival_fraction"] == 1
    assert all(v == 1 for v in result["region_survival"].values())
    assert result["edge_margin_px"] >= 8


def test_clipped_source_counts_offscreen_pixels_in_denominator(tmp_path):
    cache, obj, visible = fixture(tmp_path, x=2.4)
    result = whole_actor_visibility(cache, obj, MATRIX, CAMERA, visible, REGIONS)
    assert result["valid"]
    assert result["projected_total_pixels"] > result["in_frame_pixels"]
    assert result["in_frame_fraction"] < 0.95 and result["edge_margin_px"] < 0
    assert result["visible_pixels"] == int(visible.sum())


def test_depth_or_body_mask_hiding_feet_cannot_pass_region_gate(tmp_path):
    cache, obj, visible = fixture(tmp_path)
    visible[42:] = False
    result = whole_actor_visibility(cache, obj, MATRIX, CAMERA, visible, REGIONS)
    assert result["region_survival"]["head"] == 1
    assert result["region_survival"]["feet"] == 0
    assert result["survival_fraction"] < 0.85


def test_hidden_actor_has_no_centroid_or_surviving_height(tmp_path):
    cache, obj, visible = fixture(tmp_path)
    result = whole_actor_visibility(cache, obj, MATRIX, CAMERA, visible & False, REGIONS)
    assert result["visible_pixels"] == result["visible_height_px"] == 0
    assert result["visible_centroid_px"] is None
    assert all(v == 0 for v in result["region_survival"].values())


@pytest.mark.parametrize(
    "regions",
    [
        {},
        REGIONS | {"tail": [0, 0, 1, 1]},
        REGIONS | {"head": [0, 0, 1, float("nan")]},
        REGIONS | {"feet": [0, 1, 1, 0]},
    ],
)
def test_missing_or_invalid_source_annotations_fail_closed(regions):
    with pytest.raises(ValueError):
        region_rectangles(regions)
