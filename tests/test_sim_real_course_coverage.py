import copy
import math

import numpy as np
import pytest
from drivingbench.sim.course import Course
from drivingbench.sim.real_course_study.coverage import (
    CoverageCertificate,
    connect_gaps,
    local_pose,
    stations,
)


def certificate():
    return {
        "schema": 1,
        "map_frame": "site1_full_ground",
        "approved": True,
        "cells": [
            {
                "id": "first",
                "polygon_xy": [[0, 0], [2, 0], [2, 1], [0, 1]],
                "heading_deg": 179,
                "heading_tolerance_deg": 15,
                "narrow": "pass",
                "wide": "pass",
            }
        ],
    }


def test_reviewed_cell_requires_position_heading_and_both_cameras():
    data = certificate()
    coverage = CoverageCertificate(data)
    assert coverage((1, 0.5, -179))["supported"]
    assert coverage((0, 0, 194))["supported"]
    assert not coverage((1, 0.5, 194.001))["supported"]
    assert not coverage((2.001, 0.5, 179))["supported"]
    data["cells"][0]["wide"] = "unknown"
    assert not CoverageCertificate(data)((1, 0.5, 179))["supported"]


def test_unapproved_certificate_fails_closed():
    data = certificate()
    data["approved"] = False
    assert CoverageCertificate(data)((1, 0.5, 179))["reason"] == "coverage_not_approved"


@pytest.mark.parametrize("case", ["missing_wide", "duplicate", "bowtie", "nan_heading"])
def test_invalid_certificates(case):
    data = certificate()
    cell = data["cells"][0]
    if case == "missing_wide":
        del cell["wide"]
    elif case == "duplicate":
        data["cells"].append(copy.deepcopy(cell))
    elif case == "bowtie":
        cell["polygon_xy"] = [[0, 0], [2, 1], [2, 0], [0, 1]]
    else:
        cell["heading_deg"] = math.nan
    with pytest.raises(ValueError):
        CoverageCertificate(data)


def test_local_renderer_pose_uses_degrees_and_round_trips():
    start = (10, -3, 90)
    local = local_pose((8, 1, 120), start)
    assert np.allclose(local, [4, 2, 30])
    course = object.__new__(Course)
    course.start = start
    assert np.allclose(course.map_pose(local), [8, 1, 120])


def test_stations_handle_standstill_and_wrap_without_shortcutting_curve():
    result = stations([0, 0, 1, 1], [0, 0, 0, 1], np.radians([179, 179, -179, -177]), 0.5)
    assert np.allclose(result[:, 3], [0, 0.5, 1, 1.5, 2])
    assert np.all(np.abs(np.diff(result[:, 2])) < math.radians(3))
    assert np.allclose(result[-1, :2], [1, 1])


def test_asset_and_review_hashes_are_checked(tmp_path):
    from drivingbench.sim.real_course_study.coverage import file_hash

    asset, review = tmp_path / "asset", tmp_path / "review"
    asset.write_text("fixed")
    review.write_text("reviewed")
    data = certificate()
    data.update(
        source_hashes={"asset": file_hash(asset)},
        review_manifest="review",
        review_manifest_sha256=file_hash(review),
    )
    coverage = CoverageCertificate(data).validate_assets(tmp_path)
    assert coverage((1, 0.5, 179))["supported"]
    review.write_text("changed")
    with pytest.raises(ValueError, match="review changed"):
        coverage.validate_assets(tmp_path)


def two_cells(gap=0.5):
    data = certificate()
    data["cells"][0]["polygon_xy"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    second = copy.deepcopy(data["cells"][0])
    second["id"] = "second"
    second["polygon_xy"] = [[1 + gap, 0], [2 + gap, 0], [2 + gap, 1], [1 + gap, 1]]
    data["cells"].append(second)
    return data


def test_bounded_connection_fills_only_the_intercell_gap_without_mutating_source():
    data = two_cells()
    original = copy.deepcopy(data)
    connected = connect_gaps(data)
    assert data == original
    assert connected["cells"] == original["cells"]
    assert connected["gap_connections"] == {"max_gap_m": 0.5, "pairs": [["first", "second"]]}
    coverage = CoverageCertificate(connected)
    assert coverage((1.25, 0.5, -179))["reason"] == "reviewed_connection"
    assert coverage((1.25, 1, 179))["supported"]
    assert not coverage((1.25, 1.00001, 179))["supported"]
    assert not coverage((-0.001, 0.5, 179))["supported"]
    assert not CoverageCertificate(connect_gaps(data, 0.49))((1.25, 0.5, 179))["supported"]
    assert not CoverageCertificate(connect_gaps(two_cells(0.50001)))((1.25, 0.5, 179))["supported"]


def test_connection_requires_sum_of_distances_not_two_independent_dilations():
    data = two_cells(0.4)
    data["cells"][1]["polygon_xy"] = [[1.4, 1], [2.4, 1], [2.4, 2], [1.4, 2]]
    coverage = CoverageCertificate(connect_gaps(data))
    assert coverage((1.2, 0.8, 179))["supported"]
    assert not coverage((1.2, 0.6, 179))["supported"]


def test_connection_requires_both_heading_guards_including_wrap():
    data = two_cells()
    data["cells"][1]["heading_deg"] = -165
    coverage = CoverageCertificate(connect_gaps(data))
    assert coverage((1.25, 0.5, -175))["supported"]
    assert not coverage((1.25, 0.5, 179))["supported"]
    assert not coverage((1.25, 0.5, -160))["supported"]
    data["cells"][1]["heading_deg"] = 0
    assert connect_gaps(data)["gap_connections"]["pairs"] == []


@pytest.mark.parametrize("camera,status", [("narrow", "fail"), ("wide", "unknown")])
def test_failed_or_unknown_cells_cannot_source_connections(camera, status):
    data = two_cells()
    data["cells"][1][camera] = status
    connected = connect_gaps(data)
    assert connected["gap_connections"]["pairs"] == []
    assert not CoverageCertificate(connected)((1.25, 0.5, 179))["supported"]


@pytest.mark.parametrize("camera,status", [("narrow", "fail"), ("wide", "unknown")])
def test_failed_or_unknown_pose_region_vetoes_a_connection(camera, status):
    data = two_cells()
    excluded = copy.deepcopy(data["cells"][0])
    excluded.update(id="excluded", polygon_xy=[[1.1, 0], [1.4, 0], [1.4, 1], [1.1, 1]])
    excluded[camera] = status
    data["cells"].append(excluded)
    coverage = CoverageCertificate(connect_gaps(data))
    assert not coverage((1.25, 0.5, 179))["supported"]
    assert coverage((1.45, 0.5, 179))["supported"]
    data["cells"][-1]["heading_deg"] = 0
    assert CoverageCertificate(connect_gaps(data))((1.25, 0.5, 179))["supported"]


@pytest.mark.parametrize("bound", [0, -1, 0.500001, math.nan, math.inf])
def test_invalid_connection_bound_fails_before_dispatch(bound):
    with pytest.raises(ValueError, match="bounded"):
        connect_gaps(two_cells(), bound)


@pytest.mark.parametrize("case", ["duplicate", "same", "missing", "far", "unreviewed", "yaw"])
def test_invalid_declared_connections_are_rejected(case):
    data = connect_gaps(two_cells())
    pairs = data["gap_connections"]["pairs"]
    if case == "duplicate":
        pairs.append(["second", "first"])
    elif case == "same":
        pairs[0] = ["first", "first"]
    elif case == "missing":
        pairs[0] = ["first", "missing"]
    elif case == "far":
        data["cells"][1]["polygon_xy"] = [[3, 0], [4, 0], [4, 1], [3, 1]]
    elif case == "unreviewed":
        data["cells"][1]["wide"] = "unknown"
    else:
        data["cells"][1]["heading_deg"] = 0
    with pytest.raises(ValueError):
        CoverageCertificate(data)


def test_connection_cannot_make_unapproved_certificate_supported():
    data = connect_gaps(two_cells())
    data["approved"] = False
    assert not CoverageCertificate(data)((1.25, 0.5, 179))["supported"]


def test_measure_preserves_strict_truth_and_does_not_use_yaw_for_spatial_distance():
    coverage = CoverageCertificate(certificate())
    inside = coverage.measure((1, 0.5, -179))
    assert {key: inside[key] for key in ("supported", "reason", "cells")} == coverage(
        (1, 0.5, -179)
    )
    wrong_yaw = coverage.measure((1, 0.5, 0))
    assert not wrong_yaw["supported"]
    assert wrong_yaw["distance_to_any_paired_pass_parent_m"] == 0
    assert wrong_yaw["yaw_excess_at_nearest_parent_deg"] == 164
    assert wrong_yaw["nearest_yaw_compatible_parent_distance_m"] is None
    assert coverage.measure((7, 0.5, 179))["distance_to_any_paired_pass_parent_m"] == 5
    assert coverage.measure((7.001, 0.5, 179))["distance_to_any_paired_pass_parent_m"] > 5


def test_measure_uses_nearest_spatial_review_not_farther_yaw_compatible_review():
    data = two_cells(2)
    data["cells"][1]["heading_deg"] = 0
    result = CoverageCertificate(data).measure((0.5, 0.5, 0))
    assert result["distance_to_any_paired_pass_parent_m"] == 0
    assert result["nearest_spatial_parent"] == "first"
    assert result["nearest_yaw_compatible_parent_distance_m"] == 2.5


def test_measure_deterministic_ties_and_connection_distances():
    data = connect_gaps(two_cells())
    result = CoverageCertificate(data).measure((1.25, 0.5, 179))
    assert result["supported"] and result["reason"] == "reviewed_connection"
    assert result["distance_to_any_paired_pass_parent_m"] == 0.25
    assert result["nearest_spatial_parent"] == "first"
    data["cells"].reverse()
    assert CoverageCertificate(data).measure((1.25, 0.5, 179)) == result


@pytest.mark.parametrize("case", ["unapproved", "empty", "unknown", "failed"])
def test_measure_requires_real_approved_paired_regions(case):
    data = certificate()
    if case == "unapproved":
        data["approved"] = False
    elif case == "empty":
        data["cells"] = []
    else:
        data["cells"][0]["wide"] = "unknown" if case == "unknown" else "fail"
    coverage = CoverageCertificate(data)
    assert not coverage((1, 0.5, 179))["supported"]
    with pytest.raises(ValueError, match="approved nonempty paired"):
        coverage.measure((1, 0.5, 179))


@pytest.mark.parametrize("pose", [(1, 0, math.nan), (1, math.inf, 0), (1, 0)])
def test_measure_rejects_nonfinite_or_malformed_pose(pose):
    with pytest.raises(ValueError, match="finite map"):
        CoverageCertificate(certificate()).measure(pose)
