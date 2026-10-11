"""The authored scenes leave a continuous route for the full Corolla footprint."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from drivingbench.sim.course import Course

SCENES = Path(__file__).parents[1] / "sim"


@pytest.mark.parametrize("name", ["site1-course.json", "site1-pilot-course.json"])
def test_authored_course_has_clear_start_and_body_clear_route_to_finish(name):
    course = Course(json.loads((SCENES / name).read_text()))
    start = course.assess(course.map_pose((0, 0, 0)))
    assert not start["outside_course"] and not start["collision_ids"]
    assert not start["finish_reached"]

    data = json.loads((SCENES / name).read_text())
    left = np.asarray(data["boundaries"]["left"])
    right = np.asarray(data["boundaries"]["right"])
    center = (left + right) / 2
    direction = center[2:] - center[:-2]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
    results = []
    for index, tangent in enumerate(direction, 1):
        if arc[index] < 5 or arc[-1] - arc[index] < 3:
            continue
        heading = math.degrees(math.atan2(tangent[1], tangent[0]))
        results.append(course.assess((*center[index], heading)))
    assert results
    assert not any(result["outside_course"] for result in results)
    assert not any(result["collision_ids"] for result in results)
    assert any(result["valid_finish"] for result in results)
