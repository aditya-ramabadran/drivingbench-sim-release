import numpy as np
import pytest
from drivingbench.sim.faithfulness_video import median_window, register_trajectory


def test_selection_is_session_median_not_minimum():
    rows = [
        {"session": "a", "t": t, "sim": {"30s": {"position_m": error}}}
        for t, error in [(1, 0.1), (2, 0.4), (3, 0.8), (4, 3)]
    ]
    rows.append({"session": "b", "t": 0, "sim": {"30s": {"position_m": 0.6}}})
    assert median_window(rows, "a") is rows[1]
    with pytest.raises(ValueError, match="complete"):
        median_window(rows, "missing")


def test_fixed_registration_does_not_remove_later_drift():
    local = [[10, 20, np.pi / 2], [10, 22, np.pi / 2], [9, 24, np.pi]]
    registered = register_trajectory(local, [100, 200, 0])
    np.testing.assert_allclose(registered, [[100, 200, 0], [102, 200, 0], [104, 201, np.pi / 2]])
    np.testing.assert_allclose(local[0], [10, 20, np.pi / 2])


@pytest.mark.parametrize(
    "local,start", [([], [0, 0, 0]), ([[0, 0]], [0, 0, 0]), ([[0, 0, np.nan]], [0, 0, 0])]
)
def test_registration_rejects_invalid_poses(local, start):
    with pytest.raises(ValueError):
        register_trajectory(local, start)
