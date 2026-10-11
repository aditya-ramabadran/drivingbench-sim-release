import math

from drivingbench.sim.inventory import SPEED_BINS, histogram, local_xy, sites


def test_histogram_puts_tiny_negative_speeds_in_the_lowest_bin():
    out = histogram([(-0.0001, 1.0), (0.3, 2.0), (12.0, 0.5)], SPEED_BINS)
    assert out["0"] == 1.0 and out["0.2"] == 2.0 and out["10"] == 0.5


def test_local_xy_is_metres_east_and_north():
    x, y = local_xy(37.0001, -122.0, 37.0, -122.0)
    assert abs(x) < 1e-6 and math.isclose(y, 11.12, rel_tol=0.01)


def track(lat, lon):
    return {"gps_track": [(lat, lon, 1.0, True, True)], "engaged_s": 10}


def test_sites_group_nearby_segments_and_order_by_engaged_time():
    a, b, far = track(37.0, -122.0), track(37.0005, -122.0), track(37.03, -122.0)
    far["engaged_s"] = 100
    groups = sites([a, b, far, {"gps_track": [], "engaged_s": 0}])
    assert groups == [[far], [a, b]]
