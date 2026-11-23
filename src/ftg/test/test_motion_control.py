import math

from ftg.motion_control import compute_steering

import pytest


def test_straight_target_produces_zero_steering():
    result = compute_steering(0.0)

    assert result.steering_angle_rad == pytest.approx(0.0)
    assert result.curvature_m_inv == pytest.approx(0.0)
    assert not result.saturated


def test_left_and_right_steering():
    left = compute_steering(math.radians(20))
    right = compute_steering(math.radians(-20))

    assert left.steering_angle_rad > 0
    assert right.steering_angle_rad < 0

    assert left.steering_angle_rad == pytest.approx(-right.steering_angle_rad)


def test_known_target_angle():
    result = compute_steering(math.radians(-25))

    assert math.degrees(result.steering_angle_rad) == pytest.approx(
        -7.65,
        abs=0.1,
    )

    assert not result.saturated


def test_maximum_steering_saturation():
    result = compute_steering(math.radians(-76))

    assert result.saturated
    assert result.steering_angle_rad == pytest.approx(-0.35)


def test_invalid_lookahead():
    with pytest.raises(ValueError):
        compute_steering(0.0, lookahead_m=0.0)


def test_target_behind_rear_axle():
    with pytest.raises(ValueError):
        compute_steering(math.pi)
