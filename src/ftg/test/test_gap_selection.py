from ftg.gap_algorithm import select_target_direction

import numpy as np
import pytest


def make_scan():
    angles = np.deg2rad(np.arange(-100, 101))
    blocked = np.ones(angles.size, dtype=bool)
    return angles, blocked


def test_forward_gap():
    angles, blocked = make_scan()

    # Gap de -20° à +20°.
    blocked[80:121] = False

    target = select_target_direction(angles, blocked)

    assert target is not None
    assert np.isclose(target.angle_rad, 0.0)


def test_side_gap_target_is_not_center():
    angles, blocked = make_scan()

    # Gap de -100° à -20°.
    blocked[0:81] = False

    target = select_target_direction(angles, blocked)

    assert target is not None
    assert np.isclose(np.rad2deg(target.angle_rad), -25.0)


def test_forward_gap_preferred_over_wider_side_gap():
    angles, blocked = make_scan()

    blocked[88:113] = False  # -12° à +12°
    blocked[130:191] = False  # +30° à +90°

    target = select_target_direction(angles, blocked)

    assert target is not None
    assert np.isclose(target.angle_rad, 0.0)


def test_narrow_gap_rejected():
    angles, blocked = make_scan()

    blocked[95:100] = False

    assert select_target_direction(angles, blocked) is None


def test_all_directions_blocked():
    angles, blocked = make_scan()

    assert select_target_direction(angles, blocked) is None


def test_invalid_parameters():
    with pytest.raises(ValueError):
        select_target_direction(
            [0.1, 0.0],
            [False, False],
        )

    with pytest.raises(ValueError):
        select_target_direction(
            [0.0, 0.1],
            [False, False],
            edge_margin_deg=-1,
        )
