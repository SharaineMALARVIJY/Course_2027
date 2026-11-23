from ftg.gap_algorithm import (
    compute_safety_mask,
    find_closest_obstacle,
    find_free_gaps,
)

import numpy as np
import pytest


def test_closest_obstacle_ignores_invalid_measurements():
    distances = [np.nan, 2.0, 0.4, np.inf]
    hits = [False, True, True, False]

    assert find_closest_obstacle(distances, hits) == 2
    assert find_closest_obstacle([np.nan, np.inf], [False, False]) is None


def test_obstacle_expands_to_neighboring_angles():
    angles = [-0.6, -0.2, 0.0, 0.2, 0.6]
    distances = [3, 3, 0.5, 3, 3]

    mask = compute_safety_mask(
        angles,
        distances,
        [True] * 5,
        safety_radius_m=0.12,
        horizon_m=1.0,
    )

    np.testing.assert_array_equal(
        mask,
        [False, True, True, True, False],
    )


def test_two_walls_leave_a_center_gap():
    angles = [-0.8, -0.4, 0, 0.4, 0.8]
    distances = [3, 0.7, 3, 0.7, 3]

    mask = compute_safety_mask(
        angles,
        distances,
        [True] * 5,
        safety_radius_m=0.1,
        horizon_m=1.0,
    )

    np.testing.assert_array_equal(
        mask,
        [False, True, False, True, False],
    )

    assert [(g.start, g.stop) for g in find_free_gaps(mask)] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]


def test_distant_walls_do_not_block_every_direction():
    mask = compute_safety_mask(
        [-0.5, 0, 0.5],
        [3.0, 3.0, 3.0],
        [True] * 3,
        safety_radius_m=0.2,
        horizon_m=1.0,
    )

    assert not mask.any()


def test_unknown_rays_are_blocked():
    mask = compute_safety_mask(
        [-0.2, 0, 0.2],
        [np.nan, np.inf, 3.0],
        [False, False, True],
        safety_radius_m=0.2,
        horizon_m=1.0,
    )

    np.testing.assert_array_equal(
        mask,
        [True, True, False],
    )


def test_very_close_obstacle_blocks_all_directions():
    mask = compute_safety_mask(
        [-0.2, 0, 0.2],
        [3.0, 0.1, 3.0],
        [True] * 3,
        safety_radius_m=0.2,
        horizon_m=1.0,
    )

    assert mask.all()
    assert find_free_gaps(mask) == []


def test_rejects_bad_angles_and_parameters():
    with pytest.raises(ValueError):
        compute_safety_mask(
            [0.1, 0.0],
            [1, 1],
            [True, True],
        )

    with pytest.raises(ValueError):
        compute_safety_mask(
            [0.0],
            [1.0],
            [True],
            safety_radius_m=0,
        )
