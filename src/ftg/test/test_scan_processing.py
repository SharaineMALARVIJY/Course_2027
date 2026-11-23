from ftg.scan_processing import preprocess_scan

import numpy as np

import pytest


def test_negative_angle_increment_reorders_ranges():
    scan = preprocess_scan(0.2, -0.1, [1, 2, 3, 4, 5], 0.1, 12.0)

    np.testing.assert_allclose(
        scan.angles,
        [-0.2, -0.1, 0, 0.1, 0.2],
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        scan.distances,
        [5, 4, 3, 2, 1],
    )

    assert scan.hit_mask.all()


def test_measurement_states_are_not_merged():
    values = [1.5, np.inf, np.nan, 0.05, 20.0, -np.inf]

    scan = preprocess_scan(-0.25, 0.1, values, 0.1, 12.0)

    np.testing.assert_array_equal(
        scan.hit_mask,
        [1, 0, 0, 0, 0, 0],
    )

    np.testing.assert_array_equal(
        scan.no_return_mask,
        [0, 1, 0, 0, 0, 0],
    )

    np.testing.assert_array_equal(
        scan.invalid_mask,
        [0, 0, 1, 1, 1, 1],
    )


def test_front_sector_keeps_correct_ranges():
    scan = preprocess_scan(
        -np.pi / 2,
        np.pi / 4,
        [1, 2, 3, 4, 5],
        0.1,
        12.0,
        half_fov_deg=50,
    )

    np.testing.assert_allclose(
        scan.angles,
        [-np.pi / 4, 0, np.pi / 4],
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        scan.distances,
        [2, 3, 4],
    )


def test_invalid_increment_is_rejected():
    with pytest.raises(ValueError):
        preprocess_scan(0, 0, [1.0], 0.1, 12.0)
