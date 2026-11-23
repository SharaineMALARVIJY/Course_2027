"""LaserScan resampling used by the ROS2/Gym interface (no ROS dependency)."""
from __future__ import annotations

import numpy as np

TARGET_ANGLES = np.deg2rad(np.arange(-100.0, 101.0, 1.0))


def resample_scan(ranges, angle_min, angle_increment, range_min=0.1, max_distance=12.0):
    """Return (201 normalized rays, forward distance, diagnostic dict).

    Handles positive/negative LaserScan.angle_increment, arbitrary angle_min,
    partial FOV, +/-pi wrap, NaN/inf/too-near raw readings.
    """
    source = np.asarray(ranges, dtype=np.float32)
    if source.size == 0 or not np.isfinite(angle_increment) or abs(angle_increment) < 1e-10:
        raise ValueError('Scan vide ou increment angulaire invalide')
    source_angles = float(angle_min) + np.arange(source.size) * float(angle_increment)
    # Find nearest angular match on the circle, without assuming ROS scan order.
    circular_difference = (
        source_angles[None, :] - TARGET_ANGLES[:, None] + np.pi
    ) % (2 * np.pi) - np.pi
    abs_difference = np.abs(circular_difference)
    indices = abs_difference.argmin(axis=1)
    nearest_error = abs_difference[np.arange(TARGET_ANGLES.size), indices]
    angle_covered = nearest_error <= max(1.5 * abs(float(angle_increment)), np.deg2rad(1.5))

    chosen = source[indices]
    raw_valid = np.isfinite(chosen) & (chosen >= max(float(range_min), 0.01))
    good = angle_covered & raw_valid
    output = np.full(TARGET_ANGLES.shape, float(max_distance), dtype=np.float32)
    output[good] = np.clip(chosen[good], 0.0, max_distance)

    # +/-40 degrees is 60:141 in [-100,100].
    front_min_m = float(output[60:141].min())
    stats = {
        'raw_total': int(source.size),
        'raw_finite': int(np.isfinite(source).sum()),
        'covered_201': int(angle_covered.sum()),
        'valid_201': int(good.sum()),
        'valid_front': int(good[60:141].sum()),
        'sample_m': {str(deg): round(float(output[deg + 100]), 3)
                     for deg in (-90, -60, -30, 0, 30, 60, 90)},
    }
    return output / np.float32(max_distance), front_min_m, stats


if __name__ == '__main__':
    # Synthetic tests for both scan rotations and arbitrary start offsets.
    for direction in (1, -1):
        for angle0 in (-np.pi, 0.0, 1.1):
            n = 360
            angles = angle0 + np.arange(n) * direction * (2 * np.pi / n)
            signal = 2.0 + 0.5 * np.cos(angles)  # 2.5m in front
            normalized, front, stats = resample_scan(signal, angle0, direction * (2 * np.pi/n))
            assert 2.0 <= front <= 2.5, (front, direction, angle0)
            assert stats['covered_201'] == 201, stats
            assert abs(normalized[100] * 12 - 2.5) < 0.05
    print('Self-test LiDAR OK (+/- increment, arbitrary angle_min)')
