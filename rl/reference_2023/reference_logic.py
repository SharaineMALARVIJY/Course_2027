"""Reproduction of the 2023 CoVAPSy PPO observation/action/reward conventions.

Based on the source TT02_controller_RL.py in the user-provided ENS Paris-Saclay
annex (2024). This module is simulator/ROS independent and unit-testable.
"""
from __future__ import annotations

import numpy as np

MAX_RANGE_M = 12.0
MAX_SPEED_M_S = 12.0 / 3.6   # upstream MAXSPEED is 12 km/h
MAX_STEER_DEG = 18.0
SPEED_DELTA_M_S = 0.05      # upstream, NOT 0.5 m/s
STEER_DELTA_DEG = 9.0
CRASH_NORMALIZED_DISTANCE = 0.014  # upstream approx. 168mm


def scan_to_reference(ranges, angle_min, angle_increment, range_min=0.1,
                      flip_lidar=False):
    """LaserScan -> upstream PPO 201-beam layout, normalized, float64.

    Output order: [0deg, +1deg, ..., +100deg, -100deg, ..., -1deg].
    Upstream uses 0 for lidar returns >=12m, NaN, inf or not sampled.
    Angle positions in ROS LaserScan are remapped independently of scan sign.
    """
    raw = np.asarray(ranges, dtype=np.float64)
    if not raw.size or not np.isfinite(angle_increment) or abs(angle_increment) < 1e-12:
        raise ValueError('LaserScan empty or invalid angle_increment')
    targets_deg = np.arange(-100, 101, dtype=np.float64)
    targets = np.deg2rad(-targets_deg if flip_lidar else targets_deg)
    source = float(angle_min) + np.arange(raw.size) * float(angle_increment)
    difference = (source[None, :] - targets[:, None] + np.pi) % (2.0 * np.pi) - np.pi
    indices = np.abs(difference).argmin(axis=1)
    err = np.abs(difference[np.arange(201), indices])
    covered = err < max(np.deg2rad(1.5), abs(float(angle_increment)) * 1.5)
    selected = raw[indices]
    valid = covered & np.isfinite(selected) & (selected >= max(0.01, float(range_min))) & (selected < MAX_RANGE_M)
    angular_order = np.zeros(201, dtype=np.float64)
    angular_order[valid] = selected[valid] / MAX_RANGE_M
    reference_order = np.concatenate((angular_order[100:], angular_order[:100]))
    diagnostics = {
        'covered': int(covered.sum()),
        'valid': int(valid.sum()),
        'valid_front': int(valid[60:140].sum()),
        'at_zero_deg_m': float(angular_order[100] * MAX_RANGE_M),
        'front_min_m': float(np.min(angular_order[60:140][angular_order[60:140] > 0]) * MAX_RANGE_M)
                       if np.any(angular_order[60:140] > 0) else None,
    }
    return reference_order, diagnostics


def update_action(speed_m_s, steering_deg, action):
    """Upstream TT02_controller_RL.step(): apply normalized *increments*."""
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if a.shape != (2,):
        raise ValueError('Expected two continuous actions')
    speed_m_s = float(np.clip(speed_m_s + float(a[0]) * SPEED_DELTA_M_S,
                              0.1, MAX_SPEED_M_S))
    steering_deg = float(np.clip(steering_deg + float(a[1]) * STEER_DELTA_DEG,
                                 -MAX_STEER_DEG, MAX_STEER_DEG))
    return speed_m_s, steering_deg


def make_observation(current_lidar, previous_lidar, speed_m_s, steering_deg,
                     initial=False):
    """Upstream keys, shapes and float64 types; required for archived PPO."""
    current = np.asarray(current_lidar, dtype=np.float64).copy()
    previous = current.copy() if initial else np.asarray(previous_lidar, dtype=np.float64).copy()
    return {
        'current_lidar': current,
        'previous_lidar': previous,
        'previous_speed': np.array([0.0 if initial else speed_m_s / MAX_SPEED_M_S], dtype=np.float64),
        'previous_angle': np.array([0.0 if initial else steering_deg / MAX_STEER_DEG], dtype=np.float64),
    }


def reference_reward(current_lidar, speed_m_s):
    """Upstream TT02_controller_RL.get_reward() including normalized units.

    Original reward: 12*(mini - .014) + 3*getTargetCruisingSpeed(),
    where 'mini' is a *normalized* lidar distance and speed is in km/h.
    """
    lidar = np.asarray(current_lidar, dtype=np.float64)
    if lidar.shape != (201,):
        raise ValueError('Expected 201 lidar beams')
    # Upstream checks indices [-40..39] of [0,+1..+100,-100..-1].
    forward = lidar[np.arange(-40, 40)]
    positive = forward[forward > 0]
    mini = float(positive.min()) if positive.size else 1.0
    acquisition_failure = bool(lidar[0] == 0 and lidar[1] == 0 and lidar[-1] == 0)
    collision = acquisition_failure or (mini < CRASH_NORMALIZED_DISTANCE)
    # Webots Driver.getTargetCruisingSpeed() returns km/h.
    reward = -300.0 if collision else 12.0 * (mini - CRASH_NORMALIZED_DISTANCE) + 3.0 * speed_m_s * 3.6
    return float(reward), bool(collision), {
        'front_min_m': float(mini * MAX_RANGE_M),
        'acquisition_failure': acquisition_failure,
    }
