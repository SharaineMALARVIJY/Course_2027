"""Pure Python/Numpy tests: python -m unittest discover -s rl/reference_2023 -p 'test_*.py'"""
import math
import unittest
import numpy as np

from reference_logic import (MAX_SPEED_M_S, make_observation, reference_reward,
                             scan_to_reference, update_action)


class TestReferenceLogic(unittest.TestCase):
    def test_lidar_order_and_sign(self):
        for direction in (1, -1):
            angles = math.pi * 0.95 + np.arange(360) * direction * 2 * math.pi / 360
            ranges = 2.0 + np.sin(angles)
            data, stats = scan_to_reference(ranges, angles[0], direction * 2 * math.pi / 360)
            self.assertEqual(data.shape, (201,))
            self.assertEqual(stats['covered'], 201)
            self.assertAlmostEqual(data[0] * 12, 2.0, delta=0.02)
            self.assertAlmostEqual(data[30] * 12, 2.5, delta=0.03)
            self.assertAlmostEqual(data[-30] * 12, 1.5, delta=0.03)
            flipped, _ = scan_to_reference(ranges, angles[0], direction * 2 * math.pi / 360, flip_lidar=True)
            self.assertAlmostEqual(flipped[30] * 12, 1.5, delta=0.03)

    def test_missing_values_zero_not_twelve(self):
        rays = np.ones(360) * np.inf
        got, stats = scan_to_reference(rays, -math.pi, 2*math.pi/360)
        self.assertTrue(np.all(got == 0))
        self.assertEqual(stats['valid'], 0)

    def test_increment_and_limits(self):
        v, delta = update_action(0.1, 0.0, [1, 1])
        self.assertAlmostEqual(v, 0.15, places=5)
        self.assertEqual(delta, 9)
        v, delta = update_action(MAX_SPEED_M_S, 18, [1, 1])
        self.assertEqual(v, MAX_SPEED_M_S)
        self.assertEqual(delta, 18)

    def test_reward_units(self):
        lidar = np.full(201, 0.5, dtype=np.float64)
        reward, collision, info = reference_reward(lidar, 1.0)
        self.assertAlmostEqual(reward, 12 * (0.5-0.014) + 3*3.6)
        self.assertFalse(collision)
        lidar[0] = 0.01
        reward, collision, _ = reference_reward(lidar, 1.0)
        self.assertEqual(reward, -300)
        self.assertTrue(collision)

    def test_observation_signature(self):
        data = np.ones(201)
        obs = make_observation(data, data, 1.0, 9.0)
        self.assertTrue(all(x.dtype == np.float64 for x in obs.values()))
        self.assertAlmostEqual(obs['previous_angle'][0], 0.5)
        self.assertAlmostEqual(obs['previous_speed'][0], 1.0/MAX_SPEED_M_S)


if __name__ == '__main__':
    unittest.main()
