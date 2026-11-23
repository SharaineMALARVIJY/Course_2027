"""ROS 2 Jazzy adapter of ENS Paris-Saclay's 2023 CoVAPSy Webots PPO.

Run on Course_2027 Webots R2025a blue car, *without* changing Webots controller.
The 2023 model is usable because the observation keys, shapes, float64 dtype,
LiDAR beam ordering, action increments and reward convention are reproduced.

Limitations: R2023b and R2025a tracks/vehicle physics are not identical.
"""
from __future__ import annotations

import math
import threading
import time

import gymnasium as gym
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from gymnasium import spaces
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger

from reference_logic import (MAX_SPEED_M_S, MAX_STEER_DEG, make_observation,
                             reference_reward, scan_to_reference, update_action)


class ReferenceBlueCarEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, step_ms=32, max_episode_steps=16384, random_resets=False,
                 flip_lidar=False, flip_steering=False):
        super().__init__()
        self.step_ns = int(step_ms * 1_000_000)
        if self.step_ns < 8_000_000:
            raise ValueError('step_ms must be >= 8 for the Course_2027 world')
        self.max_episode_steps = int(max_episode_steps)
        self.flip_lidar = bool(flip_lidar)
        self.flip_steering = bool(flip_steering)
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node('reference_2023_blue_rl')
        self._condition = threading.Condition()
        self._last_lidar = None
        self._scan_seq = 0
        self._clock_ns = None
        self._scan_info = {}
        self.node.create_subscription(LaserScan, '/blue_car/scan', self._on_scan,
                                      qos_profile_sensor_data)
        self.node.create_subscription(Clock, '/clock', self._on_clock,
                                      qos_profile_sensor_data)
        self.command_pub = self.node.create_publisher(AckermannDrive,
                                                       '/blue_car/cmd_ackermann', 10)
        service = '/rl/reset_random' if random_resets else '/rl/reset'
        self.reset_client = self.node.create_client(Trigger, service)
        self.reset_service_name = service
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # Use original 2023 Gymnasium space *dtype* (float64), so the archived
        # PPO can be loaded with env= without observation-space mismatch.
        self.observation_space = spaces.Dict({
            'current_lidar': spaces.Box(np.zeros(201), np.ones(201), dtype=np.float64),
            'previous_lidar': spaces.Box(np.zeros(201), np.ones(201), dtype=np.float64),
            'previous_speed': spaces.Box(np.array([0.]), np.array([1.]), dtype=np.float64),
            'previous_angle': spaces.Box(np.array([-1.]), np.array([1.]), dtype=np.float64),
        })
        self.action_space = spaces.Box(
            low=np.array([-1., -1.], dtype=np.float32),
            high=np.array([1., 1.], dtype=np.float32), dtype=np.float32)
        self.speed = 0.0
        self.angle_deg = 0.0
        self._previous_lidar = np.zeros(201, dtype=np.float64)
        self.steps = 0
        self.crashes = 0
        self._closed = False

    def _spin(self):
        try:
            self._executor.spin()
        except (ExternalShutdownException, KeyboardInterrupt):
            pass
        except Exception as exc:
            if rclpy.ok():
                self.node.get_logger().error(f'ROS executor stopped: {exc}')

    def _on_clock(self, msg):
        ns = int(msg.clock.sec) * 1_000_000_000 + int(msg.clock.nanosec)
        with self._condition:
            self._clock_ns = ns
            self._condition.notify_all()

    def _on_scan(self, msg):
        try:
            lidar, stats = scan_to_reference(msg.ranges, msg.angle_min, msg.angle_increment,
                                             msg.range_min, self.flip_lidar)
        except (ValueError, IndexError) as exc:
            self.node.get_logger().warning(f'LaserScan rejected: {exc}')
            return
        with self._condition:
            self._last_lidar = lidar
            self._scan_info = stats
            self._scan_seq += 1
            self._condition.notify_all()

    def _wait(self, predicate, reason, timeout=6.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'{reason}. Webots is running/unpaused?')
                self._condition.wait(timeout=remaining)

    def _get_scan(self):
        with self._condition:
            return self._last_lidar.copy(), self._scan_info.copy()

    def _publish(self, speed, angle_deg):
        if not rclpy.ok():
            return
        cmd = AckermannDrive()
        cmd.speed = float(speed)
        # blue_driver.py applies Webots.setSteeringAngle(-command). The
        # original controller used Webots.setSteeringAngle(-angle_degre).
        steer = -angle_deg if self.flip_steering else angle_deg
        cmd.steering_angle = float(np.clip(math.radians(steer), -0.28, 0.28))
        try:
            self.command_pub.publish(cmd)
        except Exception:
            if rclpy.ok():
                raise

    def _wait_simulation_step(self):
        self._wait(lambda: self._clock_ns is not None, '/clock absent')
        with self._condition:
            t0 = self._clock_ns
        self._wait(lambda: self._clock_ns is not None and self._clock_ns >= t0 + self.step_ns,
                   'Simulated clock did not advance')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._publish(0.0, 0.0)
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'Missing {self.reset_service_name}; recompile supervisor plugin')
        request = self.reset_client.call_async(Trigger.Request())
        deadline = time.monotonic() + 5.0
        while not request.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not request.done():
            raise TimeoutError(f'{self.reset_service_name} did not answer')
        result = request.result()
        if result is None or not result.success:
            raise RuntimeError(f'Reset failed: {getattr(result, "message", "no response")}')
        self.speed = 0.0
        self.angle_deg = 0.0
        self.steps = 0
        self._publish(0.0, 0.0)
        with self._condition:
            previous_seq = self._scan_seq
        # Wait for a couple of NEW lidar frames after a teleport (stale scans
        # from before the reset must not become the episode's first state).
        self._wait(lambda: self._scan_seq >= previous_seq + 2,
                   'No fresh lidar scans after reset', timeout=8.0)
        current, stats = self._get_scan()
        if stats['covered'] != 201 or stats['valid_front'] < 10:
            raise RuntimeError(f'Lidar data incomplete/invalid after reset: {stats}')
        self._previous_lidar = current.copy()
        obs = make_observation(current, current, 0.0, 0.0, initial=True)
        return obs, {'reset_service': self.reset_service_name,
                     'front_min_m': stats['front_min_m'], 'lidar': stats}

    def step(self, action):
        self.speed, self.angle_deg = update_action(self.speed, self.angle_deg, action)
        old_lidar, _ = self._get_scan()
        self._publish(self.speed, self.angle_deg)
        # 2023 Webots default basicTimeStep was 32ms. This synchronizes
        # actions to simulated time, rather than waiting 83ms for each new
        # 12Hz scan (which was the key mismatch of the first ROS prototype).
        self._wait_simulation_step()
        current, stats = self._get_scan()
        obs = make_observation(current, old_lidar, self.speed, self.angle_deg)
        reward, collision, details = reference_reward(current, self.speed)
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        if collision:
            self.crashes += 1
        info = {
            **details,
            'speed_cmd_m_s': self.speed,
            'steer_cmd_deg': self.angle_deg,
            'step': self.steps,
            'collision': collision,
            'lidar': stats,
        }
        return obs, reward, collision, truncated, info

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._publish(0.0, 0.0)
        except Exception:
            pass
        try:
            if rclpy.ok():
                self._executor.shutdown(timeout_sec=2.0)
        except Exception:
            pass
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        try:
            self.node.destroy_node()
        except Exception:
            pass
