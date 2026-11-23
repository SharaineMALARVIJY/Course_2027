"""ROS2 <-> Gymnasium environment for Course_2027 blue car.

Do not launch any competing controller (FTG/teleop) on /blue_car/cmd_ackermann.
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
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger

from lidar_utils import resample_scan


class BlueCarEnv(gym.Env):
    metadata = {"render_modes": []}
    MAX_RANGE = 12.0
    MAX_SPEED = 1.5  # m/s (prototype conservative limit)
    MAX_STEER = 0.28  # rad (from Webots vehicle model)
    CRASH_DISTANCE = 0.20  # m, check measured front distance on your course

    def __init__(self, max_episode_steps: int = 400):
        super().__init__()
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node: Node = rclpy.create_node('blue_car_rl_environment')
        self._condition = threading.Condition()
        self._scan_seq = 0
        self._ranges = None
        self._last_front_min = self.MAX_RANGE
        self._scan_error = None

        self.node.create_subscription(
            LaserScan, '/blue_car/scan', self._scan_callback, qos_profile_sensor_data
        )
        self.command_pub = self.node.create_publisher(
            AckermannDrive, '/blue_car/cmd_ackermann', 10
        )
        self.reset_client = self.node.create_client(Trigger, '/rl/reset')
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._spin_ros, daemon=True
        )
        self._spin_thread.start()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            'current_lidar': spaces.Box(0, 1, shape=(201,), dtype=np.float32),
            'previous_lidar': spaces.Box(0, 1, shape=(201,), dtype=np.float32),
            'previous_speed': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'previous_angle': spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        })
        self.speed = 0.0
        self.steer = 0.0
        self.previous_lidar = np.ones(201, dtype=np.float32)
        self.steps = 0
        self.max_episode_steps = int(max_episode_steps)

    def _spin_ros(self):
        # ROS context may be shut down by Ctrl+C before main() calls close().
        from rclpy.executors import ExternalShutdownException
        try:
            self._executor.spin()
        except (ExternalShutdownException, KeyboardInterrupt):
            pass
        except Exception as exc:
            if rclpy.ok():
                self.node.get_logger().error(f'ROS executor stopped: {exc!r}')

    def _scan_callback(self, msg: LaserScan):
        if not msg.ranges or not np.isfinite(msg.angle_increment) or msg.angle_increment == 0:
            return
        # Works even when LaserScan.angle_increment < 0 or angle_min != -pi.
        # Unlike modulo-by-2pi/index calculation, this correctly handles
        # clockwise scanners and partial fields of view.
        current, front_min, _stats = resample_scan(
            msg.ranges, msg.angle_min, msg.angle_increment,
            msg.range_min, self.MAX_RANGE,
        )
        with self._condition:
            self._ranges = current
            self._last_front_min = front_min
            self._scan_seq += 1
            self._condition.notify_all()

    def _await_scan(self, after_seq: int, count: int = 1, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._scan_seq < after_seq + count:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('Pas de /blue_car/scan (Webots est-il en marche ?)')
                self._condition.wait(timeout=left)
            return self._ranges.copy(), self._last_front_min

    def _command(self, speed: float, steer: float):
        msg = AckermannDrive()
        msg.speed = float(speed)
        msg.steering_angle = float(steer)
        if not rclpy.ok():
            return
        try:
            self.command_pub.publish(msg)
        except Exception as exc:
            # On Ctrl+C ROS sometimes destroys the context before our finally.
            if rclpy.ok():
                raise exc

    def _observation(self, current: np.ndarray):
        return {
            'current_lidar': current.astype(np.float32, copy=True),
            'previous_lidar': self.previous_lidar.astype(np.float32, copy=True),
            'previous_speed': np.array([self.speed / self.MAX_SPEED], dtype=np.float32),
            'previous_angle': np.array([self.steer / self.MAX_STEER], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._command(0.0, 0.0)
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Service /rl/reset absent. Vérifier le launch du superviseur RL.')
        future = self.reset_client.call_async(Trigger.Request())
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise TimeoutError('Service /rl/reset ne répond pas')
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('Reset refusé: ' + str(getattr(result, 'message', None)))
        self.speed = 0.0
        self.steer = 0.0
        self.steps = 0
        self._command(0.0, 0.0)
        with self._condition:
            seq = self._scan_seq
        current, front_min = self._await_scan(seq, count=2, timeout=5.0)
        self.previous_lidar = current.copy()
        return self._observation(current), {'front_min_m': front_min}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        assert action.shape == (2,), 'Action must have shape (2,)'
        # The paper uses increments: 0.5 m/s and 9deg per action.
        self.speed = float(np.clip(
            self.speed + 0.5 * float(action[0]), 0.1, self.MAX_SPEED
        ))
        self.steer = float(np.clip(
            self.steer + math.radians(9) * float(action[1]),
            -self.MAX_STEER, self.MAX_STEER
        ))
        with self._condition:
            old_lidar = None if self._ranges is None else self._ranges.copy()
            seq = self._scan_seq
        self._command(self.speed, self.steer)
        current, front_min = self._await_scan(seq, timeout=5.0)
        self.previous_lidar = old_lidar if old_lidar is not None else current.copy()
        self.steps += 1

        terminated = front_min < self.CRASH_DISTANCE
        truncated = self.steps >= self.max_episode_steps
        # Reward reproduced from the paper (p. 20), except front_min is the +/-40deg sector.
        reward = -300.0 if terminated else 12.0 * (front_min - 0.014) + 3.0 * self.speed
        obs = self._observation(current)
        if terminated or truncated:
            self._command(0.0, 0.0)
        info = {
            'front_min_m': float(front_min),
            'speed_cmd_m_s': float(self.speed),
            'steer_cmd_rad': float(self.steer),
            'episode_step': self.steps,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        if rclpy.ok():
            self._command(0.0, 0.0)
        self._executor.shutdown(timeout_sec=2.0)
        self._spin_thread.join(timeout=2.0)
        self.node.destroy_node()
