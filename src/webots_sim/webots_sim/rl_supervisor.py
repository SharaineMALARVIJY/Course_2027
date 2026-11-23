"""Webots ROS2 supervisor for Course_2027 (blue car).

Maintains existing /rl/reset (fixed blue-car spawn) and adds /rl/reset_random
for reference-2023 PPO training. Spawn samples are a narrow region near the
known working blue-car start on the Piste_CoVAPSy_2025a track; other source
2023 track coordinates MUST NOT be copied directly to this different world.
"""
import random

import rclpy
from std_srvs.srv import Trigger


class RLSupervisor:
    def init(self, webots_node, properties):
        self.robot = webots_node.robot
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node('rl_reset_supervisor')
        self.car = self.robot.getFromDef('RL_CAR')
        if self.car is None:
            self.node.get_logger().error('DEF RL_CAR missing in Webots world')
        self.fixed_service = self.node.create_service(Trigger, '/rl/reset', self._reset_fixed)
        self.random_service = self.node.create_service(Trigger, '/rl/reset_random', self._reset_random)
        self.node.get_logger().info('Services /rl/reset and /rl/reset_random available')

    def _place(self, x, y, yaw, response):
        if self.car is None:
            response.success = False
            response.message = 'No DEF RL_CAR in world'
            return response
        self.car.getField('translation').setSFVec3f([x, y, 0.04])
        self.car.getField('rotation').setSFRotation([0.0, 0.0, 1.0, yaw])
        self.car.resetPhysics()
        response.success = True
        response.message = f'Blue car repositioned: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}'
        return response

    def _reset_fixed(self, request, response):
        return self._place(-1.0, -2.0, 0.0, response)

    def _reset_random(self, request, response):
        # Intentionally constrained first-pass spawn distribution, rather than
        # adapting the 2023 supervisor's 7 spawn zones to an unrelated track.
        x = random.uniform(-1.25, -0.85)
        y = random.uniform(-2.035, -1.965)
        yaw = random.uniform(-0.08, 0.08)
        return self._place(x, y, yaw, response)

    def step(self):
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def stop(self):
        if self.node is not None:
            self.node.destroy_node()
