"""Webots ROS 2 plugin: reset the blue car without restarting the entire world."""

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
            self.node.get_logger().error('DEF RL_CAR introuvable dans le world !')
        self.service = self.node.create_service(Trigger, '/rl/reset', self._reset)
        self.node.get_logger().info('Service /rl/reset disponible')

    def _reset(self, request, response):
        if self.car is None:
            response.success = False
            response.message = 'No DEF RL_CAR in world'
            return response
        # Position initiale de la blue_car dans le dépôt Course_2027.
        self.car.getField('translation').setSFVec3f([-1.0, -2.0, 0.04])
        self.car.getField('rotation').setSFRotation([0.0, 0.0, 1.0, 0.0])
        self.car.resetPhysics()
        response.success = True
        response.message = 'Blue car repositioned'
        return response

    def step(self):
        # Le driver Webots exécute périodiquement cette fonction.
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def stop(self):
        if self.node is not None:
            self.node.destroy_node()
