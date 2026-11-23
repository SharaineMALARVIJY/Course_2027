import time

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from tf2_ros import Buffer, TransformListener, TransformException


class PointCloudTest(Node):

    def __init__(self):
        super().__init__('pointcloud_test')

        self.declare_parameter(
            'cloud_topic',
            '/yellow_car/camera/colored_points'
        )

        topic = self.get_parameter('cloud_topic').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            topic,
            self.callback,
            qos_profile_sensor_data
        )

        self.last_print = 0.0

        self.get_logger().info(f'Listening to: {topic}')

    def callback(self, msg):

        # Affichage à 1 Hz
        now = time.monotonic()

        if now - self.last_print < 1.0:
            return

        self.last_print = now

        # Lecture des coordonnées XYZ
        points = point_cloud2.read_points_numpy(
            msg,
            field_names=['x', 'y', 'z']
        )

        # Sous-échantillonnage pour alléger le calcul
        points = points[::8]

        # Supprimer les points invalides
        points = points[np.isfinite(points).all(axis=1)]

        if len(points) == 0:
            self.get_logger().info('No valid points')
            return

        # Transformation vers camera_link
        if msg.header.frame_id != 'camera_link':

            try:
                tf = self.tf_buffer.lookup_transform(
                    'camera_link',
                    msg.header.frame_id,
                    Time()
                )

            except TransformException as e:
                self.get_logger().warning(f'TF: {e}')
                return

            q = tf.transform.rotation
            t = tf.transform.translation

            # Rotation quaternion (normalisé)
            qv = np.array([q.x, q.y, q.z], dtype=float)
            norm = np.sqrt(np.dot(qv, qv) + q.w**2)
            qv /= norm
            w = q.w / norm

            points = (
                points
                + 2 * np.cross(
                    qv,
                    w * points + np.cross(qv, points)
                )
                + np.array([t.x, t.y, t.z])
            )

        # Zone devant la caméra
        mask = (
            (points[:, 0] > 0.2) &
            (points[:, 0] < 4.0) &
            (np.abs(points[:, 1]) < 0.25) &
            (points[:, 2] > -0.03) &
            (points[:, 2] < 0.30)
        )

        front_points = points[mask]

        if len(front_points) == 0:
            self.get_logger().info('No obstacle detected')
            return

        # Distance robuste aux points isolés aberrants
        distance = np.percentile(front_points[:, 0], 10)

        self.get_logger().info(
            f'Distance: {distance:.2f} m '
            f'| Points: {len(front_points)}'
        )


def main():
    rclpy.init()
    node = PointCloudTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()  