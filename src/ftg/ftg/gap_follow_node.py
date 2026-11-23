"""Nœud ROS 2 de diagnostic et visualisation Follow the Gap."""

import time


from ftg.gap_algorithm import (
    compute_safety_mask,
    find_free_gaps,
    select_target_direction,
)
from ftg.motion_control import compute_steering
from ftg.rviz_visualization import create_gap_markers
from ftg.scan_processing import preprocess_scan

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray


class GapFollowNode(Node):
    """Analyse les scans LiDAR et publie les résultats dans RViz."""

    def __init__(self):
        super().__init__('ftg_node')

        # Paramètres ROS 2
        self.declare_parameter('scan_topic', '/yellow_car/scan')
        self.declare_parameter('front_half_fov_deg', 100.0)
        self.declare_parameter('safety_radius_m', 0.20)
        self.declare_parameter('horizon_m', 1.20)
        self.declare_parameter('min_gap_width_deg', 4.0)
        self.declare_parameter('target_min_gap_width_deg', 12.0)
        self.declare_parameter('target_edge_margin_deg', 5.0)

        # Géométrie du véhicule
        self.declare_parameter('wheelbase_m', 0.257)
        self.declare_parameter('lidar_offset_x_m', 0.21)
        self.declare_parameter('steering_lookahead_m', 1.20)
        self.declare_parameter('max_steering_rad', 0.35)

        scan_topic = self.get_parameter('scan_topic').value

        # Publisher RViz
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/ftg/debug/markers',
            10,
        )

        # Subscriber LiDAR
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.scan_count = 0
        self.last_log_time = 0.0

        self.get_logger().info('Nœud FTG initialisé')
        self.get_logger().info(f'LiDAR : {scan_topic}')
        self.get_logger().info('Visualisation : /ftg/debug/markers')

    def scan_callback(self, msg):
        """Prétraite un scan et calcule les directions libres."""
        self.scan_count += 1

        # Lecture à chaque scan : paramètres modifiables à chaud.
        half_fov = float(self.get_parameter('front_half_fov_deg').value)
        safety_radius = float(self.get_parameter('safety_radius_m').value)
        horizon = float(self.get_parameter('horizon_m').value)
        min_gap_width = float(self.get_parameter('min_gap_width_deg').value)

        try:
            # 1. Prétraitement LiDAR
            scan = preprocess_scan(
                angle_min=msg.angle_min,
                angle_increment=msg.angle_increment,
                ranges=msg.ranges,
                range_min=msg.range_min,
                range_max=msg.range_max,
                half_fov_deg=half_fov,
            )

            # 2. Calcul des directions interdites
            blocked_mask = compute_safety_mask(
                angles=scan.angles,
                distances=scan.distances,
                hit_mask=scan.hit_mask,
                safety_radius_m=safety_radius,
                horizon_m=horizon,
            )

            # 3. Détection des gaps
            gaps = find_free_gaps(blocked_mask)

            # 4. Sélection de la direction cible
            target = select_target_direction(
                scan.angles,
                blocked_mask,
                min_gap_width_deg=float(self.get_parameter('target_min_gap_width_deg').value),
                edge_margin_deg=float(self.get_parameter('target_edge_margin_deg').value),
            )

            # 5. Calcul du braquage théorique
            steering = None

            if target is not None:
                steering = compute_steering(
                    target_angle_rad=target.angle_rad,
                    wheelbase_m=float(self.get_parameter('wheelbase_m').value),
                    lidar_offset_x_m=float(self.get_parameter('lidar_offset_x_m').value),
                    lookahead_m=float(self.get_parameter('steering_lookahead_m').value),
                    max_steering_rad=float(self.get_parameter('max_steering_rad').value),
                )

        except ValueError as error:
            self.get_logger().error(f'Erreur de traitement LiDAR : {error}')
            return

        # 4. Calcul de la direction centrale des gaps
        gap_centers = []
        gap_info = []

        angular_step = abs(float(msg.angle_increment))

        for gap in gaps:
            start_angle = scan.angles[gap.start]
            end_angle = scan.angles[gap.stop - 1]

            center_angle = (start_angle + end_angle) / 2.0

            width_rad = end_angle - start_angle + angular_step
            width_deg = np.rad2deg(width_rad)

            gap_info.append((float(center_angle), float(width_deg)))

            if width_deg >= min_gap_width:
                gap_centers.append(float(center_angle))

        # 5. Visualisation RViz
        markers = create_gap_markers(
            header=msg.header,
            angles=scan.angles,
            blocked_mask=blocked_mask,
            gap_centers=gap_centers,
            horizon_m=horizon,
            target_angle=None if target is None else target.angle_rad,
        )

        self.marker_publisher.publish(markers)

        # 6. Diagnostics à environ 1 Hz
        now = time.monotonic()

        if now - self.last_log_time >= 1.0:
            self.last_log_time = now

            n_free = int(np.count_nonzero(~blocked_mask))
            n_blocked = int(np.count_nonzero(blocked_mask))

            self.get_logger().info(
                f'Scan #{self.scan_count} | '
                f'Libres : {n_free} | '
                f'Interdites : {n_blocked} | '
                f'Gaps détectés : {len(gaps)}'
            )

            if target is None:
                self.get_logger().warning('Aucune direction cible admissible')
            else:
                self.get_logger().info(f'Cible FTG : {np.rad2deg(target.angle_rad):.1f}°')

            if steering is not None:
                self.get_logger().info(
                    f'Braquage calculé : '
                    f'{np.rad2deg(steering.steering_angle_rad):.2f}° | '
                    f'Courbure : {steering.curvature_m_inv:.3f} m^-1 | '
                    f'Saturation : {steering.saturated}'
                )

            # Affichage des trois gaps les plus larges
            largest_gaps = sorted(
                gap_info,
                key=lambda item: item[1],
                reverse=True,
            )[:3]

            for i, (center, width) in enumerate(largest_gaps, start=1):
                self.get_logger().info(
                    f'Gap {i} : centre = {np.rad2deg(center):.1f}° | largeur = {width:.1f}°'
                )


def main(args=None):
    rclpy.init(args=args)

    node = GapFollowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
