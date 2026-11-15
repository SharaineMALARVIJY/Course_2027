#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from ackermann_msgs.msg import AckermannDrive


class FollowGap3DNode(Node):
    

    def __init__(self):
        super().__init__('follow_gap_3d')

        # Topics
        self.declare_parameter('points_topic', '/yellow_car/rgbd/points')
        self.declare_parameter('ackermann_topic', '/yellow_car/cmd_ackermann')

        # Filtrage 3D (repère camera_optical_frame : x=droite, y=bas, z=avant)
        self.declare_parameter('min_height_y', -0.25)        # Coupe au-dessus de la voiture (m)
        self.declare_parameter('max_height_y', 0.11)         # Coupe le sol (m)
        self.declare_parameter('min_range_m', 0.15)          # Bruit proche (m)
        self.declare_parameter('max_range_m', 7.0)           # Portée étendue pour lignes droites (m)

        # Paramètres géométriques et Disparity Extender
        self.declare_parameter('num_bins', 180)
        self.declare_parameter('fov_deg', 86.0)
        self.declare_parameter('car_half_width_m', 0.15)     # Demi-largeur physique TT-02 + marge (m)
        self.declare_parameter('disparity_threshold_m', 0.45)# Saut de profondeur déclenchant l'extension
        self.declare_parameter('bubble_radius_bins', 12)     # Bulle autour du point le plus proche
        self.declare_parameter('free_threshold_m', 1.25)     # Seuil d'espace libre

        # Dynamique et vitesse (Itération 2)
        self.declare_parameter('max_speed', 3.0)             # Vitesse pointe en ligne droite (m/s)
        self.declare_parameter('corner_speed', 1.15)         # Vitesse en virage normal (m/s)
        self.declare_parameter('u_turn_speed', 0.85)         # Vitesse de dégagement dans les U serrés (m/s)
        self.declare_parameter('max_steering_deg', 16.0)     # Sous la butée mécanique Webots (0.28 rad)

        points_topic = self.get_parameter('points_topic').value
        ackermann_topic = self.get_parameter('ackermann_topic').value

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.sub_points = self.create_subscription(
            PointCloud2, points_topic, self.points_callback, qos
        )
        self.pub_cmd = self.create_publisher(AckermannDrive, ackermann_topic, 1)

        self.get_logger().info(
            f'FollowTheGap3D (Itération 2) prêt | Max speed: {self.get_parameter("max_speed").value} m/s'
        )

    def points_callback(self, msg: PointCloud2):
        # 1. Lecture rapide du nuage PointCloud2
        fields = {f.name: f for f in msg.fields}
        if any(name not in fields or fields[name].datatype != PointField.FLOAT32 for name in ('x', 'y', 'z')):
            return

        dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': ['<f4', '<f4', '<f4'],
            'offsets': [fields['x'].offset, fields['y'].offset, fields['z'].offset],
            'itemsize': msg.point_step,
        })

        raw_points = np.frombuffer(msg.data, dtype=dtype)
        x = raw_points['x']
        y = raw_points['y']
        z = raw_points['z']

        # 2. Filtrage 3D : suppression du sol, du plafond et des points hors portée
        max_range = self.get_parameter('max_range_m').value
        min_range = self.get_parameter('min_range_m').value
        min_y = self.get_parameter('min_height_y').value
        max_y = self.get_parameter('max_height_y').value

        valid_mask = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            & (z > min_range) & (z < max_range)
            & (y > min_y) & (y < max_y)
        )

        x_valid = x[valid_mask]
        z_valid = z[valid_mask]

        num_bins = self.get_parameter('num_bins').value
        fov_rad = math.radians(self.get_parameter('fov_deg').value)
        half_fov = fov_rad / 2.0
        angle_per_bin = fov_rad / num_bins

        ranges = np.full(num_bins, max_range, dtype=np.float32)

        if len(x_valid) > 0:
            angles = np.arctan2(-x_valid, z_valid)
            distances = np.hypot(x_valid, z_valid)

            in_fov = (angles >= -half_fov) & (angles <= half_fov)
            angles = angles[in_fov]
            distances = distances[in_fov]

            bin_indices = ((half_fov - angles) / fov_rad * (num_bins - 1)).astype(np.int32)
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            np.minimum.at(ranges, bin_indices, distances)

        # Sauvegarde du profil brut avant masquage pour le diagnostic des virages en U
        raw_ranges = ranges.copy()
        front_clearance = float(np.mean(raw_ranges[num_bins // 2 - 10 : num_bins // 2 + 10]))

        # 3. Disparity Extender : élargit virtuellement les bords de murs de la demi-largeur de la voiture
        ranges = self._apply_disparity_extender(ranges, angle_per_bin)

        # 4. Bulle de sécurité autour du point le plus proche
        closest_idx = int(np.argmin(ranges))
        bubble_r = self.get_parameter('bubble_radius_bins').value
        start_b = max(0, closest_idx - bubble_r)
        end_b = min(num_bins - 1, closest_idx + bubble_r)
        ranges[start_b:end_b + 1] = 0.0

        # 5. Recherche du plus grand espace libre (Gap)
        free_thresh = self.get_parameter('free_threshold_m').value
        free_mask = ranges > free_thresh
        gap_start, gap_end = self._find_max_gap(free_mask)

        max_steer = math.radians(self.get_parameter('max_steering_deg').value)

        # 6. Gestion des virages en U serrés (aucun point > free_threshold_m dans le champ de vision)
        if gap_end <= gap_start:
            left_space = float(np.mean(raw_ranges[:num_bins // 2]))
            right_space = float(np.mean(raw_ranges[num_bins // 2:]))
            # Braquage maximum vers le côté qui offre le plus de profondeur
            steering_angle = max_steer if left_space >= right_space else -max_steer
            speed = self.get_parameter('u_turn_speed').value
            self._publish_ackermann(speed, steering_angle)
            return

        # 7. Visée hybride (70% point le plus profond + 30% centre du gap pour éviter de raser le mur)
        gap_slice = ranges[gap_start:gap_end + 1]
        deepest_idx = gap_start + int(np.argmax(gap_slice))
        center_gap_idx = (gap_start + gap_end) / 2.0
        target_idx = 0.7 * deepest_idx + 0.3 * center_gap_idx

        target_angle = half_fov - (target_idx / (num_bins - 1)) * fov_rad
        steering_angle = float(np.clip(target_angle, -max_steer, max_steer))

        # 8. Calcul de la vitesse dynamique (combine angle de braquage ET dégagement frontal)
        max_speed = self.get_parameter('max_speed').value
        corner_speed = self.get_parameter('corner_speed').value

        steer_ratio = abs(steering_angle) / max_steer if max_steer > 0 else 0.0
        # Facteur de dégagement : 1.0 si voie libre > 4.5m devant, diminue à l'approche d'un mur
        clearance_ratio = float(np.clip((front_clearance - 1.0) / 3.5, 0.0, 1.0))

        speed_from_steer = max_speed - (steer_ratio ** 1.2) * (max_speed - corner_speed)
        speed_from_dist = corner_speed + clearance_ratio * (max_speed - corner_speed)
        speed = float(min(speed_from_steer, speed_from_dist))

        self._publish_ackermann(speed, steering_angle)

    def _apply_disparity_extender(self, ranges: np.ndarray, angle_per_bin: float) -> np.ndarray:
        """Étend chaque discontinuité de profondeur de la demi-largeur de la voiture."""
        extended = ranges.copy()
        disp_thresh = self.get_parameter('disparity_threshold_m').value
        half_width = self.get_parameter('car_half_width_m').value
        n = len(ranges)

        for i in range(n - 1):
            diff = ranges[i + 1] - ranges[i]
            if abs(diff) > disp_thresh:
                near_dist = min(ranges[i], ranges[i + 1])
                if near_dist <= 0.05:
                    continue
                # Nombre de bins correspondant à la demi-largeur physique à cette distance
                arc_angle = math.atan2(half_width, near_dist)
                extend_bins = int(math.ceil(arc_angle / angle_per_bin))

                if diff > 0:
                    # Obstacle à gauche (i), vide à droite (i+1) -> étendre vers la droite
                    end = min(n, i + 1 + extend_bins)
                    extended[i + 1 : end] = np.minimum(extended[i + 1 : end], near_dist)
                else:
                    # Vide à gauche (i), obstacle à droite (i+1) -> étendre vers la gauche
                    start = max(0, i + 1 - extend_bins)
                    extended[start : i + 1] = np.minimum(extended[start : i + 1], near_dist)

        return extended

    def _find_max_gap(self, mask: np.ndarray):
        max_len = 0
        best_start = 0
        best_end = 0
        current_start = None

        for i, is_free in enumerate(mask):
            if is_free and current_start is None:
                current_start = i
            elif not is_free and current_start is not None:
                length = i - current_start
                if length > max_len:
                    max_len = length
                    best_start = current_start
                    best_end = i - 1
                current_start = None

        if current_start is not None:
            length = len(mask) - current_start
            if length > max_len:
                best_start = current_start
                best_end = len(mask) - 1

        return best_start, best_end

    def _publish_ackermann(self, speed: float, steering_angle: float):
        msg = AckermannDrive()
        msg.speed = float(speed)
        msg.steering_angle = float(steering_angle)
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FollowGap3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_ackermann(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()