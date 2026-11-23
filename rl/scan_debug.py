"""Print compact LaserScan diagnostics; run while Webots is playing."""
import math
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from lidar_utils import resample_scan


def main():
    rclpy.init()
    node = rclpy.create_node('rl_lidar_debug')
    count = 0

    def callback(msg):
        nonlocal count
        if count >= 5:
            return
        count += 1
        obs, front, stats = resample_scan(
            msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min)
        print(f'SCAN {count}: N={stats["raw_total"]}, min_angle={math.degrees(msg.angle_min):.1f}deg, '
              f'increment={math.degrees(msg.angle_increment):.2f}deg, '
              f'finite_raw={stats["raw_finite"]}/{stats["raw_total"]}, '
              f'covered={stats["covered_201"]}/201, valid_front={stats["valid_front"]}/81, '
              f'front_min={front:.3f}m; sectors={stats["sample_m"]}', flush=True)

    sub = node.create_subscription(LaserScan, '/blue_car/scan', callback, qos_profile_sensor_data)
    deadline = time.monotonic() + 12.0
    try:
        while rclpy.ok() and count < 5 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        if count == 0:
            print('Aucun scan recu : verifier que Webots est en mode Play, pas Pause')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
