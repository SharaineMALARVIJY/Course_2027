"""Contrôleur simple des voitures compagnons CoVAPSy (Webots).

Reprend le principe du contrôleur violet : différence des deux mesures
LiDAR brutes 240 et 120, sans conversion mètres -> millimètres.

Les distances sont en mètres, le braquage en radians et la consigne
Driver.setCruisingSpeed() est en km/h.
"""

import math

from controller import Lidar
from vehicle import Driver

# Paramètres faciles à ajuster
LIDAR_NAME = "RpLidarA2"
RAY_LEFT = 240     # Indice issu du contrôleur violet (orientation à valider)
RAY_RIGHT = 120
STEERING_GAIN = 1.0  # rad/m
MAX_STEERING = 0.28  # rad (~16°)
CRUISE_SPEED_KMH = 1.8  # équivaut à 0,5 m/s, comme le jaune original
MAX_VALID_RANGE_M = 20.0
PRINT_INTERVAL_S = 1.0


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def valid_range(value):
    return math.isfinite(value) and 0.0 < value < MAX_VALID_RANGE_M


def main():
    driver = Driver()
    timestep_ms = int(driver.getBasicTimeStep())
    lidar = Lidar(LIDAR_NAME)
    lidar.enable(4 * timestep_ms)

    driver.setCruisingSpeed(0.0)
    driver.setSteeringAngle(0.0)

    last_print = -PRINT_INTERVAL_S
    print(f"[{driver.getName()}] Contrôleur jaune démarré, LiDAR {lidar.getHorizontalResolution()} rayons")

    while driver.step() != -1:
        ranges = lidar.getRangeImage()  # distances brutes en mètres

        if not ranges or len(ranges) <= max(RAY_LEFT, RAY_RIGHT):
            driver.setCruisingSpeed(0.0)
            driver.setSteeringAngle(0.0)
            continue

        left_m = float(ranges[RAY_LEFT])
        right_m = float(ranges[RAY_RIGHT])
        valid = valid_range(left_m) and valid_range(right_m)

        if valid:
            # Même formule et signe que le contrôleur violet original
            steering_rad = clamp(
                STEERING_GAIN * (left_m - right_m),
                -MAX_STEERING,
                MAX_STEERING,
            )
            target_speed_kmh = CRUISE_SPEED_KMH
        else:
            # Sécurité de diagnostic : ne pas rouler avec une entrée invalide
            steering_rad = 0.0
            target_speed_kmh = 0.0

        driver.setSteeringAngle(steering_rad)
        driver.setCruisingSpeed(target_speed_kmh)

        now = driver.getTime()
        if now - last_print >= PRINT_INTERVAL_S:
            last_print = now


if __name__ == "__main__":
    main()
