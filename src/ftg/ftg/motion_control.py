"""Calcul géométrique de la direction pour la voiture F110."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SteeringResult:
    """Résultat du calcul de direction."""

    steering_angle_rad: float
    raw_steering_angle_rad: float
    curvature_m_inv: float
    target_x_m: float
    target_y_m: float
    saturated: bool


def compute_steering(
    target_angle_rad: float,
    *,
    wheelbase_m: float = 0.257,
    lidar_offset_x_m: float = 0.21,
    lookahead_m: float = 1.20,
    max_steering_rad: float = 0.35,
) -> SteeringResult:
    """
    Compute the steering angle using a Pure Pursuit bicycle model.

    Express the target relative to the rear axle.
    Use positive steering angles for left turns.
    """
    parameters = (
        target_angle_rad,
        wheelbase_m,
        lidar_offset_x_m,
        lookahead_m,
        max_steering_rad,
    )

    if not all(math.isfinite(value) for value in parameters):
        raise ValueError('Les paramètres doivent être finis')

    if wheelbase_m <= 0:
        raise ValueError('Empattement invalide')

    if lookahead_m <= 0:
        raise ValueError('Distance de visée invalide')

    if not (0 < max_steering_rad < math.pi / 2):
        raise ValueError('Limite de braquage invalide')

    if abs(target_angle_rad) > math.pi:
        raise ValueError('Angle cible hors de [-pi, pi]')

    # 1. Point cible dans le repère de l'essieu arrière.
    target_x = lidar_offset_x_m + lookahead_m * math.cos(target_angle_rad)

    target_y = lookahead_m * math.sin(target_angle_rad)

    # Pure Pursuit suppose un point cible devant l'essieu arrière.
    if target_x <= 0:
        raise ValueError("Point cible derrière l'essieu arrière")

    # 2. Distance au carré entre l'essieu arrière et la cible.
    distance_squared = target_x**2 + target_y**2

    # 3. Courbure géométrique.
    curvature = 2.0 * target_y / distance_squared

    # 4. Angle de braquage demandé.
    raw_steering = math.atan(wheelbase_m * curvature)

    # 5. Saturation de la commande.
    steering = max(
        -max_steering_rad,
        min(max_steering_rad, raw_steering),
    )

    return SteeringResult(
        steering_angle_rad=steering,
        raw_steering_angle_rad=raw_steering,
        curvature_m_inv=curvature,
        target_x_m=target_x,
        target_y_m=target_y,
        saturated=abs(raw_steering) > max_steering_rad,
    )
