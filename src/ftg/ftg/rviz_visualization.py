"""Visualisation RViz des directions libres et interdites."""

from geometry_msgs.msg import Point
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray


def make_point(x, y, z=0.04):
    """Construit un point 3D pour RViz."""
    return Point(x=float(x), y=float(y), z=float(z))


def make_marker(header, marker_id, marker_type, color, scale):
    """Construit un marqueur RViz avec une configuration commune."""
    marker = Marker()

    marker.header = header
    marker.ns = 'ftg_debug'
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD

    marker.pose.orientation.w = 1.0

    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0

    # Effacement automatique si le flux s'interrompt.
    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 500_000_000

    return marker


def create_gap_markers(header, angles, blocked_mask, gap_centers, horizon_m=1.2, target_angle=None
                       ):
    """Crée les marqueurs RViz correspondant au résultat du FTG."""
    angles = np.asarray(angles, dtype=float)
    blocked_mask = np.asarray(blocked_mask, dtype=bool)

    if angles.shape != blocked_mask.shape or angles.ndim != 1:
        raise ValueError('Angles et masque incompatibles')

    if not np.isfinite(horizon_m) or horizon_m <= 0:
        raise ValueError('Horizon invalide')

    # Directions interdites : rouge.
    blocked_marker = make_marker(
        header,
        marker_id=0,
        marker_type=Marker.POINTS,
        color=(1.0, 0.1, 0.1),
        scale=0.045,
    )

    # Directions libres : vert.
    free_marker = make_marker(
        header,
        marker_id=1,
        marker_type=Marker.POINTS,
        color=(0.1, 1.0, 0.1),
        scale=0.045,
    )

    # Centres des gaps : lignes cyan.
    gaps_marker = make_marker(
        header,
        marker_id=2,
        marker_type=Marker.LINE_LIST,
        color=(0.0, 0.8, 1.0),
        scale=0.015,
    )

    # Direction cible sélectionnée : jaune.
    target_marker = make_marker(
        header,
        marker_id=3,
        marker_type=Marker.LINE_LIST,
        color=(1.0, 0.9, 0.0),
        scale=0.035,
    )

    # Arc représentant les directions du LiDAR.
    for angle, blocked in zip(angles, blocked_mask):
        x = horizon_m * np.cos(angle)
        y = horizon_m * np.sin(angle)

        point = make_point(x, y)

        if blocked:
            blocked_marker.points.append(point)
        else:
            free_marker.points.append(point)

    # Une ligne pour chaque centre de gap.
    for angle in gap_centers:
        gaps_marker.points.append(make_point(0.0, 0.0))

        gaps_marker.points.append(
            make_point(
                horizon_m * np.cos(angle),
                horizon_m * np.sin(angle),
            )
        )

    if target_angle is not None:
        target_marker.points.append(make_point(0.0, 0.0))
        target_marker.points.append(
            make_point(
                horizon_m * np.cos(target_angle),
                horizon_m * np.sin(target_angle),
            )
        )

    result = MarkerArray()
    result.markers = [
        blocked_marker,
        free_marker,
        gaps_marker,
        target_marker,
    ]

    return result
