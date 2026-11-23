from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gap:
    """Intervalle d'indices [start, stop[ dans un scan trié par angle."""

    start: int
    stop: int


def find_closest_obstacle(distances, hit_mask):
    """Renvoie l'indice du retour valide le plus proche, ou None."""
    distances = np.asarray(distances, dtype=float)
    hit_mask = np.asarray(hit_mask, dtype=bool)

    if distances.ndim != 1 or distances.shape != hit_mask.shape:
        raise ValueError('Distances et masque de retours incompatibles')

    valid = np.flatnonzero(hit_mask & np.isfinite(distances))
    if valid.size == 0:
        return None

    return int(valid[np.argmin(distances[valid])])


def compute_safety_mask(angles, distances, hit_mask, *, safety_radius_m=0.20, horizon_m=1.20):
    """
    Interdit les rayons proches d'obstacles dans l'horizon local.

    Les rayons sans retour valide restent interdits.
    Les angles doivent être triés strictement croissants en radians.
    """
    angles = np.asarray(angles, dtype=float)
    distances = np.asarray(distances, dtype=float)
    hit_mask = np.asarray(hit_mask, dtype=bool)

    if (
        angles.ndim != 1
        or angles.size == 0
        or angles.shape != distances.shape
        or angles.shape != hit_mask.shape
    ):
        raise ValueError('Angles, distances et masque doivent avoir même taille 1D non vide')

    if not np.all(np.isfinite(angles)) or not np.all(np.diff(angles) > 0):
        raise ValueError('Angles non finis ou non strictement croissants')

    if (
        not np.isfinite(safety_radius_m)
        or safety_radius_m <= 0
        or not np.isfinite(horizon_m)
        or horizon_m <= 0
    ):
        raise ValueError('Rayon de sécurité et horizon doivent être positifs')

    if np.any(hit_mask & (~np.isfinite(distances) | (distances <= 0))):
        raise ValueError('Retours déclarés valides mais distances invalides')

    # À ce stade, mesure inconnue = direction interdite
    blocked = ~hit_mask.copy()

    # On considère les obstacles susceptibles d'interférer avec l'horizon de navigation
    nearby = np.flatnonzero(hit_mask & (distances <= horizon_m + safety_radius_m))

    for idx in nearby:
        d = distances[idx]

        # Obstacle déjà à l'intérieur du rayon de sécurité
        if d <= safety_radius_m:
            return np.ones_like(blocked)

        # Ouverture angulaire de la zone de sécurité
        half_angle = np.arcsin(safety_radius_m / d)

        start = np.searchsorted(angles, angles[idx] - half_angle, side='left')

        stop = np.searchsorted(angles, angles[idx] + half_angle, side='right')

        blocked[start:stop] = True

    return blocked


def find_free_gaps(blocked_mask):
    """Renvoie les suites continues libres, sous forme [start, stop[."""
    blocked = np.asarray(blocked_mask, dtype=bool)

    if blocked.ndim != 1:
        raise ValueError('blocked_mask doit être un tableau 1D')

    gaps = []
    start = None

    for i, is_blocked in enumerate(blocked):
        if not is_blocked and start is None:
            start = i

        elif is_blocked and start is not None:
            gaps.append(Gap(start, i))
            start = None

    if start is not None:
        gaps.append(Gap(start, blocked.size))

    return gaps


@dataclass(frozen=True)
class GapTarget:
    """Direction cible sélectionnée dans un gap."""

    gap: Gap
    index: int
    angle_rad: float
    width_rad: float


def select_target_direction(angles, blocked_mask, *, min_gap_width_deg=12.0, edge_margin_deg=5.0):
    """
    Select a target direction within an admissible gap.

    Prioritize forward motion while maintaining a margin
    from the boundaries of each free gap.
    """
    angles = np.asarray(angles, dtype=float)
    blocked = np.asarray(blocked_mask, dtype=bool)

    if (
        angles.ndim != 1
        or angles.shape != blocked.shape
        or angles.size < 2
        or not np.all(np.isfinite(angles))
        or not np.all(np.diff(angles) > 0)
    ):
        raise ValueError('Angles ou masque invalides')

    if (
        not np.isfinite(min_gap_width_deg)
        or min_gap_width_deg <= 0
        or not np.isfinite(edge_margin_deg)
        or edge_margin_deg < 0
    ):
        raise ValueError('Paramètres de sélection invalides')

    min_width = np.deg2rad(min_gap_width_deg)
    margin = np.deg2rad(edge_margin_deg)

    resolution = float(np.median(np.diff(angles)))

    candidates = []

    for gap in find_free_gaps(blocked):
        width = float(angles[gap.stop - 1] - angles[gap.start] + resolution)

        if width < min_width:
            continue

        # Les directions proches des limites du gap sont exclues.
        interior = np.arange(gap.start, gap.stop)

        interior = interior[
            (angles[interior] >= angles[gap.start] + margin - 1e-12)
            & (angles[interior] <= angles[gap.stop - 1] - margin + 1e-12)
        ]

        if interior.size == 0:
            continue

        # Direction admissible la plus proche de l'avant (0 rad).
        idx = int(interior[np.argmin(np.abs(angles[interior]))])

        candidates.append(
            GapTarget(
                gap=gap,
                index=idx,
                angle_rad=float(angles[idx]),
                width_rad=width,
            )
        )

    if not candidates:
        return None

    # Priorité à l'alignement avec l'avant.
    # À égalité, on préfère le gap le plus large.
    return min(
        candidates,
        key=lambda c: (abs(c.angle_rad), -c.width_rad),
    )
