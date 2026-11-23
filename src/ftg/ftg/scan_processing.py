from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProcessedScan:
    angles: np.ndarray
    distances: np.ndarray
    hit_mask: np.ndarray
    no_return_mask: np.ndarray
    invalid_mask: np.ndarray


def preprocess_scan(
    angle_min: float,
    angle_increment: float,
    ranges,
    range_min: float,
    range_max: float,
    half_fov_deg: float = 100.0,
) -> ProcessedScan:
    """Réordonne les faisceaux par angle et conserve un secteur frontal."""
    distances = np.asarray(ranges, dtype=float)

    if distances.ndim != 1 or distances.size == 0:
        raise ValueError('ranges doit être un tableau 1D non vide')

    if not np.isfinite(angle_min) or not np.isfinite(angle_increment) or angle_increment == 0:
        raise ValueError('Géométrie angulaire invalide')

    if (
        not np.isfinite(range_min)
        or not np.isfinite(range_max)
        or not (0 <= range_min < range_max)
    ):
        raise ValueError('Bornes de distance invalides')

    if not np.isfinite(half_fov_deg) or not (0 < half_fov_deg <= 180):
        raise ValueError('Champ de vision frontal invalide')

    # Reconstruction des angles
    angles = angle_min + np.arange(distances.size) * angle_increment

    # Normalisation dans [-pi, pi[
    angles = (angles + np.pi) % (2 * np.pi) - np.pi

    # Tri des mesures par angle croissant
    order = np.argsort(angles)
    angles = angles[order]
    distances = distances[order]

    # Sélection du secteur frontal
    front = np.abs(angles) <= np.deg2rad(half_fov_deg)

    angles = angles[front]
    distances = distances[front]

    # Classification des mesures
    hit_mask = np.isfinite(distances) & (distances >= range_min) & (distances <= range_max)

    no_return_mask = np.isposinf(distances)

    invalid_mask = ~(hit_mask | no_return_mask)

    return ProcessedScan(
        angles=angles,
        distances=distances,
        hit_mask=hit_mask,
        no_return_mask=no_return_mask,
        invalid_mask=invalid_mask,
    )
