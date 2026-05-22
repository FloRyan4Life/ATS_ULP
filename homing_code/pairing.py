"""Feature matching and homing vector computation."""

import math
import warnings
from typing import List, Tuple

import numpy as np

from .geometry import angle_difference
from .models import FeaturePair, Retina, RetinaFeature


def match_feature(snapshot_idx: int, snap: RetinaFeature, retinas: List[RetinaFeature]) -> FeaturePair:
    candidates = [(index, feature) for index, feature in enumerate(retinas) if feature.is_landmark == snap.is_landmark]
    if not candidates:
        return FeaturePair(snapshot_idx, 0, 0.0, 0.0)

    best_index, best_diff = candidates[0][0], float("inf")
    for index, feature in candidates:
        diff = abs(angle_difference(snap.center_angle, feature.center_angle))
        if diff < best_diff:
            best_diff, best_index = diff, index

    matched = retinas[best_index]
    angle_diff = angle_difference(matched.center_angle, snap.center_angle)
    arc_len_diff = matched.arc_length - snap.arc_length
    return FeaturePair(snapshot_idx, best_index, angle_diff, arc_len_diff)


def compute_vectors(snapshot: Retina, current: Retina):
    """Compute turn and approach vector sums together with the matched pairs."""
    Vt, Vp = np.zeros(2), np.zeros(2)
    pairs = []

    snapshot_landmarks = sum(1 for feature in snapshot.features if feature.is_landmark)
    current_landmarks = sum(1 for feature in current.features if feature.is_landmark)
    snapshot_gaps = len(snapshot.features) - snapshot_landmarks
    current_gaps = len(current.features) - current_landmarks
    if snapshot_landmarks != current_landmarks or snapshot_gaps != current_gaps:
        warnings.warn(
            "Feature count mismatch between snapshot and current retina: "
            f"snapshot landmarks/gaps = {snapshot_landmarks}/{snapshot_gaps}, "
            f"current landmarks/gaps = {current_landmarks}/{current_gaps}. "
            "Pairing continues, but results may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    for index, feature in enumerate(snapshot.features):
        pair = match_feature(index, feature, current.features)
        pairs.append(pair)

        theta = current.features[pair.retina_idx].center_angle
        tangential = np.array([-math.sin(theta), math.cos(theta)])
        radial = np.array([math.cos(theta), math.sin(theta)])

        Vt += pair.angle_diff * tangential
        Vp += (-pair.arc_len_diff) * radial

    return Vt, Vp, pairs


def homing_vector(snapshot: Retina, current: Retina) -> np.ndarray:
    Vt, Vp, _ = compute_vectors(snapshot, current)
    vector = Vt + 3.0 * Vp
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-10 else np.zeros(2)