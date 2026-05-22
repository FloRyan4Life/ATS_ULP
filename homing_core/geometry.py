"""Geometric helpers for projecting landmarks into retinal features."""

import math
from typing import Tuple

from .models import Retina, RetinaFeature, World


def angle_difference(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def project_landmark(pos: Tuple[float, float], lm: dict) -> RetinaFeature:
    dx = lm["pos"][0] - pos[0]
    dy = lm["pos"][1] - pos[1]
    dist = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    ratio = min(1.0, max(-1.0, lm["radius"] / dist if dist > 1e-9 else 1.0))
    return RetinaFeature(theta, 2.0 * math.asin(ratio), True)


def build_retina(world: World, pos: Tuple[float, float]) -> Retina:
    retina = Retina(pos)
    landmarks = [project_landmark(pos, lm) for lm in world.landmarks]
    landmarks.sort(key=lambda feature: feature.center_angle)

    count = len(landmarks)
    for index in range(count):
        current_feature, next_feature = landmarks[index], landmarks[(index + 1) % count]
        right_edge = (current_feature.center_angle + current_feature.arc_length / 2.0) % (2.0 * math.pi)
        left_edge = (next_feature.center_angle - next_feature.arc_length / 2.0) % (2.0 * math.pi)
        gap_arc = (left_edge - right_edge) % (2.0 * math.pi)
        gap = RetinaFeature((right_edge + gap_arc / 2.0) % (2.0 * math.pi), max(0.0, gap_arc), False)
        retina.add_feature(current_feature)
        retina.add_feature(gap)

    return retina