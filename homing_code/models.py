"""Data models used by the homing implementation."""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np


class FeaturePair(NamedTuple):
    """Matched snapshot/current feature pair and its differences."""

    snapshot_idx: int
    retina_idx: int
    angle_diff: float
    arc_len_diff: float


@dataclass
class RetinaFeature:
    center_angle: float
    arc_length: float
    is_landmark: bool


class Retina:
    def __init__(self, position: Tuple[float, float], features: List[RetinaFeature] = None):
        self.position = np.array(position, dtype=float)
        self.features: List[RetinaFeature] = features if features is not None else []

    def add_feature(self, feature: RetinaFeature):
        self.features.append(feature)


class World:
    def __init__(self):
        self.home = (0.0, 0.0)
        self.start = (-3.0, -1.0)
        self.landmarks = [
            {"pos": (3.5, 2.0), "radius": 0.5},
            {"pos": (3.5, -2.0), "radius": 0.5},
            {"pos": (0.0, -4.0), "radius": 0.5},
        ]
        self.home_retina: Optional[Retina] = None
        self.retinas: List[Retina] = []