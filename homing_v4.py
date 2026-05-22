"""Compatibility wrapper for the homing demo.

The implementation lives in `homing_core`. This file preserves the legacy
entry point so existing commands like `python homing_v4.py` continue to work.
"""

from homing_core import (
    FeaturePair,
    RetinaFeature,
    Retina,
    World,
    angle_difference,
    project_landmark,
    build_retina,
    match_feature,
    compute_vectors,
    homing_vector,
    HomingDiashow,
)
from homing_core.main import run_demo


__all__ = [
    "FeaturePair",
    "RetinaFeature",
    "Retina",
    "World",
    "angle_difference",
    "project_landmark",
    "build_retina",
    "match_feature",
    "compute_vectors",
    "homing_vector",
    "HomingDiashow",
    "run_demo",
]


if __name__ == "__main__":
    run_demo()