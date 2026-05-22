import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, NamedTuple
import matplotlib.pyplot as plt
"""Compatibility wrapper for the homing demo."""

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


if __name__ == "__main__":
    run_demo()
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title("Homing-Vektorfeld")
    ax2.legend(loc='upper left')
    fig2.canvas.draw()
    fig2.show()
    
    # Fenster 2: Diashow (blockierend)
    plt.ioff()
    print("=" * 50)
    print("Zwei Fenster geoffnet:")
    print("  [1] Vektorfeld (statisch)")
    print("  [2] Diashow (interaktiv — hier klicken + LEERTASTE)")
    print("=" * 50)
    
    diashow = HomingDiashow(world, snapshot, world.start, step_size, tolerance)