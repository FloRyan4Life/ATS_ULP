"""Executable demo entrypoint for the homing package."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from homing_code.geometry import build_retina
    from homing_code.models import World
    from homing_code.pairing import homing_vector
    from homing_code.visualization import HomingDiashow
else:
    from .geometry import build_retina
    from .models import World
    from .pairing import homing_vector
    from .visualization import HomingDiashow


def _precompute_path(world: World, snapshot, step_size: float, tolerance: float, max_steps: int = 50):
    path = [np.array(world.start, dtype=float)]
    for _ in range(max_steps):
        position = path[-1]
        distance = np.linalg.norm(position - np.array(world.home))
        if distance < tolerance:
            break
        current = build_retina(world, tuple(position))
        vector = homing_vector(snapshot, current)
        path.append(position + vector * min(step_size, distance * 0.8))
    return path


def _build_vector_field(world: World, snapshot):
    x_values = np.arange(-7, 8)
    y_values = np.arange(-7, 8)
    X, Y = np.meshgrid(x_values, y_values)
    U, V = np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)

    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            px, py = float(X[row, col]), float(Y[row, col])
            if abs(px) < 0.01 and abs(py) < 0.01:
                continue
            retina = build_retina(world, (px, py))
            vector = homing_vector(snapshot, retina)
            U[row, col], V[row, col] = vector[0], vector[1]

    return X, Y, U, V


def run_demo():
    world = World()
    snapshot = build_retina(world, world.home)
    world.home_retina = snapshot

    step_size, tolerance = 0.4, 0.15
    path = _precompute_path(world, snapshot, step_size, tolerance)

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    X, Y, U, V = _build_vector_field(world, snapshot)
    ax.quiver(X, Y, U, V, scale=25, width=0.003, color="black")

    for landmark in world.landmarks:
        ax.add_patch(plt.Circle(landmark["pos"], landmark["radius"], color="skyblue", ec="navy", zorder=5))

    ax.plot(0, 0, "kX", markersize=12, zorder=6, label="Home")
    ax.plot(world.start[0], world.start[1], "mo", markersize=8, zorder=6, label="Start")

    path_array = np.array(path)
    ax.plot(path_array[:, 0], path_array[:, 1], "r-", linewidth=2.5, zorder=4, label="Suchpfad")
    ax.plot(path_array[:, 0], path_array[:, 1], "r.", markersize=5, zorder=4)

    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Homing-Vektorfeld")
    ax.legend(loc="upper left")
    fig.canvas.draw()
    fig.show()

    plt.ioff()
    print("=" * 50)
    print("Zwei Fenster geoffnet:")
    print("  [1] Vektorfeld (statisch)")
    print("  [2] Diashow (interaktiv — hier klicken + LEERTASTE)")
    print("=" * 50)

    HomingDiashow(world, snapshot, world.start, step_size, tolerance)


if __name__ == "__main__":
    run_demo()