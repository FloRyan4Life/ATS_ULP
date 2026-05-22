"""Matplotlib visualization for the homing model."""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Wedge

from .pairing import compute_vectors, homing_vector
from .geometry import build_retina
from .models import Retina, World


class HomingDiashow:
    """Interactive step-by-step visualization of the homing process."""

    def __init__(self, world: World, snapshot: Retina, start, step_size=0.4, tolerance=0.15):
        self.world = world
        self.snapshot = snapshot
        self.path = [np.array(start, dtype=float)]
        self.step_size = step_size
        self.tolerance = tolerance

        self.r_snap = 0.50
        self.r_cur = 1.00

        self.state = 0
        self.labels = ["Retinas", "Pairing", "Komponenten", "Vektor", "Bewegung"]

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._draw()
        plt.show()

    def _draw_ring(self, center, retina: Retina, radius: float, hollow=False):
        for feature in retina.features:
            angle_deg = np.degrees(feature.center_angle)
            half_width = max(np.degrees(feature.arc_length / 2.0), 1.0)
            color = "red" if feature.is_landmark else "green"

            if hollow:
                arc = Arc(center, 2 * radius, 2 * radius, angle=0, theta1=angle_deg - half_width, theta2=angle_deg + half_width, color=color, linewidth=10, zorder=10)
                self.ax.add_patch(arc)
            else:
                wedge = Wedge(center, radius, angle_deg - half_width, angle_deg + half_width, facecolor=color, edgecolor="black", linewidth=1, alpha=0.8, zorder=10)
                self.ax.add_patch(wedge)

    def _draw(self):
        self.ax.clear()
        position = self.path[-1]
        current = build_retina(self.world, tuple(position))

        for landmark in self.world.landmarks:
            self.ax.add_patch(plt.Circle(landmark["pos"], landmark["radius"], color="skyblue", ec="navy", zorder=5))
        self.ax.plot(0, 0, "kX", markersize=12, zorder=6, label="Home")
        if len(self.path) > 1:
            path = np.array(self.path)
            self.ax.plot(path[:, 0], path[:, 1], "r.-", linewidth=2, zorder=4, label="Pfad")
        else:
            self.ax.plot(position[0], position[1], "mo", markersize=8, zorder=6, label="Start")

        if self.state in (0, 1, 2, 3):
            self._draw_ring(position, self.snapshot, self.r_snap, hollow=False)
            self._draw_ring(position, current, self.r_cur, hollow=True)

        if self.state == 1:
            _, _, pairs = compute_vectors(self.snapshot, current)
            for pair in pairs:
                snapshot_feature = self.snapshot.features[pair.snapshot_idx]
                retina_feature = current.features[pair.retina_idx]
                color = "red" if snapshot_feature.is_landmark else "green"
                x1 = position[0] + self.r_snap * math.cos(snapshot_feature.center_angle)
                y1 = position[1] + self.r_snap * math.sin(snapshot_feature.center_angle)
                x2 = position[0] + self.r_cur * math.cos(retina_feature.center_angle)
                y2 = position[1] + self.r_cur * math.sin(retina_feature.center_angle)
                self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.5, alpha=0.8, zorder=11)
                self.ax.plot([x1, x2], [y1, y2], "o", color=color, markersize=4, zorder=12)

        if self.state == 2:
            turn_total, approach_total, pairs = compute_vectors(self.snapshot, current)

            for pair in pairs:
                retina_feature = current.features[pair.retina_idx]
                theta = retina_feature.center_angle
                start_x = position[0] + self.r_cur * math.cos(theta)
                start_y = position[1] + self.r_cur * math.sin(theta)

                tangential = np.array([-math.sin(theta), math.cos(theta)])
                radial = np.array([math.cos(theta), math.sin(theta)])

                turn_vector = pair.angle_diff * tangential
                approach_vector = (-pair.arc_len_diff) * radial

                self.ax.quiver(start_x, start_y, turn_vector[0], turn_vector[1], scale=3, color="purple", width=0.006, headwidth=4, headlength=5, zorder=15, alpha=0.9)
                self.ax.quiver(start_x, start_y, approach_vector[0], approach_vector[1], scale=3, color="orange", width=0.006, headwidth=4, headlength=5, zorder=15, alpha=0.9)

            self.ax.quiver(position[0], position[1], turn_total[0], turn_total[1], scale=3, color="deepskyblue", width=0.01, headwidth=5, headlength=6, zorder=16, alpha=0.55, label="Vt (Turn, Summe)")
            self.ax.quiver(position[0], position[1], approach_total[0], approach_total[1], scale=3, color="saddlebrown", width=0.01, headwidth=5, headlength=6, zorder=16, alpha=0.55, label="Vp (Approach, Summe)")

        if self.state == 3:
            vector = homing_vector(self.snapshot, current)
            self.ax.quiver(position[0], position[1], vector[0], vector[1], scale=5, color="black", width=0.008, zorder=15)

        self.ax.set_xlim(-7.5, 7.5)
        self.ax.set_ylim(-7.5, 7.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.set_title(f"Schritt {len(self.path)-1} | {self.labels[self.state]} | Pos: ({position[0]:.2f}, {position[1]:.2f}) — LEERTASTE")
        self.ax.legend(loc="upper left")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key != " ":
            return

        if self.state == 4:
            position = self.path[-1]
            home = np.array(self.world.home)
            distance = np.linalg.norm(position - home)
            if distance < self.tolerance:
                self.ax.set_title("ZIEL ERREICHT", fontsize=16, color="green")
                self.fig.canvas.draw_idle()
                return

            current = build_retina(self.world, tuple(position))
            vector = homing_vector(self.snapshot, current)
            step = min(self.step_size, distance * 0.8)
            self.path.append(position + vector * step)
            self.state = 0
        else:
            self.state += 1

        self._draw()