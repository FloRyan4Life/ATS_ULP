import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc


# =============================================================================
# Datenklassen
# =============================================================================

@dataclass
class RetinaFeature:
    center_angle: float   # rad [0, 2π)
    arc_length: float     # rad
    is_landmark: bool     # True = Landmarke, False = Gap


class Retina:
    def __init__(self, position: Tuple[float, float], features: List[RetinaFeature] = None):
        self.position = np.array(position, dtype=float)
        self.features: List[RetinaFeature] = features if features is not None else []
    
    def add_feature(self, feature: RetinaFeature):
        self.features.append(feature)


class World:
    def __init__(self):
        self.home = (0.0, 0.0)
        self.start = (-3.0, -3.0)
        self.landmarks = [
            {"pos": (3.5, 2.0),  "radius": 0.5},
            {"pos": (3.5, -2.0), "radius": 0.5},
            {"pos": (0.0, -4.0), "radius": 0.5},
        ]
        self.home_retina: Optional[Retina] = None
        self.retinas: List[Retina] = []


# =============================================================================
# Trigonometrie-Hilfsfunktionen
# =============================================================================

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
    lms = [project_landmark(pos, lm) for lm in world.landmarks]
    lms.sort(key=lambda f: f.center_angle)
    
    n = len(lms)
    for i in range(n):
        curr, nxt = lms[i], lms[(i + 1) % n]
        r = (curr.center_angle + curr.arc_length / 2.0) % (2.0 * math.pi)
        l = (nxt.center_angle - nxt.arc_length / 2.0) % (2.0 * math.pi)
        gap_arc = (l - r) % (2.0 * math.pi)
        gap = RetinaFeature((r + gap_arc / 2.0) % (2.0 * math.pi), max(0.0, gap_arc), False)
        retina.add_feature(curr)
        retina.add_feature(gap)
    
    return retina


# =============================================================================
# Matching & Vektorberechnung (nur gleiche Typen)
# =============================================================================

def match_feature(snap: RetinaFeature, retinas: List[RetinaFeature]) -> Tuple[int, float, float]:
    candidates = [(j, f) for j, f in enumerate(retinas) if f.is_landmark == snap.is_landmark]
    if not candidates:
        return 0, 0.0, 0.0
    
    best_idx, best_diff = candidates[0][0], float('inf')
    for j, f in candidates:
        diff = abs(angle_difference(snap.center_angle, f.center_angle))
        if diff < best_diff:
            best_diff, best_idx = diff, j
    
    m = retinas[best_idx]
    return best_idx, angle_difference(m.center_angle, snap.center_angle), m.arc_length - snap.arc_length


def compute_vectors(snapshot: Retina, current: Retina):
    Vt, Vp = np.zeros(2), np.zeros(2)
    pairs = []
    
    for i, sf in enumerate(snapshot.features):
        ri, ad, ard = match_feature(sf, current.features)
        pairs.append((i, ri, ad, ard))
        th = current.features[ri].center_angle
        Vt += ad * np.array([-math.sin(th), math.cos(th)])
        Vp += (-ard) * np.array([math.cos(th), math.sin(th)])
    
    return Vt, Vp, pairs


def homing_vector(snapshot: Retina, current: Retina) -> np.ndarray:
    Vt, Vp, _ = compute_vectors(snapshot, current)
    V = Vt + 3.0 * Vp
    n = np.linalg.norm(V)
    return V / n if n > 1e-10 else np.zeros(2)


# =============================================================================
# Visualisierung 1: Interaktive Diashow
# =============================================================================

class HomingDiashow:
    """
    Interaktive Schritt-für-Schritt-Anzeige.
    Leertaste schaltet durch: Retinas → Pairing → Vektor → Bewegung
    """
    
    def __init__(self, world: World, snapshot: Retina, start: Tuple[float, float],
                 step_size=0.4, tolerance=0.15):
        self.world = world
        self.snapshot = snapshot
        self.path = [np.array(start, dtype=float)]
        self.step_size = step_size
        self.tolerance = tolerance
        
        # Retina-Radien
        self.r_snap = 0.50   # innerer Ring (Home-Retina, gefüllt)
        self.r_cur = 1.00    # äußerer Ring (aktuelle Retina, hohl)
        
        self.state = 0
        self.labels = ["Retinas", "Pairing", "Vektor", "Bewegung"]
        
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self._draw()
        plt.show()
    
    def _draw_ring(self, center, retina: Retina, radius: float, hollow=False):
        """Zeichnet Features als Kreissegmente. Rot=LM, Grün=Gap."""
        for f in retina.features:
            c = np.degrees(f.center_angle)
            h = max(np.degrees(f.arc_length / 2.0), 1.0)
            color = 'red' if f.is_landmark else 'green'
            
            if hollow:
                # Äußerer Ring: nur Linienbogen (keine Füllung)
                arc = Arc(center, 2*radius, 2*radius,
                          angle=0, theta1=c-h, theta2=c+h,
                          color=color, linewidth=10, zorder=10)
                self.ax.add_patch(arc)
            else:
                # Innerer Ring: gefüllte Sektoren
                w = Wedge(center, radius, c-h, c+h,
                          facecolor=color, edgecolor='black', linewidth=1,
                          alpha=0.8, zorder=10)
                self.ax.add_patch(w)
    
    def _draw(self):
        self.ax.clear()
        pos = self.path[-1]
        cur = build_retina(self.world, tuple(pos))
        
        # Welt
        for lm in self.world.landmarks:
            self.ax.add_patch(plt.Circle(lm["pos"], lm["radius"],
                           color='skyblue', ec='navy', zorder=5))
        self.ax.plot(0, 0, 'kX', markersize=12, zorder=6, label='Home')
        if len(self.path) > 1:
            p = np.array(self.path)
            self.ax.plot(p[:,0], p[:,1], 'r.-', linewidth=2, zorder=4, label='Pfad')
        else:
            self.ax.plot(pos[0], pos[1], 'mo', markersize=8, zorder=6, label='Start')
        
        # Zustände 0-2: Retinas
        if self.state in (0, 1, 2):
            self._draw_ring(pos, self.snapshot, self.r_snap, hollow=False)
            self._draw_ring(pos, cur, self.r_cur, hollow=True)
        
        # Zustand 1: Pairing-Linien
        if self.state == 1:
            _, _, pairs = compute_vectors(self.snapshot, cur)
            for si, ri, _, _ in pairs:
                sf, rf = self.snapshot.features[si], cur.features[ri]
                c = 'red' if sf.is_landmark else 'green'
                x1 = pos[0] + self.r_snap * math.cos(sf.center_angle)
                y1 = pos[1] + self.r_snap * math.sin(sf.center_angle)
                x2 = pos[0] + self.r_cur * math.cos(rf.center_angle)
                y2 = pos[1] + self.r_cur * math.sin(rf.center_angle)
                self.ax.plot([x1, x2], [y1, y2], color=c, linewidth=2.5, alpha=0.8, zorder=11)
                self.ax.plot([x1, x2], [y1, y2], 'o', color=c, markersize=4, zorder=12)
        
        # Zustand 2: Homing-Vektor
        if self.state == 2:
            v = homing_vector(self.snapshot, cur)
            self.ax.quiver(pos[0], pos[1], v[0], v[1],
                           scale=5, color='black', width=0.008, zorder=15)
        
        # Layout
        self.ax.set_xlim(-7.5, 7.5)
        self.ax.set_ylim(-7.5, 7.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_title(f"Schritt {len(self.path)-1} | {self.labels[self.state]} | "
                          f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}) — LEERTASTE")
        self.ax.legend(loc='upper left')
        self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        if event.key != ' ':
            return
        
        if self.state == 3:
            pos = self.path[-1]
            home = np.array(self.world.home)
            d = np.linalg.norm(pos - home)
            if d < self.tolerance:
                self.ax.set_title("ZIEL ERREICHT", fontsize=16, color='green')
                self.fig.canvas.draw_idle()
                return
            
            cur = build_retina(self.world, tuple(pos))
            v = homing_vector(self.snapshot, cur)
            step = min(self.step_size, d * 0.8)
            self.path.append(pos + v * step)
            self.state = 0
        else:
            self.state += 1
        
        self._draw()


# =============================================================================
# Hauptprogramm: Beide Fenster gleichzeitig
# =============================================================================

if __name__ == "__main__":
    world = World()
    snapshot = build_retina(world, world.home)
    world.home_retina = snapshot
    
    # Pfad vorberechnen
    path = [np.array(world.start, dtype=float)]
    step_size, tolerance = 0.4, 0.15
    
    for _ in range(50):
        pos = path[-1]
        d = np.linalg.norm(pos - np.array(world.home))
        if d < tolerance:
            break
        cur = build_retina(world, tuple(pos))
        v = homing_vector(snapshot, cur)
        path.append(pos + v * min(step_size, d * 0.8))
    
    # Fenster 1: Vektorfeld (nicht-blockierend)
    plt.ion()
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    
    xr = np.arange(-7, 8)
    yr = np.arange(-7, 8)
    X, Y = np.meshgrid(xr, yr)
    U, V = np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            px, py = float(X[i,j]), float(Y[i,j])
            if abs(px) < 0.01 and abs(py) < 0.01:
                continue
            r = build_retina(world, (px, py))
            v = homing_vector(snapshot, r)
            U[i,j], V[i,j] = v[0], v[1]
    
    ax2.quiver(X, Y, U, V, scale=25, width=0.003, color='black')
    
    for lm in world.landmarks:
        ax2.add_patch(plt.Circle(lm["pos"], lm["radius"],
                                 color='skyblue', ec='navy', zorder=5))
    
    ax2.plot(0, 0, 'kX', markersize=12, zorder=6, label='Home')
    ax2.plot(world.start[0], world.start[1], 'mo', markersize=8, zorder=6, label='Start')
    
    p = np.array(path)
    ax2.plot(p[:,0], p[:,1], 'r-', linewidth=2.5, zorder=4, label='Suchpfad')
    ax2.plot(p[:,0], p[:,1], 'r.', markersize=5, zorder=4)
    
    ax2.set_xlim(-7.5, 7.5)
    ax2.set_ylim(-7.5, 7.5)
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