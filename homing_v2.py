import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


# =============================================================================
# 1. Datenklassen & Welt
# =============================================================================

@dataclass
class RetinaFeature:
    center_angle: float   # Mittelpunkt des Bogens in rad [0, 2π)
    arc_length: float     # Winkelgröße des Bogens in rad
    is_landmark: bool     # True = Landmarke, False = Lücke


class Retina:
    def __init__(self, position: Tuple[float, float], features: List[RetinaFeature] = None):
        self.position = np.array(position, dtype=float)
        self.features: List[RetinaFeature] = features if features is not None else []
    
    def add_feature(self, feature: RetinaFeature):
        self.features.append(feature)


class World:
    def __init__(self):
        self.home: Tuple[float, float] = (0.0, 0.0)
        self.start: Tuple[float, float] = (6.0, 2.5)
        self.landmarks = [
            {"pos": (3.5, 2.0),  "radius": 0.5},
            {"pos": (3.5, -2.0), "radius": 0.5},
            {"pos": (0.0, -4.0), "radius": 0.5},
        ]
        self.home_retina: Optional[Retina] = None
        self.retinas: List[Retina] = []


# =============================================================================
# 2. Hilfsfunktionen & Kernalgorithmus (unverändert)
# =============================================================================

def angle_difference(angle_a: float, angle_b: float) -> float:
    diff = (angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi
    return diff


def project_single_landmark(robot_pos: Tuple[float, float], landmark: dict) -> RetinaFeature:
    px, py = robot_pos
    lx, ly = landmark["pos"]
    r = landmark["radius"]
    dx, dy = lx - px, ly - py
    dist = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    ratio = min(1.0, max(-1.0, (r / dist if dist > 1e-9 else 1.0)))
    arc = 2.0 * math.asin(ratio)
    return RetinaFeature(theta, arc, True)


def build_retina_for_position(world: World, position: Tuple[float, float]) -> Retina:
    retina = Retina(position)
    lm_features: List[RetinaFeature] = []
    for lm in world.landmarks:
        lm_features.append(project_single_landmark(position, lm))
    
    lm_features.sort(key=lambda f: f.center_angle)
    n = len(lm_features)
    
    for i in range(n):
        curr = lm_features[i]
        nxt = lm_features[(i + 1) % n]
        
        curr_right = (curr.center_angle + curr.arc_length / 2.0) % (2.0 * math.pi)
        nxt_left   = (nxt.center_angle - nxt.arc_length / 2.0) % (2.0 * math.pi)
        gap_arc    = (nxt_left - curr_right) % (2.0 * math.pi)
        gap_arc    = max(0.0, gap_arc)
        gap_center = (curr_right + gap_arc / 2.0) % (2.0 * math.pi)
        
        retina.add_feature(curr)
        retina.add_feature(RetinaFeature(gap_center, gap_arc, False))
    
    return retina

# =============================================================================
# PAIRING – NUR GLEICHE TYPEN (LM↔LM, Gap↔Gap)
# =============================================================================

def pair_single_snapshot_feature(snap_feature: RetinaFeature,
                                 retina_features: List[RetinaFeature]) -> Tuple[int, float, float]:
    """
    Matcht ein Snapshot-Feature zu seinem nächsten Nachbarn auf der Retina.
    
    WICHTIG: Es werden nur Features desselben Typs betrachtet!
      - Landmarke (is_landmark=True)  sucht nur unter Retina-Landmarken
      - Gap       (is_landmark=False) sucht nur unter Retina-Gaps
    """
    # Filter: Nur Kandidaten gleichen Typs
    candidates = [(j, f) for j, f in enumerate(retina_features)
                  if f.is_landmark == snap_feature.is_landmark]
    
    if not candidates:
        # Sollte bei korrekter Initialisierung nie passieren
        return 0, 0.0, 0.0
    
    best_idx, best_diff = candidates[0][0], float('inf')
    for j, ret_feat in candidates:
        diff = abs(angle_difference(snap_feature.center_angle, ret_feat.center_angle))
        if diff < best_diff:
            best_diff, best_idx = diff, j
    
    matched = retina_features[best_idx]
    ang_diff = angle_difference(matched.center_angle, snap_feature.center_angle)
    arc_diff = matched.arc_length - snap_feature.arc_length
    return best_idx, ang_diff, arc_diff



def compute_summed_vectors(snapshot: Retina, current: Retina):
    Vt = np.array([0.0, 0.0])
    Vp = np.array([0.0, 0.0])
    all_pairs = []
    
    for i, snap_feat in enumerate(snapshot.features):
        ret_idx, ang_diff, arc_diff = pair_single_snapshot_feature(snap_feat, current.features)
        all_pairs.append((i, ret_idx, ang_diff, arc_diff))
        
        theta = current.features[ret_idx].center_angle
        t = np.array([-math.sin(theta), math.cos(theta)])
        u = np.array([math.cos(theta), math.sin(theta)])
        
        Vt += ang_diff * t
        Vp += (-arc_diff) * u
    
    return Vt, Vp, all_pairs


def calculate_homing_vector(snapshot: Retina, current: Retina) -> np.ndarray:
    Vt, Vp, _ = compute_summed_vectors(snapshot, current)
    V = Vt + 3.0 * Vp
    norm = np.linalg.norm(V)
    return (V / norm if norm > 1e-10 else np.array([0.0, 0.0]))


# =============================================================================
# 3. INTERAKTIVE DIASHOW-KLASSE
# =============================================================================

class HomingDiashow:
    """
    Interaktive Schritt-für-Schritt-Visualisierung des Homing-Pfads.
    
    Steuerung:
        LEERTASTE  -> schaltet den nächsten Zustand frei
    
    Zustandszyklus pro Position:
        [0] Retinas      : Zeigt Home-Retina (klein, innen) + aktuelle Retina (groß, außen)
        [1] Pairing      : + blaue Linien zwischen gematchten Features
        [2] Vektor       : + schwarzen Homing-Pfeil
        [3] Bewegung     : Roboter bewegt sich, Zyklus startet neu an neuer Position
    """
    
    def __init__(self, world: World, snapshot: Retina, start_pos: Tuple[float, float],
                 step_size: float = 0.5, tolerance: float = 0.15):
        self.world = world
        self.snapshot = snapshot          # ← immer die HOME-Retina!
        self.path = [np.array(start_pos, dtype=float)]
        self.step_size = step_size
        self.tolerance = tolerance
        
        # Radien für die konzentrischen Retina-Ringe (im Welt-Koordinatensystem)
        self.r_snap = 0.35   # Home-Retina = halb so groß
        self.r_cur = 0.70    # Aktuelle Retina = doppelt so groß
        
        self.state = 0       # 0=Retinas, 1=Pairing, 2=Vektor, 3=Move
        self.titles = [
            "Retinas anzeigen (Home=innen, Aktuell=außen)",
            "Feature-Pairing (blaue Linien = Matches)",
            "Homing-Vektor (schwarzer Pfeil)",
            "Bewegung ausführen → nächster Schritt"
        ]
        
        # Figure vorbereiten
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self._draw_frame()
        plt.show()
    
    # -------------------------------------------------------------------------
    # Hilfs: Retina als farbige Kreissegmente zeichnen
    # -------------------------------------------------------------------------
    def _draw_retina_ring(self, center: np.ndarray, retina: Retina, radius: float):
        """
        Zeichnet jedes Feature als Wedge (Kreissegment).
        Rot = Landmarke, Grün = Gap.
        Für sehr kleine Bögen wird visuell ein Mindestwinkel von 2° erzwungen,
        damit man überhaupt etwas sieht (rein visuell, Berechnung bleibt exakt).
        """
        for feat in retina.features:
            c_deg = np.degrees(feat.center_angle)
            # Mindestwinkel für die Darstellung, damit winzige Features sichtbar sind
            half_vis = max(np.degrees(feat.arc_length / 2.0), 1.0)
            t1, t2 = c_deg - half_vis, c_deg + half_vis
            color = 'red' if feat.is_landmark else 'green'
            
            w = Wedge(center, radius, t1, t2,
                      facecolor=color, edgecolor='black', linewidth=1.2,
                      alpha=0.75, zorder=10)
            self.ax.add_patch(w)
    
    # -------------------------------------------------------------------------
    # Hilfs: Komplette Szene neu zeichnen
    # -------------------------------------------------------------------------
    def _draw_frame(self):
        self.ax.clear()
        
        # --- Statische Welt ---
        for lm in self.world.landmarks:
            c = plt.Circle(lm["pos"], lm["radius"], color='skyblue',
                           ec='navy', linewidth=1.5, zorder=5)
            self.ax.add_patch(c)
        
        # Home-Punkt (X) und Start-Markierung
        self.ax.plot(0, 0, 'kX', markersize=14, markeredgewidth=2.5, zorder=6, label='Home')
        if len(self.path) == 1:
            self.ax.plot(self.path[0][0], self.path[0][1], 'mo',
                         markersize=8, zorder=6, label='Start')
        
        # Bereits gelaufener Pfad
        if len(self.path) > 1:
            p_arr = np.array(self.path)
            self.ax.plot(p_arr[:,0], p_arr[:,1], 'r.-',
                         linewidth=2, markersize=6, zorder=4, label='Gelaufener Pfad')
        
        # --- Aktuelle Position ---
        pos = self.path[-1]
        
        # Aktuelle Retina berechnen (frisch für diesen Frame)
        current_retina = build_retina_for_position(self.world, tuple(pos))
        
        # --- Zustand 0,1,2: Retinas zeichnen ---
        if self.state in (0, 1, 2):
            # Home-Retina (Snapshot) = kleiner, innerer Ring
            self._draw_retina_ring(pos, self.snapshot, self.r_snap)
            # Aktuelle Retina = größerer, äußerer Ring
            self._draw_retina_ring(pos, current_retina, self.r_cur)
        
        # --- Zustand 1: Pairing-Linien ---
        if self.state == 1:
            _, _, pairs = compute_summed_vectors(self.snapshot, current_retina)
            for si, ri, _, _ in pairs:
                sf = self.snapshot.features[si]
                rf = current_retina.features[ri]
                
                # Punkt auf dem inneren Ring (Snapshot-Feature)
                x1 = pos[0] + self.r_snap * math.cos(sf.center_angle)
                y1 = pos[1] + self.r_snap * math.sin(sf.center_angle)
                # Punkt auf dem äußeren Ring (Retina-Feature)
                x2 = pos[0] + self.r_cur * math.cos(rf.center_angle)
                y2 = pos[1] + self.r_cur * math.sin(rf.center_angle)
                
                self.ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2,
                             alpha=0.7, zorder=11)
                # Kleine Punkte an den Enden der Linien
                self.ax.plot([x1, x2], [y1, y2], 'bo', markersize=3, zorder=12)
        
        # --- Zustand 2: Homing-Vektor ---
        if self.state == 2:
            vec = calculate_homing_vector(self.snapshot, current_retina)
            # Pfeil etwas länger zeichnen, damit man ihn gut sieht
            self.ax.quiver(pos[0], pos[1], vec[0], vec[1],
                           scale=6, color='black', width=0.009,
                           headwidth=5, headlength=6, zorder=15)
        
        # Achsen & Titel
        self.ax.set_xlim(-7.5, 7.5)
        self.ax.set_ylim(-7.5, 7.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        step_nr = len(self.path) - 1
        self.ax.set_title(f"Schritt {step_nr} | Zustand: {self.titles[self.state]}\n"
                          f"Position: ({pos[0]:.2f}, {pos[1]:.2f})  —  "
                          f"LEERTASTE für nächsten Zustand")
        self.ax.legend(loc='upper left', fontsize=9)
        
        self.fig.canvas.draw_idle()
    
    # -------------------------------------------------------------------------
    # Event-Handler: Leertaste schaltet weiter
    # -------------------------------------------------------------------------
    def on_key(self, event):
        if event.key != ' ':
            return
        
        # Zustand 3 = Bewegung ausführen
        if self.state == 3:
            pos = self.path[-1]
            home = np.array(self.world.home)
            dist = np.linalg.norm(pos - home)
            
            # Ziel erreicht?
            if dist < self.tolerance:
                self.ax.set_title("✅  ZIEL ERREICHT!  ✅", fontsize=16, color='darkgreen')
                self.fig.canvas.draw_idle()
                return
            
            # Homing-Vektor berechnen (immer gegen Snapshot!)
            cur_ret = build_retina_for_position(self.world, tuple(pos))
            vec = calculate_homing_vector(self.snapshot, cur_ret)
            
            # Adaptive Schrittweite: nie über Home hinausschießen
            adaptive = min(self.step_size, dist * 0.8)
            new_pos = pos + vec * adaptive
            self.path.append(new_pos)
            
            # Zyklus neu starten an neuer Position
            self.state = 0
        else:
            # Einfach zum nächsten Zustand schalten
            self.state += 1
        
        self._draw_frame()


# =============================================================================
# 4. Hauptprogramm
# =============================================================================

if __name__ == "__main__":
    # Welt aufbauen
    world = World()
    
    # Snapshot EINMALIG am Home-Punkt erstellen
    snapshot = build_retina_for_position(world, world.home)
    world.home_retina = snapshot
    world.retinas.append(snapshot)
    
    print("=" * 60)
    print("INTERAKTIVE HOMING-DIASHOW")
    print("=" * 60)
    print("Steuerung:")
    print("  LEERTASTE  = naechster Zustand (Retina -> Pairing -> Vektor -> Move)")
    print("Das Fenster muss den Fokus haben (hineinklicken).")
    print("=" * 60)
    
    # Diashow starten
    HomingDiashow(world, snapshot, world.start, step_size=0.4, tolerance=0.15)