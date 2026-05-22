import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# =============================================================================
# 1. RetinaFeature class
# =============================================================================
@dataclass
class RetinaFeature:
    """Ein einzelnes Feature (Landmarke oder Gap) auf der kreisförmigen Retina."""
    center_angle: float   # O_i: Mittelpunkt des Bogens in rad [0, 2π)
    arc_length: float     # L_i: Winkelgröße des Bogens in rad
    is_landmark: bool     # True = Landmarke, False = Lücke


# =============================================================================
# 2. Retina class
# =============================================================================
class Retina:
    """Repräsentiert die kreisförmige Retina an einer bestimmten Position."""
    def __init__(self, position: Tuple[float, float], features: List[RetinaFeature] = None):
        self.position = np.array(position, dtype=float)
        self.features: List[RetinaFeature] = features if features is not None else []
    
    def add_feature(self, feature: RetinaFeature):
        self.features.append(feature)


# =============================================================================
# 3. Datenstruktur für Liste mehrerer Retinas & Spielfeld-Definition
# =============================================================================
class World:
    """Definiert das Spielfeld, die Landmarken, Home und speichert alle Retinas."""
    def __init__(self):
        # Home-Punkt
        self.home: Tuple[float, float] = (0.0, 0.0)
        
        # Startpunkt (Beispiel, kann geändert werden)
        self.start: Tuple[float, float] = (-3.0, -1.0)
        
        # Liste der Landmarken (Position und Radius)
        # Anzahl ist hier frei wählbar – das System arbeitet mit beliebig vielen
        self.landmarks = [
            {"pos": (3.5, 2.0),  "radius": 0.5},
            {"pos": (3.5, -2.0), "radius": 0.5},
            {"pos": (0.0, -4.0), "radius": 0.5},
        ]
        
        # Retina des Home-Punktes (wird später initialisiert)
        self.home_retina: Optional[Retina] = None
        
        # Datenstruktur: Liste aller jemals berechneten Retinas
        self.retinas: List[Retina] = []


# =============================================================================
# Hilfsfunktion: Zyklische Winkeldifferenz
# =============================================================================
def angle_difference(angle_a: float, angle_b: float) -> float:
    """
    Berechnet die signed Differenz (a - b) im Bereich [-π, +π].
    Wichtig für korrektes Matching über die 0°-Grenze hinweg.
    """
    diff = (angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi
    return diff


# =============================================================================
# 5. Funktion für Projektion EINER Landmarke auf die Retina
# =============================================================================
def project_single_landmark(
    robot_pos: Tuple[float, float],
    landmark: dict
) -> RetinaFeature:
    """
    Projiziert eine einzelne Landmarke auf den Panoramakreis (Retina).
    Berechnet Kompasswinkel (O) und Winkelgröße (L) via 2*asin(r/d).
    """
    px, py = robot_pos
    lx, ly = landmark["pos"]
    r = landmark["radius"]
    
    dx = lx - px
    dy = ly - py
    dist = math.hypot(dx, dy)
    
    # Kompasswinkel zur Landmarke (0 = +x, gegen Uhrzeigersinn)
    theta = math.atan2(dy, dx)
    
    # Winkelgröße (Bogenlänge) der Landmarke auf der Retina
    if dist < 1e-9:
        ratio = 1.0
    else:
        ratio = r / dist
    ratio = min(1.0, max(-1.0, ratio))  # Numerische Sicherheit für asin
    arc = 2.0 * math.asin(ratio)
    
    return RetinaFeature(theta, arc, True)


# =============================================================================
# 6. Oberfunktion: Projektion ALLER Landmarken + Gaps für eine Position
# =============================================================================
def build_retina_for_position(
    world: World,
    position: Tuple[float, float]
) -> Retina:
    """
    Oberfunktion, die für eine Roboterposition:
      1. Alle Landmarken projiziert (Aufruf von project_single_landmark)
      2. Die Gap-Features dazwischen berechnet (unter Berücksichtigung der Landmarken-Breite)
      3. Eine vollständige Retina (LM, Gap, LM, Gap, ...) zurückgibt
    """
    retina = Retina(position)
    
    # --- Projektion aller Landmarken (Aufruf der Einzelfunktion) ---
    lm_features: List[RetinaFeature] = []
    for lm in world.landmarks:
        feat = project_single_landmark(position, lm)
        lm_features.append(feat)
    
    # --- Sortieren nach Winkel, damit Reihenfolge festliegt ---
    lm_features.sort(key=lambda f: f.center_angle)
    
    # --- Berechnung der Gap-Features zwischen den Landmarken ---
    n = len(lm_features)
    for i in range(n):
        curr = lm_features[i]
        nxt = lm_features[(i + 1) % n]  # Zyklisch: letzter Gap schließt an ersten an
        
        # Ränder der Landmarken (in mathematischer Winkelrichtung = gegen Uhrzeigersinn)
        # curr reicht von (curr.O - curr.L/2) bis (curr.O + curr.L/2)
        curr_right_edge = (curr.center_angle + curr.arc_length / 2.0) % (2.0 * math.pi)
        nxt_left_edge  = (nxt.center_angle - nxt.arc_length / 2.0) % (2.0 * math.pi)
        
        # Winkelabstand vom rechten Rand von curr zum linken Rand von nxt
        # (positiv = gegen Uhrzeigersinn)
        gap_arc = (nxt_left_edge - curr_right_edge) % (2.0 * math.pi)
        gap_arc = max(0.0, gap_arc)  # Sicherheitshalber
        
        # Mitte des Gaps liegt exakt in der Mitte zwischen den beiden Rändern
        gap_center = (curr_right_edge + gap_arc / 2.0) % (2.0 * math.pi)
        
        # Reihenfolge: Landmarke, dann Gap
        retina.add_feature(curr)
        retina.add_feature(RetinaFeature(gap_center, gap_arc, False))
    
    return retina

# =============================================================================
# 7. Funktion für Pairing von Features (mehrfaches Matching erlaubt)
# =============================================================================
def pair_single_snapshot_feature(
    snap_feature: RetinaFeature,
    retina_features: List[RetinaFeature]
) -> Tuple[int, float, float]:
    """
    Matcht EIN Snapshot-Feature zu seinem nächsten Retina-Feature.
    Gibt zurück: (retina_index, winkeldifferenz, bogenlaengen_differenz)
    """
    best_idx = 0
    best_diff = float('inf')
    
    for j, ret_feat in enumerate(retina_features):
        diff = abs(angle_difference(snap_feature.center_angle, ret_feat.center_angle))
        if diff < best_diff:
            best_diff = diff
            best_idx = j
    
    matched_ret = retina_features[best_idx]
    
    # 8. Winkeldifferenz des Pairs
    ang_diff = angle_difference(matched_ret.center_angle, snap_feature.center_angle)
    
    # 9. Bogenlängen-Differenz des Pairs (Retina - Snapshot)
    arc_diff = matched_ret.arc_length - snap_feature.arc_length
    
    return best_idx, ang_diff, arc_diff


# =============================================================================
# 10. Oberfunktion: Mehrfaches Pairing, Diff-Berechnung, Aufsummierung
# =============================================================================
def compute_summed_vectors(
    snapshot: Retina,
    current: Retina
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float, float]]]:
    """
    Ruft für jedes Snapshot-Feature die Pairing-Funktion auf.
    Summiert alle Turn-Vektoren (Vt) und Approach-Vektoren (Vp).
    Gibt zurück: (Vt, Vp, list_of_pairs)
    """
    Vt = np.array([0.0, 0.0])  # Turn-Komponente
    Vp = np.array([0.0, 0.0])  # Approach-Komponente
    all_pairs: List[Tuple[int, int, float, float]] = []  # (snap_idx, ret_idx, ang, arc)
    
    for i, snap_feat in enumerate(snapshot.features):
        ret_idx, ang_diff, arc_diff = pair_single_snapshot_feature(
            snap_feat, current.features
        )
        all_pairs.append((i, ret_idx, ang_diff, arc_diff))
        
        theta = current.features[ret_idx].center_angle
        
        # Turn-Vektor: tangential zur Sichtlinie (-sin, cos)
        t = np.array([-math.sin(theta), math.cos(theta)])
        Vt += ang_diff * t
        
        # Approach-Vektor: radial zur Sichtlinie (cos, sin)
        # Wenn arc_diff < 0 (Retina kleiner -> zu weit weg -> + radial)
        u = np.array([math.cos(theta), math.sin(theta)])
        Vp += (-arc_diff) * u  # = (L_snap - L_ret) * u
    
    return Vt, Vp, all_pairs


# =============================================================================
# 11. Oberoberfunktion: Berechnung einer Retina -> Homing-Vektor
# =============================================================================
def calculate_homing_vector(
    snapshot: Retina,
    current: Retina
) -> np.ndarray:
    """
    Komplettberechnung für eine Retina:
      - Extrahiert Features (muss vorher geschehen, Retina übergeben)
      - Ruft Pairing & Summierung auf
      - Kombiniert Vt + 3*Vp zu finalen normierten Homing-Vektor
    """
    Vt, Vp, _ = compute_summed_vectors(snapshot, current)
    
    # Kombination nach Cartwright & Collett / Yuan-Folien
    V = Vt + 3.0 * Vp
    
    norm = np.linalg.norm(V)
    if norm < 1e-10:
        return np.array([0.0, 0.0])
    
    return V / norm


# =============================================================================
# 12. Funktion: Umwandlung Homing-Vektor -> Positionsänderung
# =============================================================================
def vector_to_position_change(
    vector: np.ndarray,
    step_size: float = 0.5
) -> np.ndarray:
    """
    Skaliert den normierten Homing-Vektor zu einer konkreten Schrittweite.
    """
    return vector * step_size


# =============================================================================
# 13. Iteratives Aufrufen: Suchpfad bis Home erreicht
# =============================================================================
def iterate_homing(
    world: World,
    snapshot: Retina,
    start_pos: Tuple[float, float],
    step_size: float = 0.5,
    max_steps: int = 100,
    tolerance: float = 0.1
) -> List[np.ndarray]:
    """
    Ruft wiederholt die Oberoberfunktion (calculate_homing_vector) auf,
    ändert die Position, berechnet die nächste Retina, usw.
    Endbedingung: Abstand zu Home < tolerance.
    Gibt den Pfad als Liste von Positions-Vektoren zurück.
    """
    path: List[np.ndarray] = [np.array(start_pos, dtype=float)]
    current_pos = np.array(start_pos, dtype=float)
    home_vec = np.array(world.home, dtype=float)
    
    for _ in range(max_steps):
        # Abbruch: Home erreicht?
        dist_to_home = np.linalg.norm(current_pos - home_vec)
        if dist_to_home < tolerance:
            break
        
        # Aktuelle Retina berechnen (Oberfunktion)
        current_retina = build_retina_for_position(world, tuple(current_pos))
        world.retinas.append(current_retina)
        
        # Homing-Vektor berechnen (Oberoberfunktion)
        vec = calculate_homing_vector(snapshot, current_retina)
        
        # Abbruch: Stehenbleiben (keine Richtung)
        if np.linalg.norm(vec) < 1e-10:
            break
        
        # Position ändern (Funktion 12)
        change = vector_to_position_change(vec, step_size)
        current_pos = current_pos + change
        path.append(current_pos.copy())
    
    return path


# =============================================================================
# Hauptprogramm / Demonstration
# =============================================================================
if __name__ == "__main__":
    # Welt initialisieren
    world = World()
    
    # Snapshot am Home-Punkt bauen (Oberfunktion)
    snapshot = build_retina_for_position(world, world.home)
    world.home_retina = snapshot
    world.retinas.append(snapshot)
    
    # --- Einzelner Homing-Vektor vom Startpunkt ---
    start = world.start
    start_retina = build_retina_for_position(world, start)
    world.retinas.append(start_retina)
    
    homing_vec = calculate_homing_vector(snapshot, start_retina)
    print(f"Start: {start}")
    print(f"Homing-Vektor (normiert): {homing_vec}")
    
    # --- Iterativer Pfad zurück nach Home ---
    path = iterate_homing(world, snapshot, start, step_size=0.3, tolerance=0.15)
    print(f"\nAnzahl Schritte bis Home: {len(path) - 1}")
    print("Pfad (x, y):")
    for p in path:
        print(f"  ({p[0]:.3f}, {p[1]:.3f})")
    
    # --- Optional: Gesamt-Quiver-Plot über das Grid ---
    try:
        import matplotlib.pyplot as plt
        
        x_range = np.arange(-7, 8)
        y_range = np.arange(-7, 8)
        X, Y = np.meshgrid(x_range, y_range)
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                px, py = float(X[i,j]), float(Y[i,j])
                # Home selbst überspringen (Division by zero in atan2 irrelevant, aber Vektor = 0)
                if abs(px) < 0.01 and abs(py) < 0.01:
                    continue
                r = build_retina_for_position(world, (px, py))
                vec = calculate_homing_vector(snapshot, r)
                U[i,j] = vec[0]
                V[i,j] = vec[1]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.quiver(X, Y, U, V, scale=20, width=0.003)
        
        # Landmarken zeichnen
        for lm in world.landmarks:
            circle = plt.Circle(lm["pos"], lm["radius"], color='skyblue', ec='black')
            ax.add_patch(circle)
        
        # Home und Start
        ax.plot(0, 0, 'kX', markersize=10, label='Home')
        ax.plot(start[0], start[1], 'ko', markersize=6, label='Start')
        
        # Pfad zeichnen
        path_arr = np.array(path)
        ax.plot(path_arr[:,0], path_arr[:,1], 'r-', linewidth=2, label='Suchpfad')
        ax.plot(path_arr[:,0], path_arr[:,1], 'r.', markersize=4)
        
        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Snapshot-Modell: Homing-Vektorfeld und Suchpfad')
        plt.show()
        
    except ImportError:
        print("\n(matplotlib nicht installiert – Plot übersprungen)")