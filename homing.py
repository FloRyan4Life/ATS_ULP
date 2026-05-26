import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc
import argparse
import os
import json

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RetinaFeature:
    """Represents a single feature on the circular retina.
    
    Attributes:
        center_angle: Angular center of the feature in radians [-π, π).
        arc_length: Angular width of the feature in radians.
        is_landmark: True if the feature is a landmark projection, False if it is a gap.
    """
    center_angle: float
    arc_length: float
    is_landmark: bool


class Retina:
    """Circular retina representation at a specific world position.
    
    Stores a list of alternating landmark and gap features sorted by angle.
    """
    def __init__(self, position: Tuple[float, float], features: List[RetinaFeature] = None):
        self.position = np.array(position, dtype=float)
        self.features: List[RetinaFeature] = features if features is not None else []
    
    def add_feature(self, feature: RetinaFeature):
        self.features.append(feature)


class World:
    """Container for the simulation environment.
    
    Defines the home position, the three circular landmarks, and stores
    the snapshot retina captured at home.
    """
    def __init__(self):
        self.home = (0.0, 0.0)
        self.landmarks = [
            {"pos": (3.5, 2.0),  "radius": 0.5},
            {"pos": (3.5, -2.0), "radius": 0.5},
            {"pos": (0.0, -4.0), "radius": 0.5},
        ]
        self.home_retina: Optional[Retina] = None


# =============================================================================
# Trigonometry Helpers
# =============================================================================

def angle_difference(angle_a: float, angle_b: float) -> float:
    """Compute the signed smallest difference between two angles in radians.
    
    Result is wrapped to the interval [-π, π].
    """
    return (angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi


def project_landmark(robot_position: Tuple[float, float], landmark: dict) -> RetinaFeature:
    """Project a circular landmark onto the retina from a given robot position.
    
    The projection is an arc whose angular size depends on the landmark radius
    and the distance to the robot.
    """
    delta_x = landmark["pos"][0] - robot_position[0]
    delta_y = landmark["pos"][1] - robot_position[1]
    distance = math.hypot(delta_x, delta_y)
    
    # Angular direction to the landmark center
    direction = math.atan2(delta_y, delta_x)
    
    # Angular radius of the landmark as seen from the robot
    if distance > 1e-9:
        ratio = min(1.0, max(-1.0, landmark["radius"] / distance))
    else:
        ratio = 1.0
    half_arc = math.asin(ratio)
    
    return RetinaFeature(direction, 2.0 * half_arc, True)


def build_retina(world: World, position: Tuple[float, float]) -> Retina:
    """Construct the full retina for a given robot position.
    
    Projects all landmarks, sorts them by angle, and inserts gap features
    between consecutive landmarks.
    """
    retina = Retina(position)
    landmark_features = [project_landmark(position, lm) for lm in world.landmarks]
    landmark_features.sort(key=lambda feature: feature.center_angle)
    
    num_landmarks = len(landmark_features)
    for index in range(num_landmarks):
        current = landmark_features[index]
        next_feature = landmark_features[(index + 1) % num_landmarks]
        
        # Right edge of current landmark arc
        right_edge = (current.center_angle + current.arc_length / 2.0) % (2.0 * math.pi)
        # Left edge of next landmark arc
        left_edge = (next_feature.center_angle - next_feature.arc_length / 2.0) % (2.0 * math.pi)
        
        # Angular size of the gap between the two landmarks
        gap_arc = (left_edge - right_edge) % (2.0 * math.pi)
        gap_center = (right_edge + gap_arc / 2.0) % (2.0 * math.pi)
        
        gap = RetinaFeature(gap_center, max(0.0, gap_arc), False)
        retina.add_feature(current)
        retina.add_feature(gap)
    
    return retina


# =============================================================================
# Feature Matching & Vector Computation
# =============================================================================

def match_feature(snapshot_feature: RetinaFeature, current_features: List[RetinaFeature]) -> Tuple[int, float, float]:
    """Find the best matching feature in the current retina for a snapshot feature.
    
    Matching is restricted to features of the same type (landmark-to-landmark,
    gap-to-gap). Returns the index of the match, the angular difference, and
    the arc-length difference.
    """
    candidates = [(index, feature) for index, feature in enumerate(current_features)
                  if feature.is_landmark == snapshot_feature.is_landmark]
    
    if not candidates:
        return 0, 0.0, 0.0
    
    best_index = candidates[0][0]
    best_difference = float('inf')
    
    for index, feature in candidates:
        difference = abs(angle_difference(snapshot_feature.center_angle, feature.center_angle))
        if difference < best_difference:
            best_difference = difference
            best_index = index
    
    matched = current_features[best_index]
    angular_diff = angle_difference(matched.center_angle, snapshot_feature.center_angle)
    arc_diff = matched.arc_length - snapshot_feature.arc_length
    
    return best_index, angular_diff, arc_diff


def compute_component_vectors(snapshot: Retina, current: Retina):
    """Compute the tangential (turn) and radial (approach) component vectors.
    
    For each matched feature pair, a tangential vector Vt and a radial vector Vp
    are calculated based on the current retina feature angles. The total vectors
    are the sums over all pairs.
    
    Returns:
        total_turn_vector: Sum of all tangential components (Vt).
        total_approach_vector: Sum of all radial components (Vp).
        pairs: List of tuples (snapshot_index, current_index, angular_diff, arc_diff).
    """
    total_turn_vector = np.zeros(2)
    total_approach_vector = np.zeros(2)
    pairs = []
    
    for snapshot_index, snapshot_feature in enumerate(snapshot.features):
        current_index, angular_diff, arc_diff = match_feature(snapshot_feature, current.features)
        pairs.append((snapshot_index, current_index, angular_diff, arc_diff))
        
        # Use the angle of the CURRENT retina feature for the basis vectors
        current_angle = current.features[current_index].center_angle
        
        # Tangential basis vector (perpendicular to line of sight, counter-clockwise)
        tangent = np.array([-math.sin(current_angle), math.cos(current_angle)])
        # Radial basis vector (line of sight towards the feature)
        radial = np.array([math.cos(current_angle), math.sin(current_angle)])
        
        total_turn_vector += angular_diff * tangent
        total_approach_vector += (-arc_diff) * radial
    
    return total_turn_vector, total_approach_vector, pairs


def homing_vector(snapshot: Retina, current: Retina) -> np.ndarray:
    """Calculate the final normalized homing direction vector.
    
    The vector is a weighted sum of the turn and approach components.
    The weight factor 3.0 for the approach component follows the original model.
    """
    turn_vector, approach_vector, _ = compute_component_vectors(snapshot, current)
    combined = turn_vector + 3.0 * approach_vector
    norm = np.linalg.norm(combined)
    
    if norm > 1e-10:
        return combined / norm
    else:
        return np.zeros(2)


# =============================================================================
# Evaluation: Homing Precision
# =============================================================================

def ideal_homing_direction(position: np.ndarray) -> np.ndarray:
    """Return the ideal (ground-truth) homing vector pointing from position to home.
    
    The ideal direction is simply the negative normalized position vector.
    """
    norm = np.linalg.norm(position)
    if norm > 1e-10:
        return -position / norm
    return np.zeros(2)


def angle_between_vectors(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute the unsigned angle in radians between two vectors."""
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    
    cosine = np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0)
    return math.acos(cosine)


def compute_homing_precision(world: World, snapshot: Retina, grid_range=range(-7, 8)) -> dict:
    """Evaluate the homing precision over the entire grid.
    
    For every grid point except home, the ideal homing direction is compared
    with the direction calculated by the snapshot model. The average angular
    error is returned together with detailed statistics.
    
    Returns:
        A dictionary containing:
        - mean_error_deg: Average angular error in degrees.
        - max_error_deg: Maximum angular error in degrees.
        - std_dev_deg: Standard deviation of the error in degrees.
        - errors: List of (x, y, error_deg) for every evaluated point.
    """
    errors = []
    
    for grid_x in grid_range:
        for grid_y in grid_range:
            if grid_x == 0 and grid_y == 0:
                continue  # Home has no homing direction
            
            position = np.array([float(grid_x), float(grid_y)])
            ideal = ideal_homing_direction(position)
            
            current_retina = build_retina(world, (grid_x, grid_y))
            calculated = homing_vector(snapshot, current_retina)
            
            error_rad = angle_between_vectors(ideal, calculated)
            error_deg = math.degrees(error_rad)
            errors.append((grid_x, grid_y, error_deg))
    
    error_values = [entry[2] for entry in errors]
    mean_error_deg = float(np.mean(error_values))
    max_error_deg = float(np.max(error_values))
    std_dev_deg = float(np.std(error_values))
    
    return {
        "mean_error_deg": mean_error_deg,
        "max_error_deg": max_error_deg,
        "std_dev_deg": std_dev_deg,
        "errors": errors,
    }


# =============================================================================
# Visualization
# =============================================================================

def draw_retina_ring(ax, center, retina: Retina, radius: float, hollow=False):
    """Draw a retina ring (landmarks in red, gaps in green) on a matplotlib axis."""
    for feature in retina.features:
        center_deg = np.degrees(feature.center_angle)
        half_width_deg = max(np.degrees(feature.arc_length / 2.0), 1.0)
        color = 'red' if feature.is_landmark else 'green'
        
        if hollow:
            arc = Arc(center, 2 * radius, 2 * radius,
                      angle=0, theta1=center_deg - half_width_deg,
                      theta2=center_deg + half_width_deg,
                      color=color, linewidth=10, zorder=10)
            ax.add_patch(arc)
        else:
            wedge = Wedge(center, radius,
                          center_deg - half_width_deg,
                          center_deg + half_width_deg,
                          facecolor=color, edgecolor='black', linewidth=1,
                          alpha=0.8, zorder=10)
            ax.add_patch(wedge)


def plot_vector_field(world: World, snapshot: Retina, output_path: str):
    """Generate and save the homing vector field over the 14x14 grid.
    
    This reproduces a map similar to Fig. 4e of the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    grid_coordinates = np.arange(-7, 8)
    x_grid, y_grid = np.meshgrid(grid_coordinates, grid_coordinates)
    u_components = np.zeros_like(x_grid, dtype=float)
    v_components = np.zeros_like(y_grid, dtype=float)
    
    for row in range(x_grid.shape[0]):
        for col in range(x_grid.shape[1]):
            pos_x = float(x_grid[row, col])
            pos_y = float(y_grid[row, col])
            
            if abs(pos_x) < 0.01 and abs(pos_y) < 0.01:
                continue
            
            current_retina = build_retina(world, (pos_x, pos_y))
            vector = homing_vector(snapshot, current_retina)
            u_components[row, col] = vector[0]
            v_components[row, col] = vector[1]
    
    ax.quiver(x_grid, y_grid, u_components, v_components,
              scale=25, width=0.003, color='black')
    
    # Draw landmarks
    for landmark in world.landmarks:
        ax.add_patch(plt.Circle(landmark["pos"], landmark["radius"],
                                color='skyblue', ec='navy', zorder=5))
    
    # Mark home
    ax.plot(0, 0, 'kX', markersize=12, zorder=6, label='Home')
    
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_title("Homing Vector Field (Snapshot Model)")
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Vector field saved to: {output_path}")


def plot_precision_heatmap(precision_data: dict, output_path: str):
    """Optional: Draw a heatmap of angular errors over the grid."""
    errors = precision_data["errors"]
    grid_size = 15  # -7 to 7 inclusive
    heatmap = np.full((grid_size, grid_size), np.nan)
    
    for grid_x, grid_y, error_deg in errors:
        row = grid_y + 7
        col = grid_x + 7
        heatmap[row, col] = error_deg
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, origin='lower', cmap='viridis',
                   extent=[-7.5, 7.5, -7.5, 7.5], aspect='equal')
    plt.colorbar(im, ax=ax, label='Angular Error (degrees)')
    
    for landmark in World().landmarks:
        ax.add_patch(plt.Circle(landmark["pos"], landmark["radius"],
                                color='white', ec='navy', alpha=0.7, zorder=5))
    ax.plot(0, 0, 'rX', markersize=12, zorder=6)
    
    ax.set_title("Homing Precision Heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Precision heatmap saved to: {output_path}")


# =============================================================================
# Interactive Diashow
# =============================================================================

class HomingDiashow:
    """Interactive step-by-step visualization of the homing process.
    
    Press SPACE to cycle through the states:
    1. Retinas         – show snapshot and current retina rings
    2. Pairing         – draw matching lines between features
    3. Individual      – draw Vt and Vp for each matched pair
    4. Sum Vectors     – draw total Vt and Vp from robot center
    5. Homing Vector   – draw the final normalized homing direction
    6. Movement        – move one step towards home
    """
    
    def __init__(self, world: World, snapshot: Retina, start: Tuple[float, float],
                 step_size=0.4, tolerance=0.15):
        self.world = world
        self.snapshot = snapshot
        self.path = [np.array(start, dtype=float)]
        self.step_size = step_size
        self.tolerance = tolerance
        
        self.radius_snapshot = 0.50
        self.radius_current = 1.00
        
        self.state = 0
        self.labels = [
            "Retinas",
            "Pairing",
            "Individual Vectors",
            "Sum Vectors",
            "Homing Vector",
            "Movement"
        ]
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self._draw()
        plt.show()
    
    def _draw(self):
        self.ax.clear()
        current_position = self.path[-1]
        current_retina = build_retina(self.world, tuple(current_position))
        
        # Draw world landmarks
        for landmark in self.world.landmarks:
            self.ax.add_patch(plt.Circle(landmark["pos"], landmark["radius"],
                           color='skyblue', ec='navy', zorder=5))
        self.ax.plot(0, 0, 'kX', markersize=12, zorder=6, label='Home')
        
        if len(self.path) > 1:
            path_array = np.array(self.path)
            self.ax.plot(path_array[:, 0], path_array[:, 1],
                         'r.-', linewidth=2, zorder=4, label='Path')
        else:
            self.ax.plot(current_position[0], current_position[1],
                         'mo', markersize=8, zorder=6, label='Start')
        
        # States 0-4: Draw retina rings
        if self.state in (0, 1, 2, 3, 4):
            draw_retina_ring(self.ax, current_position, self.snapshot,
                             self.radius_snapshot, hollow=False)
            draw_retina_ring(self.ax, current_position, current_retina,
                             self.radius_current, hollow=True)
        
        # State 1: Draw pairing lines
        if self.state == 1:
            _, _, pairs = compute_component_vectors(self.snapshot, current_retina)
            for snapshot_index, current_index, _, _ in pairs:
                snapshot_feature = self.snapshot.features[snapshot_index]
                current_feature = current_retina.features[current_index]
                color = 'red' if snapshot_feature.is_landmark else 'green'
                
                x1 = current_position[0] + self.radius_snapshot * math.cos(snapshot_feature.center_angle)
                y1 = current_position[1] + self.radius_snapshot * math.sin(snapshot_feature.center_angle)
                x2 = current_position[0] + self.radius_current * math.cos(current_feature.center_angle)
                y2 = current_position[1] + self.radius_current * math.sin(current_feature.center_angle)
                
                self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.5,
                             alpha=0.8, zorder=11)
                self.ax.plot([x1, x2], [y1, y2], 'o', color=color,
                             markersize=4, zorder=12)
        
          # State 2: Draw individual Vt (purple) and Vp (orange) per pair
        if self.state == 2:
            _, _, pairs = compute_component_vectors(self.snapshot, current_retina)
            
            for _, current_index, angular_diff, arc_diff in pairs:
                current_feature = current_retina.features[current_index]
                feature_angle = current_feature.center_angle
                
                # Center point of the feature on the retina ring
                center_x = current_position[0] + self.radius_current * math.cos(feature_angle)
                center_y = current_position[1] + self.radius_current * math.sin(feature_angle)
                
                # Arrow start point slightly outside the ring
                arrow_start_radius = self.radius_current + 0.5
                start_x = current_position[0] + arrow_start_radius * math.cos(feature_angle)
                start_y = current_position[1] + arrow_start_radius * math.sin(feature_angle)
                
                # Dashed connector line for clear assignment
                self.ax.plot([center_x, start_x], [center_y, start_y],
                             color='black', linewidth=1.5, linestyle='--',
                             alpha=0.6, zorder=14)
                
                tangent = np.array([-math.sin(feature_angle), math.cos(feature_angle)])
                radial = np.array([math.cos(feature_angle), math.sin(feature_angle)])
                
                turn_component = angular_diff * tangent * 0.5 # Scaled for visibility
                approach_component = (-arc_diff) * radial * 0.5 # Scaled for visibility
                
                self.ax.quiver(start_x, start_y, turn_component[0], turn_component[1],
                               scale=3, color='purple', width=0.006,
                               headwidth=4, headlength=5, zorder=15, alpha=0.9)
                self.ax.quiver(start_x, start_y, approach_component[0], approach_component[1],
                               scale=3, color='orange', width=0.006,
                               headwidth=4, headlength=5, zorder=15, alpha=0.9)
        
        # State 3: Draw sum vectors Vt and Vp from robot center
        if self.state == 3:
            total_turn, total_approach, _ = compute_component_vectors(self.snapshot, current_retina)

            #scaling for better visibility in the diashow
            total_turn *= 0.5
            total_approach *= 0.5  

            self.ax.quiver(current_position[0], current_position[1],
                           total_turn[0], total_turn[1],
                           scale=3, color='purple', width=0.01,
                           headwidth=5, headlength=6, zorder=16, label='Vt (Turn)')
            self.ax.quiver(current_position[0], current_position[1],
                           total_approach[0], total_approach[1],
                           scale=3, color='orange', width=0.01,
                           headwidth=5, headlength=6, zorder=16, label='Vp (Approach)')
        
        # State 4: Final homing vector
        if self.state == 4:
            vector = homing_vector(self.snapshot, current_retina)
            self.ax.quiver(current_position[0], current_position[1],
                           vector[0], vector[1],
                           scale=5, color='black', width=0.008, zorder=15,
                           label='Homing Vector')
        
        self.ax.set_xlim(-7.5, 7.5)
        self.ax.set_ylim(-7.5, 7.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_title(f"Step {len(self.path)-1} | {self.labels[self.state]} | "
                          f"Pos: ({current_position[0]:.2f}, {current_position[1]:.2f}) — SPACE")
        self.ax.legend(loc='upper left')
        self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        if event.key != ' ':
            return
        
        if self.state == 5:
            current_position = self.path[-1]
            home = np.array(self.world.home)
            distance_to_home = np.linalg.norm(current_position - home)
            
            if distance_to_home < self.tolerance:
                self.ax.set_title("GOAL REACHED", fontsize=16, color='green')
                self.fig.canvas.draw_idle()
                return
            
            current_retina = build_retina(self.world, tuple(current_position))
            vector = homing_vector(self.snapshot, current_retina)
            step_length = min(self.step_size, distance_to_home * 0.8)
            self.path.append(current_position + vector * step_length)
            self.state = 0
        else:
            self.state += 1
        
        self._draw()


# =============================================================================
# Main Entry Point
# =============================================================================

def run_automatic_evaluation(output_directory: str):
    """Execute the full automatic verification pipeline.
    
    Computes homing vectors for the entire grid, evaluates precision,
    generates plots, and writes the precision data to a JSON file.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    world = World()
    snapshot = build_retina(world, world.home)
    world.home_retina = snapshot
    
    print("[INFO] Computing homing precision over the 14x14 grid...")
    precision = compute_homing_precision(world, snapshot)
    
    print(f"[RESULT] Mean angular error: {precision['mean_error_deg']:.4f}°")
    print(f"[RESULT] Max angular error:   {precision['max_error_deg']:.4f}°")
    print(f"[RESULT] Std deviation:      {precision['std_dev_deg']:.4f}°")
    
    plot_vector_field(world, snapshot, os.path.join(output_directory, "homing_vectors.png"))
    plot_precision_heatmap(precision, os.path.join(output_directory, "precision_heatmap.png"))
    
    # Save precision data as JSON for external README generation
    json_path = os.path.join(output_directory, "precision.json")
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(precision, file, indent=2)
    print(f"[INFO] Precision data saved to: {json_path}")
    
    print("[INFO] Automatic evaluation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Snapshot-based visual homing algorithm (Cartwright & Collett model)."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "diashow"],
        default="auto",
        help="Execution mode: 'auto' runs the full evaluation and saves plots; "
             "'diashow' opens the interactive step-by-step viewer."
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Directory for generated output files (plots and precision data)."
    )
    parser.add_argument(
        "--start",
        nargs=2,
        type=float,
        default=[-3.0, -1.0],
        help="Starting position for the diashow mode (default: -3.0 -1.0)."
    )
    
    args = parser.parse_args()
    
    if args.mode == "auto":
        run_automatic_evaluation(args.output)
    elif args.mode == "diashow":
        world = World()
        snapshot = build_retina(world, world.home)
        world.home_retina = snapshot
        start_position = tuple(args.start)
        print("[INFO] Starting interactive diashow. Press SPACE to advance.")
        print("[INFO] States: Retinas -> Pairing -> Individual Vectors -> Sum Vectors -> Homing Vector -> Movement")
        HomingDiashow(world, snapshot, start_position)


if __name__ == "__main__":
    main()