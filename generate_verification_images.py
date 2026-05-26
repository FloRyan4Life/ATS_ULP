import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc

# Import core logic from the main submission file
from homing import (
    World, Retina, RetinaFeature, build_retina, compute_component_vectors,
    homing_vector, draw_retina_ring, angle_difference
)

# =============================================================================
# Configurable Parameters
# =============================================================================

# Four start positions for verification (easily editable)
START_POSITIONS = [
    (-3.0, -1.0),
    (5.0, 5.0),
    (-5.0, 3.0),
    (2.0, -6.0),
]

OUTPUT_DIR = "./verification"
STEP_SIZE = 0.4
TOLERANCE = 0.15
RADIUS_SNAPSHOT = 0.50
RADIUS_CURRENT = 1.00

STATE_LABELS = [
    "Retinas",
    "Pairing",
    "Individual Vectors",
    "Sum Vectors",
    "Homing Vector",
    "Movement"
]


# =============================================================================
# Drawing Logic (non-interactive, headless version of the diashow states)
# =============================================================================

def draw_diashow_state(world: World, snapshot: Retina, position: np.ndarray,
                       state_index: int, ax):
    """Render a single diashow state onto a matplotlib axis.
    
    This is a headless, non-blocking equivalent of HomingDiashow._draw()
    for automated image generation.
    """
    current_retina = build_retina(world, tuple(position))
    
    # World landmarks and home
    for landmark in world.landmarks:
        ax.add_patch(plt.Circle(landmark["pos"], landmark["radius"],
                                color='skyblue', ec='navy', zorder=5))
    ax.plot(0, 0, 'kX', markersize=12, zorder=6, label='Home')
    ax.plot(position[0], position[1], 'mo', markersize=8, zorder=6, label='Start')
    
    # States 0-4: Retina rings visible
    if state_index in (0, 1, 2, 3, 4):
        draw_retina_ring(ax, position, snapshot, RADIUS_SNAPSHOT, hollow=False)
        draw_retina_ring(ax, position, current_retina, RADIUS_CURRENT, hollow=True)
    
    # State 1: Feature pairing lines
    if state_index == 1:
        _, _, pairs = compute_component_vectors(snapshot, current_retina)
        for snapshot_index, current_index, _, _ in pairs:
            snapshot_feature = snapshot.features[snapshot_index]
            current_feature = current_retina.features[current_index]
            color = 'red' if snapshot_feature.is_landmark else 'green'
            
            x1 = position[0] + RADIUS_SNAPSHOT * math.cos(snapshot_feature.center_angle)
            y1 = position[1] + RADIUS_SNAPSHOT * math.sin(snapshot_feature.center_angle)
            x2 = position[0] + RADIUS_CURRENT * math.cos(current_feature.center_angle)
            y2 = position[1] + RADIUS_CURRENT * math.sin(current_feature.center_angle)
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.5,
                    alpha=0.8, zorder=11)
            ax.plot([x1, x2], [y1, y2], 'o', color=color, markersize=4, zorder=12)
    
    # State 2: Individual component vectors per matched pair
    if state_index == 2:
        _, _, pairs = compute_component_vectors(snapshot, current_retina)
        
        for _, current_index, angular_diff, arc_diff in pairs:
            current_feature = current_retina.features[current_index]
            feature_angle = current_feature.center_angle
            
            # Feature center on the ring
            center_x = position[0] + RADIUS_CURRENT * math.cos(feature_angle)
            center_y = position[1] + RADIUS_CURRENT * math.sin(feature_angle)
            
            # Arrow start slightly outside the ring
            arrow_start_radius = RADIUS_CURRENT + 0.5
            start_x = position[0] + arrow_start_radius * math.cos(feature_angle)
            start_y = position[1] + arrow_start_radius * math.sin(feature_angle)
            
            # Connector line for clear assignment
            ax.plot([center_x, start_x], [center_y, start_y],
                    color='black', linewidth=1.5, linestyle='--',
                    alpha=0.6, zorder=14)
            
            tangent = np.array([-math.sin(feature_angle), math.cos(feature_angle)])
            radial = np.array([math.cos(feature_angle), math.sin(feature_angle)])
            
            turn_component = angular_diff * tangent * 0.5  # scaled for visibility
            approach_component = (-arc_diff) * radial * 0.5
            
            ax.quiver(start_x, start_y, turn_component[0], turn_component[1],
                      scale=3, color='purple', width=0.006,
                      headwidth=4, headlength=5, zorder=15, alpha=0.9)
            ax.quiver(start_x, start_y, approach_component[0], approach_component[1],
                      scale=3, color='orange', width=0.006,
                      headwidth=4, headlength=5, zorder=15, alpha=0.9)
    
    # State 3: Summed Vt and Vp from robot center
    if state_index == 3:
        total_turn, total_approach, _ = compute_component_vectors(snapshot, current_retina)
        total_turn *= 0.5
        total_approach *= 0.5
        
        ax.quiver(position[0], position[1], total_turn[0], total_turn[1],
                  scale=3, color='purple', width=0.01,
                  headwidth=5, headlength=6, zorder=16, label='Vt (Turn)')
        ax.quiver(position[0], position[1], total_approach[0], total_approach[1],
                  scale=3, color='orange', width=0.01,
                  headwidth=5, headlength=6, zorder=16, label='Vp (Approach)')
    
    # State 4: Final normalized homing vector
    if state_index == 4:
        vector = homing_vector(snapshot, current_retina)
        ax.quiver(position[0], position[1], vector[0], vector[1],
                  scale=5, color='black', width=0.008, zorder=15,
                  label='Homing Vector')
    
    # State 5: Movement step visualization
    if state_index == 5:
        vector = homing_vector(snapshot, current_retina)
        home = np.array(world.home)
        distance_to_home = np.linalg.norm(position - home)
        step_length = min(STEP_SIZE, distance_to_home * 0.8)
        new_position = position + vector * step_length
        
        # Draw the movement vector at the original position
        ax.quiver(position[0], position[1], vector[0], vector[1],
                  scale=5, color='black', width=0.008, zorder=15,
                  label='Homing Vector')
        
        # Path from old to new position
        ax.plot([position[0], new_position[0]], [position[1], new_position[1]],
                'r-', linewidth=2.5, zorder=4)
        ax.plot(new_position[0], new_position[1], 'r.', markersize=8,
                zorder=6, label='New Position')
    
    # Layout
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect('equal')
    ax.set_title(f"Location {tuple(position)} | {STATE_LABELS[state_index]}")
    ax.legend(loc='upper left')


# =============================================================================
# Main Automation Loop
# =============================================================================

def main():
    """Generate all verification images for every configured start position."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    world = World()
    snapshot = build_retina(world, world.home)
    
    for location_index, start_pos in enumerate(START_POSITIONS):
        position = np.array(start_pos, dtype=float)
        
        for state_index in range(6):
            fig, ax = plt.subplots(figsize=(10, 10))
            draw_diashow_state(world, snapshot, position, state_index, ax)
            
            filename = f"homing_verification_image_LM-{location_index}_step-{state_index}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"[INFO] Saved {filename}")
        
        print(f"[INFO] Completed all steps for location {location_index}: {start_pos}")
    
    print(f"[INFO] All verification images saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()