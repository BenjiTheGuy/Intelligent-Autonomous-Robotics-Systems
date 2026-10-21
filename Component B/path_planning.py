import heapq
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon

from target_visualization import (
    ARENA_MIN,
    ARENA_MAX,
    GRID_SPACING,
    CLEARANCE,
    ROBOT_LENGTH,
    ROBOT_WIDTH,
    obstacles,
    robots,
    get_obstacle_polygon,
    distance_to_polygon
)

# Project Settings
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# A finer grid helps to improve the path detail.
PATH_GRID_SPACING = 0.25

GRID_ORIGIN = ARENA_MIN + math.hypot(ROBOT_LENGTH / 2, ROBOT_WIDTH / 2) + CLEARANCE

# Conservative circular approximation of the robot's footprint.
ROBOT_RADIUS = math.hypot(ROBOT_LENGTH / 2, ROBOT_WIDTH / 2)

# Including an additional safety margin around obstacles.
OBSTACLE_CLEARANCE = ROBOT_RADIUS + CLEARANCE

# Eight Movement Directions: horizontal, vertical, diagonal
MOVEMENTS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]

OBSTACLE_POLYGONS = [get_obstacle_polygon(obstacle) for obstacle in obstacles]

# Grid Conversion
def world_to_grid(point):
    """
    Converting world coordinates into integer grid coordinates.
    """
    x, y = point

    return (
        round((x - GRID_ORIGIN) / PATH_GRID_SPACING),
        round((y - GRID_ORIGIN) / PATH_GRID_SPACING)
    )

def grid_to_world(node):
    """
    Converting integer grid coordinates into world coordinates.
    """
    gx, gy = node

    return(
        GRID_ORIGIN + gx * PATH_GRID_SPACING,
        GRID_ORIGIN + gy * PATH_GRID_SPACING
    )

def grid_dimensions():
    """
    Returning the number of grid cells along each axis within the safe arena bounds.
    """
    wall_margin = ROBOT_RADIUS + CLEARANCE

    maximum = ARENA_MAX - wall_margin

    count = math.floor(
        (maximum - GRID_ORIGIN) / PATH_GRID_SPACING + 1e-9
    )

    return count + 1, count + 1

# Collision Checking
def is_valid_node(node):
    """
    Checking whether a grid location has sufficient clearance from arena walls and obstacles for the robot.
    """
    gx, gy = node

    width, height = grid_dimensions()

    if not (0 <= gx < width) or not (0 <= gy < height):
        return False

    x, y = grid_to_world(node)

    # Ensuring the robot remains inside the arena.
    wall_margin = ROBOT_RADIUS + CLEARANCE

    epsilon = 1e-9

    if (
        x < ARENA_MIN + wall_margin - epsilon
        or x > ARENA_MAX - wall_margin + epsilon
        or y < ARENA_MIN + wall_margin - epsilon
        or y > ARENA_MAX - wall_margin + epsilon
    ):
        return False

    # Rejcting all locations too close to any obstacle.
    for polygon in OBSTACLE_POLYGONS:
        if (distance_to_polygon(x, y, polygon) < OBSTACLE_CLEARANCE):
            return False

    return True

def movement_is_clear(current, neighbour):
    """
    Preventing diagonal moves from cutting across blocked corners between grid cells.
    """
    dx = neighbour[0] - current[0]
    dy = neighbour[1] - current[1]

    if dx != 0 and dy != 0:
        horizontal = (current[0] + dx, current[1])
        vertical = (current[0], current[1] + dy)

        if (not is_valid_node(horizontal)) or (not is_valid_node(vertical)):
            return False

    return is_valid_node(neighbour)

# A* Search Algorithm
def heuristic(node, goal):
    """
    Using an Octile Distance Heuristic for an Eight-Direction Grid.
    """
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])

    diagonal = min(dx, dy)
    straight = max(dx, dy) - diagonal

    return (
        diagonal * PATH_GRID_SPACING * math.sqrt(2) + straight * PATH_GRID_SPACING
    )

def reconstruct_path(came_from, current):
    """
    Reconstructing the path by following the parent nodes.
    """
    path = [current]

    while current in came_from:
        current = came_from[current]

        path.append(current)

    path.reverse()

    return [grid_to_world(node) for node in path]

def astar(start, goal):
    """
    Finding a collision-aware path between two world positions

    Returns a list of (x, y) world coordinates, or None if not path can be found.
    """
    start_node = world_to_grid(start)
    goal_node = world_to_grid(goal)

    if not is_valid_node(start_node):
        raise ValueError(
            f"Starting Position {start} is too close to an obstacle or arena wall."
        )

    if not is_valid_node(goal_node):
        snapped_goal = grid_to_world(goal_node)

        print(f"\nDEBUGGING: Original Goal: {goal}")
        print(f"\nDEBUGGING: Snapped Goal: {snapped_goal}")
        print(f"\nDEBUGGING: Required Clearance: {OBSTACLE_CLEARANCE:.3f} m")

        for index, polygon in enumerate(OBSTACLE_POLYGONS, start=1):
            original_distance = distance_to_polygon(goal[0], goal[1], polygon)

            snapped_distance = distance_to_polygon(snapped_goal[0], snapped_goal[1], polygon)

            print(
                f"Obstacle {index}:"
                f"Original={original_distance:.3f} m,"
                f"Snapped={snapped_distance:.3f} m"
            )

        raise ValueError(
            f"Goal Position {goal} is too close to an obstacle or arena wall."
        )

    open_heap = []

    heapq.heappush(open_heap, (heuristic(start_node, goal_node), start_node))

    came_from = {}
    g_score = {start_node: 0.0}
    closed_set = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        if current == goal_node:
            return reconstruct_path(came_from, current)

        closed_set.add(current)

        for dx, dy in MOVEMENTS:
            neighbour = (
                current[0] + dx,
                current[1] + dy
            )

            if neighbour in closed_set:
                continue

            if not movement_is_clear(current, neighbour):
                continue

            step_cost = PATH_GRID_SPACING * (
                math.sqrt(2) if dx != 0 and dy != 0 else 1.0
            )

            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g

                f_score = (tentative_g + heuristic(neighbour, goal_node))

                heapq.heappush(open_heap, (f_score, neighbour))

    return None

# Path Visualization
def plot_path(path, start, goal):
    """
    Plotting a planned path alongside the existing obstacles.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Drawing obstacles using the existing project geometry
    for obstacle, polygon in zip(obstacles, OBSTACLE_POLYGONS):
        ax.add_patch(PlotPolygon(
            polygon,
            closed=True,
            facecolor="gray",
            edgecolor="black",
            alpha=0.7
        ))

    # Drawing the path
    if path:
        xs = [point[0] for point in path]
        ys = [point[1] for point in path]

        ax.plot(
            xs,
            ys,
            "-o",
            markersize=2,
            lw=1.5,
            label="A* Path"
        )

        ax.scatter(
            *start,
            marker="s",
            s=100,
            label="Start",
            zorder=3
        )

        ax.scatter(
            *goal,
            marker="*",
            s=100,
            label="Goal",
            zorder=3
        )

        ax.set_title("A* Obstacle-Aware Path Planning")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")

        ax.set_xlim(ARENA_MIN, ARENA_MAX)
        ax.set_ylim(ARENA_MIN, ARENA_MAX)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()

        output_file = RESULTS_DIR / "astar_path.png"

        plt.savefig(output_file, dpi=200)

        plt.show()

        print(f"Path Plot Saved To:{output_file}")

# Testing a Single Path
def main():
    # Testing a path for robot_2 towards a candidate target.
    start = robots["robot_2"]["position"]
    goal =(0.0, 3.0)

    print("A* Path Planning Test")
    print(f"Start: {start}")
    print(f"Goal: {goal}")

    path = astar(start, goal)

    if path is None:
        print("No collision-free path was found.")

        return

    path_length = sum(
        math.dist(path[i], path[i + 1])

        for i in range(len(path) - 1)
    )

    print(f"Path Nodes: {len(path)}")
    print(f"Path Length: {path_length:.2f} m")

    plot_path(path, start, goal)

if __name__ == "__main__":
    main()