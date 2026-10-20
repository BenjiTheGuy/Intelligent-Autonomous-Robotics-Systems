# Importing libraries to use this in the file.
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

# Arena Configuration
ARENA_SIZE = 12.0
ARENA_MIN = -ARENA_SIZE / 2
ARENA_MAX = ARENA_SIZE / 2

GRID_SPACING = 0.5
CLEARANCE = 0.35

ROBOT_LENGTH = 0.8
ROBOT_WIDTH = 0.46

# Robot Starting Positions
robots = {
    "robot_1": {
        "position": (-5.0, 0.0),
        "rotation": 0.0
    },
    "robot_2": {
        "position": (4.7, -5.0),
        "rotation": 1.57
    },
    "robot_3": {
        "position": (0.5, 5.0),
        "rotation": -1.57
    }
}

# Obstacle Configuration (Setting out the X, Y, and Z positions of the obstacles.
obstacles = [
    {
        "name": "Obstacle 1",
        "position": (-0.8, 0.0, 0.25),
        "rotation": (0, 0, 1, 0),
        "size": (1.5, 0.5, 0.5)
    },
    {
        "name": "Obstacle 2",
        "position": (2.0, -2.5, 1.01),
        "rotation": (0, 1, 0, 1.57),
        "size": (2.0, 0.5, 0.5)
    },
    {
        "name": "Obstacle 3",
        "position": (3.0, 3.5, 0.25),
        "rotation": (1, 0, 0, 0),
        "size": (0.5, 2.5, 0.5)
    },
    {
        "name": "Obstacle 4",
        "position": (-3.2, 3.2, 0.25),
        "rotation": (0, 0, 1, 0),
        "size": (1.5, 0.5, 0.5)
    },
    {
        "name": "Obstacle 5",
        "position": (-3.5, -3.5, 1.0),
        "rotation": (0, 1, 0, 1.57),
        "size": (2.0, 2.0, 1.0)
    }
]

# Rotating a Point in 3D
def rotate_point(point, axis, angle):
    """
    Rotating a point using the axis-angle rotation.
    """
    x, y, z = point

    ax, ay, az = axis

    length = math.sqrt(ax**2 + ay**2 + az**2)

    if (length == 0) or (angle == 0):
        return x, y, z

    ax = ax / length
    ay = ay / length
    az = az / length

    cosine = math.cos(angle)
    sine = math.sin(angle)

    dot = (ax * x) + (ay * y) + (az * z)

    cross_x = (ay * z) - (az * y)
    cross_y = (az * x) - (ax * z)
    cross_z = (ax * y) - (ay * x)

    return (
    (x * cosine) + (cross_x * sine) + (ax * dot * (1 - cosine)),
    (y * cosine) + (cross_y * sine) + (ay * dot * (1 - cosine)),
    (z * cosine) + (cross_z * sine) + (az * dot * (1 - cosine))
    )

# Project Each Obstacle's Box onto the Floor
def get_obstacle_polygon(obstacle):
    cx, cy, cz = obstacle["position"]
    ax, ay, az, angle = obstacle["rotation"]
    sx, sy, sz = obstacle["size"]

    proj_points = []

    for dx in (-sx / 2, sx / 2):
        for dy in (-sy / 2, sy / 2):
            for dz in (-sz / 2, sz / 2):
                rx, ry, rz = rotate_point(
                    (dx, dy, dz),
                    (ax, ay, az),
                    angle
                )

                proj_points.append((round(cx + rx, 6), round(cy + ry, 6)))

    # Convex the hull of the projected corners.
    points = sorted(set(proj_points))

    if len(points) <= 2:
        return points
    
    def cross(origin, a, b):
        return (
            (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
        )

    lower = []

    for point in points:
        while (len(lower) >= 2) and (cross(lower[-2], lower[-1], point) <= 0):
            lower.pop()

        lower.append(point)

    upper = []

    for point in reversed(points):
        while (len(upper) >= 2) and (cross(upper[-2], upper[-1], point) <= 0):
            upper.pop()

        upper.append(point)

    return lower[:-1] + upper[:-1]

# Checking whether a point is found inside of a polygon or obstacle.
def point_inside_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if (yi > y) != (yj > y):
            intersection = (
            (xj -xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            )

            if x < intersection:
                inside = not inside

        j = i

    return inside

# Calculating the distance to an obstacle.
def point_segment_distance(px, py, a, b):
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    length_squared = (dx ** 2) + (dy ** 2)

    if length_squared == 0:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / length_squared

    t = max(0.0, min(1.0, t))

    closest_x = ax + (t * dx)
    closest_y = ay + (t * dy)

    return math.hypot(
        px - closest_x,
        py - closest_y
    )

def distance_to_polygon(x, y, polygon):
    if point_inside_polygon(x, y, polygon):
        return 0.0

    distances = []

    for i, point_a in enumerate(polygon):
        point_b = polygon[(i + 1) % len(polygon)]

        distances.append(
            point_segment_distance(
                x, y, point_a, point_b
            )
        )

    return min(distances)

# Generating Candidate Coverage Targets
def generate_targets(polygons):
    targets = []

    wall_margin = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2 + CLEARANCE)

    minimum = ARENA_MIN + wall_margin
    maximum = ARENA_MAX - wall_margin

    # Generate Grid coordinates using the specified spacing
    x_values = []
    value = minimum

    while value <= maximum + 1e-9:
        x_values.append(value)
        value += GRID_SPACING

    # Checking every possible (x, y) position
    for x in x_values:
        for y in x_values:
            blocked = False

            # Rejecting targets which are too close to obstacles
            for polygon in polygons:
                distance = distance_to_polygon(x, y, polygon)

                if distance < CLEARANCE:
                    blocked = True
                    break

            # Storing only targets that will pass the clearance checks.
            if not blocked:
                targets.append((x, y))

    return targets

# Visualizing the Arena and Targets
def main():
    polygons = [
        get_obstacle_polygon(obstacle)

        for obstacle in obstacles
    ]

    targets = generate_targets(polygons)

    print("=" * 25)
    print("COMPONENT B - CANDIDATE TARGET GENERATION")
    print("=" * 25)

    print(f"Arena: {ARENA_SIZE} x {ARENA_SIZE} metres")
    print(f"Grid Spacing: {GRID_SPACING} metres")
    print(f"Clearance: {CLEARANCE} metres")
    print(f"Candidate Targets: {len(targets)}")

    print("\nRobot Starting Positions: ")

    for name, robot_data in robots.items():
        print(f"{name}: {robot_data["position"]}")

    # Generating the Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Drawing the Arena Boundary
    ax.add_patch(Rectangle(
        (ARENA_MIN, ARENA_MIN),
        ARENA_SIZE,
        ARENA_SIZE,
        fill=False,
        edgecolor="black",
        lw=2
    ))

    # Obstacles
    for obstacle, polygon in zip(obstacles, polygons):
        ax.add_patch(Polygon(
            polygon,
            closed=True,
            facecolor="gray",
            edgecolor="black",
            alpha=0.7
        ))

        x, y, z = obstacle["position"]

        ax.text(
            x,
            y,
            obstacle["name"],
            ha="center",
            va="center",
            fontsize=8
        )

    # Candidate Targets
    if targets:
        xs = [point[0] for point in targets]
        ys = [point[1] for point in targets]

        ax.scatter(
            xs,
            ys,
            s=12,
            marker=".",
            label="Candidate Targets"
        )

    # Robots with their Webots Orientations
    for name, robot_data in robots.items():
        x, y = robot_data["position"]
        angle = robot_data["rotation"]

        # Four Corners of the Robot before Rotation
        corners = [
            (-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
            (ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
            (ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
            (-ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
        ]

        # Rotating the corners around the robot's center
        rotated_corners = []

        for px, py in corners:
            rx = (px * math.cos(angle)) - (py * math.sin(angle))
            ry = (px * math.sin(angle)) + (py * math.cos(angle))

            rotated_corners.append((x + rx, y + ry))

        ax.add_patch(Polygon(
            rotated_corners,
            closed=True,
            fill=False,
            edgecolor="black",
            lw=1.5
        ))

        ax.scatter(
            x,
            y,
            marker="s",
            s=100,
            label=name
        )

    ax.set_title("Component B: Candidate Coverage Targets")

    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")

    ax.set_xlim(ARENA_MIN - 0.25, ARENA_MAX + 0.25)
    ax.set_ylim(ARENA_MIN - 0.25, ARENA_MAX + 0.25)

    ax.set_xticks(range(-6, 7))
    ax.set_yticks(range(-6, 7))

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()


    # Saving the graph to Component B/results folder
    results_directory = (
        Path(__file__).resolve().parent / "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        results_directory / "candidate_targets.png"
    )

    plt.savefig(output_path, dpi=250)

    print(f"\nGraph Saved to: {output_path}")

    plt.show()

if __name__ == "__main__":
    main()