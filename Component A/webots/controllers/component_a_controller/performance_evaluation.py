# Importing libraries
import csv
import math
from pathlib import Path
from tabulate import tabulate

from waypoints import A_STAR_WAYPOINTS

# Project Directories
CONTROLLER_DIR = Path(__file__).resolve().parent
COMPONENT_A_DIR = CONTROLLER_DIR.parents[2]
DATA_DIR = COMPONENT_A_DIR / "data"

TRAJECTORY_FILE = DATA_DIR / "trajectory_data.csv"
OUTPUT_FILE = DATA_DIR / "performance_evaluation.csv"

# Goal Settings
GOAL_X = 2.0
GOAL_Y = -2.0
GOAL_TOLERANCE = 0.25

# Planned A* Grid-Path Length from planning.py
PLANNED_PATH_LENGTH = 7.80

# Simplified route from the generated A* waypoints.
WAYPOINTS = A_STAR_WAYPOINTS

def distance(x1, y1, x2, y2):
    """
    Calculating straight-line distance between two positions.
    """
    return math.hypot(x2 - x1, y2 - y1)


def point_to_segment_distance(point, start, end):
    """
    Calculating the shortest distance from a point to a route segment.
    """
    px, py = point
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    length_squared = dx**2 + dy**2

    if length_squared == 0:
        return distance(px, py, x1, y1)

    t = ((px - x1) * dx + (py - y1) * dy) / length_squared
    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return distance(px, py, closest_x, closest_y)

def main():
    # Checking for trajectory data
    if not TRAJECTORY_FILE.exists():
        raise FileNotFoundError(
            f"Trajectory data not found:\n{TRAJECTORY_FILE}\n"
            "Run the Webots Simulation first."
        )

    # Loading the actual robot trajectory
    trajectory = []

    with open(TRAJECTORY_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            trajectory.append({
                "time": float(row["Time"]),
                "x": float(row["X Position"]),
                "y": float(row["Y Position"]),
            })

    if len(trajectory) < 2:
        raise ValueError("Not enough trajectory samples to evaluate.")

    # Calculating the actual distance travelled
    actual_distance = 0.0

    for i in range(1, len(trajectory)):
        previous = trajectory[i - 1]
        current = trajectory[i]

        actual_distance += distance(
            previous["x"], previous["y"],
            current["x"], current["y"]
        )

    # Finding the first recorded time inside the goal tolerance
    time_to_goal = None

    for sample in trajectory:
        goal_error = distance(
            sample["x"],
            sample["y"],
            GOAL_X,
            GOAL_Y
        )

        if goal_error <= GOAL_TOLERANCE:
            time_to_goal = sample["time"]

            break

    # Calculating the final goal error
    final_position = trajectory[-1]

    final_goal_error = distance(
        final_position["x"],
        final_position["y"],
        GOAL_X,
        GOAL_Y
    )

    # Calculating deviation from the simplified waypoint route
    deviations = []

    for sample in trajectory:
        point = (sample["x"], sample["y"])

        segment_distances = [
            point_to_segment_distance(point, WAYPOINTS[i], WAYPOINTS[i + 1])

            for i in range(len(WAYPOINTS) - 1)
        ]

        deviations.append(min(segment_distances))

        mean_deviation = sum(deviations) / len(deviations)
        maximum_deviation = max(deviations)

        # Building performance evaluation table
        results = [
            [
                "Planned A* Grid-Path Length",
                f"{PLANNED_PATH_LENGTH:.2f}",
                "m"
            ],
            [
                "Actual Distance Travelled",
                f"{actual_distance:.2f}",
                "m"
            ],
            [
                "Time to Goal Region",
                f"{time_to_goal:.3f}" if time_to_goal is not None else "Not reached",
                "s"
            ],
            [
                "Final Goal Error",
                f"{final_goal_error:.3f}",
                "m"
            ],
            [
                "Mean Route Deviation",
                f"{mean_deviation:.3f}",
                "m"
            ],
            [
                "Maximum Route Deviation",
                f"{maximum_deviation:.3f}",
                "m"
            ]
        ]

        headers = ["Performance Metric", "Result", "Unit"]

        print("\nPerformance Evaluation Table")
        print(tabulate(results, headers=headers, tablefmt="grid"))

        # Saving the table as a CSV file
        with open(OUTPUT_FILE, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(headers)
            writer.writerows(results)

        print(f"\nTable Saved To: {OUTPUT_FILE}")
        print(f"Trajectory Samples Analyzed: {len(trajectory)}")

if __name__ == "__main__":
    main()