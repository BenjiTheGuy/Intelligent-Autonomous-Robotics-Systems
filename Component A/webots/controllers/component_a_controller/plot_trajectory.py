# Importing libraries to use in this file
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from waypoints import A_STAR_WAYPOINTS

# Project Directories
CONTROLLER_DIR = Path(__file__).resolve().parent
COMPONENT_A_DIR = CONTROLLER_DIR.parents[2]
DATA_DIR = COMPONENT_A_DIR / "data"
RESULTS_DIR = COMPONENT_A_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAJECTORY_FILE = DATA_DIR / "trajectory_data.csv"
OUTPUT_FILE = RESULTS_DIR / "planned_vs_actual_trajectory.png"

# Checking if Webots has produced trajectory data.
if not TRAJECTORY_FILE.exists():
    raise FileNotFoundError(
        f"Trajectory Data not Found: {TRAJECTORY_FILE}\n"
        "Run the Webots Simulation First and Allow the CSV to be Saved."
    )

# Reading the robot's actual trajectory
actual_x = []
actual_y = []

with open(TRAJECTORY_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        x = float(row["X Position"])
        y = float(row["Y Position"])

        if math.isfinite(x) and math.isfinite(y):
            actual_x.append(x)
            actual_y.append(y)

if not actual_x:
    raise ValueError("The trajectory CSV contains no valid position data.")

# Extracting the planned A* Waypoints
planned_x = [point[0] for point in A_STAR_WAYPOINTS]
planned_y = [point[1] for point in A_STAR_WAYPOINTS]

# Obstacle positions and dimensions from the project world
obstacles = [
    (-0.8, 0.8, 1.2, 0.35, "Obstacle 1"),
    (0.8, 0.0, 0.35, 1.2, "Obstacle 2"),
    (1.4, 1.2, 0.7, 0.35, "Obstacle 3")
]

# Plotting the planned route and actual trajectory
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(
    planned_x,
    planned_y,
    ls="--",
    marker="o",
    lw=2,
    label="Planned A* Waypoint Route"
)

ax.plot(
    actual_x,
    actual_y,
    lw=2,
    label="Actual Robot Trajectory"
)

# Marking the Start and Goal
ax.scatter(
    planned_x[0],
    planned_y[0],
    marker="s",
    s=100,
    label="Start",
    zorder=5
)

ax.scatter(
    planned_x[-1],
    planned_y[-1],
    marker="*",
    s=180,
    label="Goal",
    zorder=5
)

# Drawing obstacles
for x, y, width, height, name in obstacles:
    rectangle = plt.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        fill=True,
        alpha=0.35,
        edgecolor="black",
        label=name
    )

    ax.add_patch(rectangle)

# Arena and Plot Formatting
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Planned A* Route vs Actual Robot Trajectory")
ax.grid(True, ls=":")
ax.legend()

fig.tight_layout()

fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

plt.show()

print(f"Plot Saved To: {OUTPUT_FILE}")
print(f"Actual Trajectory Samples: {len(actual_x)}")