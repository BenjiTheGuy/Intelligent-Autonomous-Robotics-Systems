# Importing libraries
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project Directories
CONTROLLER_DIR = Path(__file__).resolve().parent
COMPONENT_A_DIR = CONTROLLER_DIR.parents[2]
RESULTS_DIR = COMPONENT_A_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2D Occupancy Grid Size
GRID_SIZE = 20

# Webots Arena Size: 6m x 6m, centered at (0, 0)
WORLD_SIZE = 6.0

WORLD_MIN_X = -WORLD_SIZE / 2
WORLD_MAX_X = WORLD_SIZE / 2

WORLD_MIN_Y = -WORLD_SIZE / 2
WORLD_MAX_Y = WORLD_SIZE / 2

# Each Occupancy-Grid Cell is represented by dividing the world size by the grid size.
CELL_SIZE = WORLD_SIZE / GRID_SIZE

# Leave at zero initially to avoid changing the established route.
OBSTACLE_MARGIN = CELL_SIZE

def world_to_grid(x, y):
    """
    Converting a Webots World Coordinate (x, y) into an Occupancy-Grid Coordinate (row, column).

    Webots Arena spans:
        X = -2.5 to +2.5m
        Y = -2.5 to +2.5m

    Occupancy Grid spans:
        20 x 20 cells

    Each cell is represented as:
        0.25m x 0.25m
    """

    column = int((x - WORLD_MIN_X) / CELL_SIZE)

    row = int((WORLD_MAX_Y - y) / CELL_SIZE)

    # Keeping coordinates inside the grid
    row = max(0, min(GRID_SIZE - 1, row))
    column = max(0, min(GRID_SIZE - 1, column))

    return row, column

def grid_to_world(row, column):
    """
    Converting the Occupancy-Grid Coordinate (row, column) into the center coordinates corresponding
    to the Webots Grid Cell.
    """

    x = (WORLD_MIN_X + (column + 0.5) * CELL_SIZE)
    y = (WORLD_MAX_Y - (row + 0.5) * CELL_SIZE)

    return x, y

# Occupancy Grid
def mark_obstacle(grid, center_x, center_y, size_x, size_y):
    """
    Marking the physical rectangular Webots obstacles as occupied cells within the occupancy grid.

    A safety margin is added around these physical obstacles to ensure the robot doesn't come in close proximity to them.
    """
    min_x = center_x - size_x / 2 - OBSTACLE_MARGIN
    max_x = center_x + size_x / 2 + OBSTACLE_MARGIN

    min_y = center_y - size_y / 2 - OBSTACLE_MARGIN
    max_y = center_y + size_y / 2 + OBSTACLE_MARGIN

    min_row, min_col = world_to_grid(min_x, max_y)
    max_row, max_col = world_to_grid(max_x, min_y)

    # Identify grid cells whose centers fall within the obstacle bounds.
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            x, y = grid_to_world(row, column)

            if min_x <= x <= max_x and min_y <= y <= max_y:
                grid[row, column] = 1

def create_occupancy_grid():
    """
    Creating a 20 x 20 2D Occupancy Grid which represents the robot's environment.

    0 = Free Space
    1 = Occupied (An obstacle is present)
    """
    # Creating a 20 x 20 grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Creating the boundary walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Physical Webots Obstacles
    # Obstacle 1
    mark_obstacle(grid, center_x=-0.8, center_y=0.8, size_x=1.2, size_y=0.35)

    # Obstacle 2
    mark_obstacle(grid, center_x=0.8, center_y=0.0, size_x=0.35, size_y=1.2)

    # Obstacle 3
    mark_obstacle(grid, center_x=1.4, center_y=1.2, size_x=0.7, size_y=0.35)

    return grid

# Start and Goal Positions
def define_start_and_end_goal():
    """
    Defining the Webots Robot's starting position and determining the end goal

    Coordinates are represented as the following: (row, column)

    Robot (-2.0, -2.0)
    Goal (+2.0, +2.0)
    """

    start_world = (-2.0, 2.0)
    goal_world = (2.0, -2.0)

    start = world_to_grid(start_world[0], start_world[1])
    goal = world_to_grid(goal_world[0], goal_world[1])

    return start, goal, start_world, goal_world

# Dijkstra's Algorithm
def min_distance(distances, spt_set):
    """
    Finding the Unprocessed Grid Cells with the Smallest Distance from the Starting Position
    """

    minimum = float("inf")
    minimum_index = None

    for position in distances:
        if ((not spt_set[position]) and (distances[position] < minimum)):
            minimum = distances[position]
            minimum_index = position

    return minimum_index

def dijkstra(grid, start, goal):
    """
    Dijkstra's Shortest-Path Algorithm Adapted for a 2D Occupancy Grid

    0 = Free Space
    1 = Obstacle

    :param
        grid: The 2D Occupancy Grid
        start: The Robot's Starting Position (row, column)
        goal: The End Goal (row, column)

    :return:
        path: List of Grid positions which will form the shortest path.
        nodes_explored: Number of grid cells explored.
    """

    # Creating a list of all the free grid cell positions
    vertices = []

    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if grid[row, column] == 0:
                vertices.append((row, column))

    # Initializing Distances
    distances = {
        vertex: float("inf")

        for vertex in vertices
    }

    if start not in distances:
        return [], 0

    if goal not in distances:
        return [], 0

    distances[start] = 0

    # Shortest Path Tree Set
    # False - Not Yet Processed
    # True - Shortest Distance Has Been Calculated
    spt_set = {
        vertex: False

        for vertex in vertices
    }

    # Storing the previous position which was used in order to reconstruct the path after reaching the Goal Position
    previous = {}

    # Directional Movement: Up, Down, Left, and Right
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    # Initializing visited nodes to 0.
    nodes_explored = 0

    # Processing vertices until the Goal Position is reached or until there are no more reachable vertices remaining.
    for _ in range(len(vertices)):
        current = min_distance(distances, spt_set)

        # No reachable vertex remains.
        if current is None:
            break

        # Finalize the current vertex.
        spt_set[current] = True
        nodes_explored += 1

        # Stop when the goal has been reached.
        if current == goal:
            break

        current_row, current_column = current

        # Examine Adjacent Grid Cells.
        for row_change, column_change in directions:
            neighbour = (current_row + row_change, current_column + column_change)

            # Check that the neighbour exists in the grid.
            if neighbour not in distances:
                continue

            # Ignore the vertices already found in the shortest-path tree
            if spt_set[neighbour]:
                continue

            # Every movement between adjacent cells has produced a total cost of 1.
            edge_weight = 1

            new_distance = (distances[current] + edge_weight)

            # Relax the Edge.
            if new_distance < distances[neighbour]:
                distances[neighbour] = new_distance

                previous[neighbour] = current

    # If the goal was never reached, no path would exist.
    if distances[goal] == float("inf"):
        return [], nodes_explored

    # Reconstructing the Shortest Path
    path = []

    current = goal

    while (current != start):
        path.append(current)

        current = previous[current]

    path.append(start)

    # Reverse the path so that it goes: Start TO Goal
    path.reverse()

    return path, nodes_explored

# A* Heuristic
def heuristic(current, goal):
    """
    Calculating the Manhattan distance between the Current Position and the Goal Position.

    The Manhattan Distance is suitable here since the robot only moves horizontally and vertically.
    """
    return (abs(current[0] - goal[0]) + abs(current[1] - goal[1]))

def a_star(grid, start, goal):
    """
    A* Shortest-Path Algorithm Adapted for a 2D Occupancy Grid

    The Algorithm will combine the following:
        g(n) which represents the total cost from the starting position through to the current position
        h(n) which represents the estimated cost fron the current position to the goal position.

    The formula is represented as follows: f(n) = g(n) + h(n)

    :param
        grid: The 2D Occupancy Grid
        start: The Robot's Starting Position
        goal: The Goal Position

    :return:
        path: List of grid positions which form the path
        nodes_explored: Number of grid cells which have been explored
    """

    # Storing the cost from the Starting Position through to each position
    distances = {start: 0}

    # Storing the previous position for path reconstruction
    previous = {}

    # Storing the positions that still have to be evaluated
    open_set = [start]

    # Keeping track of all positions that have been explored
    explored = set()

    # Directional Movement: Up, Down, Left, and Right
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    # Initializing visited nodes to 0.
    nodes_explored = 0

    while open_set:
        # Selecting the position with the lowest estimated total cost.
        current = min(
            open_set,
            key=lambda position: distances[position] + heuristic(position, goal)
        )

        open_set.remove(current)

        # Ignoring all positions that have already been explored.
        if current in explored:
            continue

        explored.add(current)
        nodes_explored += 1

        # Stop once the goal is reached.
        if current == goal:
            break

        current_row, current_column = current

        # Examining Neighbouring Cells
        for row_change, column_change in directions:
            neighbour = (current_row + row_change, current_column + column_change)

            row, column = neighbour

            # Check that the neighbour is inside the grid
            if not ((0 <= row < grid.shape[0]) and (0 <= column < grid.shape[1])):
                continue

            # Ignoring obstacles
            if grid[row, column] == 1:
                continue

            # Ignoring already explored cells
            if neighbour in explored:
                continue

            # Movement cost between adjacent cells.
            movement_cost = 1

            new_distance = (distances[current] + movement_cost)

            # Add the neighbour if it hasn't been evaluated yet
            if ((neighbour not in distances) or (new_distance < distances[neighbour])):
                distances[neighbour] = new_distance
                previous[neighbour] = current

                if neighbour not in open_set:
                    open_set.append(neighbour)

    # No path was found
    if goal not in distances:
        return [], nodes_explored

    # Reconstructing the path
    path = []

    current = goal

    while (current != start):
        path.append(current)
        current = previous[current]

    path.append(start)

    # Reverse the path so it runs from Starting Position to the Goal Position
    path.reverse()

    return path, nodes_explored

# Path Conversion
def grid_path_to_world(path):
    """
    Converting a complete grid path into Webots World Coordinates
    :return: [(x1, y1), (x2, y2), ...]
    """

    return [
        grid_to_world(row, column)

        for row, column in path
    ]

def extract_waypoints(path, start_world, goal_world):
    """
    Reducing the raw grid path to meaningful turning points.

    A waypoint becomes retained whenever the direction of travel changes. The exact Webots start and goal
    coordinates are used as the first and last waypoints.
    """
    if not path:
        return []

    if len(path) == 1:
        return [
            start_world,
            goal_world
        ]

    waypoints = []

    # Always use the actual physical robot start
    waypoints.append(start_world)

    # Determining the direction between consecutive grid cells.
    previous_direction = None

    for i in range(1, len(path)):
        previous_row, previous_col = path[i - 1]
        current_row, current_col = path[i]

        direction = (
            current_row - previous_row,
            current_col - previous_col
        )

        # A direction change means that the robot needs to turn.
        if (previous_direction is not None) and (direction != previous_direction):
            waypoint = grid_to_world(
                path[i - 1][0],
                path[i - 1][1]
            )

            waypoints.append(waypoint)

        previous_direction = direction

    # Always us the actual physical goal
    waypoints.append(goal_world)

    return waypoints

def save_waypoints(waypoints, filename=None):
    """
    Saving the extracted A* waypoints into a Python Module.

    By saving it into a new python module, it can be imported by the Webots controller.
    """
    if filename is None:
        filename = CONTROLLER_DIR / "waypoints.py"
    else:
        filename = Path(filename)

        if not filename.is_absolute():
            filename = CONTROLLER_DIR / filename

    with open(filename, "w", encoding="utf-8") as file:
        file.write("# A* Waypoints are generated automatically.\n")
        file.write("A_STAR_WAYPOINTS = [\n")

        for x, y in waypoints:
            file.write(f"    ({x:.2f}, {y:.2f}),\n")

        file.write("]\n")

# Path Length
def calculate_path_length(path):
    """
    Calculating the physical path length in metres. Each grid-cell transition represents CELL_SIZE metres.
    """
    if len(path) < 2:
        return 0.0

    return (len(path) - 1) * CELL_SIZE

def display_occupancy_grid(grid, start, goal, dijk_path=None, a_path=None):
    """
    Display the Occupancy Grid together with both algorithms through Matplotlib.

    :param
        grid: the results obtained from the create_occupancy_grid function.

    :return: Displaying a visualization of the occupancy grid through Matplotlib.
    """
    from pathlib import Path
    from matplotlib.colors import ListedColormap

    # Finding the Component A directory from this script's location
    component_a_dir = Path(__file__).resolve().parents[2]
    results_dir = component_a_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    occupancy_cmap = ListedColormap(["#F4F4F4", "#333333"])

    ax.imshow(
        grid,
        cmap=occupancy_cmap,
        origin="upper",
        extent=[
            WORLD_MIN_X,
            WORLD_MAX_X,
            WORLD_MIN_Y,
            WORLD_MAX_Y
        ],
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    obstacles = [
        {
            "x": -0.8,
            "y": 0.8,
            "sx": 1.2,
            "sy": 0.35
        },
        {
            "x": 0.8,
            "y": 0.0,
            "sx": 0.35,
            "sy": 1.2
        },
        {
            "x": 1.4,
            "y": 1.2,
            "sx": 0.8,
            "sy": 0.35
        }
    ]

    for obstacle in obstacles:
        lower_left_x = obstacle["x"] - (obstacle["sx"] / 2)
        lower_left_y = obstacle["y"] - (obstacle["sy"] / 2)

        rectangle = plt.Rectangle(
            (
                lower_left_x,
                lower_left_y
            ),
            obstacle["sx"],
            obstacle["sy"],
            fill=False,
            lw=3
        )

        ax.add_patch(rectangle)

    # Drawing the Dijkstra Path
    if dijk_path:
        path_world = grid_path_to_world(dijk_path)

        ax.plot(
            [p[0] for p in path_world],
            [p[1] for p in path_world],
            color="royalblue",
            lw=2.5,
            label="Dijkstra Path",
            zorder=3
        )

    # Drawing the Dijkstra Path
    if a_path:
        a_world = grid_path_to_world(a_path)

        ax.plot(
            [p[0] for p in a_world],
            [p[1] for p in a_world],
            color="darkorange",
            lw=2.5,
            ls="--",
            label="A* Path",
            zorder=4
        )

        # Mark A* waypoints where the direction changes
        waypoint_indices = [0]

        for i in range(1, len(a_path) - 1):
            previous = a_path[i - 1]
            current = a_path[i]
            following = a_path[i + 1]

            direction_before = (
                current[0] - previous[0],
                current[1] - previous[1]
            )

            direction_after = (
                following[0] - current[0],
                following[1] - current[1]
            )

            if direction_before != direction_after:
                waypoint_indices.append(i)

        if len(a_path) > 1:
            waypoint_indices.append(len(a_path) - 1)

        waypoint_world = [
            grid_to_world(*a_path[i])

            for i in waypoint_indices
        ]

        ax.scatter(
            [p[0] for p in waypoint_world],
            [p[1] for p in waypoint_world],
            color="darkorange",
            marker="o",
            s=45,
            label="A* Waypoints",
            zorder=5
        )

    # Drawing the Starting Position
    start_x, start_y = grid_to_world(*start)
    goal_x, goal_y = grid_to_world(*goal)

    ax.scatter(
        start_x,
        start_y,
        color="green",
        marker="o",
        s=120,
        edgecolors="black",
        label="Starting Position",
        zorder=6
    )

    ax.scatter(
        goal_x,
        goal_y,
        color="red",
        marker="*",
        s=200,
        edgecolors="black",
        label="Goal Position",
        zorder=6
    )

    ax.set_title("Webots Robot Occupancy Grid and Planned Paths")
    ax.set_xlabel("Webots X Position (m)")
    ax.set_ylabel("Webots Y Position (m)")

    ax.set_xlim(WORLD_MIN_X, WORLD_MAX_X)
    ax.set_ylim(WORLD_MIN_Y, WORLD_MAX_Y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(np.arange(WORLD_MIN_X, WORLD_MAX_X + 0.01, 0.5))
    ax.set_yticks(np.arange(WORLD_MIN_Y, WORLD_MAX_Y + 0.01, 0.5))

    ax.grid(True, alpha=0.2)

    ax.legend(loc="upper right")

    plt.tight_layout()

    figure_path = RESULTS_DIR / "occupancy_grid_paths.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")

    print(f"\nPlanning Visualization Saved To: {figure_path}")

    plt.show()

if __name__ == "__main__":
    occupancy_grid = create_occupancy_grid()

    start, goal, start_world, goal_world = define_start_and_end_goal()

    print("=" * 50)
    print("Component A - Motion Planning")
    print("=" * 50)

    print(
        f"\nWorld Starting Position:"
        f"X={start_world[0]:.2f} m, "
        f"Y={start_world[1]:.2f} m"
    )

    print(
        f"Grid Start Position: "
        f"{start}"
    )

    print(
        f"\nWorld Goal Position"
        f"X={goal_world[0]:.2f} m, "
        f"Y={goal_world[1]:.2f} m"
    )

    print(
        f"Grid Goal Position: "
        f"{goal}"
    )

    print(
        f"\nGrid Resolution: "
        f"{CELL_SIZE:.2f} m/cell"
    )

    print("\nOccupancy Grid:")
    print(occupancy_grid)

    # Running the Dijkstra's Shortest-Path Algorithm
    dijk_path, dijk_nodes = dijkstra(occupancy_grid, start, goal)

    print("\n")
    print("=" * 50)
    print("Dijkstra Path:")
    print("=" * 50)

    if dijk_path:
        print(
            f"Grid Path Length: "
            f"{len(dijk_path)} cells"
        )

        print(
            f"Physical Path Length: "
            f"{calculate_path_length(dijk_path):.2f} m"
        )

        print(
            f"Nodes Explored: "
            f"{dijk_nodes}"
        )

        print("Path Found: YES")
    else:
        print("Path Found: NO")

    # Running the A* Shortest-Path Algorithm
    a_path, a_nodes = a_star(occupancy_grid, start, goal)

    # Extracting A* Waypoints
    a_waypoints = extract_waypoints(a_path, start_world, goal_world)

    # Saving A* Waypoints for the Webots Controller
    save_waypoints(a_waypoints)

    print("\n")
    print("=" * 50)
    print("A* Waypoints:")
    print("=" * 50)

    for index, waypoint in enumerate(a_waypoints, start=1):
        print(
            f"Waypoint {index:02d}: "
            f"X={waypoint[0]:.2f} m, "
            f"Y={waypoint[1]:.2f} m"
        )

    print(
        f"\nTotal Waypoints: "
        f"{len(a_waypoints)}"
    )

    print("\n")
    print("=" * 50)
    print("A* Path:")
    print("=" * 50)

    if a_path:
        print(
            f"Grid Path Length: "
            f"{len(a_path)} cells"
        )

        print(
            f"Physical Path Length: "
            f"{calculate_path_length(a_path):.2f} m"
        )

        print(
            f"Nodes Explored: "
            f"{a_nodes}"
        )

        print("Path Found: YES")
    else:
        print("Path Found: NO")

    display_occupancy_grid(occupancy_grid, start, goal, dijk_path, a_path)