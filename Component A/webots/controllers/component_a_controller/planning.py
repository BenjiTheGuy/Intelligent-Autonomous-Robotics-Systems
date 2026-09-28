import heapq
import numpy as np
import matplotlib.pyplot as plt

def create_occupancy_grid():
    """
    Creating a 2D Occupancy Grid which represents the robot's environment.

    0 = Free Space
    1 = Occupied (An obstacle is present)
    """

    # Creating a 20 x 20 grid
    grid = np.zeros((20, 20), dtype=int)

    # Creating the boundary walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Adding an internal obstacle representing an object that has been detected by the robot's perception system.
    grid[2, 7:14] = 1
    grid[6:11, 13:16] = 1

    # Adding another obstacle onto the grid in order to make path planning meaningful.
    grid[10:15, 2:6] = 1
    grid[13:17, 9:13] = 1

    return grid

def define_start_and_end_goal():
    """
    Defining the Robot's starting position and determining the end goal

    Coordinates are represented as the following: (row, column)
    """

    start = (18, 2)
    goal = (2, 17)

    return start, goal

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

    # Distance from the Start Position to every vertex.
    distances = {
        vertex: float("inf")

        for vertex in vertices
    }

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

def display_occupancy_grid(grid, start, goal, path=None):
    """
    Display the Occupancy Grid through Matplotlib.

    :param
        grid: the results obtained from the create_occupancy_grid function.

    :return: Displaying a visualization of the occupancy grid through Matplotlib.
    """

    plt.figure(figsize=(9, 6))

    plt.imshow(grid, cmap="viridis", origin="upper")

    # Drawing the Dijkstra Path
    if path:
        path_rows = [position[0] for position in path]
        path_columns = [position[1] for position in path]

        plt.plot(path_columns, path_rows, lw=3, label="Dijkstra Path")

    # Drawing the A* Path
    if a_star_path:
        path_rows = [
            position[0]

            for position in a_star_path
        ]

        path_columns = [
            position[1]

            for position in a_star_path
        ]

        plt.plot(path_columns, path_rows, lw=2, ls="--", label="A* Path")

    # Drawing the Starting Position
    if start is not None:
        plt.scatter(start[1], start[0], marker="o", s=150, label="Starting Position")

    # Drawing the Goal Position
    if goal is not None:
        plt.scatter(goal[1], goal[0], marker="*", s=200, label="Goal Position")

    plt.title("Robot Occupancy Grid")
    plt.xlabel("X Grid Position")
    plt.ylabel("Y Grid Position")

    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    occupancy_grid = create_occupancy_grid()

    start, goal = define_start_and_end_goal()

    print(f"Starting Position: {start}")
    print(f"Goal Position: {goal}\n")

    print("Occupancy Grid: ")
    print(occupancy_grid)

    # Running the Dijkstra's Shortest-Path Algorithm
    path, nodes_explored = dijkstra(occupancy_grid, start, goal)

    print("\nDijkstra Path:")
    print(path)

    print(f"\nPath Length: {len(path)}")
    print(f"Nodes Explored: {nodes_explored}")

    # Running the A* Shortest-Path Algorithm
    a_star_path, a_star_nodes_explored = a_star(occupancy_grid, start, goal)

    print("\nA* Path:")
    print(a_star_path)

    print(f"\nA* Path Length: {len(a_star_path)}")
    print(f"A* Nodes Explored: {a_star_nodes_explored}")

    display_occupancy_grid(occupancy_grid, start, goal, path)