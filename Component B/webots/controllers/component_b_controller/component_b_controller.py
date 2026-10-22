# Importing libraries
from controller import Robot
import csv
import os
import json
import math
import joblib

# Creating this controller instance and obtaining Webots' basic simulation time step.
# Declaring the Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Identifying the Robot
robot_name = robot.getName()
print(f"Starting {robot_name}")

# These handles are initialized here; only robot_1 opens and writes the shared consensus log.
# CSV Logging
log_file = None
csv_writer = None

# Cache the latest position and route status received from each other robot.
robot_positions = {}

# Use robot_1 as the single CSV writer to avoid three controllers overwriting the same file.
if robot_name == "robot_1":
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    data_directory = os.path.join(root, "data")
    os.makedirs(data_directory, exist_ok=True)

    # CSV File Path
    log_file_path = os.path.join(data_directory, "consensus_data.csv")

    # Open CSV File
    log_file = open(log_file_path, "w", newline="")

    csv_writer = csv.writer(log_file)

    # Write CSV Header
    csv_writer.writerow([
        "time",
        "robot_1",
        "robot_2",
        "robot_3"
    ])

    log_file.flush()

# Each robot starts with a different scalar value so the distributed consensus process can be observed.
# Setting the Identity of the Robot and the Initial Consensus Values
initial_values = {
    "robot_1": 20.0,
    "robot_2": 50.0,
    "robot_3": 80.0
}

consensus_value = initial_values.get(robot_name, 0.0)

print(f"{robot_name} started with value: {consensus_value:.2f}")

# None means that this controller has not yet received a value from that robot.
# Storing the latest known consensus value of each robot
consensus_values = {
    "robot_1": None,
    "robot_2": None,
    "robot_3": None
}

consensus_values[robot_name] = consensus_value

# alpha controls how strongly each robot adjusts its value toward the values received from neighbours.
# Consensus Parameters
alpha = 0.2

# The emitter broadcasts this robot's state; the receiver collects state messages from other robots.
# Communication Devices
emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")

# Enable Receiver
receiver.enable(timestep)

# GPS supplies the robot's position in the simulation world.
# GPS Device
gps = robot.getDevice("gps")
gps.enable(timestep)

# The IMU provides roll, pitch, yaw; yaw is used for planner steering.
# Enabling the inertial unit for orientation readings
imu = robot.getDevice("imu")
imu.enable(timestep)

# Waiting for sensor readings to become available
robot.step(timestep)

print(f"{robot_name}: Initial GPS Position = {gps.getValues()}")
print(f"{robot_name}: IMU = {imu.getRollPitchYaw()}")

# Loading the model trained by machine_learning.py; its expected input is the same four-feature order used during
# Loading the Trained Navigation Neural Network
controller_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(controller_dir, "..", "..", ".."))

model_path = os.path.join(
    project_root,
    "results",
    "navigation_model.joblib"
)

navigation_model = joblib.load(model_path)

print(f"{robot_name}: Navigation Model Loaded Successfully")

# Each controller reads the precomputed routes and selects only the route assigned to its own robot.
# Loading the Robot's A* Path
controller_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(controller_dir, "..", "..", ".."))

route_file = os.path.join(project_root, "data", "astar_full_paths.json")

with open(route_file, "r", encoding="utf-8") as file:
    all_routes = json.load(file)

if robot_name not in all_routes:
    raise ValueError(f"No Planned Route Found for {robot_name}")

# Skip the first route point because it represents the starting position, not a waypoint the robot needs to drive
# forward.
route = all_routes[robot_name][1:]

waypoint_index = 0

print(f"{robot_name}: Loaded {len(route)} A* Waypoints")

# Retrieving all four wheel motors used by this robot.
right_front = robot.getDevice("right_front_motor")
right_rear = robot.getDevice("right_rear_motor")
left_front = robot.getDevice("left_front_motor")
left_rear = robot.getDevice("left_rear_motor")

motors = [right_front, right_rear, left_front, left_rear]

# Infinite position enables velocity control rather than moving to a fixed motor angle.
# Configuring motors for velocity control
for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# Tune maximum wheel speed, waypoint arrival tolerance, steering strength, and heading tolerance.
# Navigation Parameters
MAX_SPEED = 2.0
MIN_SPEED = 0.0
WAYPOINT_TOLERANCE = 0.25
TURN_GAIN = 2.0
HEADING_TOLERANCE = 0.15

# Define the crossing location and separation thresholds used by the special yielding and collision avoidance rules
# Robot 2 yields to Robot 3 near their shared crossing
CONFLICT_ZONE = (-2.2, -1.6)
CONFLICT_RADIUS = 1.0
ROBOT_CLEARANCE_DISTANCE = 1.2
EMERGENCY_DISTANCE = 0.45

print(f"{robot_name}: Wheel Motors Initialized")

# Wrapping an angle into [-pi, pi], preventing steering from taking the long way around.
def normalize_angle(angle):
    """
    Keep an angle between -pi and +pi.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

# Compute straight-line distance to a waypoint and the signed heading error needed to face it.
def calculate_navigation(position, target, yaw):
    """
    Calculate distance and heading error to a waypoint.
    """
    dx = target[0] - position[0]
    dy = target[1] - position[1]

    distance = math.hypot(dx, dy)

    # In this world, yaw=0 points along +X
    target_heading = math.atan2(dy, dx)
    heading_error = normalize_angle(target_heading - yaw)

    return distance, heading_error

# Using received robot positions to estimate the nearest neighbour and its bearing relative to this robot.
def get_nearest_robot(position, yaw):
    """
    Finding the nearest other robot whose position is known.

    This function returns:
        distance: distance to nearest robot
        bearing: relative angle to nearest robot
    """
    nearest_distance = 5.0
    nearest_bearing = 0.0

    for other_name, info in robot_positions.items():
        if other_name == robot_name or info is None:
            continue

        other_position = info["position"]

        dx = other_position[0] - position[0]
        dy = other_position[1] - position[1]

        distance = math.hypot(dx, dy)

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_bearing = normalize_angle(math.atan2(dy, dx) - yaw)

    return nearest_distance, nearest_bearing

# Passing the live navigation features to the saved classifier and return its predicted action label.
def predict_navigation_action(
        heading_error,
        waypoint_distance,
        nearest_robot_distance,
        neighbour_bearing
):
    """
    Predicting the next local navigation action
    """
    features = [[
        heading_error,
        waypoint_distance,
        nearest_robot_distance,
        neighbour_bearing
    ]]

    prediction = navigation_model.predict(features)[0]

    return prediction

# Special-case crossing rule: robot_2 pauses when robot_3 is near the conflict zone and the robots are close.
def should_robot_2_yield():
    """
    Stopping robot 2 while robot 3 occupies the crossing area.
    """
    if robot_name != "robot_2":
        return False

    robot_3 = robot_positions.get("robot_3")

    # If robot 3's position is not known yet, do not yield.
    if robot_3 is None:
        return False

    robot_3_pos = robot_3["position"]

    zone_x, zone_y = CONFLICT_ZONE

    distance_to_zone = math.hypot(
        robot_3_pos[0] - zone_x,
        robot_3_pos[1] - zone_y
    )

    distance_between_robots = math.hypot(
        current_position[0] - robot_3_pos[0],
        current_position[1] - robot_3_pos[1]
    )

    return (
        distance_to_zone < CONFLICT_RADIUS and
        distance_between_robots < ROBOT_CLEARANCE_DISTANCE
    )

# The controller repeats communication, consensus updates, waypoint navigation, and wheel control each simulation step.
# Main Loop
step_counter = 0

# Update the robot's latest consensus value
consensus_values[robot_name] = consensus_value

robot_positions = {
    "robot_1": None,
    "robot_2": None,
    "robot_3": None,
}

# Webots returns -1 when the simulation ends otherwise process one control cycle.
while robot.step(timestep) != -1:
    # Broadcasting a message with the identity of the robot, the consensus value, and its current position.
    gps_values = gps.getValues()
    current_position = (gps_values[0], gps_values[1])

    current_x = gps_values[0]
    current_y = gps_values[1]

    if waypoint_index < len(route):
        next_target = route[waypoint_index]
        distance_to_target = math.hypot(
            next_target[0] - current_x,
            next_target[1] - current_y
        )

        route_active = 1
    else:
        distance_to_target = 1e9
        route_active = 0

    message = (
        f"{robot_name}, {consensus_value}, "
        f"{current_x}, {current_y}, "
        f"{distance_to_target}, {route_active}"
    )

    emitter.send(message.encode("utf-8"))

    # Collecting the consensus values from messages received during this cycle.
    # Storing values received from other robots.
    neighbour_values = []

    # Drain the receiver queue so all waiting packets are processed in this simulation step.
    # Checking for incoming messages
    while receiver.getQueueLength() > 0:
        received_msg = receiver.getString()

        receiver.nextPacket()

        # Parse each packet defensively, malformed or incomplete messages are ignored below.
        try:
            sender, value, x, y, target_distance, active = [
                item.strip()

                for item in received_msg.split(",")
            ]

            # Ignoring the robot's own message
            if sender != robot_name and sender in consensus_values:
                value = float(value)

                x = float(x)
                y = float(y)
                target_distance = float(target_distance)
                active = bool(int(active))

                # Storing the latest value received from this robot.
                consensus_values[sender] = value
                neighbour_values.append(value)

                # Storing the sender's last position
                robot_positions[sender] = {
                    "position": (x, y),
                    "target_distance": target_distance,
                    "active": active
                }

        except (ValueError, TypeError):
            continue

    # Move this robot's value a fraction alpha toward the mean of the neighbour values received this cycle.
    # Average Consensus Update
    if neighbour_values:
        average_difference = sum(
            value - consensus_value

            for value in neighbour_values
        ) / len(neighbour_values)

        consensus_value += alpha * average_difference

        # Keeping the robot's stored consensus value up to date.
        consensus_values[robot_name] = consensus_value

    # Periodic console output provides a lightweight progress trace without printing every time step.
    # Logging the Result
    if step_counter % 1000 == 0:
        print(
            f"{robot_name} | "
            f"Consensus Value: {consensus_value:.4f}"
        )

    # Log only when values from all three robots are known, so each CSV row contains a complete sample.
    # Only robot_1 writes to the csv
    if step_counter % 10 == 0:
        if (robot_name == "robot_1") and (csv_writer is not None) and (all(
            value is not None

            for value in consensus_values.values()
        )
        ):
            # Write Simulation Time and all known robot consensus values
            csv_writer.writerow([
                robot.getTime(),
                consensus_values["robot_1"],
                consensus_values["robot_2"],
                consensus_values["robot_3"]
            ])

            # Make sure the data is written immediately
            log_file.flush()

    # Following the robot-specific A* route, applying the special crossing rule before normal navigation.
    # A* Waypoint Navigation
    if waypoint_index < len(route):
        # Robot 2 waits for Robot 3 to clear the crossing.
        if should_robot_2_yield():
            if step_counter % 100 == 0:
                robot_3 = robot_positions.get("robot_3")

                if robot_3 is not None:
                    robot_3_pos = robot_3["position"]

                    distance_to_robot_3 = math.hypot(
                        current_position[0] - robot_3_pos[0],
                        current_position[1] - robot_3_pos[1]
                    )

                    distance_to_zone = math.hypot(
                        robot_3_pos[0] - CONFLICT_ZONE[0],
                        robot_3_pos[1] - CONFLICT_ZONE[1]
                    )

                    print(
                        f"{robot_name} | SPECIAL YIELD RULE | "
                        f"Robot 3 distance={distance_to_robot_3:.2f} m | "
                        f"Robot 3 to conflict zone={distance_to_zone:.2f} m"
                    )
                else:
                    print(f"{robot_name} | Waiting: Robot 3 position unknown")

            for motor in motors:
                motor.setVelocity(0.0)
        else:
            # Extracting the yaw from the IMU for heading-error calculation.
            # Reading the Robot's Current Heading
            yaw = imu.getRollPitchYaw()[2]

            # Getting the next waypoint
            target = route[waypoint_index]
            target_position = (target[0], target[1])

            # Calculating the distance and heading error
            distance, heading_error = calculate_navigation(current_position, target_position, yaw)

            # Get the nearest known robot
            nearest_distance, neighbour_bearing = get_nearest_robot(current_position, yaw)

            # The neural network predicts a local action from heading, target distance, and neighbour information.
            # Predicting the local navigation action
            predicted_action = predict_navigation_action(
                heading_error,
                distance,
                nearest_distance,
                neighbour_bearing
            )

            # Override the learned prediction if a known robot is closer than the configured clearance distance
            # Independent proximity safety check
            if nearest_distance < ROBOT_CLEARANCE_DISTANCE:
                predicted_action = "YIELD"

            # Log decisions periodically
            if step_counter % 100 == 0:
                print(
                    f"{robot_name} | "
                    f"Action: {predicted_action} | "
                    f"Nearest Robot: {nearest_distance:.2f} m"
                )

            # Once within tolerance, advance the waypoint index and briefly stop the wheels.
            # Moving to the next waypoint when close enough
            if distance < WAYPOINT_TOLERANCE:
                print(
                    f"{robot_name}: REACHED Waypoint "
                    f"{waypoint_index + 1}/{len(route)} "
                    f"at ({target_position[0]:.2f}, "
                    f"{target_position[1]:.2f}) | "
                    f"Distance: {distance:.3f} m"
                )

                waypoint_index += 1

                # Stopping briefly while changing waypoint
                for motor in motors:
                    motor.setVelocity(0.0)

            else:
                # Resolving close encounters using a fixed priority order, with a special allowance for robot_3
                # to clear the crossing.
                # Deterministic right-of-way with a conflict-zone override.
                # Normal priority: robot_1 > robot_2 > robot_3
                # Exception: robot_3 may clear the crossing while robot_2 waits.
                robot_priority = {
                    "robot_1": 0,
                    "robot_2": 1,
                    "robot_3": 2
                }

                # Find the nearest robot for emergency safety, and the nearest ACTIVE robot for right-of-way.
                nearest_any_distance = float("inf")
                nearest_active_name = None
                nearest_active_distance = float("inf")

                for other_name, info in robot_positions.items():
                    if other_name == robot_name or info is None:
                        continue

                    other_position = info["position"]
                    dx = other_position[0] - current_position[0]
                    dy = other_position[1] - current_position[1]
                    other_distance = math.hypot(dx, dy)

                    # All robots count for emergency separation
                    nearest_any_distance = min(
                        nearest_any_distance, other_distance
                    )

                    # Only robots with active routes determine priority.
                    if info.get("active", False):
                        if other_distance < nearest_active_distance:
                            nearest_active_distance = other_distance
                            nearest_active_name = other_name

                # Determining if robot_3 is currently inside the conflict zone
                distance_to_conflict_zone = math.hypot(
                    current_position[0] - CONFLICT_ZONE[0],
                    current_position[1] - CONFLICT_ZONE[1]
                )

                robot_3_clearing_crossing = (
                    robot_name == "robot_3"
                    and distance_to_conflict_zone < CONFLICT_RADIUS
                    and nearest_active_name == "robot_2"
                    and nearest_active_distance > EMERGENCY_DISTANCE
                )

                # Emergency separation applies to all robots including robots that have completed their routes.
                emergency_stop = (
                    nearest_any_distance <= EMERGENCY_DISTANCE
                )

                # Right of way applies only to active robots.
                must_yield = (
                    nearest_active_name is not None
                    and nearest_active_distance < ROBOT_CLEARANCE_DISTANCE
                    and robot_priority[robot_name] > robot_priority[nearest_active_name]
                )

                # Let robot_3 clear the crossing to avoid deadlock.
                if robot_3_clearing_crossing:
                    must_yield = False

                # Applying the final movement decision.
                if emergency_stop or must_yield:
                    left_speed = 0.0
                    right_speed = 0.0

                else:
                    # Preserving the original heading-based steering
                    turn = max(
                        -MAX_SPEED,
                        min(MAX_SPEED, TURN_GAIN * heading_error)
                    )

                    # Reducing the forward speed when the robot is not facing the waypoint
                    forward = MAX_SPEED * max(
                        0.0,
                        1.0 - abs(heading_error) / math.pi
                    )

                    # Calculating differential wheel speeds.
                    left_speed = max(
                        -MAX_SPEED,
                        min(MAX_SPEED, forward - turn)
                    )

                    right_speed = max(
                        -MAX_SPEED,
                        min(MAX_SPEED, forward + turn)
                    )

                # EXTRA CODE
                if step_counter % 50 == 0:
                    print(
                        f"{robot_name} | "
                        f"Position=({current_position[0]:.2f}, "
                        f"{current_position[1]:.2f}) | "
                        f"Target=({target_position[0]:.2f}, "
                        f"{target_position[1]:.2f}) | "
                        f"Yaw={yaw:.2f} | "
                        f"Heading Error={heading_error:.2f} | "
                        f"Distance={distance:.2f} | "
                        f"Left Speed={left_speed:.2f} | "
                        f"Right Speed={right_speed:.2f}"
                    )

                # Applying wheel velocities
                left_front.setVelocity(left_speed)
                left_rear.setVelocity(left_speed)
                right_front.setVelocity(right_speed)
                right_rear.setVelocity(right_speed)
    else:
        # All waypoints completed
        for motor in motors:
            motor.setVelocity(0.0)

        if waypoint_index == len(route):
            print(
                f"{robot_name}: A* Route Completed "
                f"({len(route)}/{len(route)} waypoints reached)"
            )

            waypoint_index += 1

    # Increment Simulation Step Counter
    step_counter += 1

# Close CSV File
if log_file is not None:
    log_file.close()