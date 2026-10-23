# Importing libraries to use in file
import math
import csv

from controller import Robot

from control import (
    PIDController,
    choose_avoidance_direction,
    get_side_clearance,
    avoidance_wheel_speeds,
    calculate_heading_alignment_command
)

from perception import (
    process_lidar_data,
    process_camera_image_data,
    sensor_fusion_data
)

from waypoints import A_STAR_WAYPOINTS

from pathlib import Path

# Locate Component A from this controller's directory.
COMPONENT_A_DIR = Path(__file__).resolve().parents[3]

# Directory for CSV data
DATA_DIR = COMPONENT_A_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Function used to control the speed of the motors
def set_motor_speeds(left_speed, right_speed):
    """
    Applying the same left/right speeds on all 4 wheels.
    """
    left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

    left_front_motor.setVelocity(left_speed)
    left_rear_motor.setVelocity(left_speed)

    right_front_motor.setVelocity(right_speed)
    right_rear_motor.setVelocity(right_speed)

# Create the Webots Robot
robot = Robot()

# Webots Simulation Timestep
timestep = int(robot.getBasicTimeStep())

initial_x = None
initial_y = None
initial_heading = None

# Creating the Devices
lidar = robot.getDevice("lidar")
camera = robot.getDevice("camera")

left_front_motor = robot.getDevice("left_front_motor")
left_rear_motor = robot.getDevice("left_rear_motor")
right_rear_motor = robot.getDevice("right_rear_motor")
right_front_motor = robot.getDevice("right_front_motor")

gps = robot.getDevice("gps")
imu = robot.getDevice("imu")

# Setting up the motors
left_front_motor.setPosition(float("inf"))
left_front_motor.setVelocity(0.0)

left_rear_motor.setPosition(float("inf"))
left_rear_motor.setVelocity(0.0)

right_rear_motor.setPosition(float("inf"))
right_rear_motor.setVelocity(0.0)

right_front_motor.setPosition(float("inf"))
right_front_motor.setVelocity(0.0)

# Sensor Setup
camera_width = camera.getWidth()
camera_height = camera.getHeight()

lidar.enable(timestep)
camera.enable(timestep)
gps.enable(timestep)
imu.enable(timestep)

# PID Controller
pid = PIDController(
    kp=3.0,
    ki=0.05,
    kd=0.4
)

# Robot Settings
# Base Wheel Speed
base_speed = 1.0

# A* Waypoint Navigation
waypoints = A_STAR_WAYPOINTS
current_waypoint_index = 0

# Distance at which the robot considers a waypoint reached.
WAYPOINT_TOLERANCE = 0.20

# Used to prevent the robot from stopping prematurely at the final waypoint
GOAL_TOLERANCE = 0.25

# PID Gains
heading_kp = 1.0

# Maximum Webots Motor Speed
MAX_SPEED = 6.28

# Obstacle Avoidance Settings
OBSTACLE_TURN_SPEED = 2.0
OBSTACLE_FORWARD_SPEED = 3.0

FRONT_OBSTACLE_ENTER = 0.90
FRONT_OBSTACLE_CLEAR = 1.50

# LiDAR Emergency Stop Distance. Independent of the normal front-fused obstacle threshold.
EMERGENCY_DISTANCE = 0.40

# Minimum number of steps required for commitment to a selected turn.
MIN_TURN_STEPS = 40

# Safety Timeout.
MAX_AVOIDANCE_STEPS = 100

# Required clearance on the selected side before rejoining the A* path.
MIN_SIDE_CLEARANCE = 0.90

CLEAR_CONFIRMATION_STEPS = 8
clear_confirmation_count = 0

FORWARD_CLEAR_STEPS = 30
forward_clear_count = 0

HEADING_ALIGN_ENTER = 0.80
HEADING_ALIGN_EXIT = 0.30
HEADING_ALIGN_SPEED = 2.2
heading_alignment_active = False

SIDE_CLEARANCE_TARGET = 0.90
SIDE_CLEARANCE_GAIN = 3.0

# True immediately after obstacle avoidance until the robot is correctly oriented towards the current A* Waypoint
rejoin_alignment_active = False

# Rejoin state. Immediately returns to closed-loop waypoint tracking.
rejoin_steps = 0

# Obstacle Avoidance State
# False = Normal PID Navigation
# True = Currently Avoiding an Obstacle
avoiding_obstacle = False

# Storing the chosen direction: Left or Right
avoidance_direction = None

# Storing the chosen action.
avoidance_phase = "TURN"

# Counting how many simulation steps the robot has been avoiding
avoidance_steps = 0

# PID Response Data Logging
time_data = []
position_error_data = []
heading_error_data = []
correction_data = []

simulation_time = 0.0

# A* Trajectory Logging
traj_time = []
traj_x = []
traj_y = []
traj_waypoint = []

# Saving the Trajectory Path Data.
def save_trajectory_data():
    with open(DATA_DIR / "trajectory_data.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Time",
            "X Position",
            "Y Position",
            "Waypoint"
        ])

        for i in range(len(traj_time)):
            writer.writerow([
                traj_time[i],
                traj_x[i],
                traj_y[i],
                traj_waypoint[i]
            ])

    print(f"Final Trajectory Saved: {len(traj_time)} samples.")

# Save PID data after every 5 seconds
last_save_time = 0.0

# Main Simulation Loop
while robot.step(timestep) != -1:
    # LiDAR Perception
    # Reading the LiDAR Data
    lidar_data = lidar.getRangeImage()

    # Processing the LiDAR data
    perception = process_lidar_data(lidar_data)

    # Handling all missing LiDAR Measurements
    min_distance = perception["min_distance"]
    front_distance = perception["front_distance"]
    left_distance = perception["left_distance"]
    right_distance = perception["right_distance"]

    # Camera Perception
    # Reading the camera image
    camera_image = camera.getImage()

    # Processing the Camera Image data
    camera_perception = process_camera_image_data(
        camera_image,
        camera_width,
        camera_height
    )

    # Sensor Fusion
    fused_perception = sensor_fusion_data(perception, camera_perception)

    front_fused = fused_perception["front"]
    left_fused = fused_perception["left"]
    right_fused = fused_perception["right"]

    front_blocked = fused_perception["front_blocked"]
    left_blocked = fused_perception["left_blocked"]
    right_blocked = fused_perception["right_blocked"]

    # Camera Obstacle Information
    left_camera_obstacle = camera_perception["left_obstacle"]
    front_camera_obstacle = camera_perception["front_obstacle"]
    right_camera_obstacle = camera_perception["right_obstacle"]

    left_edges = camera_perception["left_edges"]
    front_edges = camera_perception["front_edges"]
    right_edges = camera_perception["right_edges"]

    # Displaying sensor information
    min_display = (
        f"{min_distance:.2f}m"

        if min_distance is not None
        else "N/A"
    )

    front_display = (
        f"{front_distance:.2f}m"

        if front_distance is not None
        else "N/A"
    )

    left_display = (
        f"{left_distance:.2f}m"

        if left_distance is not None
        else "N/A"
    )

    right_display = (
        f"{right_distance:.2f}m"

        if right_distance is not None
        else "N/A"
    )

    # Print Basic Sensor Information
    print(
        f"LiDAR Minimum: {min_display} | "
        f"Front: {front_display} | "
        f"Left: {left_display} | "
        f"Right: {right_display} | "
        f"Camera Edges: "
        f"L={left_edges} "
        f"F={front_edges} "
        f"R={right_edges} | "
        f"Obstacles: "
        f"L={left_camera_obstacle} "
        f"F={front_camera_obstacle} "
        f"R={right_camera_obstacle}"
    )

    # Printing Sensor Fusion Results and displaying them
    print(
        f"Sensor Fusion Environment: "
        f"Front: {'Occupied' if front_fused['occupied'] else 'Clear'}"
        f"({front_fused['confidence']}) | "
        f"Left: {'Occupied' if left_fused['occupied'] else 'Clear'}"
        f"({left_fused['confidence']}) | "
        f"Right: {'Occupied' if right_fused['occupied'] else 'Clear'}"
        f"({right_fused['confidence']}) | "
    )

    # GPS and IMU Set Up
    gps_values = gps.getValues()
    imu_values = imu.getRollPitchYaw()

    x_position = gps_values[0]
    y_position = gps_values[1]
    z_position = gps_values[2]
    heading = imu_values[2]

    if initial_x is None:
        initial_x = x_position

    if initial_y is None:
        initial_y = y_position

    if initial_heading is None:
        initial_heading = heading

    # Update simulation time and record the actual robot trajectory
    simulation_time += timestep / 1000.0

    traj_time.append(simulation_time)
    traj_x.append(x_position)
    traj_y.append(y_position)
    traj_waypoint.append(current_waypoint_index + 1)

    if current_waypoint_index < len(waypoints):
        current_target = waypoints[current_waypoint_index]

        print(
            f"NAVIGATION TARGET: "
            f"Initial Heading={initial_heading:.3f} | "
            f"Waypoint={current_waypoint_index + 1}/{len(waypoints)} | "
            f"Target X={current_target[0]:.2f}"
            f"Target Y={current_target[1]:.2f}"
        )

    print(
        f"Position: X={x_position:.2f}m, "
        f"Y={y_position:.2f}m, "
        f"Z={z_position:.2f}m | "
        f"Heading={heading:.2f} rad"
    )

    # Obstacle Detection
    """
    Normal obstacle trigger comes from the fused perception layer. The emergency trigger remains raw LiDAR because an
    imminent collision must override the normal fusion threshold.
    """
    emergency_obstacle = (min_distance is not None and min_distance < EMERGENCY_DISTANCE)

    front_obstacle = front_blocked or emergency_obstacle

    front_clear = (front_distance is not None) and (front_distance >= FRONT_OBSTACLE_CLEAR)

    safety_obstacle = front_obstacle or emergency_obstacle

    if not avoiding_obstacle and (clear_confirmation_count < CLEAR_CONFIRMATION_STEPS):
        if emergency_obstacle:
            clear_confirmation_count = CLEAR_CONFIRMATION_STEPS
        elif front_clear:
            clear_confirmation_count += 1
        else:
            clear_confirmation_count = 0

    obstacle_trigger_allowed = (clear_confirmation_count >= CLEAR_CONFIRMATION_STEPS) or (clear_confirmation_count == 0)

    # Entering Obstacle Avoidance Mode
    if (safety_obstacle and not avoiding_obstacle and obstacle_trigger_allowed
        and (not rejoin_alignment_active or emergency_obstacle)):
        avoiding_obstacle = True
        avoidance_steps = 0
        avoidance_phase = "TURN"

        avoidance_direction = choose_avoidance_direction(left_distance, right_distance)

        heading_alignment_active = False
        rejoin_alignment_active = False
        clear_confirmation_count = 0
        forward_clear_count = 0

        print("OBSTACLE DETECTED -- PID PATH FOLLOWING OVERRIDDEN")

        print(
            f"AVOIDANCE START | "
            f"Direction={avoidance_direction} | "
            f"Phase={avoidance_phase}"
        )

    # Obstacle Avoidance State
    if avoiding_obstacle:
        avoidance_steps += 1

        # If the original direction has no usable measurement, choose again.
        if avoidance_direction is None:
            avoidance_direction = choose_avoidance_direction(left_distance, right_distance)

        side_distance = get_side_clearance(avoidance_direction, left_distance, right_distance)

        side_clear = (side_distance is not None) and (side_distance >= MIN_SIDE_CLEARANCE)

        # Phase 1 in which we commit to the selected turn.
        if avoidance_phase == "TURN":
            if avoidance_steps <= MIN_TURN_STEPS:
                left_speed, right_speed = avoidance_wheel_speeds(
                    avoidance_direction,
                    "TURN",
                    OBSTACLE_TURN_SPEED,
                    OBSTACLE_FORWARD_SPEED
                )

                print(
                    f"Avoidance: TURN {avoidance_direction.upper()} "
                    f"({avoidance_steps}/{MIN_TURN_STEPS})"
                )
            else:
                # Minimum Turn Completed. Robot won't move forward if the front is still blocked.
                if safety_obstacle:
                    left_speed, right_speed = avoidance_wheel_speeds(
                        avoidance_direction,
                        "TURN",
                        OBSTACLE_TURN_SPEED,
                        OBSTACLE_FORWARD_SPEED
                    )

                    print(
                        f"Avoidance: FRONT STILL BLOCKED -- "
                        f"CONTINUING TURN"
                    )
                else:
                    avoidance_phase = "CLEAR"

                    left_speed, right_speed = avoidance_wheel_speeds(
                        avoidance_direction,
                        "CLEAR",
                        OBSTACLE_TURN_SPEED,
                        OBSTACLE_FORWARD_SPEED
                    )

                    print(
                        f"Avoidance: TURN COMPLETE -- "
                        f"ENTERING CLEARANCE PHASE"
                    )
        elif avoidance_phase == "CLEAR":
            if safety_obstacle:
                avoidance_phase = "TURN"
                forward_clear_count = 0

                left_speed, right_speed = avoidance_wheel_speeds(
                    avoidance_direction,
                    "TURN",
                    OBSTACLE_TURN_SPEED,
                    OBSTACLE_FORWARD_SPEED
                )

                print(
                    f"Avoidance: OBSTACLE REDETECTED -- "
                    f"RETURNING TO TURN PHASE"
                )
            else:
                forward_clear_count += 1

                left_speed, right_speed = avoidance_wheel_speeds(
                    avoidance_direction,
                    "CLEAR",
                    OBSTACLE_TURN_SPEED,
                    OBSTACLE_FORWARD_SPEED
                )

                print(
                    f"Avoidance: MOVING FORWARD | "
                    f"Forward Steps:{forward_clear_count}/{FORWARD_CLEAR_STEPS} | "
                    f"Side={side_distance if side_distance is not None else 'N/A'}"
                )
        else:
            left_speed = 0.0
            right_speed = 0.0

        # Leaving Avoidance as soon as all conditions are satisfied.
        if (
            avoidance_phase == "CLEAR" and not safety_obstacle and side_clear
                and forward_clear_count >= FORWARD_CLEAR_STEPS
        ):
            clear_confirmation_count += 1

            if clear_confirmation_count >= CLEAR_CONFIRMATION_STEPS:
                avoiding_obstacle = False
                avoidance_phase = "TURN"
                avoidance_steps = 0
                forward_clear_count = 0

                pid.reset()

                clear_confirmation_count = 0
                heading_alignment_active = False

                rejoin_alignment_active = True
                rejoin_steps = 0

            print(
                f"OBSTACLE CLEARED -- "
                f"RETURNING TO CLOSED-LOOP WAYPOINT TRACKING"
            )

            # Recalculating the target immediately on the next controller section. No fixed straight-line rejoin is
            # utilized.
            rejoin_steps = 0
        # Safety Timeout which ensures the robot doesn't blindly return to PID while an obstacle is present.
        elif avoidance_steps >= MAX_AVOIDANCE_STEPS:
            print(
                f"AVOIDANCE TIMEOUT -- "
                f"CONTINUING LOCKED DIRECTION: {avoidance_direction}"
            )

            avoidance_steps = 0
            avoidance_phase = "TURN"

            pid.reset()
            heading_alignment_active = False

            left_speed, right_speed = avoidance_wheel_speeds(
                avoidance_direction,
                "TURN",
                OBSTACLE_TURN_SPEED,
                OBSTACLE_FORWARD_SPEED
            )

        set_motor_speeds(left_speed, right_speed)

        print(
            f"Obstacle Control | "
            f"Phase={avoidance_phase} | "
            f"Direction={avoidance_direction} | "
            f"Steps={avoidance_steps} | "
            f"Front Blocked={front_obstacle} | "
            f"Side Clear={side_clear} | "
            f"Left={left_speed:.3f} | "
            f"Right={right_speed:.3f} | "
        )
    # Normal PID Path Following
    else:
        if current_waypoint_index >= len(waypoints):
            set_motor_speeds(0.0, 0.0)

            save_trajectory_data()

            print("A* Goal Reached -- Robot Stopped")

            continue

        # A* Waypoint Navigation
        target_x, target_y = waypoints[current_waypoint_index]

        # Calculating the vector from the robot to the current waypoint
        dx = target_x - x_position
        dy = target_y - y_position

        waypoint_distance = math.sqrt(dx ** 2 + dy ** 2)

        # Advance to the next waypoint when the current waypoint is reached.
        if waypoint_distance <= WAYPOINT_TOLERANCE:
            print(
                f"WAYPOINT {current_waypoint_index + 1} REACHED | "
                f"{target_x:.2f} | {target_y:.2f}"
            )

            current_waypoint_index += 1

            # Stop the robot when the final A* waypoint is reached.
            if current_waypoint_index >= len(waypoints):
                set_motor_speeds(0.0, 0.0)

                print("A* Goal Reached -- Robot Stopped")

                continue

            target_x, target_y = waypoints[current_waypoint_index]

            dx = target_x - x_position
            dy = target_y - y_position

            waypoint_distance = math.sqrt(dx ** 2 + dy ** 2)

        # Webots IMU Yaw is used with X/Y World Coordinates
        target_heading = math.atan2(dy, dx)

        # Heading Error
        heading_error = target_heading - heading

        side_correction = 0.0

        # Keeping heading error within -pi to +pi
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        elif heading_error < -math.pi:
            heading_error += 2 * math.pi

        # Heading Alignment Mode
        if rejoin_alignment_active:
            # Robot has cleared an obstacle and must first establish a safe heading toward the current A* waypoint
            if abs(heading_error) <= HEADING_ALIGN_EXIT:
                rejoin_alignment_active = False
                heading_alignment_active = False
                pid.reset()

                print(
                    f"REJOIN ALIGNMENT COMPLETE -- "
                    f"HEADING ERROR={heading_error:.3f}rad -- "
                    f"RESUMING PID"
                )
            else:
                heading_alignment_active = True
        elif heading_alignment_active and (abs(heading_error) <= HEADING_ALIGN_EXIT):
            heading_alignment_active = False
            pid.reset()

            print("HEADING ALIGNMENT COMPLETE -- RESUMING PID")
        elif not heading_alignment_active and (abs(heading_error) >= HEADING_ALIGN_ENTER):
            heading_alignment_active = True
            pid.reset()

            print(f"HEADING ALIGNMENT START | Error={heading_error:.3f}rad")

        if heading_alignment_active:
            left_speed, right_speed = calculate_heading_alignment_command(heading_error, HEADING_ALIGN_SPEED)

            # Position Error, which allows the PID controller to correct the robot toward the current A* waypoint rather
            # than using the old straight-line path.
            position_error = waypoint_distance * math.sin(heading_error)
            position_correction = 0.0
            heading_correction = heading_error
            correction = heading_correction
        else:
            # Position Error, which allows the PID controller to correct the robot toward the current A* waypoint rather
            # than using the old straight-line path.
            position_error = waypoint_distance * math.sin(heading_error)

            # PID Correction for the Path Position
            position_correction = pid.update(position_error, timestep / 1000.0)

            # Additional Correction to bring the robot back to the target heading
            heading_correction = heading_kp * heading_error

            side_correction = 0.0

            # Combining Position and Heading Corrections
            correction = heading_correction + 0.2 * position_correction + side_correction

            # Limiting the correction to prevent the robot from spinning aggressively
            correction = max(-2.0, min(2.0, correction))

            # When the waypoint is far behind the robot, reduce the forward motion through the reorientation. By doing
            # so, it prevents the robot from pushing forward.
            heading_factor = max(0.35, 1.0 - (abs(heading_error) / 0.8))
            navigation_speed = base_speed * heading_factor

            # Calculate Left and Right Wheel Speeds
            left_speed = navigation_speed - correction
            right_speed = navigation_speed + correction

        # Recording PID Response Data
        time_data.append(simulation_time)
        position_error_data.append(position_error)
        heading_error_data.append(heading_error)
        correction_data.append(correction)

        if simulation_time - last_save_time >= 5.0:
            with open(DATA_DIR / "pid_response_data.csv", "w", newline="") as file:
                writer = csv.writer(file)

                writer.writerow([
                    "Time",
                    "Position Error",
                    "Heading Error",
                    "PID Correction"
                ])

                for i in range(len(time_data)):
                    writer.writerow([
                        time_data[i],
                        position_error_data[i],
                        heading_error_data[i],
                        correction_data[i]
                    ])

            with open(DATA_DIR / "trajectory_data.csv", "w", newline="") as file:
                writer = csv.writer(file)

                writer.writerow([
                    "Time",
                    "X Position",
                    "Y Position",
                    "Waypoint"
                ])

                for i in range(len(traj_time)):
                    writer.writerow([
                        traj_time[i],
                        traj_x[i],
                        traj_y[i],
                        traj_waypoint[i]
                    ])

            last_save_time = simulation_time

            print("PID Response Data and Trajectory Data Saved.")

        # Apply Motor Speeds
        set_motor_speeds(left_speed, right_speed)

        print(
            f"PID | "
            f"Waypoint={current_waypoint_index + 1}/{len(waypoints)} | "
            f"Distance={waypoint_distance:.3f}m | "
            f"Side Correction={side_correction:.3f} | "
            f"Target Heading={target_heading:.3f}rad | "
            f"Heading={heading:.3f}rad | "
            f"Heading Error={heading_error:.3f}rad | "
            f"Position Correction={position_correction:.3f} | "
            f"Heading Correction={heading_correction:.3f} | "
            f"Left={left_speed:.3f} | "
            f"Right={right_speed:.3f}"
        )