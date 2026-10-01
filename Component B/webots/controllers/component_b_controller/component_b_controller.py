# Importing libraries
from controller import Robot
import csv
import os

# Declaring the Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Identifying the Robot
robot_name = robot.getName()
print(f"Starting {robot_name}")

# CSV Logging
log_file = None
csv_writer = None

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

# Setting the Identity of the Robot and the Initial Consensus Values
initial_values = {
    "robot_1": 20.0,
    "robot_2": 50.0,
    "robot_3": 80.0
}

consensus_value = initial_values.get(robot_name, 0.0)

print(f"{robot_name} started with value: {consensus_value:.2f}")

# Storing the latest known consensus value of each robot
consensus_values = {
    "robot_1": None,
    "robot_2": None,
    "robot_3": None
}

consensus_values[robot_name] = consensus_value

# Consensus Parameters
alpha = 0.2

# Communication Devices
emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")

# Enable Receiver
receiver.enable(timestep)

'''
# Get Motors
right_front = robot.getDevice("right_front_motor")
right_rear = robot.getDevice("right_rear_motor")
left_front = robot.getDevice("left_front_motor")
left_rear = robot.getDevice("left_rear_motor")

motors = [right_front, right_rear, left_front, left_rear]

for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)


speed = 3.0

right_front.setVelocity(speed)
right_rear.setVelocity(speed)
left_front.setVelocity(speed)
left_rear.setVelocity(speed)
'''

# Main Loop
step_counter = 0

while robot.step(timestep) != -1:
    # Broadcasting a message with the consensus value from the robot to identify which robot sent it.
    message = f"{robot_name}, {consensus_value}"

    emitter.send(message.encode("utf-8"))

    # Storing values received from other robots
    neighbour_values = []

    # Checking for incoming messages
    while receiver.getQueueLength() > 0:
        received_msg = receiver.getString()

        receiver.nextPacket()

        try:
            sender, value = received_msg.split(",")

            # Ignoring the robot's own message
            if sender != robot_name:
                value = float(value)

                neighbour_values.append(value)

                # Storing the latest value received from this robot.
                consensus_values[sender] = value

        except ValueError:
            continue

    # Average Consensus Update
    if neighbour_values:
        average_difference = sum(
            value - consensus_value

            for value in neighbour_values
        ) / len(neighbour_values)

        consensus_value += alpha * average_difference

    # Update the robot's latest consensus value
    consensus_values[robot_name] = consensus_value

    # Logging the Result
    if step_counter % 10 == 0:
        print(
            f"{robot_name} | "
            f"Consensus Value: {consensus_value:.4f}"
        )

        # Only robot_1 writes to the csv
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

    # Increment Simulation Step Counter
    step_counter += 1

# Close CSV File
if log_file is not None:
    log_file.close()