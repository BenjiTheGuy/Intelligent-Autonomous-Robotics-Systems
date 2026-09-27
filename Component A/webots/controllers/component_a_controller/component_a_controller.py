from controller import Robot
from perception import process_lidar_data

# Create the Webots Robot
robot = Robot()

# Webots Simulation Timestep
timestep = int(robot.getBasicTimeStep())

# Create the Sensors
lidar = robot.getDevice("lidar")
camera = robot.getDevice("camera")

# Enable the sensors which we created
lidar.enable(timestep)
camera.enable(timestep)

# Main Simulation Loop
while robot.step(timestep) != -1:
    # Reading the LiDAR data
    lidar_data = lidar.getRangeImage()

    # Processing the LiDAR data
    perception = process_lidar_data(lidar_data)

    # Reading the camera image
    camera_image = camera.getImage()

    # Handling all missing LiDAR Measurements
    min_distance = perception["min_distance"]
    front_distance = perception["front_distance"]
    left_distance = perception["left_distance"]
    right_distance = perception["right_distance"]

    if min_distance is None:
        min_display = "N/A"
        front_display = "N/A"
        left_display = "N/A"
        right_display = "N/A"
    else:
        min_display = f"{min_distance:.2f}m"
        front_display = f"{front_distance:.2f}m"
        left_display = f"{left_distance:.2f}m"
        right_display = f"{right_distance:.2f}m"

    # Print Basic Sensor Information
    print(
        f"Minimum: {perception['min_distance']:.2f}m | "
        f"Front: {perception['front_distance']:.2f}m | "
        f"Left: {perception['left_distance']:.2f}m | "
        f"Right: {perception['right_distance']:.2f}m | "
        f"Camera Image Received: {camera_image is not None}"
    )