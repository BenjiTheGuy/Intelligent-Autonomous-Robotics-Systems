from controller import Robot
from perception import process_lidar_data, process_camera_image_data

# Create the Webots Robot
robot = Robot()

# Webots Simulation Timestep
timestep = int(robot.getBasicTimeStep())

# Create the Sensors
lidar = robot.getDevice("lidar")
camera = robot.getDevice("camera")

camera_width = camera.getWidth()
camera_height = camera.getHeight()

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

    camera_edges = process_camera_image_data(
        camera_image,
        camera_width,
        camera_height
    )

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
        f"Minimum: {min_display} | "
        f"Front: {front_distance}m | "
        f"Left: {left_distance}m | "
        f"Right: {right_distance}m | "
        f"Camera Image Received: {camera_image is not None} | "
        f"Camera Processing Shape: {camera_edges.shape}"
    )