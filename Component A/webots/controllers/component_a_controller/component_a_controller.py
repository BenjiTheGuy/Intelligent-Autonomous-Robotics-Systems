from controller import Robot
from perception import process_lidar_data, process_camera_image_data, sensor_fusion_data

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

    camera_perception = process_camera_image_data(
        camera_image,
        camera_width,
        camera_height
    )

    fused_perception = sensor_fusion_data(perception, camera_perception)

    front_fused = fused_perception["front"]
    left_fused = fused_perception["left"]
    right_fused = fused_perception["right"]

    left_camera_obstacle = camera_perception["left_obstacle"]
    front_camera_obstacle = camera_perception["front_obstacle"]
    right_camera_obstacle = camera_perception["right_obstacle"]

    left_edges = camera_perception["left_edges"]
    front_edges = camera_perception["front_edges"]
    right_edges = camera_perception["right_edges"]

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

    # Printing Sensor Fusion Results
    print(
        f"Sensor Fusion Environment: "
        f"Front: {'Occupied' if front_fused['occupied'] else 'Clear'}"
        f"({front_fused['confidence']}) | "
        f"Left: {'Occupied' if left_fused['occupied'] else 'Clear'}"
        f"({left_fused['confidence']}) | "
        f"Right: {'Occupied' if right_fused['occupied'] else 'Clear'}"
        f"({right_fused['confidence']}) | "
    )