import math
import cv2
import numpy as np

def process_lidar_data(lidar_data):
    """
    Process LiDAR Range Measurements and Identify Nearby Obstacles.

    :param:
        lidar_data: List of LiDAR Distance Measurements.

    :return:
        A dictionary containing useful obstacle-distance information.
    """

    # Remove Invalid Measurements
    valid_ranges = [
        distance for distance in lidar_data

        if math.isfinite(distance)
    ]

    if not valid_ranges:
        return {
            "min_distance": None,
            "front_distance": None,
            "left_distance": None,
            "right_distance": None
        }

    min_distance = min(valid_ranges)

    # LiDAR has 360 measurements.
    # Using groups of measurements rather than relying on a single ray.

    front_ranges = lidar_data[170:189]
    left_ranges = lidar_data[80:99]
    right_ranges = lidar_data[260:279]

    # Remove invalid measurements from each region
    front_valid = [
        distance for distance in front_ranges
        if math.isfinite(distance)
    ]

    left_valid = [
        distance for distance in left_ranges
        if math.isfinite(distance)
    ]

    right_valid = [
        distance for distance in right_ranges
        if math.isfinite(distance)
    ]

    front_distance = min(front_valid) if front_valid else None
    left_distance = min(left_valid) if left_valid else None
    right_distance = min(right_valid) if right_valid else None

    return {
        "min_distance": min_distance,
        "front_distance": front_distance,
        "left_distance": left_distance,
        "right_distance": right_distance
    }

def process_camera_image_data(camera_image, width, height):
    """
    Converting the Webots Camera Image into an OpenCV Image and Performing a Basic Detection on Edges.

    :param:
        camera_image: Raw Image Returned by the Webots Camera.
        width: Width size of the Camera Image.
        height: Height size of the Camera Image.

    :return:
        A dictionary containing the edge image and camera-based obstacle information.
    """

    # Converting Webots BGRA Image into an array converted via NumPy
    image = np.frombuffer(
        camera_image,
        dtype=np.uint8
    ).reshape((height, width, 4))

    # Converting BGRA to BGR
    bgr_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGRA2BGR
    )

    # Converting to Grayscale
    gray_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2GRAY
    )

    # Detecting Edges
    edges = cv2.Canny(
        gray_image,
        100,
        200
    )

    # Dividing the image into 3 Regions
    third = width // 3

    top = int(height * 0.25)
    bottom = int(height * 0.75)

    left_region = edges[top:bottom, :third]
    front_region = edges[top:bottom, third:2 * third]
    right_region = edges[top:bottom, 2 * third:]

    # Count Detected Edge Pixels
    left_edges = cv2.countNonZero(left_region)
    front_edges = cv2.countNonZero(front_region)
    right_edges = cv2.countNonZero(right_region)

    # Threshold which is utilized in identifying significant visual structure
    edge_threshold = 500

    return {
        "edges": edges,
        "left_edges": left_edges,
        "front_edges": front_edges,
        "right_edges": right_edges,
        "left_obstacle": left_edges > edge_threshold,
        "front_obstacle": front_edges > edge_threshold,
        "right_obstacle": right_edges > edge_threshold
    }