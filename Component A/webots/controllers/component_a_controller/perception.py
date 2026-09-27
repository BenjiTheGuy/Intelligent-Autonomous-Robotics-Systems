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
    # Approximate Directions:
    # 0 = Front
    # 90 = Left
    # 180 = Rear
    # 270 = Right

    front_distance = lidar_data[0]
    left_distance = lidar_data[90]
    right_distance = lidar_data[270]

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
        A processed OpenCV Edge Image.
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

    return edges