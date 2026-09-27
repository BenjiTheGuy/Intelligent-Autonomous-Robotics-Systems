import math

def process_lidar_data(lidar_data):
    """
    Process LiDAR Range Measurements and Identify Nearby Obstacles.
    :Parameters:
        lidar_data: List of LiDAR Distance Measurements
    :return:
        A dictionary containing useful obstacle-distance information
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