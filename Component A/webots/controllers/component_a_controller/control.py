def calculate_motor_speeds(left_speed, right_speed):
    """
    Calculating the left and right motor speeds.

    :param
        left_speed: Speed for the left motor
        right_speed: Speed for the right motor

    :return: A tuple which consists of the speeds for both the left and right motors.
    """

    return left_speed, right_speed

if __name__ == "__main__":
    left_speed, right_speed = calculate_motor_speeds(2.0, 2.0)

    print(f"Left Motor Speed: {left_speed}")
    print(f"Right Motor Speed: {right_speed}")