# Importing library to use in file
import math

class PIDController:
    """
    PID Controller for Correcting the Robot when finding a Path Error.
    """

    def __init__(self, kp, ki, kd, integral_limit=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = abs(integral_limit)

        self.previous_error = 0.0
        self.integral = 0.0

    def reset(self):
        """
        Resetting the PID Memory after a navigation-mode change.
        """
        self.previous_error = 0.0
        self.integral = 0.0

    def update(self, error, timestep):
        """
        Calculating the PID correction

        :param
            error: Current Tracking Error
            timestep: Simulation Timestep measured in seconds

        :return: PID Correction Value
        """

        if timestep <= 0:
            return 0.0

        self.integral += error * timestep
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        if self.previous_error == 0.0:
            derivative = 0.0
        else:
            derivative = (error - self.previous_error) / timestep

        correction = ((self.kp * error) + (self.ki * self.integral) + (self.kd * derivative))

        self.previous_error = error

        return correction

def normalize_angle(angle):
    """
    Wrapping an angle to the range [-pi, pi]
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle

def choose_avoidance_direction(left_distance, right_distance):
    """
    Selecting the side with the greatest available clearance.
    :return:
        'left', 'right', or 'None' when neither side has a usable measurement.
    """
    if left_distance is None and right_distance is None:
        return None

    if left_distance is None:
        return "right"

    if right_distance is None:
        return "left"

    if right_distance >= left_distance:
        return "right"

    return ("left")

def get_side_clearance(direction, left_distance, right_distance):
    """
    Returning the LiDAR clearance on the currently selected side.
    """
    if direction == "left":
        return left_distance

    if direction == "right":
        return right_distance

    return None

def avoidance_wheel_speeds(direction, phase, turn_speed, forward_speed):
    """
    Generating differential-drive commands for obstacle avoidance.
        TURN: Pivot toward the selected free side
        CLEAR: Move forward slowly while the sensors are checked again
        STOP: Stop the Robot when a safe direction hasn't been found
    """
    if direction == "left":
        if phase == "TURN":
            return -turn_speed, turn_speed
        if phase == "CLEAR":
            return forward_speed, forward_speed
    elif direction == "right":
        if phase == "TURN":
            return turn_speed, -turn_speed
        if phase == "CLEAR":
            return forward_speed, forward_speed

    return 0.0, 0.0

def avoidance_is_clear(avoidance_steps, minimum_turn_steps, front_blocked, side_distance, minimum_side_clearance):
    """
    Determine whether the robot can safely leave avoidance mode.

    The Robot needs to achieve the following:
        It must first complete its minimum turn, then it can rejoin the waypoint navigation as soon as
        both the front and selected side are clear. A max step count is not used; instead, the contrller
        can plan its avoidance direction if a timeout occurs.
    """
    side_clear = (side_distance is not None) and (side_distance >= minimum_side_clearance)

    return (avoidance_steps >= minimum_turn_steps) and not front_blocked and side_clear

def calculate_rejoin_command(
        target_x,
        target_y,
        x_position,
        y_position,
        heading,
        pid,
        timestep,
        base_speed,
        heading_kp,
        max_correction=2.0
):
    """
    Immediately return from avoidance to closed-loop waypoint tracking.

    This function utilizes the same waypoint geometry as normal PID navigation. Allowing for the robot
    to rotate toward the waypoint immediately after an obstacle is cleared instead of driving blindly for several
    simulation steps.
    """
    dx = target_x - x_position
    dy = target_y - y_position

    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, distance

    target_heading = math.atan2(dy, dx)
    heading_error = normalize_angle(target_heading - heading)

    position_error = distance * math.sin(heading_error)
    position_correction = pid.update(position_error, timestep)

    heading_correction = heading_kp * heading_error

    correction = heading_correction + (0.2 * position_correction)
    correction = max(-max_correction, min(max_correction, correction))

    # When the target is almost directly behind the robot, reduce the forward motion so the robot can reorient without
    # driving deeper into an obstacle.
    if abs(heading_error) > 1.0:
        rejoin_speed = base_speed * 0.35
    else:
        rejoin_speed = base_speed

    left_speed = rejoin_speed - correction
    right_speed = rejoin_speed + correction

    return(
        left_speed,
        right_speed,
        heading_error,
        position_correction,
        distance
    )

def calculate_heading_alignment_command(heading_error, turn_speed):
    """
    Generating a rotation command toward the target heading.
    """
    if heading_error > 0:
        return -turn_speed, turn_speed

    if heading_error < 0:
        return turn_speed, -turn_speed

    return 0.0, 0.0

def calculate_motor_speeds(left_speed, right_speed):
    """
    Returning the requested left and right motor speeds.

    :param
        left_speed: Speed for the left motor
        right_speed: Speed for the right motor

    :return: A tuple which consists of the speeds for both the left and right motors.
    """

    return left_speed, right_speed

def calculate_robot_velocity(left_angular_velocity, right_angular_velocity):
    """
    Calculating the Linear and Angular Velocity of the Robot
    """

    wheel_radius = 0.08
    wheel_separation = 0.34

    linear_velocity = (wheel_radius / 2) * (right_angular_velocity + left_angular_velocity)
    angular_velocity = (wheel_radius / wheel_separation) * (right_angular_velocity - left_angular_velocity)

    return linear_velocity, angular_velocity

def calculate_robot_motion(x, y, heading, left_angular_velocity, right_angular_velocity, timestep):
    """
    Calculating the next position of the Robot by utilizing the Differential-Drive Kinematic Model.
    """
    linear_velocity, angular_velocity = calculate_robot_velocity(left_angular_velocity, right_angular_velocity)

    new_x = x + (linear_velocity * math.cos(heading) * timestep)
    new_y = y + (linear_velocity * math.sin(heading) * timestep)
    new_heading = heading + (angular_velocity * timestep)

    return new_x, new_y, new_heading

if __name__ == "__main__":
    left_speed, right_speed = calculate_motor_speeds(2.0, 2.0)

    print(f"Left Motor Speed: {left_speed}")
    print(f"Right Motor Speed: {right_speed}")

    linear_velocity, angular_velocity = calculate_robot_velocity(2.0, 2.0)

    print(f"Linear Velocity: {linear_velocity:.3f} m/s")
    print(f"Angular Velocity: {angular_velocity:.3f} rad/s")

    pid = PIDController(kp=1.0, ki=0.0, kd=0.1)

    error = 0.5
    timestep = 0.032

    correction = pid.update(error, timestep)

    print(f"PID Controller: {correction:.3f}")