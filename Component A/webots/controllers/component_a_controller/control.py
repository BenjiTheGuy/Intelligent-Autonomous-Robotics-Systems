import math

class PIDController:
    """
    PID Controller for Correcting the Robot when finding a Path Error.
    """

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

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

        derivative = (error - self.previous_error) / timestep

        correction = ((self.kp * error) + (self.ki * self.integral) + (self.kd * derivative))

        self.previous_error = error

        return correction

def calculate_motor_speeds(left_speed, right_speed):
    """
    Calculating the left and right motor speeds.

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
    timestep=0.032

    correction = pid.update(error, timestep)

    print(f"PID Controller: {correction:.3f}")