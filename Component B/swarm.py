import random
import math
import csv
import os
import matplotlib.pyplot as plt

# Robot Starting Positions
robots = {
    "robot_1": (-2.0, 2.0),
    "robot_2": (-0.0, 0.0),
    "robot_3": (2.0, -2.0),
}

# Target Locations
targets = {
    "target_1": (-4.0, 4.0),
    "target_2": (4.0, 4.0),
    "target_3": (0.0, 3.0),
    "target_4": (-4.0, -3.0),
    "target_5": (4.0, -3.0),
}

# Distance Function
def distance(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1] ** 2)
    )

# Creating Random Particle
def create_particle():
    """
    Each value represents the index of the target selected by one robot.

    Example: [2, 0, 4] = Robot 1 (Target 3), Robot 2 (Target 1), Robot 3 (Target 5)
    """

    return [
        random.randint(0, len(targets) - 1)

        for _ in robots
    ]

class Particle:
    def __init__(self, robot_position, targets):
        self.robot_position = robot_position
        self.targets = targets

        # Randomly selecting an initial target
        self.position = random.randint(0, len(targets) - 1)

        # PSO Velocity
        self.velocity = 0.0

        # Personal Best
        self.best_position = self.position
        self.best_fitness = float("inf")

class PSO:
    def __init__(
            self,
            robot_positions,
            targets,
            num_particles=10,
            iterations=30,
            inertia=0.7,
            cognitive=1.5,
            social=1.5
    ):
        self.robot_positions = robot_positions
        self.targets = targets

        self.num_particles = num_particles
        self.iterations = iterations

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.particles = []

        # Setting the Global Best
        self.global_best_position = None
        self.global_best_fitness = float("inf")

        # Create Particles
        for _ in range(num_particles):
            robot_position = random.choice(robot_positions)

            particle = Particle(robot_position, targets)

            self.particles.append(particle)

        # Distance Calculation
