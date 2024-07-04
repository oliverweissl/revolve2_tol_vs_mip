"""Evaluator class."""

<<<<<<< HEAD
from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.interactive_objects import Ball
=======
from revolve2.ci_group import fitness_functions, terrains, interactive_objects
>>>>>>> 259b226 (better structure)
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.experimentation.evolution.abstract_elements import Evaluator
from revolve2.simulation.scene import Pose
from pyrr import Vector3


class EvaluatorObjectManipulation(Evaluator):
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()

    def evaluate(
        self,
        robots: list[ModularRobot],
    ) -> list[float]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param robots: The robots to simulate.
        :returns: Fitnesses of the robots.
        """
        # Create the scenes.

<<<<<<< HEAD
        scenes, balls = [], []
        for robot in robots:
            ball = Ball(radius=0.1, mass=0.1, pose=Pose(position=Vector3([0.35, 0.35, 0.0])))
            balls.append(ball)
=======
        scenes = []
        ball = interactive_objects.Ball(radius=0.1, mass=0.1, pose=Pose(Vector3([-0.5, 0.5, 0])))
        for robot in robots:
>>>>>>> 259b226 (better structure)
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scene.add_interactive_object(ball)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        # Calculate the xy displacements of the interactive objects.
        xy_displacements = [
            fitness_functions.xy_displacement(
                states[0].get_interactive_object_simulation_state(ball),
                states[-1].get_interactive_object_simulation_state(ball),
            )
<<<<<<< HEAD
            for states, ball in zip(scene_states, balls)
=======
            for robot, states in zip(robots, scene_states)
>>>>>>> 259b226 (better structure)
        ]
        return xy_displacements
