"""Main environment class."""
import contextlib
import sys

import gym
import numpy as np
from gym import spaces

from gym_missile_command.config import CONFIG
from gym_missile_command.game.batteries import Batteries
from gym_missile_command.game.cities import Cities
from gym_missile_command.game.enemy_missiles import EnemyMissiles
from gym_missile_command.game.friendly_missiles import FriendlyMissiles
from gym_missile_command.game.target import Target
from gym_missile_command.utils import rgetattr, rsetattr

# Import Pygame and remove welcome message
with contextlib.redirect_stdout(None):
    import pygame


class MissileCommandEnv(gym.Env):
    """Missile Command Gym environment.

    Attributes:
        NB_ACTIONS (int): the 6 possible actions. (0) do nothing, (1) target
            up, (2) target down, (3) target left, (4) target right, (5) fire
            missile.
    """
    NB_ACTIONS = 6
    metadata = {"render.modes": ["human"]}

    def __init__(self, custom_config={}):
        """Initialize MissileCommand environment.

        Args:
            custom_config (dict): optional, custom configuration dictionary
                with configuration attributes (strings) as keys (for example
                "FRIENDLY_MISSILES.NUMBER") and values as... Well, values (for
                example 42).

        Attributes:
            timestep (int): current timestep, starts from 0.

            observation (numpy.array): of size (WIDTH, HEIGHT, 3). The
                observation of the current timestep, representing the RGB
                values of each pixel.

            reward_timestep (float): reward of the current timestep.

            reward_total (float): reward sum from first timestep to current
                one.

            display (pygame.Surface): pygame surface, only for the render
                method.
        """
        super(MissileCommandEnv, self).__init__()
        self.action_space = spaces.Discrete(self.NB_ACTIONS)

        # Custom configuration
        # ------------------------------------------

        # For each custom attributes
        for key, value in custom_config.items():

            # Check if attributes is valid
            try:
                _ = rgetattr(CONFIG, key)
            except AttributeError as e:
                print("Invalid custom configuration: {}".format(e))
                sys.exit(1)

            # Modify it
            rsetattr(CONFIG, key, value)

        # Initializing objects
        # ------------------------------------------

        # No display while no render
        self.display = None

        # Objects
        self.batteries = Batteries()
        self.cities = Cities()
        self.enemy_missiles = EnemyMissiles()
        self.friendly_missiles = FriendlyMissiles()
        self.target = Target()

    def _collisions_cities(self):
        """Check for cities collisions.

        Check cities destroyed by enemy missiles.
        """
        # Cities
        cities = self.cities.cities

        # Enemy missiles current positions
        enemy_m = self.enemy_missiles.enemy_missiles[:, [2, 3]]

        # Align cities and enemy missiles
        cities_dup = np.repeat(cities, enemy_m.shape[0], axis=0)
        enemy_m_dup = np.tile(enemy_m, reps=[cities.shape[0], 1])

        # Compute distances
        dx = enemy_m_dup[:, 0] - cities_dup[:, 0]
        dy = enemy_m_dup[:, 1] - cities_dup[:, 1]
        distances = np.sqrt(np.square(dx) + np.square(dy))

        # Get cities destroyed by enemy missiles
        exploded = distances <= (
            CONFIG.ENEMY_MISSILES.RADIUS + CONFIG.CITIES.RADIUS)
        exploded = exploded.astype(int)
        exploded = np.reshape(exploded, (cities.shape[0], enemy_m.shape[0]))

        # Get destroyed cities
        cities_out = np.argwhere(
            (np.sum(exploded, axis=1) >= 1) &
            (cities[:, 2] > 0.0)
        )

        # Update timestep reward
        self.reward_timestep += CONFIG.REWARD.DESTROYED_CITY * \
            cities_out.shape[0]

        # Remove these cities
        self.cities.cities = np.delete(
            self.cities.cities,
            np.squeeze(cities_out),
            axis=0,
        )

    def _collisions_missiles(self):
        """Check for missiles collisions.

        Check enemy missiles destroyed by friendly exploding missiles.
        """
        # Friendly exploding missiles
        friendly_exploding = self.friendly_missiles.missiles_explosion

        # Enemy missiles current positions
        enemy_missiles = self.enemy_missiles.enemy_missiles[:, [2, 3]]

        # Align enemy missiles and friendly exploding ones
        enemy_m_dup = np.repeat(enemy_missiles,
                                friendly_exploding.shape[0],
                                axis=0)
        friendly_e_dup = np.tile(friendly_exploding,
                                 reps=[enemy_missiles.shape[0], 1])

        # Compute distances
        dx = friendly_e_dup[:, 0] - enemy_m_dup[:, 0]
        dy = friendly_e_dup[:, 1] - enemy_m_dup[:, 1]
        distances = np.sqrt(np.square(dx) + np.square(dy))

        # Get enemy missiles inside an explosion radius
        inside_radius = distances <= (
            friendly_e_dup[:, 2] + CONFIG.ENEMY_MISSILES.RADIUS)
        inside_radius = inside_radius.astype(int)
        inside_radius = np.reshape(
            inside_radius,
            (enemy_missiles.shape[0], friendly_exploding.shape[0]),
        )

        # Remove theses missiles
        missiles_out = np.argwhere(np.sum(inside_radius, axis=1) >= 1)
        self.enemy_missiles.enemy_missiles = np.delete(
            self.enemy_missiles.enemy_missiles,
            np.squeeze(missiles_out),
            axis=0,
        )

        # Compute current reward
        nb_missiles_destroyed = missiles_out.shape[0]
        self.reward_timestep += CONFIG.REWARD.DESTROYED_ENEMEY_MISSILES * \
            nb_missiles_destroyed

    def _reset_observation(self):
        """Reset observation."""
        self.observation = np.zeros(
            (CONFIG.WIDTH, CONFIG.HEIGHT, 3), dtype=CONFIG.DTYPE)
        self.observation[:, :, 0] = CONFIG.COLORS.BACKGROUND[0]
        self.observation[:, :, 1] = CONFIG.COLORS.BACKGROUND[1]
        self.observation[:, :, 2] = CONFIG.COLORS.BACKGROUND[2]

    def reset(self):
        """Reset the environment.

        Returns:
            observation (numpy.array): the representation of the environment.
        """
        self.timestep = 0
        self.reward_total = 0.0
        self.reward_timestep = 0.0

        self._reset_observation()

        self.batteries.reset()
        self.cities.reset()
        self.enemy_missiles.reset()
        self.friendly_missiles.reset()
        self.target.reset()

        return self.observation

    def step(self, action):
        """Go from current step to next one.

        Args:
            action (int): 0, 1, 2, 3, 4 or 5, the different actions.

        Returns:
            observation (numpy.array): the representation of the environment.

            reward (float): reward of the current time step.

            done (bool): True if the episode is finished, False otherwise.

            info (dict): additional information on the current time step.
        """
        # Reset current reward
        # ------------------------------------------

        self.reward_timestep = 0.0

        # Step functions
        # ------------------------------------------

        _, battery_reward, _, can_fire_dict = self.batteries.step(action)
        _, _, done_cities, _ = self.cities.step(action)
        _, _, done_enemy_missiles, _ = self.enemy_missiles.step(action)
        _, _, _, _ = self.friendly_missiles.step(action)
        _, _, _, _ = self.target.step(action)

        # Launch a new missile
        # ------------------------------------------

        if action == 5 and can_fire_dict["can_fire"]:
            self.friendly_missiles.launch_missile(self.target)
            self.reward_timestep += CONFIG.REWARD.FRIENDLY_MISSILE_LAUNCHED

        # Check for collisions
        # ------------------------------------------

        self._collisions_missiles()
        self._collisions_cities()

        # Check if episode is finished
        # ------------------------------------------

        done = done_cities or done_enemy_missiles

        # Render every objects
        # ------------------------------------------

        self._reset_observation()
        self.batteries.render(self.observation)
        self.cities.render(self.observation)
        self.enemy_missiles.render(self.observation)
        self.friendly_missiles.render(self.observation)
        self.target.render(self.observation)

        # Return everything
        # ------------------------------------------

        self.reward_total += self.reward_timestep
        return self.observation, self.reward_timestep, done, None

    def render(self, mode="human"):
        """Render the environment."""
        if not self.display:
            pygame.init()
            pygame.display.set_caption("MissileCommand")
            self.display = pygame.display.set_mode(
                (CONFIG.WIDTH, CONFIG.HEIGHT))
            pygame.mouse.set_visible(False)

        surface = pygame.surfarray.make_surface(self.observation)
        self.display.blit(surface, (0, 0))
        pygame.display.update()

    def close(self):
        """Close the environment."""
        if self.display:
            pygame.quit()
