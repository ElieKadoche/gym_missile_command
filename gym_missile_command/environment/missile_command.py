"""Main environment class."""

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from gym_missile_command.configuration import CONFIG, update_config
from gym_missile_command.game.batteries import Batteries
from gym_missile_command.game.cities import Cities
from gym_missile_command.game.enemy_missiles import EnemyMissiles
from gym_missile_command.game.friendly_missiles import FriendlyMissiles
from gym_missile_command.game.target import Target


class MissileCommandEnv(gym.Env):
    """Missile Command Gym environment.

    Attributes:
        NB_ACTIONS (int): the 6 possible actions. (0) do nothing, (1) target
            up, (2) target down, (3) target left, (4) target right, (5) fire
            missile.

        action_space (gym.spaces.discrete.Discrete): OpenAI Gym action space.
        observation_space (gym.spaces.Box): OpenAI Gym observation space.
        reward_range (tuple): reward range.

        batteries (Batteries): Batteries game object.
        cities (Cities): Cities game object.
        enemy_missiles (EnemyMissiles): EnemyMissiles game object.
        friendly_missiles (FriendlyMissiles): FriendlyMissiles game object.

        observation (numpy.array): of size (CONFIG.EPISODE.WIDTH,
            CONFIG.EPISODE.HEIGHT, 3). The observation of the current time]
            step, representing the RGB values of each pixel.
        reward (float): reward of the current time step.
        reward_total (float): reward sum from first time step to current one.
        time_step (int): current time step, starts from 0.
    """

    NB_ACTIONS = 6

    def __init__(self, env_context=None):
        """Initialize environment.

        Args:
            env_context (dict): environment configuration.
        """
        # Update configuration
        if env_context is not None:
            update_config(env_context)

        # Action and observation spaces
        self.action_space = spaces.Discrete(self.NB_ACTIONS)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(CONFIG.OBSERVATION.HEIGHT, CONFIG.OBSERVATION.WIDTH, 3),
            dtype=np.uint8,
        )

        # TODO: compute reward bounds
        self.reward_range = (-float("inf"), float("inf"))

        # Objects
        self.batteries = Batteries()
        self.cities = Cities()
        self.enemy_missiles = EnemyMissiles()
        self.friendly_missiles = FriendlyMissiles()
        self.target = Target()

        # No display while no render
        self._clock = None
        self._display = None

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

        # Update time step reward
        self.reward += CONFIG.REWARD.DESTROYED_CITY * \
            cities_out.shape[0]

        # Destroy the cities
        self.cities.cities[cities_out, 2] -= 1

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

        # Remove these missiles
        missiles_out = np.argwhere(np.sum(inside_radius, axis=1) >= 1)
        self.enemy_missiles.enemy_missiles = np.delete(
            self.enemy_missiles.enemy_missiles,
            np.squeeze(missiles_out),
            axis=0,
        )

        # Compute current reward
        nb_missiles_destroyed = missiles_out.shape[0]
        self.reward += CONFIG.REWARD.DESTROYED_ENEMEY_MISSILES * \
            nb_missiles_destroyed

    def _compute_observation(self):
        """Compute observation."""
        # Reset observation
        self.observation = np.zeros(
            (CONFIG.EPISODE.WIDTH, CONFIG.EPISODE.HEIGHT, 3), dtype=np.uint8)
        self.observation[:, :, 0] = CONFIG.COLORS.BACKGROUND[0]
        self.observation[:, :, 1] = CONFIG.COLORS.BACKGROUND[1]
        self.observation[:, :, 2] = CONFIG.COLORS.BACKGROUND[2]

        # Draw objects
        self.batteries.render(self.observation)
        self.cities.render(self.observation)
        self.enemy_missiles.render(self.observation)
        self.friendly_missiles.render(self.observation)
        self.target.render(self.observation)

    def _process_observation(self):
        """Process observation.

        This function could be implemented into the agent model, but for
        commodity this environment can do it directly.

        The interpolation mode INTER_AREA seems to give the best results. With
        other methods, every objects could not be seen at all time steps.

        Returns:
            processed_observation (numpy.array): of size
                (CONFIG.OBSERVATION.HEIGHT, CONFIG.OBSERVATION.WIDTH, 3), the
                resized (or not) observation.
        """
        # Process observation
        processed_observation = cv2.resize(
            self.observation,
            (CONFIG.OBSERVATION.HEIGHT, CONFIG.OBSERVATION.WIDTH),
            interpolation=cv2.INTER_AREA,
        )
        return processed_observation.astype(np.float32)

    def reset(self, seed=None):
        """Reset the environment.

        Args:
            seed (int): seed for reproducibility.

        Returns:
            observation (numpy.array): the processed observation.
        """
        # Reset time step and rewards
        self.time_step = 0
        self.reward_total = 0.0
        self.reward = 0.0

        # Reset objects
        self.batteries.reset(seed=seed)
        self.cities.reset(seed=seed)
        self.enemy_missiles.reset(seed=seed)
        self.friendly_missiles.reset(seed=seed)
        self.target.reset(seed=seed)

        # Compute observation
        self._compute_observation()

        return self._process_observation()

    def step(self, action):
        """Go from current step to next one.

        Args:
            action (int): 0, 1, 2, 3, 4 or 5, the different actions.

        Returns:
            observation (numpy.array): the processed observation.

            reward (float): reward of the current time step.

            done (bool): True if the episode is finished, False otherwise.

            info (dict): additional information on the current time step.
        """
        # Reset current reward
        self.reward = 0.0

        # Step functions
        _, battery_reward, _, can_fire_dict = self.batteries.step(action)
        _, _, done_cities, _ = self.cities.step(action)
        _, _, done_enemy_missiles, _ = self.enemy_missiles.step(action)
        _, _, _, _ = self.friendly_missiles.step(action)
        _, _, _, _ = self.target.step(action)

        # Launch a new missile
        if action == 5 and can_fire_dict["can_fire"]:
            self.friendly_missiles.launch_missile(self.target)
            self.reward += CONFIG.REWARD.FRIENDLY_MISSILE_LAUNCHED

        # Check for collisions
        self._collisions_missiles()
        self._collisions_cities()

        # Check if episode is finished
        done = done_cities or done_enemy_missiles

        # Compute observation
        self._compute_observation()

        # Update values
        self.time_step += 1
        self.reward_total += self.reward

        return self._process_observation(), self.reward, done, {}

    def render(self, mode="raw_observation"):
        """Render the environment.

        This function renders the environment observation. To check what the
        processed observation looks like, it can also renders it.

        Args:
            mode (str): the render mode. Possible values are "raw_observation"
                and "processed_observation".
        """
        # Get width and height
        w, h = CONFIG.EPISODE.WIDTH, CONFIG.EPISODE.HEIGHT

        # Initialize PyGame
        if self._display is None:
            pygame.init()
            pygame.mouse.set_visible(False)
            self._clock = pygame.time.Clock()
            pygame.display.set_caption("MissileCommand")
            self._display = pygame.display.set_mode((w, h))

        # For debug only, display processed observation
        if mode == "processed_observation":
            observation = self._process_observation()
            surface = pygame.surfarray.make_surface(observation)
            surface = pygame.transform.scale(surface, (h, w))

        # Normal mode
        else:
            observation = self.observation
            surface = pygame.surfarray.make_surface(observation)

        # Display all
        self._display.blit(surface, (0, 0))
        pygame.display.update()

        # Limit max FPS
        self._clock.tick(CONFIG.EPISODE.FPS)

    def close(self):
        """Close the environment."""
        if self._display is not None:
            pygame.quit()
