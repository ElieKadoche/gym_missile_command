"""Cities."""

import sys

import cv2
import numpy as np

import gym_missile_command.config as CONFIG
from gym_missile_command.utils import get_render_coordinates


class Cities():
    """Cities class.

    Attributes:
        MAX_HEALTH (float): value corresponding to the max health of a city.
            Each time an enemy missile destroys a city, it loses 1 point of
            heath. At 0, the city is completely destroyed.
    """
    MAX_HEALTH = 1.0

    def __init__(self):
        """Initialize cities.

        Attributes:
            cities (numpy array): of size (N, 3) with N the number of cities.
                The features are: (0) x position, (1) y position and (2)
                integrity level (0 if destroyed else 1).
        """
        # First initializations
        # ------------------------------------------

        # The length of free space for the cities (only for one side)
        free_space = 0.5 * CONFIG.WIDTH - CONFIG.BATTERY_RADIUS

        # Creation of the main numpy array
        self.cities = np.zeros((CONFIG.CITIES_NUMBER, 3), dtype=CONFIG.DTYPE)

        # Check for errors
        # ------------------------------------------

        # Even number of cities for symmetry, bigger or equal to 2
        if CONFIG.CITIES_NUMBER % 2 != 0 or CONFIG.CITIES_NUMBER < 2:
            sys.exit("Please choose an even number of cities, bigger or equal "
                     "to 2.")

        # Enough space is needed for objects not to not overlap
        if free_space < CONFIG.CITIES_NUMBER * CONFIG.CITY_RADIUS:
            sys.exit("Not enough space for the cities. Increase width, "
                     "decrease the number of cities or decrease objects "
                     "radiuses.")

        # Compute x positions
        # ------------------------------------------

        # Gap between cities
        gap = (free_space - CONFIG.CITIES_NUMBER * CONFIG.CITY_RADIUS) \
            / (0.5 * CONFIG.CITIES_NUMBER + 1)

        # First position, last position and step between cities centers
        start = CONFIG.BATTERY_RADIUS + gap + CONFIG.CITY_RADIUS
        step = gap + 2 * CONFIG.CITY_RADIUS
        stop = 0.5 * CONFIG.WIDTH - gap
        half_cities_nb = int(CONFIG.CITIES_NUMBER / 2)

        # Cities on the left side
        self.cities[:half_cities_nb, 0] = -np.arange(start=start,
                                                     stop=stop,
                                                     step=step,
                                                     dtype=CONFIG.DTYPE)

        # Cities on the right side
        self.cities[half_cities_nb:, 0] = np.arange(start=start,
                                                    stop=stop,
                                                    step=step,
                                                    dtype=CONFIG.DTYPE)

    def get_remaining_cities(self):
        """Compute healthy cities number.

        Returns:opencv draw multiple circles
            nb_remaining_cities (int): the number of remaining cities.
        """
        return np.sum(self.cities[:, 2] == self.MAX_HEALTH)

    def reset(self):
        """Reset cities.

        Integrity is reset to 1 for all cities.

        Warning:
            To fully initialize a Cities object, init function and reset
            function musts be called.
        """
        self.cities[:, 2] = self.MAX_HEALTH

    def step(self, action):
        """Go from current step to next one.

        Destructions by enemy missiles are checked in the main environment
        class.

        Args:
            action (int): (0) do nothing, (1) target up, (2) target down, (3)
                target left, (4) target right, (5) fire missile.

        returns:
            observation: None.

            reward: None.

            done (bool): True if the episode is finished, i.d. all cities are
                destroyed. False otherwise.

            info: None.
        """
        done = np.all(self.cities[:, 2] == 0.0)
        return None, None, done, None

    def render(self, observation):
        """Render cities.

        Todo:
            Include the integrity level.

        Args:
            observation (numpy.array): the current environment observation
                representing the pixels. See the object description in the main
                environment class for information.
        """
        x_centers = self.cities[:, 0]
        y_centers = self.cities[:, 1]
        integrities = self.cities[:, 2]
        get_render_coordinates(x_centers, y_centers)

        for x, y, integrity in zip(x_centers, y_centers, integrities):
            cv2.circle(
                img=observation,
                center=(int(y), int(x)),
                radius=int(CONFIG.CITY_RADIUS),
                color=CONFIG.COLOR_CITY,
                thickness=-1,
            )
