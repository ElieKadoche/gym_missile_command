"""Cities."""

import sys

import numpy as np

import gym_missile_command.config as CONFIG


class Cities():
    """Cities class."""

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

        # Cities on the left side
        self.cities[:CONFIG.CITIES_NUMBER, 0] = -np.arange(start=start,
                                                           stop=stop,
                                                           step=step,
                                                           dtype=CONFIG.DTYPE)

        # Cities on the right side
        self.cities[CONFIG.CITIES_NUMBER:, 0] = np.arange(start=start,
                                                          stop=stop,
                                                          step=step,
                                                          dtype=CONFIG.DTYPE)

    def reset(self):
        """Reset cities.

        Integrity is reset to 1 for all cities.

        Warning:
            To fully initialize a Cities object, init function and reset
            function musts be called.
        """
        self.cities[:, 2] = 1.0
