"""Anti-missiles batteries."""

import numpy as np

import gym_missile_command.config as CONFIG


class Batteries():
    """Anti-missiles batteries class."""

    def __init__(self):
        """Initialize 1 battery.

        Attributes:
            batteries (numpy array): of size (N, 1) with N the number of
                batteries, i.d. 1. The feature is: (0) number of available
                missiles.
        """
        self.batteries = np.zeros((1, 1), dtype=CONFIG.DTYPE)

    def reset(self):
        """Reset batteries.

        Total number of missiles is reset to default.

        Warning:
            To fully initialize a Batteries object, init function and reset
            function musts be called.
        """
        self.batteries[:, 0] = CONFIG.BATTERY_MISSILES_NUMBER
