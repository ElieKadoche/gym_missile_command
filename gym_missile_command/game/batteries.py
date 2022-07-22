"""Anti-missiles batteries."""

import cv2
import numpy as np

from gym_missile_command.configuration import CONFIG
from gym_missile_command.utils import get_cv2_xy


class Batteries():
    """Anti-missiles batteries class.

    Attributes:
        NB_BATTERY (int): the number of batteries. It only supports 1 battery.

        batteries (numpy array): of size (N, 1) with N the number of batteries,
            i.d., 1. The feature is: (0) number of available missiles.
        nb_missiles_launched (int): the number of missiles launched.
    """

    NB_BATTERIES = 1

    def __init__(self):
        """Initialize 1 battery."""
        self.batteries = np.zeros((self.NB_BATTERIES, 1), dtype=np.float32)

    def reset(self, seed=None):
        """Reset batteries.

        Total number of missiles is reset to default.

        Warning:
            To fully initialize a Batteries object, init function and reset
            function musts be called.

        Args:
            seed (int): seed for reproducibility.
        """
        self.batteries[:, 0] = CONFIG.FRIENDLY_MISSILES.NUMBER
        self.nb_missiles_launched = 0

    def step(self, action):
        """Go from current step to next one.

        The missile launched is done in the main environment class.

        Args:
            action (int): (0) do nothing, (1) target up, (2) target down, (3)
                target left, (4) target right, (5) fire missile.

        Returns:
            observation: None.

            reward: None.

            done: None.

            info (dict): additional information of the current time step. It
                contains key "can_fire" with associated value "True" if the
                anti-missile battery can fire a missile and "False" otherwise.
        """
        can_fire = self.batteries[0, 0] > 0

        if action == 5 and can_fire:
            self.batteries[0, 0] -= 1

        return None, None, None, {"can_fire": can_fire}

    def render(self, observation):
        """Render anti-missiles batteries.

        Todo:
            Include the number of available missiles.

        Args:
            observation (numpy.array): the current environment observation
                representing the pixels. See the object description in the main
                environment class for information.
        """
        cv2.circle(
            img=observation,
            center=(get_cv2_xy(CONFIG.EPISODE.HEIGHT,
                               CONFIG.EPISODE.WIDTH,
                               0.0,
                               0.0)),
            radius=int(CONFIG.BATTERY.RADIUS),
            color=CONFIG.COLORS.BATTERY,
            thickness=-1,
        )
