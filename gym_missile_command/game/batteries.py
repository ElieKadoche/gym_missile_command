"""Anti-missiles batteries."""

import numpy as np

import gym_missile_command.config as CONFIG


class Batteries():
    """Anti-missiles batteries class.

    Attributes:
        NB_BATTERY (int): the number of batteries. It only supports 1 battery.
    """
    NB_BATTERY = 1

    def __init__(self):
        """Initialize 1 battery.

        Attributes:
            batteries (numpy array): of size (N, 1) with N the number of
                batteries, i.d. 1. The feature is: (0) number of available
                missiles.
        """
        self.batteries = np.zeros((self.NB_BATTERY, 1), dtype=CONFIG.DTYPE)

    def reset(self):
        """Reset batteries.

        Total number of missiles is reset to default.

        Warning:
            To fully initialize a Batteries object, init function and reset
            function musts be called.
        """
        self.batteries[:, 0] = CONFIG.BATTERY_MISSILES_NUMBER
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
