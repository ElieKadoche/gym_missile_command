"""Friendly missiles."""

import numpy as np

import gym_missile_command.config as CONFIG


class FriendlyMissiles():
    """Friendly missiles class.

    Friendly missiles are send by the user from the anti-missiles battery.
    """

    def __init__(self):
        """Initialize friendly missiles.

        Notes:
            No need to keep trace of the origin fire position because all
            missiles are launched from the origin (0, 0).

        Attributes:
            missiles_movement (numpy array): of size (N, 4) with N the number
                of missiles in movement in the environment. The features are:
                (0) x position, (1) y position, (2) horizontal speed vx and (3)
                vertical speed (vy).

            missiles_explosion (numpy array): of size (M, 3) with M the number
                of missiles in explosion. The features are: (0) x position, (1)
                y position and (2) explosion level.
        """
        pass

    def reset(self):
        """Reset friendly missiles.

        Warning:
            To fully initialize a FriendlyMissiles object, init function and
            reset function musts be called.
        """
        self.missiles_movement = np.zeros((0, 4), dtype=CONFIG.DTYPE)
        self.missiles_explosion = np.zeros((0, 3), dtype=CONFIG.DTYPE)

    def launch_missile(self, target):
        """Launch a new missile.

        Notes:
            When executing this function, not forget to decrease the number
            of available missiles of the anti-missiles battery by 1.

        Args:
            target (list): of 2 elements, corresponding to the position (x, y)
                of the target.
        """
        # Compute speed vectors
        # ------------------------------------------

        norm = np.sqrt(np.square(target[0]) + np.square(target[1]))

        ux = target[0] / norm
        uy = target[1] / norm

        vx = CONFIG.BATTERY_MISSILE_SPEED * ux
        vy = CONFIG.BATTERY_MISSILE_SPEED * uy

        # Add the new missile
        # ------------------------------------------

        self.missiles_movement = np.vstack((
            self.missiles_movement,
            np.array([[0.0, 0.0, vx, vy]], dtype=CONFIG.DTYPE),
        ))

    def step(self):
        """Go from current step to next one."""
        # Moving missiles
        # ------------------------------------------

        self.missiles_movement[:, 0] += self.missiles_movement[:, 2]
        self.missiles_movement[:, 1] += self.missiles_movement[:, 3]

        # Exploding missiles
        # ------------------------------------------

        # Increase by 1 the explosion
        self.missiles_explosion[:, 2] += 1.0

        # Remove missiles with full explosion
        full_explosion_indices = np.squeeze(np.argwhere(
            (self.missiles_explosion[:, 2] >
             CONFIG.BATTERY_MISSILE_EXPLOSION_RADIUS)
        ))

        self.missiles_explosion = np.delete(
            self.missiles_explosion, full_explosion_indices, axis=0)
