"""Friendly missiles."""

import numpy as np

import gym_missile_command.config as CONFIG


class FriendlyMissiles():
    """Friendly missiles class.

    Friendly missiles are send by the user from the anti-missiles battery.

    Attributes:
        ORIGIN_X (float): x origin position of the anti-missiles battery.

        ORIGIN_Y (float): y origin position of the anti-missiles battery.
    """

    ORIGIN_X = 0.0
    ORIGIN_Y = 0.0

    def __init__(self):
        """Initialize friendly missiles.

        Notes:
            No need to keep trace of the origin fire position because all
            missiles are launched from the origin (0, 0).

        Attributes:
            missiles_movement (numpy array): of size (N, 6) with N the number
                of missiles in movement in the environment. The features are:
                (0) current x position, (1) current y position, (2) final x
                position, (3) final y position, (4) horizontal speed vx and (5)
                vertical speed (vy).

            missiles_explosion (numpy array): of size (M, 3) with M the number
                of missiles in explosion. The features are: (0) x position, (1)
                y position and (2) explosion level.
        """
        pass

    def launch_missile(self, target):
        """Launch a new missile.

        0) Compute speed vectors. 1) Add the new missile.

        Notes:
            When executing this function, not forget to decrease the number
            of available missiles of the anti-missiles battery by 1.

        Args:
            target (list): of 2 elements, corresponding to the position (x, y)
                of the target.
        """
        # Compute speed vectors
        # ------------------------------------------

        # Compute norm
        norm = np.sqrt(np.square(target[0]) + np.square(target[1]))

        # Compute unit vectors
        ux = target[0] / norm
        uy = target[1] / norm

        # Compute speed vectors
        vx = CONFIG.BATTERY_MISSILE_SPEED * ux
        vy = CONFIG.BATTERY_MISSILE_SPEED * uy

        # Add the new missile
        # ------------------------------------------

        # Create the missile
        new_missile = np.array(
            [[self.ORIGIN_X, self.ORIGIN_Y, target[0], target[1], vx, vy]],
            dtype=CONFIG.DTYPE,
        )

        # Add it to the others
        self.missiles_movement = np.vstack(
            (self.missiles_movement, new_missile))

    def reset(self):
        """Reset friendly missiles.

        Warning:
            To fully initialize a FriendlyMissiles object, init function and
            reset function must be called.
        """
        self.missiles_movement = np.zeros((0, 6), dtype=CONFIG.DTYPE)
        self.missiles_explosion = np.zeros((0, 3), dtype=CONFIG.DTYPE)

    def step(self):
        """Go from current step to next one.

        0) Moving missiles. 1) Exploding missiles. 2) New exploding missiles.
        3) Remove missiles with full explosion.

        Notes:
            From one step to another, a missile could exceed its final
            position, so we need to do some verification. This issue is due to
            the discrete nature of environment, decomposed in time steps.
        """
        # Moving missiles
        # ------------------------------------------

        # Compute horizontal and vertical distances to targets
        dx = np.abs(
            self.missiles_movement[:, 2] - self.missiles_movement[:, 0])
        dy = np.abs(
            self.missiles_movement[:, 3] - self.missiles_movement[:, 1])

        # Take the minimum between the actual speed and the distance to target
        movement_x = np.sign(self.missiles_movement[:, 4]) \
            * np.minimum(np.abs(self.missiles_movement[:, 4]), dx)
        movement_y = np.sign(self.missiles_movement[:, 5]) \
            * np.minimum(np.abs(self.missiles_movement[:, 5]), dy)

        # Step t to step t+1
        self.missiles_movement[:, 0] += movement_x
        self.missiles_movement[:, 1] += movement_y

        # Exploding missiles
        # ------------------------------------------

        # Increase by 1 the explosion
        self.missiles_explosion[:, 2] += 1.0

        # New exploding missiles
        # ------------------------------------------

        # Indices of new exploding missiles
        new_exploding_missiles_indices = np.squeeze(np.argwhere(
            (self.missiles_movement[:, 0] == self.missiles_movement[:, 2]) &
            (self.missiles_movement[:, 1] == self.missiles_movement[:, 3])
        ))

        # Get positions
        x = self.missiles_movement[new_exploding_missiles_indices, 0]
        y = self.missiles_movement[new_exploding_missiles_indices, 1]

        # Remove missiles
        self.missiles_movement = np.delete(
            self.missiles_movement, new_exploding_missiles_indices, axis=0)

        # Create new ones
        new_exploding_missiles = np.zeros(
            (new_exploding_missiles_indices.shape[0], 3),
            dtype=CONFIG.DTYPE,
        )

        # Affect positions
        new_exploding_missiles[:, 0] = x
        new_exploding_missiles[:, 1] = y

        # Add them
        self.missiles_explosion = np.vstack(
            (self.missiles_explosion, new_exploding_missiles))

        # Remove missiles with full explosion
        # ------------------------------------------

        full_explosion_indices = np.squeeze(np.argwhere(
            (self.missiles_explosion[:, 2] >
             CONFIG.BATTERY_MISSILE_EXPLOSION_RADIUS)
        ))

        self.missiles_explosion = np.delete(
            self.missiles_explosion, full_explosion_indices, axis=0)