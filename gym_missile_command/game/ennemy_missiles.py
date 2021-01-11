"""Enemy missiles."""

import random

import numpy as np

import gym_missile_command.config as CONFIG


class EnemyMissiles():
    """Enemy missiles class.

    Enemy missiles are created by the environment.
    """

    def __init__(self):
        """Initialize missiles.

        Attributes:
            enemy_missiles (numpy array): of size (N, 6)  with N the number of
                enemy missiles present in the environment. The features are:
                (0) initial x position, (1) initial y position, (2) current x
                position, (3) current y position, (4) horizontal speed vx and
                (5) vertical speed vy.

            nb_missiles_launched (int): the number of enemy missiles launched
                in the environment.
        """
        pass

    def _launch_missile(self):
        """Launch a new missile.

        0) Generate initial and final positions. 1) Compute speed vectors. 2)
        Add the new missile.
        """
        # Generate initial and final positions
        # ------------------------------------------

        # Initial position
        x0 = random.uniform(-0.5 * CONFIG.WIDTH, 0.5 * CONFIG.WIDTH)
        y0 = CONFIG.HEIGHT

        # Final position
        x1 = random.uniform(-0.5 * CONFIG.WIDTH, 0.5 * CONFIG.WIDTH)
        y2 = 0.0

        # Compute speed vectors
        # ------------------------------------------

        # Compute norm
        norm = np.sqrt(np.square(x1 - x0) + np.square(y1 - y0))

        # Compute unit vectors
        ux = (x1 - x0) / norm
        uy = (y1 - y0) / norm

        # Compute speed vectors
        vx = CONFIG.BATTERY_MISSILE_SPEED * ux
        vy = CONFIG.BATTERY_MISSILE_SPEED * uy

        # Add the new missile
        # ------------------------------------------

        # Create the missile
        new_missile = np.array(
            [[x0, y0, x0, y0, x1, y1, vx, vy]],
            dtype=CONFIG.DTYPE,
        )

        # Add it to the others
        self.enemy_missiles = np.vstack(
            (self.enemy_missiles, new_missile))

        # Increase number of launched missiles
        self.nb_missiles_launched += 1

    def reset(self):
        """Reset enemy missiles.

        Warning:
            To fully initialize a EnemyMissiles object, init function and reset
            function must be called.
        """
        self.enemy_missiles = np.zeros((0, 6), dtype=CONFIG.DTYPE)
        self.nb_missiles_launched = 0

    def step(self):
        """Go from current step to next one.

        0) Moving missiles. 1) Potentially launch a new missile. 2) Remove
        missiles that hit the ground.

        Warning:
            Collisions with friendly missiles are not checked here.

        Notes:
            From one step to another, a missile could exceed its final
            position, so we need to do some verification. This issue is due to
            the discrete nature of environment, decomposed in time steps.
        """
        # Moving missiles
        # ------------------------------------------

        # Compute horizontal and vertical distances to targets
        dx = np.abs(self.enemy_missiles[:, 2] - self.enemy_missiles[:, 0])
        dy = np.abs(self.enemy_missiles[:, 3] - self.enemy_missiles[:, 1])

        # Take the minimum between the actual speed and the distance to target
        movement_x = np.sign(self.enemy_missiles[:, 4]) \
            * np.minimum(np.abs(self.enemy_missiles[:, 4]), dx)
        movement_y = np.sign(self.enemy_missiles[:, 5]) \
            * np.minimum(np.abs(self.enemy_missiles[:, 5]), dy)

        # Step t to step t+1
        self.enemy_missiles[:, 0] += movement_x
        self.enemy_missiles[:, 1] += movement_y

        # Potentially launch a new missile
        # ------------------------------------------

        if self.nb_missiles_launched < CONFIG.ENEMY_MISSILES_NUMBER:
            if random.random() <= CONFIG.ENEMY_MISSILE_PROBA_IN:
                self._launch_new_missile()

        # Remove missiles that hit the ground
        # ------------------------------------------

        missiles_out_indices = np.squeeze(np.argwhere(
            (self.enemy_missiles[:, 1] < 0.0))

        self.enemy_missiles=np.delete(
            self.enemy_missiles, missiles_out_indices, axis=0)
