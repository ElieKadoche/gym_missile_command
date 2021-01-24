"""Enemy missiles."""

import random

import cv2
import numpy as np

import gym_missile_command.config as CONFIG
from gym_missile_command.utils import get_cv2_xy


class EnemyMissiles():
    """Enemy missiles class.

    Enemy missiles are created by the environment.
    """

    def __init__(self):
        """Initialize missiles.

        Attributes:
            enemy_missiles (numpy array): of size (N, 8)  with N the number of
                enemy missiles present in the environment. The features are:
                (0) initial x position, (1) initial y position, (2) current x
                position, (3) current y position, (4) final x position, (5)
                final y position, (6) horizontal speed vx and (7) vertical
                speed vy.

            nb_missiles_launched (int): the number of enemy missiles launched
                in the environment.
        """
        pass

    def _launch_missile(self):
        """Launch a new missile.

        - 0) Generate initial and final positions.
        - 1) Compute speed vectors.
        - 2) Add the new missile.
        """
        # Generate initial and final positions
        # ------------------------------------------

        # Initial position
        x0 = random.uniform(-0.5 * CONFIG.WIDTH, 0.5 * CONFIG.WIDTH)
        y0 = CONFIG.HEIGHT

        # Final position
        x1 = random.uniform(-0.5 * CONFIG.WIDTH, 0.5 * CONFIG.WIDTH)
        y1 = 0.0

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
        self.enemy_missiles = np.zeros((0, 8), dtype=CONFIG.DTYPE)
        self.nb_missiles_launched = 0

    def step(self, action):
        """Go from current step to next one.

        - 0) Moving missiles.
        - 1) Potentially launch a new missile.
        - 2) Remove missiles that hit the ground.

        Collisions with friendly missiles and / or cities are checked in the
        main environment class.

        Notes:
            From one step to another, a missile could exceed its final
            position, so we need to do some verification. This issue is due to
            the discrete nature of environment, decomposed in time steps.

        Args:
            action (int): (0) do nothing, (1) target up, (2) target down, (3)
                target left, (4) target right, (5) fire missile.

        returns:
            observation: None.

            reward: None.

            done (bool): True if the episode is finished, i.d. there are no
                more enemy missiles in the environment and no more enemy
                missiles to be launch. False otherwise.

            info: None.
        """
        # Moving missiles
        # ------------------------------------------

        # Compute horizontal and vertical distances to targets
        dx = np.abs(self.enemy_missiles[:, 4] - self.enemy_missiles[:, 2])
        dy = np.abs(self.enemy_missiles[:, 5] - self.enemy_missiles[:, 3])

        # Take the minimum between the actual speed and the distance to target
        movement_x = np.sign(self.enemy_missiles[:, 6]) \
            * np.minimum(np.abs(self.enemy_missiles[:, 6]), dx)
        movement_y = np.sign(self.enemy_missiles[:, 7]) \
            * np.minimum(np.abs(self.enemy_missiles[:, 7]), dy)

        # Step t to step t+1
        self.enemy_missiles[:, 2] += movement_x
        self.enemy_missiles[:, 3] += movement_y

        # Potentially launch a new missile
        # ------------------------------------------

        if self.nb_missiles_launched < CONFIG.ENEMY_MISSILES_NUMBER:
            if random.random() <= CONFIG.ENEMY_MISSILE_PROBA_IN:
                self._launch_missile()

        # Remove missiles that hit the ground
        # ------------------------------------------

        missiles_out_indices = np.squeeze(np.argwhere(
            (self.enemy_missiles[:, 2] == self.enemy_missiles[:, 4]) &
            (self.enemy_missiles[:, 3] == self.enemy_missiles[:, 5])
        ))

        self.enemy_missiles = np.delete(
            self.enemy_missiles, missiles_out_indices, axis=0)

        done = self.enemy_missiles.shape[0] == 0 and \
            self.nb_missiles_launched == CONFIG.ENEMY_MISSILES_NUMBER
        return None, None, done, None

    def render(self, observation):
        """Render enemy missiles.

        For each enemy missiles, draw a line of its trajectory and the actual
        missile.

        Args:
            observation (numpy.array): the current environment observation
                representing the pixels. See the object description in the main
                environment class for information.
        """
        for x0, y0, x, y in zip(self.enemy_missiles[:, 0],
                                self.enemy_missiles[:, 1],
                                self.enemy_missiles[:, 2],
                                self.enemy_missiles[:, 3]):
            cv2.line(
                img=observation,
                pt1=(get_cv2_xy(x0, y0)),
                pt2=(get_cv2_xy(x, y)),
                color=CONFIG.COLOR_ENEMY_MISSILE,
                thickness=1,
            )

            cv2.circle(
                img=observation,
                center=(get_cv2_xy(x, y)),
                radius=int(CONFIG.ENEMY_MISSILE_RADIUS),
                color=CONFIG.COLOR_ENEMY_MISSILE,
                thickness=-1,
            )
