"""Target."""

import cv2

from gym_missile_command.config import CONFIG
from gym_missile_command.utils import get_cv2_xy


class Target():
    """Target class.

    Attributes:
        x (float): x position.
        y (float): y position.
    """

    def __init__(self):
        """Initialize target."""
        pass

    def reset(self, seed=None):
        """Reset target.

        Warning:
            To fully initialize a Batteries object, init function and reset
            function musts be called.

        Args:
            seed (int): seed for reproducibility.
        """
        self.x = 0.0
        self.y = CONFIG.HEIGHT / 2

    def step(self, action):
        """Go from current step to next one.

        Args:
            action (int): (0) do nothing, (1) target up, (2) target down, (3)
                target left, (4) target right, (5) fire missile.

        Returns:
            observation: None.

            reward: None.

            done: None.

            info: None.
        """
        if action == 1:
            self.y = min(CONFIG.HEIGHT, self.y + CONFIG.TARGET.VY)

        elif action == 2:
            self.y = max(0, self.y - CONFIG.TARGET.VY)

        elif action == 3:
            self.x = max(-CONFIG.WIDTH / 2, self.x - CONFIG.TARGET.VX)

        elif action == 4:
            self.x = min(CONFIG.WIDTH / 2, self.x + CONFIG.TARGET.VX)

        return None, None, None, None

    def render(self, observation):
        """Render target.

        The target is a cross, represented by 4 coordinates, 2 for the
        horizontal line and 2 for the vertical line.

        Args:
            observation (numpy.array): the current environment observation
                representing the pixels. See the object description in the main
                environment class for information.
        """
        # Horizontal
        cv2.line(
            img=observation,
            pt1=(get_cv2_xy(CONFIG.HEIGHT,
                            CONFIG.WIDTH,
                            self.x - CONFIG.TARGET.SIZE,
                            self.y)),
            pt2=(get_cv2_xy(CONFIG.HEIGHT,
                            CONFIG.WIDTH,
                            self.x + CONFIG.TARGET.SIZE,
                            self.y)),
            color=CONFIG.COLORS.TARGET,
            thickness=1,
        )

        # Vertical
        cv2.line(
            img=observation,
            pt1=(get_cv2_xy(CONFIG.HEIGHT,
                            CONFIG.WIDTH,
                            self.x,
                            self.y + CONFIG.TARGET.SIZE)),
            pt2=(get_cv2_xy(CONFIG.HEIGHT,
                            CONFIG.WIDTH,
                            self.x,
                            self.y - CONFIG.TARGET.SIZE)),
            color=CONFIG.COLORS.TARGET,
            thickness=1,
        )
