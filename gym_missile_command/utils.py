"""Useful functions."""

import numpy as np

from gym_missile_command.config import CONFIG


def get_cv2_xy(x, y):
    """Transform x environment position into x opencv position.

    The origin of the environment is the anti-missiles battery, placed in the
    bottom center. But in the render method, the origin is in the top left
    corner. It is also good to note that in python-opencv, coordinates are
    written (y, x) and not (x, y) like for the environment.

    Args:
        x (float): x environment coordinate.
        y (float): y environment coordinate.

    Returns:
        y (int): x opencv coordinate.

        x (int): x opencv coordinate.
    """
    return int(CONFIG.HEIGHT - y), int(x + (CONFIG.WIDTH / 2))
