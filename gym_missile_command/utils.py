"""Useful functions."""

import numpy as np

import gym_missile_command.config as CONFIG


def get_render_coordinates(x, y):
    """Transform environment coordinates into render environments.

    The origin of the environment is the anti-missiles battery, placed in the
    bottom center. But in the render method, the origin is in the top left
    corner. It is also good to note that in python-opencv, coordinates are
    written (y, x) and not (x, y) like for the environment.

    Args:
        x (numpy.array): of size (N) with N being the number of x coordinates
            to convert.

        y (numpy.array): of size (N) with N being the number of y coordinates
            to convert.
    """
    x += CONFIG.WIDTH / 2
    y -= CONFIG.HEIGHT
