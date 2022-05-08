"""Useful functions."""


def get_cv2_xy(height, width, x, y):
    """Transform x environment position into x opencv position.

    The origin of the environment is the anti-missiles battery, placed in the
    bottom center. But in the render method, the origin is in the top left
    corner. It is also good to note that in python-opencv, coordinates are
    written (y, x) and not (x, y) like for the environment.

    Args:
        height (float): environment height.
        width (float): environment width.
        x (float): x environment coordinate.
        y (float): y environment coordinate.

    Returns:
        y (int): x opencv coordinate.

        x (int): x opencv coordinate.
    """
    return int(height - y), int(x + (width / 2))
