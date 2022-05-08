"""Functions to parse the configuration."""

import functools
import sys

from gym_missile_command.config import CONFIG


def _rgetattr(obj, attr, *args):
    """Recursive getattr function."""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))


def _rsetattr(obj, attr, val):
    """Recursive setattr function."""
    pre, _, post = attr.rpartition('.')
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def update_config(env_context):
    """Update global configuration.

    Args:
        env_context (dict): environment configuration.
    """
    # For each custom attributes
    for key, value in env_context.items():

        # Check if attributes is valid
        try:
            _ = _rgetattr(CONFIG, key)
        except AttributeError as e:
            print("Invalid custom configuration: {}".format(e))
            sys.exit(1)

        # Modify it
        _rsetattr(CONFIG, key, value)
