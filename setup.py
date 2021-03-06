"""Setup file."""

from setuptools import setup

setup(
    name="gym_missile_command",
    version="1.0",
    author="Elie KADOCHE",
    install_requires=["gym",
                      "numpy",
                      "opencv-python",
                      "pygame"],
    description="Gym environment of the Atari game, Missile Command.",
)
