"""Setup file."""

from setuptools import setup

setup(
    name="gym_missile_command",
    version="1.3",
    author="Elie KADOCHE",
    author_email="eliekadoche78@gmail.com",
    install_requires=["gym", "numpy", "opencv-python", "pygame"],
    description="Gym environment of the Atari game, Missile Command.",
    packages=["gym_missile_command"],
    url="https://github.com/ElieKadoche/gym_missile_command.git",
)
