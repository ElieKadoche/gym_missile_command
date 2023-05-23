"""Setup file."""

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).parent
with open(ROOT / "gym_missile_command" / "version.py") as version_file:
    VERSION = version_file.read().strip()

setup(
    name="gym_missile_command",
    version=VERSION,
    author="Elie KADOCHE",
    author_email="eliekadoche78@gmail.com",
    install_requires=["gymnasium >= 0.28.1",
                      "numpy >= 1.24.3",
                      "opencv-python >= 4.7.0.72",
                      "pygame >= 2.4.0"],
    description="Gym environment of the Atari game, Missile Command.",
    packages=["gym_missile_command"],
    url="https://github.com/ElieKadoche/gym_missile_command.git",
)
