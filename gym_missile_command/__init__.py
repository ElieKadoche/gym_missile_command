from pathlib import Path

from gymnasium.envs.registration import register

with open(Path(__file__).parent / "version.py") as _version_file:
    __version__ = _version_file.read().strip()

register(
    id="missile-command-v0",
    entry_point="gym_missile_command.environment:MissileCommandEnv",
)
