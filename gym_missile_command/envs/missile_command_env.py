import gym
import gym_missiles_command.config as CONFIG
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

from gym_missile_command.game.batteries import Batteries
from gym_missile_command.game.cities import Cities
from gym_missile_command.game.enemy_missiles import EnemyMissiles
from gym_missile_command.game.friendly_missiles import FriendlyMissiles


class MissileCommandEnv(gym.Env):
    """Missile Command Gym environment.


    Attributes:
        NB_ACTIONS (int): the 6 possible actions. (0) do nothing, (1) target
            up, (2) target down, (3) target left, (4) target right, (5) fire
            missile.
    """
    NB_ACTIONS = 6
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """Initialize MissileCommand environment.

        Attributes:
            target (list): of two elements, corresponding the position (x, y)
                of the target.
        """
        super(MissileCommandEnv, self).__init__()
        self.action_space = spaces.Discrete(self.NB_ACTIONS)

        # Objects
        self.batteries = Batteries()
        self.cities = Cities()
        self.enemy_missiles = EnemyMissiles()
        self.friendly_missiles = FriendlyMissiles()

    def _collisions(self):
        """Check for collisions.

        For all exploding missiles, check if an enemy missiles is destroyed.
        """
        # Friendly exploding missiles
        friendly_m = self.friendly_missiles.missiles_explosion

        # Enemy missiles current positions
        enemy_m = self.enemy_missiles.enemy_missiles[:, [0, 1]]

        # Align enemy missiles and friendly exploding ones
        friendly_m_dup = np.tile(enemy_m, reps=[enemy_m.shape[0], 1])
        enemy_m_dup = np.repeat(enemy_m, friendly_m.shape[0], axis=0)

        # Compute distances
        dx = friendly_m_dup[:, 0] - enemy_m_dup[:, 0]
        dy = friendly_m_dup[:, 1] - enemy_m_dup[:, 1]
        distances = np.sqrt(np.square(dx) + np.square(dy))

        # Get enemy missiles inside an explosion radius
        inside_radius = distances <= (
            friendly_m_dup[:, 2] + CONFIG.ENEMY_MISSILE_RADIUS)
        inside_radius = inside_radius.astype(int)
        inside_radius = np.reshape(
            inside_radius, (enemy_m.shape[0], friendly_m.shape[0]))

        # Remove theses missiles
        missiles_out = np.argwhere(np.sum(inside_radius, axis=1) >= 1)
        self.enemy_missiles.enemy_missiles = np.delete(
            self.enemy_missiles.enemy_missiles,
            np.squeeze(missiles_out),
            axis=0,
        )

    def reset(self):
        """Reset the environment.

        Returns:
            observation (numpy.array): the representation of the environment.
        """
        self.batteries.reset()
        self.cities.reset()
        self.enemy_missiles.reset()
        self.friendly_missiles.reset()

        # The target
        self.target = [0.0, CONFIG.HEIGHT / 2]

    def step(self, action):
        """Go from current step to next one.

        Args:
            action (int): 0, 1, 2, 3, 4 or 5, the different actions.

        Returns:
            observation (numpy.array): the representation of the environment.

            reward (float): reward of the current time step.

            done (bool): True if the episode is finished, False otherwise.

            info (dict): additional information on the current time step.
        """
        self.enemy_missiles.step()
        self.friendly_missiles.step()

        # Move target (actions 1, 2, 3 and 4)
        # ------------------------------------------

        # Launch a new missile (actions 5)
        # ------------------------------------------

        # Check for collisions
        # ------------------------------------------

    def render(self, mode="human"):
        """Render the environment."""
        pass

    def close(self):
        """Close the environment."""
        pass
