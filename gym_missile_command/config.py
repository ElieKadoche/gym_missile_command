"""Environment configuration."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CONFIG():
    """Configuration class.

    Attributes:
        HEIGHT (int): environment height.
        WIDTH (int): environment width.
        DTYPE (numpy.dtype): numpy arrays type.
    """
    HEIGHT: int = 600
    WIDTH: int = 1100
    DTYPE: np.dtype = np.float32

    @dataclass
    class BATTERY():
        """Anti-missiles battery configuration.

        Attributes:
            RADIUS (float): radius of the anti-missile battery object.
        """
        RADIUS: float = 35.0

    @dataclass
    class CITIES():
        """Cities configuration.

        Attributes:
            NUMBER (int): number of cities to defend (even integer >= 2).
            RADIUS (float): radius of a city object.
        """
        NUMBER: int = 6
        RADIUS: float = 27.0

    @dataclass
    class COLORS():
        """Colors configuration.

        Attributes:
            BACKGROUND (tuple): #000000.
            BATTERY (tuple): #ff0ff0.
            CITY (tuple): #0000ff.
            ENEMY_MISSILE (tuple): #ff0000.
            EXPLOSION (tuple): #ffff00.
            FRIENDLY_MISSILE (tuple): #00ff00.
            TARGET (tuple): #ffffff.
        """
        BACKGROUND: tuple = (0, 0, 0)
        BATTERY: tuple = (255, 15, 240)
        CITY: tuple = (0, 0, 255)
        ENEMY_MISSILE: tuple = (255, 0, 0)
        EXPLOSION: tuple = (255, 255, 0)
        FRIENDLY_MISSILE: tuple = (0, 255, 0)
        TARGET: tuple = (255, 255, 255)

    @dataclass
    class ENEMY_MISSILES():
        """Enemy missiles configuration.

        Attributes:
            NUMBER (int): total number of enemy missiles for 1 episode.
            PROBA_IN (float): probability for an enemy missile to appear at a
                time step.
            RADIUS (float): radius of an enemy missile object.
            SPEED (float): enemy missile speed.
        """
        NUMBER: int = 42
        PROBA_IN: float = 0.005
        RADIUS: float = 4.0
        SPEED: float = 1.0

    @dataclass
    class FRIENDLY_MISSILES():
        """Friendly missiles configuration.

        Attributes:
            NUMBER (int): total number of available friendly missiles.
            EXPLOSION_RADIUS (float): maximum explosion radius.
            EXPLOSION_SPEED (float): speed of the explosion.
            RADIUS (float): radius of a friendly missile object.
            SPEED (float): friendly missile speed.
        """
        NUMBER: int = 142
        EXPLOSION_RADIUS: float = 37.0
        EXPLOSION_SPEED: float = 0.5
        RADIUS: float = 7.0
        SPEED: float = 7.0

    @dataclass
    class REWARD():
        """Reward configuration.

        Attributes:
            DESTROYED_CITY (float): reward for each destroyed city.
            DESTROYED_ENEMEY_MISSILES (float): reward for each
                destroyed missile.
            REMAINING_CITY (float): reward for each remaining city.
            REMAINING_MISSILE (float): reward for each remaining
                missile.
        """
        DESTROYED_CITY: float = -10.0
        DESTROYED_ENEMEY_MISSILES: float = 2.0
        REMAINING_CITY: float = 10.0
        REMAINING_MISSILE: float = 5.0

    @dataclass
    class TARGET():
        """Target configuration.

        Attributes:
            SIZE_(int): target size (only for render).
            VX (int): horizontal shifting of the target.
            VY (int): vertical shifting of the target.
        """
        SIZE: int = 12
        VX: int = 6
        VY: int = 6
