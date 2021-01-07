"""Friendly missiles."""


class FriendlyMissiles():
    """Friendly missiles class.

    Friendly missiles are send by the user from the anti-missiles battery.
    """

    def __init__(self):
        """Initialize missiles.

        Notes:
            No need to keep trace of the origin fire position because all
            missiles are launched from the origin (0, 0).

        Attributes:
            missiles_movement (numpy array): of size (N, 4) with N the number
                of missiles in movement in the environment. The features are:
                (0) x position, (1) y position, (2) horizontal speed vx and (3)
                vertical speed (vy).

            missiles_explosion (numpy array): of size (M, 3) with M the number
                of missiles in explosion. The features are: (0) x position, (1)
                y position and (2) explosion level.
        """
        pass
