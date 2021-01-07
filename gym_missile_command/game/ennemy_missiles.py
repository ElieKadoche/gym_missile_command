"""Enemy missiles."""


class EnemyMissiles():
    """Enemy missiles class.

    Enemy missiles are created by the environment.
    """

    def __init__(self):
        """Initialize missiles.

        Attributes:
            missiles_waiting (numpy array): of size (N, 6)  with N the number
                of enemy missiles to drop on the cities. The features are: (0)
                initial position x, (1) initial position y, (2) current
                position x, (3) current position y, (4) horizontal speed vx and
                (5) vertical speed (vy).

            missiles_launched (numpy array): of size (N, 6)  with N the number
                of enemy missiles present in the environment. The features are:
                (0) initial position x, (1) initial position y, (2) current
                position x, (3) current position y, (4) horizontal speed vx and
                (5) vertical speed (vy).
        """
        pass
