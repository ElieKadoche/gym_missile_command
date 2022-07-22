"""Environment configuration."""

from dataclasses import dataclass


@dataclass
class CONFIG():
    """Configuration class."""

    @dataclass
    class EPISODE():
        """Episode global configuration."""

        FPS: int = 144
        HEIGHT: int = 466
        WIDTH: int = 966

    @dataclass
    class BATTERY():
        """Anti-missiles battery configuration."""

        RADIUS: float = 37.0

    @dataclass
    class CITIES():
        """Cities configuration."""

        NUMBER: int = 6
        RADIUS: float = 24.0

    @dataclass
    class COLORS():
        """Colors configuration."""

        BACKGROUND: tuple = (0, 0, 0)
        BATTERY: tuple = (255, 255, 255)
        CITY: tuple = (0, 0, 255)
        ENEMY_MISSILE: tuple = (255, 0, 0)
        EXPLOSION: tuple = (255, 255, 0)
        FRIENDLY_MISSILE: tuple = (0, 255, 0)
        TARGET: tuple = (255, 255, 255)

    @dataclass
    class ENEMY_MISSILES():
        """Enemy missiles configuration."""

        NUMBER: int = 19
        PROBA_IN: float = 0.005
        RADIUS: float = 4.0
        SPEED: float = 1.0

    @dataclass
    class FRIENDLY_MISSILES():
        """Friendly missiles configuration."""

        NUMBER: int = 142
        EXPLOSION_RADIUS: float = 37.0
        EXPLOSION_SPEED: float = 0.5
        RADIUS: float = 7.0
        SPEED: float = 7.0

    @dataclass
    class OBSERVATION():
        """Observation configuration."""

        HEIGHT: float = 84
        RENDER_PROCESSED_HEIGHT: int = 250
        RENDER_PROCESSED_WIDTH: int = 250
        WIDTH: float = 84

    @dataclass
    class REWARD():
        """Reward configuration."""

        DESTROYED_CITY: float = -10.0
        DESTROYED_ENEMEY_MISSILES: float = 15.0
        FRIENDLY_MISSILE_LAUNCHED: float = -4.0

    @dataclass
    class TARGET():
        """Target configuration."""

        SIZE: int = 12
        VX: int = 4
        VY: int = 4
