"""Environment configuration."""

import numpy as np
from PIL import ImageColor

# General
# ------------------------------------------

# Environment height
HEIGHT = 600

# Environment width
WIDTH = 1100

# Numpy arrays type
DTYPE = np.float32

# Reward
# ------------------------------------------

# Reward for each remaining city
REWARD_REMAINING_CITY = 10.0

# Reward for each destroyed city
REWARD_DESTROYED_CITY = -10.0

# Reward for each remaining missile
REWARD_REMAINING_MISSILE = 5.0

# Reward for each destroyed enemy missiles
REWARD_DESTROYED_ENEMEY_MISSILES = 2.0

# Target
# ------------------------------------------

# Horizontal shifting of the target
TARGET_VX = 3

# Vertical shifting of the target
TARGET_VY = 3

# Target size (only for render)
TARGET_SIZE = 6

# Cities
# ------------------------------------------

# Number of cities to defend (even integer >= 2)
CITIES_NUMBER = 6

# Radius of a city object
CITY_RADIUS = 27.0

# Anti-missile battery
# ------------------------------------------

# Radius of the anti-missile battery object
BATTERY_RADIUS = 35.0

# Enemy missiles
# ------------------------------------------

# The total number of enemy missiles for 1 episode
ENEMY_MISSILES_NUMBER = 42

# The probability for an enemy missile to appear at a time step
ENEMY_MISSILE_PROBA_IN = 0.05

# Radius of an enemy missile object
ENEMY_MISSILE_RADIUS = 3.0

# Enemy missile speed
ENEMY_MISSILE_SPEED = 7.0

# Battery missiles
# ------------------------------------------

# Total number of available battery missiles
BATTERY_MISSILES_NUMBER = 30

# Radius of an battery missile object
BATTERY_MISSILE_RADIUS = 7.0

# Maximum explosion radius
BATTERY_MISSILE_EXPLOSION_RADIUS = 21.0

# Battery missile speed
BATTERY_MISSILE_SPEED = 34.0

# Colors
# ------------------------------------------

# Keep colors in hex to display them in Vim (vim-css-color) plugin
COLOR_BACKGROUND = ImageColor.getcolor("#000000", "RGB")
COLOR_BATTERY = ImageColor.getcolor("#ff0ff0", "RGB")
COLOR_BATTERY_MISSILE = ImageColor.getcolor("#00ff00", "RGB")
COLOR_CITY = ImageColor.getcolor("#0000ff", "RGB")
COLOR_ENEMY_MISSILE = ImageColor.getcolor("#ff0000", "RGB")
COLOR_EXPLOSION = ImageColor.getcolor("#ffff00", "RGB")
COLOR_TARGET = ImageColor.getcolor("#ffffff", "RGB")
