"""Environment configuration."""

import numpy as np

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
TARGET_VX = 6

# Vertical shifting of the target
TARGET_VY = 6

# Target size (only for render)
TARGET_SIZE = 12

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
ENEMY_MISSILE_PROBA_IN = 0.005

# Radius of an enemy missile object
ENEMY_MISSILE_RADIUS = 4.0

# Enemy missile speed
ENEMY_MISSILE_SPEED = 1.0

# Battery missiles
# ------------------------------------------

# Total number of available battery missiles
BATTERY_MISSILES_NUMBER = 142

# Radius of an battery missile object
BATTERY_MISSILE_RADIUS = 7.0

# Maximum explosion radius
BATTERY_MISSILE_EXPLOSION_RADIUS = 37.0

# Sped of the explosion
BATTERY_MISSILE_EXPLOSION_SPEED = 0.5

# Battery missile speed
BATTERY_MISSILE_SPEED = 7.0

# Colors
# ------------------------------------------

# Keep colors in hex to display them in Vim (vim-css-color plugin)
COLOR_BACKGROUND = (0, 0, 0)         # #000000
COLOR_BATTERY = (255, 15, 240)       # #ff0ff0
COLOR_BATTERY_MISSILE = (0, 255, 0)  # #00ff00
COLOR_CITY = (0, 0, 255)             # #0000ff
COLOR_ENEMY_MISSILE = (255, 0, 0)    # #ff0000
COLOR_EXPLOSION = (255, 255, 0)      # #ffff00
COLOR_TARGET = (255, 255, 255)       # #ffffff
