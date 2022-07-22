# CONFIG

Below are all the editable variables.
It is highly recommended to read it carefully, as understanding each variable leads to a better understanding of the overall environment.
Default parameters are in [./gym_missile_command/configuration/config.py](./gym_missile_command/configuration/config.py).

```python
env_context = {
    "EPISODE.FPS": 144,
    "EPISODE.HEIGHT": 466,
    "EPISODE.WIDTH": 966,
    "BATTERY.RADIUS": 37.0,
    "CITIES.NUMBER": 6,
    "CITIES.RADIUS": 24.0,
    "COLORS.BACKGROUND": (0, 0, 0),
    "COLORS.BATTERY": (255, 255, 255),
    "COLORS.CITY": (0, 0, 255),
    "COLORS.ENEMY_MISSILE": (255, 0, 0),
    "COLORS.EXPLOSION": (255, 255, 0),
    "COLORS.FRIENDLY_MISSILE": (0, 255, 0),
    "COLORS.TARGET": (255, 255, 255),
    "ENEMY_MISSILES.NUMBER": 19,
    "ENEMY_MISSILES.PROBA_IN": 0.005,
    "ENEMY_MISSILES.RADIUS": 4.0,
    "ENEMY_MISSILES.SPEED": 1.0,
    "FRIENDLY_MISSILES.NUMBER": 142,
    "FRIENDLY_MISSILES.EXPLOSION_RADIUS": 37.0,
    "FRIENDLY_MISSILES.EXPLOSION_SPEED": 0.5,
    "FRIENDLY_MISSILES.RADIUS": 7.0,
    "FRIENDLY_MISSILES.SPEED": 7.0,
    "OBSERVATION.HEIGHT": 84,
    "OBSERVATION.RENDER_PROCESSED_HEIGHT": 250,
    "OBSERVATION.RENDER_PROCESSED_WIDTH": 250,
    "OBSERVATION.WIDTH": 84,
    "REWARD.DESTROYED_CITY": -10.0,
    "REWARD.DESTROYED_ENEMEY_MISSILES": 15.0,
    "REWARD.FRIENDLY_MISSILE_LAUNCHED": -4.0,
    "TARGET.SIZE": 12,
    "TARGET.VX": 4,
    "TARGET.VY": 4,
}
```

### General

`"EPISODE.FPS"`
Frame per second for the rendering function.

`"EPISODE.HEIGHT"`
Height of the rendering window.

`"EPISODE.WIDTH"`
Width of the rendering window.

### Battery

`"BATTERY.RADIUS"`
Radius of the anti-missile battery object.

### Cities

`"CITIES.NUMBER"`
Number of cities to defend (even integer >= 2).

`"CITIES.RADIUS"`
Radius of a city object.

### Colors

`"COLORS.BACKGROUND"`
`"COLORS.BATTERY"`
`"COLORS.CITY"`
`"COLORS.ENEMY_MISSILE"`
`"COLORS.EXPLOSION"`
`"COLORS.FRIENDLY_MISSILE"`
`"COLORS.TARGET"`
Colors of each object for the rendering function.

### Enemy missiles

`"ENEMY_MISSILES.NUMBER"`
Total number of enemy missiles for 1 episode.

`"ENEMY_MISSILES.PROBA_IN"`
Probability for an enemy missile to appear at a time step.

`"ENEMY_MISSILES.RADIUS"`
Radius of an enemy missile object.

`"ENEMY_MISSILES.SPEED"`
Enemy missile speed.

### Friendly missiles

`"FRIENDLY_MISSILES.NUMBER"`
Total number of available friendly missiles.

`"FRIENDLY_MISSILES.EXPLOSION_RADIUS"`
Maximum explosion radius.

`"FRIENDLY_MISSILES.EXPLOSION_SPEED"`
Speed of the explosion.

`"FRIENDLY_MISSILES.RADIUS"`
Radius of a friendly missile object.

`"FRIENDLY_MISSILES.SPEED"`
Friendly missile speed.

### Observation

An agent takes as input the screen pixels.
The resolution of the environment can be quite big: the computational cost to train an agent can then be high.
For an agent to well perform on the Missile Command Atari game, a smaller resized version of the observation can be enough.
So the environment returns a resized version of the environment observation.
If you wish to not resize the observation, fix these variables to the same values as CONFIG.EPISODE.HEIGHT and CONFIG.EPISODE.WIDTH.

`"OBSERVATION.HEIGHT"`
Observation height.

`"OBSERVATION.RENDER_PROCESSED_HEIGHT"`
Render window height of the processed observation.

`"OBSERVATION.RENDER_PROCESSED_WIDTH"`
Render window width of the processed observation.

`"OBSERVATION.WIDTH"`
Observation width.

### Reward

`"REWARD.DESTROYED_CITY"`
Reward for each destroyed city.

`"REWARD.DESTROYED_ENEMEY_MISSILES"`
Reward for each destroyed missile.

`"REWARD.FRIENDLY_MISSILE_LAUNCHED"`
Reward for each friendly missile launched.

### Target

`"TARGET.SIZE"`
Target size (only for the rendering function).

`"TARGET.VX"`
Horizontal shifting of the target.

`"TARGET.VY"`
Vertical shifting of the target.
