"""Random agent."""

import gymnasium as gym

import gym_missile_command

if __name__ == "__main__":
    # Environment configuration (see CONFIG.md)
    env_context = {"ENEMY_MISSILES.NUMBER": 42,
                   "FRIENDLY_MISSILES.EXPLOSION_RADIUS": 17}

    # Create the environment
    env = gym.make("missile-command-v0", env_context=env_context)

    # Reset it
    observation, _ = env.reset(seed=None)

    # While the episode is not finished
    terminated = False
    while not terminated:

        # Select an action (here, a random one)
        action = env.action_space.sample()

        # One step forward
        observation, reward, terminated, _, _ = env.step(action)

        # Render (or not) the environment
        env.render()
