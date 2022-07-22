"""Random agent."""

import gym

import gym_missile_command

if __name__ == "__main__":
    # Environment configuration (see CONFIG.md)
    env_context = {"ENEMY_MISSILES.NUMBER": 42,
                   "FRIENDLY_MISSILES.EXPLOSION_RADIUS": 17}

    # Create the environment
    env = gym.make("missile-command-v0")

    # Reset it
    observation = env.reset(seed=None)

    # While the episode is not finished
    done = False
    while not done:

        # Select an action (here, a random one)
        action = env.action_space.sample()

        # One step forward
        observation, reward, done, _ = env.step(action)

        # Render (or not) the environment
        env.render()
