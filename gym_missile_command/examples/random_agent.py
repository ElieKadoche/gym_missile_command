"""Environment usage for a machine."""

import gym

if __name__ == "__main__":
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
