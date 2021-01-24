"""Environment usage for a human."""

import gym

# Create the environment
env = gym.make("gym_missile_command:missile-command-v0")

# Reset it
observation = env.reset()

# While the episode is not finished
done = False
while not done:

    # Select an action (here, a random one)
    action = env.action_space.sample()

    # One step forward
    observation, reward, done, _ = env.step(action)

    # Render (or not) the environment
    env.render()
