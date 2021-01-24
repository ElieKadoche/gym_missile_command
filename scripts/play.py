"""Environment usage for a human."""

import sys

import gym
import pygame

# Create the environment
env = gym.make("gym_missile_command:missile-command-v0")

# Reset it
observation = env.reset()

# Initialize PyGame
env.render()

# While the episode is not finished
done = False
pause = False
while not done:

    for event in pygame.event.get():
        # Exit the environment
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Pause the environment
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pause = not pause

        keystate = pygame.key.get_pressed()
        action = 0

        # Target up
        if keystate[pygame.K_UP]:
            action = 1

        # Target down
        if keystate[pygame.K_DOWN]:
            action = 2

        # Target left
        if keystate[pygame.K_LEFT]:
            action = 3

        # Target right
        if keystate[pygame.K_RIGHT]:
            action = 4

        # Fire missile
        if keystate[pygame.K_SPACE]:
            action = 5

    if not pause:
        # One step forward
        observation, reward, done, _ = env.step(action)

        # Render (or not) the environment
        env.render()

# Close the environment
env.close()
