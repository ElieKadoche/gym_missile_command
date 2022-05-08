"""Environment usage for a human."""

import sys

import gym
import pygame

# Number of time step to wait before the user can send a new missile
STOP_FIRE_WAIT = 57

if __name__ == "__main__":
    # Create the environment
    env = gym.make("missile-command-v0")

    # Reset it
    observation = env.reset(seed=None)

    # Initialize PyGame
    env.render()

    # The user can naturally pause the environment
    pause = False

    # Prevent multiple fire when the user chooses action 5
    stop_fire = 0

    # While the episode is not finished
    done = False
    while not done:

        # Check if user exits
        for event in pygame.event.get():

            # Exit the environment
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Pause the environment
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pause = not pause

        # Get keys pressed by user
        keystate = pygame.key.get_pressed()

        # Default action is 0
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
            if stop_fire == 0:
                stop_fire = STOP_FIRE_WAIT
                action = 5

        # Decrease stop_fire
        if stop_fire > 0:
            stop_fire -= 1

        if not pause:
            # One step forward
            observation, reward, done, _ = env.step(action)

            # Render (or not) the environment
            env.render()

    # Close the environment
    env.close()
