# gym_missile_command

Open AI Gym environment of the [Missile Command Atari game](https://en.wikipedia.org/wiki/Missile_Command).

## What for?

To easily train and test different bots on the Missile Command Atari game.
Gym environments are well designed for Reinforcement Learning algorithms.
This environment does not reproduce an exact version of the Missile Command Atari game but a simplified one.

## The game

The player musts defend 6 cities from incoming enemy ballistic missiles.
To do so, he can fire missiles from 3 anti-missiles batteries.
An episode ends when all ennemy missiles or cities are destroyed.
Reward depends of the number of missiles used and the number of cities remaining.
There are no levels, all episodes have the same difficulty.

## Installation

[Python](https://www.python.org/) 3.6+ is required.
The installation is done with the following commands.

```shell
git clone https://github.com/ElieKadoche/gym_missile_command.git
pip install ./gym_missile_command
```

## Usage

```python
import gym
env = gym.make("gym_missile_command:missile-command-v0")
observation = env.reset()
done = False
while not done:
    observation, reward, done, _ = env.step(action)
    env.render()
```

## Authors

- Elie KADOCHE.
