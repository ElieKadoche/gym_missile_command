# gym_missile_command

Open AI Gym environment of the [Missile Command Atari game](https://en.wikipedia.org/wiki/Missile_Command).

## What for?

To easily train and test different bots on the Missile Command Atari game.
Gym environments are well designed for Reinforcement Learning algorithms.
This environment does not reproduce an exact version of the Missile Command Atari game but a simplified one.

## Game

The player musts defend 6 cities from incoming enemy ballistic missiles.
To do so, he can fire missiles from an anti-missiles battery.
An episode ends when all enemy missiles or cities are destroyed.
There are no levels, all episodes have the same difficulty and enemy missiles do not split.

## Reward

The reward is one of the most decisive value for the success of an Reinforcement Learning algorithm.
The reward depends on several variables, each one contributing to a specific wanted skills of the engine.
- Number of cities remaining, to protect the cities.
- Number of missiles launched, ti minimize the use of missiles.
- How long before all cities are destroyed, to last as long as possible before all cities are destroyed.

## Installation

[Python](https://www.python.org/) 3.6+ is required.
The installation is done with the following commands.

```shell
git clone https://github.com/ElieKadoche/gym_missile_command.git
pip install -e ./gym_missile_command
```
## Configuration

A rich configuration of the environment can be edited in [./gym_missile_command/config.py](./gym_missile_command/config.py).

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
