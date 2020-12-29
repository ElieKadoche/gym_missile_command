# gym_missile_command

Open AI Gym environment of the Missile Command Atari game.

## Prerequisites

[Python](https://www.python.org/) 3.6+ installed on your system.

## Installation

To install, execute the following commands.

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
