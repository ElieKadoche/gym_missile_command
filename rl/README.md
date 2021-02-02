# Reinforcement Learning

Here are some scripts to train and test Reinforcement Learning algorithms on the Missile Command environment.

## Installation

[Python](https://www.python.org/) 3.6+ is required.
You need to have the OpenAI Gym Missile Command environment installed.
To install dependencies, execute the following command.

```shell
pip install -r requirements.txt
```
## Usage

Use the following commands to execute the XXX algorithm.
Careful, each algorithm can have specific arguments.

```shell
python ./scripts/XXX.py --help  # Get help
python ./scripts/XXX.py train  # Launch training
python ./scripts/XXX.py test  # Launch testing
```

## Algorithms

The [RLlib](https://docs.ray.io/en/master/rllib.html) library from the [Ray](https://github.com/ray-project/ray.git) package is used.
Current implemented algorithms are:

- DQN.

There is a special case for MuZero, in the [./scripts/muzero/](./scripts/muzero/) folder.
