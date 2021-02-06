"""R2D2 with Acme."""

import argparse
import functools

import acme
import gym
import numpy as np
import tensorflow as tf
from acme import wrappers
from acme.agents.tf import r2d2
from acme.tf import networks

# Without these lines, the script returns an error in Docker
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Environment configuration
ENV_CONFIG = {
    "ENEMY_MISSILES.NUMBER": 7,
    "ENEMY_MISSILES.PROBA_IN": 0.07,
    "REWARD.DESTROYED_CITY": -16.67,
    "REWARD.DESTROYED_ENEMEY_MISSILES": 28.57,
    "REWARD.FRIENDLY_MISSILE_LAUNCHED": -0.57,
}

# Maximum episode length (to determine the right value, simulate some random
# episodes with the custom config and print env.timestep)
MAX_EPISODE_LEN = 1150


class DummyALE():
    """Dummy ALE class.

    OpenAI Gym Atari environments are built from ALE. Acme uses some of the ALE
    functions, so we implement it here, in a dummy class, to easily make
    Missile Command compatible.
    """

    def lives(self):
        """Get lives.

        Returns:
            lives (int): 0.
        """
        return 0


def make_environmment():
    """Make environment.

    Returns:
        env (acme.wrappers.single_precision.SinglePrecisionWrapper).
    """
    # Create the environment
    environment = gym.make("gym_missile_command:missile-command-v0",
                           custom_config=ENV_CONFIG)

    # Add the necessary ALE function
    environment.ale = DummyALE()

    # Acme processing
    environment = wrappers.wrap_all(environment, [
        wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariWrapper,
            to_float=True,
            max_episode_len=MAX_EPISODE_LEN,
            zero_discount_on_life_loss=True,
        ),
        wrappers.SinglePrecisionWrapper,
        wrappers.ObservationActionRewardWrapper,
    ])

    # wrappers.ObservationActionRewardWrapper adds previous actions and rewards
    # to observation

    return environment


def get_env_agent():
    """Creates env and agent.

    Returns:
        env_acme (acme.wrappers.observation_action_reward.
            ObservationActionRewardWrappe).

        agent (acme.agents.tf.r2d2.agent.R2D2).
    """
    # Get environment
    env_acme = make_environmment()
    env_spec = acme.make_environment_spec(env_acme)

    # Create agent and network
    network = networks.R2D2AtariNetwork(env_spec.actions.num_values)
    agent = r2d2.R2D2(environment_spec=env_spec,
                      network=network,
                      burn_in_length=2,
                      trace_length=6,
                      replay_period=4,)

    return env_acme, agent


def test(args):
    """Test agent.

    Args:
        args (argparse.Namespace): argparse arguments.
    """
    pass


def train(args):
    """Train agent.

    Args:
        args (argparse.Namespace): argparse arguments.
    """
    # Get env and agent
    env_acme, agent = get_env_agent()

    # Launch training
    loop = acme.EnvironmentLoop(env_acme, agent)
    loop.run(args.episodes)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="R2D2 agent training and testing with Acme.")
    subparsers = parser.add_subparsers()

    # Train parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--episodes",
                              type=int,
                              default=100000000,
                              help="Number of episodes to train for.")
    train_parser.set_defaults(func=train)

    # Test parser
    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(func=test)

    # Launch script
    args = parser.parse_args()
    args.func(args)
