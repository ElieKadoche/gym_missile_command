"""DQN with RLlib.

Checkpoints and results are saved in ~/ray_results. Inside the folder of your
choice, you can execute "tensorboard --logdir ." to monitor results.
"""

import argparse

import gym
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print


class MissileCommand(gym.Env):
    """MissileCommand environment.

    This environment allows us to use a custom configuration for the Missile
    Command OpenAI Gym environment.
    """
    ENV_CONFIG = {
        "ENEMY_MISSILES.NUMBER": 7,
        "ENEMY_MISSILES.PROBA_IN": 0.07,
        "REWARD.DESTROYED_CITY": -16.67,
        "REWARD.DESTROYED_ENEMEY_MISSILES": 28.57,
        "REWARD.FRIENDLY_MISSILE_LAUNCHED": -0.57,
    }

    def __init__(self, env_config):
        """Initialize MissileCommand environment."""
        self._env = gym.make("gym_missile_command:missile-command-v0",
                             custom_config=self.ENV_CONFIG)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        """Reset the environment.

        For more documentation, see the MissileCommandEnv class.
        """
        return self._env.reset()

    def step(self, action):
        """Go from current step to next one.

        For more documentation, see the MissileCommandEnv class.
        """
        return self._env.step(action)

    def render(self):
        """Render the environment."""
        self._env.render()

    def close(self):
        """Close the environment."""
        self._env.close()


def create_agent(args):
    """Create DQN agent.

    Args:
        args (argparse.Namespace): argparse arguments.

    Returns:
        agent (ray.rllib.agents.trainer_template.DQN): DQN agent.
    """
    # Custom configuration
    config = dqn.DEFAULT_CONFIG.copy()
    config["framework"] = "torch"
    config["num_gpus"] = 1
    config["num_workers"] = 19

    # Agent creation
    agent = dqn.DQNTrainer(env=MissileCommand, config=config)

    # To optionally load a checkpoint
    if args.checkpoint:
        agent.restore(args.checkpoint)

    return agent


def test(args):
    """Test agent.

    Args:
        args (argparse.Namespace): argparse arguments.
    """
    # Initialize Ray
    ray.init()

    # Create the agent
    agent = create_agent(args)

    # Create the environment
    env = gym.make("gym_missile_command:missile-command-v0",
                   custom_config=MissileCommand.ENV_CONFIG)

    # Reset it
    observation = env.reset()

    # While the episode is not finished
    done = False
    while not done:

        # Agent computes action
        action = agent.compute_action(observation)

        # One step forward
        observation, reward, done, _ = env.step(action)

        # Render (or not) the environment
        env.render()


def train(args):
    """Train agent.

    Args:
        args (argparse.Namespace): argparse arguments.
    """
    # Initialize Ray
    ray.init()

    # Create the agent
    agent = create_agent(args)

    # Launch training
    for i in range(args.iter):
        result = agent.train()
        # print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="DQN agent training and testing with RLlib.")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="Checkpoint path (optional).")
    subparsers = parser.add_subparsers()

    # Train parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--iter", type=int, default=100000000, help="Training iteration number.")
    train_parser.set_defaults(func=train)

    # Test parser
    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(func=test)

    # Launch script
    args = parser.parse_args()
    args.func(args)
