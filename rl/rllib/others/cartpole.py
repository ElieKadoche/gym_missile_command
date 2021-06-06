"""Agent training and testing with RLlib.

Checkpoints and results are saved in ~/ray_results. Inside the folder of your
choice, you can execute "tensorboard --logdir ." to monitor results.

Use `python cartpole.py --help` to get info.
"""

import argparse

import gym
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print


class GymEnv(gym.Env):
    """OpenAI Gym environment."""

    def __init__(self, env_config):
        """Initialize environment."""
        self._env = gym.make("CartPole-v1")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        """Reset the environment."""
        return self._env.reset()

    def step(self, action):
        """Go from current step to next one."""
        return self._env.step(action)

    def render(self):
        """Render the environment."""
        self._env.render()

    def close(self):
        """Close the environment."""
        self._env.close()


def create_agent(args):
    """Create XXX agent.

    Args:
        args (argparse.Namespace): argparse arguments.

    Returns:
        agent (ray.rllib.agents.trainer_template.XXX): XXX agent.
    """
    # A3C
    # ------------------------------------------

    if args.agent == "A2C":
        # Custom configuration
        config = a3c.DEFAULT_CONFIG.copy()
        config["framework"] = "torch"
        config["lr"] = 5e-4
        config["num_gpus"] = 1
        config["num_workers"] = 1
        config["train_batch_size"] = 128
        config["use_critic"] = True
        config["use_gae"] = False

        # Custom model
        config["model"]["fcnet_activation"] = "tanh"
        config["model"]["fcnet_hiddens"] = [64, 64, 64]

        # Agent creation
        agent = a3c.A2CTrainer(env=GymEnv, config=config)

    # DQN
    # ------------------------------------------

    elif args.agent == "DQN":
        # Custom configuration
        config = dqn.DEFAULT_CONFIG.copy()
        config["double_q"] = False
        config["dueling"] = False
        config["framework"] = "torch"
        config["lr"] = 5e-3
        config["num_gpus"] = 1
        config["num_workers"] = 1
        config["train_batch_size"] = 128

        # Custom model
        config["model"]["fcnet_activation"] = "tanh"
        config["model"]["fcnet_hiddens"] = [128, 128, 128]

        # Agent creation
        agent = dqn.DQNTrainer(env=GymEnv, config=config)

    # To optionally load a checkpoint
    if args.checkpoint:
        agent.restore(args.checkpoint)

    # Print model
    if args.verbose > 0:
        model = agent.get_policy().model
        if config["framework"] == "tf":
            print(type(model.base_model.summary()))
        elif config["framework"] == "torch":
            print(model)

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
    env = gym.make("CartPole-v1")

    while True:

        total_reward = 0

        # Reset it
        observation = env.reset()

        # While the episode is not finished
        done = False
        while not done:

            # Agent computes action
            action = agent.compute_action(observation, explore=False)

            # One step forward
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            # Render the environment
            if args.verbose > 1:
                env.render()

        print(total_reward)


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
    for i in range(args.training_steps):
        result = agent.train()

        # Print results
        if args.verbose > 0:
            print(pretty_print(result))

        # Save model
        if i % args.saving_frequency == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="Agent training and testing with RLlib.")
    parser.add_argument("--agent",
                        choices=["DQN", "A2C"],
                        default="A2C",
                        help="Agent name.",
                        required=False)
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Checkpoint path (optional).",
                        required=False,
                        type=str)
    parser.add_argument("-v",
                        "--verbose",
                        choices=range(3),
                        default=1,
                        help="Verbose mode, 0 (nothing) and 2 (everything).",
                        required=False,
                        type=int)
    subparsers = parser.add_subparsers()

    # Train parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--saving-frequency",
                              type=int,
                              default=1,
                              help="The model is saved every --saving-steps.",
                              required=False)
    train_parser.add_argument("--training-steps",
                              type=int,
                              default=1000000,
                              help="Training iteration number.",
                              required=False)
    train_parser.set_defaults(func=train)

    # Test parser
    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(func=test)

    # Launch script
    args = parser.parse_args()
    args.func(args)
