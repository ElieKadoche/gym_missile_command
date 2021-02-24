"""Actor-critic coded from scratch."""

import argparse
import os.path
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import torch
from torch import nn as nn
from torch import optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F


class Model(nn.Module):
    """Actor and critic model for CartPole."""

    def __init__(self):
        """Initialize model."""
        super(Model, self).__init__()
        self.fc0 = nn.Linear(4, 128)

        # Actor's layer (policy)
        self.action_head = nn.Linear(128, 2)

        # Critic's layer (value)
        self.value_head = nn.Linear(128, 1)

    def _preprocessor(self, observation):
        """Preprocessor function.

        Args:
            observation (numpy.array): environment observations.

        Returns:
            x (torch.tensor): preprocessed observations.
        """
        # Add batch dimension
        x = np.expand_dims(observation, 0)

        # Transform to torch.tensor
        x = torch.from_numpy(x).float()

        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x (numpy.array): environment observations.

        Returns:
            actions_prob (torch.tensor): list with the probability of each
                action over the action space.

            state_values (torch.tensor): the value from the current state.
        """
        # Preprocessor
        x = self._preprocessor(x)

        # Input layer
        x = F.relu(self.fc0(x))

        # Actor head
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # Critic head
        state_values = self.value_head(x)

        return action_prob, state_values


def create_model(args):
    """Create model.

    Args:
        args (argparse.Namespace): argparse arguments.

    Returns:
        model (PyTorch model): actor-critic neural network.
    """
    model = Model()

    # To optionally load a checkpoint
    if os.path.isfile(args.checkpoint):
        model = torch.load(args.checkpoint)

    # Print model
    if args.verbose > 0:
        print(model)

    return model


def test(args):
    """Test agent.

    Args:
        args (argparse.Namespace): argparse arguments.
    """
    # Create the model
    model = create_model(args)
    model.eval()

    # Create the environment
    env = gym.make("CartPole-v1")

    while True:

        total_reward = 0

        # Reset it
        observation = env.reset()

        # While the episode is not finished
        done = False
        while not done:

            # Select action
            actor, _ = model(observation)
            action = torch.argmax(actor).item()

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
    # Creates model
    model = create_model(args)
    model.train()

    # Create the environment
    env = gym.make("CartPole-v1")

    gamma = 0.99  # Discount factor
    horizon = 10000  # Max environment length

    eps = np.finfo(np.float32).eps.item()
    optimizer = optim.Adam(model.parameters(), lr=3e-2)
    saved_action = namedtuple("SavedAction", ["log_prob", "value"])

    # Run infinitely many episodes
    for i_episode in count(1):

        # Reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # Action and reward buffer
        saved_actions = []
        rewards = []

        # Prevent infinite loop
        for t in range(1, horizon):

            # Select action from policy
            # ------------------------------------------

            # Model prediction
            probs, state_value = model(state)

            # Create a categorical distribution over the list of probabilities
            # of actions
            m = Categorical(probs)

            # And sample an action using the distribution
            action = m.sample()

            # Save to action buffer
            # m.log_prob(a) = ln(model_out_proba_of_a)
            saved_actions.append(saved_action(m.log_prob(action), state_value))

            # Take the action
            state, reward, done, _ = env.step(action.item())

            # Save reward
            rewards.append(reward)
            ep_reward += reward

            # End episode
            if done:
                break

        # Perform backpropagation
        # ------------------------------------------

        R = 0
        policy_losses = []  # List to save actor (policy) loss
        value_losses = []  # List to save critic (value) loss
        returns = []  # List to save the true values

        # Calculate the true value using rewards returned from the environment
        for r in rewards[::-1]:
            # Calculate the discounted value
            R = r + gamma * R
            returns.insert(0, R)

        # Normalize
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            # Compute advantage
            advantage = R - value.item()

            # Calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # Calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(
                value, torch.tensor([R], dtype=torch.float32)))

        # Reset gradients
        optimizer.zero_grad()

        # Sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()

        # Perform backpropagation
        loss.backward()
        optimizer.step()

        # Log results
        if i_episode % args.saving_frequency == 0:
            # Save neural net
            torch.save(model, args.checkpoint)

            # Print results
            print("Episode {}\tLast reward: {: .2f}.".format(
                i_episode, ep_reward))


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="Custom actor-critic.")
    parser.add_argument("--checkpoint",
                        default="actor_critic_model.pt",
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
