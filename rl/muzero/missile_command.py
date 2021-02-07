"""MuZero configuration for Missile Command.

This is the game file for the Missile Command environment. It needs to be
placed in the games/ folder. To use it and train a MuZero agent, please refer
to the following project.

https://github.com/werner-duvaud/muzero-general.git
"""

import datetime
import os

import gym
import numpy
import torch

from .abstract_game import AbstractGame

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError('\nPlease run "pip install gym[atari]"')


class MuZeroConfig:
    """MuZero Missile Command configuration."""

    def __init__(self):
        """Initialize configuration.

        See https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization.
        """  # noqa
        # Seed for numpy, torch and the game
        self.seed = 0

        # Fix the maximum number of GPUs to use. It's usually faster to use a
        # single GPU (set it to 1) if it has enough memory. None will use every
        # GPUs available
        self.max_num_gpus = None

        # Game
        # ------------------------------------------

        # Dimensions of the game observation, must be 3D (channel, height,
        # width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (3, 82, 82)

        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(6))

        # List of players. You should only edit the length
        self.players = list(range(1))

        # Number of previous observations and previous actions to add to the
        # current observation
        self.stacked_observations = 3

        # Evaluate
        # ------------------------------------------

        # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays
        # second)
        self.muzero_player = 0

        # Hard coded agent that MuZero faces to assess his progress in
        # multiplayer games. It doesn't influence training. None, "random" or
        # "expert" if implemented in the Game class
        self.opponent = None

        # Self-Play
        # ------------------------------------------

        # Number of simultaneous threads/workers self-playing to feed the
        # replay buffer
        self.num_workers = 17

        # Runs selfplay on GPU
        self.selfplay_on_gpu = False

        # Maximum number of moves if game is not finished before
        self.max_moves = 1150

        # Number of future moves self-simulated
        self.num_simulations = 25

        # Chronological discount of the reward
        self.discount = 0.997

        # Number of moves before dropping the temperature given by
        # visit_softmax_temperature_fn to 0 (ie selecting the best action). If
        # None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        # ------------------------------------------

        # "resnet" / "fullyconnected"
        self.network = "resnet"

        # Value and reward are scaled (with almost sqrt) and encoded on a
        # vector with a range of -support_size to support_size. Choose it so
        # that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 15

        # Residual Network
        # ------------------------------------------

        # Downsample observations before representation network, False / "CNN"
        # (lighter) / "resnet" (See paper appendix Network Architecture)
        self.downsample = "resnet"

        # Number of blocks in the ResNet
        self.blocks = 8

        # Number of channels in the ResNet
        self.channels = 64

        # Number of channels in reward head
        self.reduced_channels_reward = 64

        # Number of channels in value head
        self.reduced_channels_value = 64

        # Number of channels in policy head
        self.reduced_channels_policy = 64

        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = [64, 64]

        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = [64, 64]

        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = [64, 64]

        # Fully Connected Network
        # ------------------------------------------

        self.encoding_size = 10

        # Define the hidden layers in the representation network
        self.fc_representation_layers = []

        # Define the hidden layers in the dynamics network
        self.fc_dynamics_layers = [16]

        # Define the hidden layers in the reward network
        self.fc_reward_layers = [16]

        # Define the hidden layers in the value network
        self.fc_value_layers = []

        # Define the hidden layers in the policy network
        self.fc_policy_layers = []

        # Training
        # ------------------------------------------

        # Path to store the model weights and TensorBoard logs
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../results", os.path.basename(__file__)[:-3],
            datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
        )

        # Save the checkpoint in results_path as model.checkpoint
        self.save_model = True

        # Total number of training steps (ie weights update according to a
        # batch)
        self.training_steps = int(1000e3)

        # Number of parts of games to train on at each training step
        self.batch_size = 128

        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = int(1e3)

        # Scale the value loss to avoid overfitting of the value function,
        # paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25

        # Train on GPU if available
        self.train_on_gpu = torch.cuda.is_available()

        # "Adam" or "SGD". Paper uses SGD
        self.optimizer = "SGD"

        # L2 weights regularization
        self.weight_decay = 1e-4

        # Used only if optimizer is SGD
        self.momentum = 0.9

        # Exponential learning rate schedule

        # Initial learning rate
        self.lr_init = 0.05

        # Set it to 1 to use a constant learning rate
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 350e3

        # Replay Buffer
        # ------------------------------------------

        # Number of self-play games to keep in the replay buffer
        self.replay_buffer_size = int(1e6)

        # Number of game moves to keep for every batch element
        self.num_unroll_steps = 5

        # Number of steps in the future to take into account for calculating
        # the target value
        self.td_steps = 10

        # Prioritized Replay (See paper appendix Training), select in priority
        # the elements in the replay buffer which are unexpected for the
        # network
        self.PER = True

        # How much prioritization is used, 0 corresponding to the uniform case,
        # paper suggests 1
        self.PER_alpha = 1

        # Reanalyze (See paper appendix Reanalyse). Use the last model to
        # provide a fresher, stable n-step value (see paper appendix Reanalyze)
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        # ------------------------------------------

        # Number of seconds to wait after each played game
        self.self_play_delay = 0

        # Number of seconds to wait after each training step
        self.training_delay = 0

        # Desired training steps per self played step ratio. Equivalent to a
        # synchronous version, training can take much longer. Set it to None to
        # disable it
        self.ratio = None

    def visit_softmax_temperature_fn(self, trained_steps):
        """Parameter to alter the visit count distribution.

        Parameter to alter the visit count distribution to ensure that the
        action selection becomes greedier as training progresses. The smaller
        it is, the more likely the best action (ie with the highest visit
        count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """Game wrapper."""

    def __init__(self, seed=None):
        """Initialize game."""
        env_config = {
            "ENEMY_MISSILES.NUMBER": 7,
            "ENEMY_MISSILES.PROBA_IN": 0.07,
            "REWARD.DESTROYED_CITY": -16.67,
            "REWARD.DESTROYED_ENEMEY_MISSILES": 28.57,
            "REWARD.FRIENDLY_MISSILE_LAUNCHED": -0.57,
        }
        self.env = gym.make("gym_missile_command:missile-command-v0",
                            custom_config=env_config)

    def step(self, action):
        """Apply action to the game.

        Args:
            action (int): action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has
            ended.
        """
        observation, reward, done, _ = self.env.step(action)
        observation = cv2.resize(
            observation, (82, 82), interpolation=cv2.INTER_AREA)
        observation = numpy.asarray(observation, dtype="float32") / 255.0
        observation = numpy.moveaxis(observation, -1, 0)
        return observation, reward, done

    def legal_actions(self):
        """Return legal actions.

        Should return the legal actions at each turn, if it is not available,
        it can return the whole action space. At each turn, the game have to be
        able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is
        to define the legal actions equal to the action space but to return a
        negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(6))

    def reset(self):
        """Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation = self.env.reset()
        observation = cv2.resize(
            observation, (82, 82), interpolation=cv2.INTER_AREA)
        observation = numpy.asarray(observation, dtype="float32") / 255.0
        observation = numpy.moveaxis(observation, -1, 0)
        return observation

    def close(self):
        """Properly close the game."""
        self.env.close()

    def render(self):
        """Display the game observation."""
        self.env.render()
