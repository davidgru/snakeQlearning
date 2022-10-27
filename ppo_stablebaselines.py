from typing import Callable, Dict, List, Tuple, Type, Union, Optional
import gym
from gym import spaces
import numpy as np

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import AcutorCriticPolicy


from display import Display
from snakeMDP import Action, SnakeMDP
from plot import GameStats, Plot

class SnakeEnv(gym.Env):

    def __init__(self, height, width, fade=False, food_reward=1.0, death_reward=0.0, living_reward=0.0, plot=None):
        super(SnakeEnv, self).__init__()
        
        self.snake = SnakeMDP(height, width, food_reward, death_reward, living_reward, fade)
        self.plot = plot
        self.stats = GameStats()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-0.1, high=3.0, shape=(1, height, width), dtype=np.float32)

        self.state = None

    def step(self, action):
        action = Action(action)
        reward = self.snake.reward(self.state, action)
        score, self.state = self.snake.next(self.state, action)
        self.stats.push(score)
        if self.state:
            obs = SnakeEnv._to_image(np.expand_dims(self.state.world, axis=0))
        else:
            obs = None
            if self.plot:
                self.plot.push(self.stats)
        done = self.state is None
        return obs, reward, done, {}

    def reset(self):
        self.state = self.snake.sample_start_state()
        self.stats = GameStats()
        return SnakeEnv._to_image(np.expand_dims(self.state.world, axis=0))

    def render(self, mode="human"):
        if self.display:
            self.display.draw(self.state.world, "ppo_stable_baselines")

    def close(self):
        pass

    def score(self):
        return len(self.state.body) - 1 if self.state else 0

    def done(self):
        return self.state is None

    @staticmethod
    def _to_image(obs):
        return obs
        # return ((obs + 0.1) * 80).astype(np.uint8)

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        _, height, width = observation_space.shape

        FEATURE_MAPS = 32
        KERNEL_SIZE = 5

        PADDING = (KERNEL_SIZE - 1) // 2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, FEATURE_MAPS, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(FEATURE_MAPS, FEATURE_MAPS, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(FEATURE_MAPS, FEATURE_MAPS, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(height * width * FEATURE_MAPS, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


HEIGHT = 10
WIDTH = 10
FADE = True

plot = Plot(1000)

from stable_baselines3.common.env_checker import check_env

# env = SnakeEnv(HEIGHT, WIDTH)
# It will check your custom environment and output additional warnings if needed
# check_env(env)

env = make_vec_env(lambda: SnakeEnv(HEIGHT, WIDTH, FADE, plot=plot), n_envs=16)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256)
)
model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=True)
model.learn(1e100)
