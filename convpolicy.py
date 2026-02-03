import torch as th
import torch.nn as nn
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Tuple, List, Union, Type, Dict
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.type_aliases import Schedule
from convrnn import ConvLSTM, ConvLSTMWrapper
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn


@dataclass
class ConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

    def make(self):
        return nn.Conv2d(
            self.in_channels, self.out_channels, 
            self.kernel_size, self.stride, self.padding
        )

class ConvFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, config: dict):
        # We will adaptively pool to fixed size feature maps
        self.adaptive_pool_size = config.get('adaptive_pool_size', (8, 8))
        self.conv_configs = config.get("conv_configs", None)

        assert self.conv_configs is not None, "conv_configs must be provided in extractor_config"

        super(ConvFeatureExtractor, self).__init__(observation_space, features_dim=1)  # Dummy features_dim

        in_channels = observation_space.shape[2]
        conv_layers = []
        for conv_conf in self.conv_configs:
            out_channels = conv_conf['out_channels']
            kernel_size = conv_conf.get('kernel_size', 3)
            stride = conv_conf.get('stride', 1)
            padding = conv_conf.get('padding', 1)

            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.adaptive_pool_size)

        self._features_dim = in_channels * self.adaptive_pool_size[0] * self.adaptive_pool_size[1]
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch_size, height, width, channels)
        x = observations.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = th.flatten(x, start_dim=1)
        return x

class CustomConvLSTMPolicy(RecurrentActorCriticPolicy):
    """
    Custom Policy that swaps the standard nn.LSTM for a ConvLSTM.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        # Specific args for your ConvLSTM
        conv_lstm_kwargs: Optional[Dict[str, Any]] = None,
        # Standard args
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 1. Initialize the Parent
        # We pass the standard arguments to let SB3 set up the MLP extractors and basic logic.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )

        self.conv_lstm_kwargs = conv_lstm_kwargs or {}
        
        c_hidden = self.conv_lstm_kwargs.get("hidden_size")
        c_shape = self.conv_lstm_kwargs.get("feature_shape")
        
        flat_hidden_state_size = c_hidden * c_shape[0] * c_shape[1]

        self.lstm_actor = ConvLSTMWrapper(
            **self.conv_lstm_kwargs
        )

        # Handle the Critic LSTM
        if self.enable_critic_lstm:
            self.lstm_critic = ConvLSTMWrapper(
                **self.conv_lstm_kwargs
            )
        elif not self.shared_lstm:
             pass

        self.lstm_hidden_state_shape = (n_lstm_layers, 1, flat_hidden_state_size)

        self.optimizer = self.optimizer_class(
            self.parameters(), 
            lr=lr_schedule(1), 
            **self.optimizer_kwargs
        )