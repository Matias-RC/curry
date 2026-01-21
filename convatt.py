import torch
import torch.nn as nn
import torch.nn.functional as F

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from dataclasses import dataclass

@dataclass
class ConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

    def make(self):
        return nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

@dataclass
class ConvAttConfig:
    working_channels: int
    qk_dim: int
    kernel_size: int

    def make(self):
        # make query, key, value conv layers and mlp for query and key.
        padding = self.kernel_size // 2
        heads = nn.ModuleList([ConvConfig(
            in_channels=self.working_channels,
            out_channels=self.qk_dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding
        ).make() for _ in range(3)])
        q_mlp = nn.Sequential(
            nn.Linear(self.working_channels, self.qk_dim*4),
            nn.ReLU(),
            nn.Linear(self.qk_dim*4, self.qk_dim)
        )
        k_mlp = nn.Sequential(
            nn.Linear(self.working_channels, self.qk_dim*4),
            nn.ReLU(),
            nn.Linear(self.qk_dim*4, self.qk_dim)
        )
        return heads, q_mlp, k_mlp

class ConvAttPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(ConvAttPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        # Currently ConvAtt layers are not supported.

