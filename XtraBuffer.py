"""
This custom rollout buffer is very simple, it stores all of what the normal buffer does and a single prefix for the whole size of the buffer
each env present in the VecEnv will show the same level always.
"""
from typing import Optional, Union, Tuple, NamedTuple, Generator, Type, Set

import warnings
import numpy as np
import torch as th
import torch.nn as nn

from gymnasium import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback, Any

from dataclasses import dataclass

@dataclass
class RlSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    prefixes: th.Tensor

@dataclass
class SupervisionSamples(NamedTuple):
    inputs: th.Tensor # shape = (B, num_envs, T, *obs_shape)
    mask: th.Tensor # shape = inputs.shape
    targets: th.Tensor # shape = (num_envs, prefix_shape) -> (B, ...)


class SinglePrefixBuffer(RolloutBuffer):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    optimizer_class: Type[th.optim.SGD]
    prefixes = Type[nn.Parameter]
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            prefix_shape: Tuple[int, int],
            prefix_lr: int,
            prefix_optimizer: Type[nn.Module] = th.optim.SGD,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        if prefix_optimizer != th.optim.SGD:
            # Warn the user
            warnings.warn("Using optimizers other than SGD may lead to unexpected behavior.\
                           Ensure that optimizer state is properly managed across iterations.\
                           And coinsider sticking to optimizers without adaptive states like\
                           momentum or Adam.")
            
        self.prefix_shape = prefix_shape
        self.prefix_learning_rate = prefix_lr
        self.optimizer_class = prefix_optimizer
        self.prefixes = None
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs
        )
    
    def simple_reset(self) -> None:
        super().reset()
    
    def hard_reset(self) -> None:
        super().reset()
        self.prefixes = self.prefixes.detach().clone()
    
    def set_prefix(self, prefixes) -> Tuple[Type[nn.Module], Type[nn.Module]]:
        """
        No longer using Ids to verify the position of a prefix.
        """
        self.prefixes = nn.Parameter(prefixes)
        optimizer = self.optimizer_class(self.prefixes, lr=self.prefix_learning_rate)
        return self.prefixes, optimizer #Should be deleated later