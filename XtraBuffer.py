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

class SinglePrefixBuffer(RolloutBuffer):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
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
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        if prefix_optimizer != th.optim.SGD:
            # Warn the user
            warnings.warn("Using optimizers other than SGD may lead to unexpected behavior.\
                           Ensure that optimizer state is properly managed across iterations.\
                           And coinsider sticking to optimizers without adaptive states like\
                           momentum or Adam.")
            
        self.prefix_shape = prefix_shape
        