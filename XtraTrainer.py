"""
Trainer, works like a double loop.
for i outer_loops:
    for j in inner loops:
        fill buffer
        Customize the memory
    update policy and vf
    update meta weigths
    update bandit
"""

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.common.type_aliases import GymEnv, VecEnv, Schedule, MaybeCallback, Any

from typing import Union

import torch as th
import torch.nn as nn

from XtraPolicy import AttBasedPolicyClass

class trainerClass(OnPolicyAlgorithm):
    policy: Union[th.nn.Module, ActorCriticPolicy]
    rollout_buffer: RolloutBuffer

    def __init__(
            self,
            policy: Union[th.nn.Module, AttBasedPolicyClass],
            env: Union[GymEnv],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 128,
            n_inner_loops: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 1,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            
    ):
