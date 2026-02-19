# On Policy Algorithm that keeps track of prefix with env info and trains consolidator

import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, BaseBuffer
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO

from gymnasium import spaces

import torch.nn.functional as F
import torch as th
import numpy as np


