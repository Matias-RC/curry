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

from alternative_maple_buffers import DynamicReplayBuffer, MapleRolloutBuffer
from consolidator_class import Consolidator


class OptimizedMaple(PPO):
    """
    This class removes the dynamic buffer, and ignores the amout or retries associated to any given level.
    Instead uses a hash map that stores current prefix for each level id. Whenever a steps says "forget" t
    -he hash map deletes the item and moves on.

    The rollout buffer stores prefixes anyways, thus there is no problem at train time.

    For training:
        . Actor predicts actions: pi(a_i^j \mid s_i^j, p_pi^j)
        . Critic predicts values: Vf(v_i^j \mid s_i^j, p_vf^j)
        . Thinker predicts prefixes: p_pi^j, p_vf^j = thinker(tau)
        . Prefixes behave as parameters and recive gradient from Advantage Regret
        . The updated prefixes are redirected to hash map
    """

    thinker: Consolidator

    def __init__(
        self,
        # ---- PPO args (explicit) ----
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        # ---- MAPLE-specific args ----
        thinker_class: type = None,
        thinker_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.thinker_class = thinker_class
        self.thinker_kwargs = thinker_kwargs

        # ---- Call PPO explicitly ----
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,   # IMPORTANT: delay setup
        )

        # ---- Maple-controlled setup ----
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                raise NotImplementedError
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        #=====================================================================
        self.thinker = self.thinker_class(**self.thinker_kwargs)
        #=====================================================================
        self.policy = self.policy.to(self.device)
        self.thinker = self.thinker.to(self.device)
        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)
        
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None

        self.policy.set_training_mode(False)
        n_steps = 0
        self.rollout_buffer.reset()
        while n_steps  < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)


