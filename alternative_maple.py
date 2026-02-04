#MAPLE: Memory Augmented Policy that Learns from Errors at test time

from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, BaseBuffer
from alternative_maple_buffers import DynamicReplayBuffer, MapleRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from consolidator_class import Consolidator
from typing import Any, Optional, Union
from stable_baselines3 import PPO
from gymnasium import spaces
import torch as th
import numpy as np



class Maple(PPO):

    consolidator:Consolidator
    dynamic_buffer_class:DynamicReplayBuffer
    rollout_buffer:RolloutBuffer

    def __init__(self, dynamic_buffer_class, dynamic_buffer_kwargs,
                 consolidator_class, consolidator_kwargs,
                 _init_setup_model: bool = True,
                  *args, **kwargs):
        self.dynamic_buffer_class = dynamic_buffer_class
        self.dynamic_buffer_kwargs = dynamic_buffer_kwargs
        self.consolidator_class = consolidator_class
        self.consolidator_kwargs = consolidator_kwargs
        super().__init__(_init_setup_model=False, *args, **kwargs)
        if _init_setup_model:
            self._setup_model()
        
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                raise NotImplementedError
            else:
                self.rollout_buffer_class = MapleRolloutBuffer

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
        #===========================================================================
        self.dynamic_buffer = self.dynamic_buffer_class(**self.dynamic_buffer_kwargs)
        #===========================================================================
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        #=====================================================================
        self.consolidator = self.consolidator_class(**self.consolidator_kwargs)
        #=====================================================================
        self.policy = self.policy.to(self.device)
        self.consolidator = self.consolidator.to(self.device)
        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def _update_dynamic_buffer(self, contents, past_infos, current_prefix):
        current_obs, actions, rewards, episode_starts, values, log_probs = contents
        for idx, info in enumerate(past_infos):
            # 1. Append current step to the dynamic buffer for this specific env
            self.dynamic_buffer.append_step(env_idx=idx, 
                                            obs=current_obs[idx], 
                                            actions=actions[idx],
                                            rewards=rewards[idx],
                                            episode_starts=episode_starts[idx],
                                            values=values[idx],
                                            log_probs=log_probs[idx],
                                            prefix=current_prefix[idx])
            
            # 2. Check if the LEVEL is finished (all retries exhausted or won)
            if info.get("retries_left") == 0 and episode_starts[idx]:
                
                # --- THE FLUSH ---
                
                # Get the trajectory of the FIRST attempt (Prefix = 0)
                # This is the input for the Consolidator training
                traj_first = self.dynamic_buffer.get_first_attempt(env_idx=idx)
                traj_last = self.dynamic_buffer.get_last_attempt(env_idx=idx)
                self.rollout_buffer.add_trajectory(traj_first)
                self.rollout_buffer.add_trajectory(traj_last)
                
                # Clear Dynamic Buffer for this env to start fresh on next level
                self.dynamic_buffer.reset_env(env_idx=idx)

    def maybe_update_prefixes(self, infos, dones):
        for idx, info in enumerate(infos):
            if info.get("retry_count") == 0 and dones[idx]:
                #generate the prefix with consolidator
            elif dones[idx]:
                #improve the prefix present inside

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MapleRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self.consolidator.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        #TODO: make maple callback
        callback.on_rollout_start()

        #Make observation prefixes
        prefixes = self.consolidator._init_empty(self.n_envs)

        assert isinstance(prefixes, th.Tensor), "prefixes must be a tensor"
        assert prefixes.shape[0] == self.n_envs

        self.dynamic_buffer.initialize_self(prefixes)
        self._past_infos = [{"retries_left":1, "retry_count":0} for _ in range(self.n_envs)]

        while n_steps < n_rollout_steps:

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor, prefixes)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, infos, dones = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs, prefixes)[0]
                    rewards[idx] += self.gamma * terminal_value
            #Check with dones if end of play check with infos idx of the play/replay
            prefixes = self.maybe_update_prefixes(self._past_infos, self._last_episode_starts) # The rest are updated with simplified gradient descent

            self._update_dynamic_buffer((self._last_obs, actions, rewards,self._last_episode_starts,
                                            values, log_probs),
                                            self._past_infos) 
            
            self._last_obs = new_obs
            self._past_infos = infos
            self._last_episode_starts = dones
            n_steps = rollout_buffer.get_current_size()

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), prefixes)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        #Not yet working TODO: CALLBACK 

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def train(self):
        return super().train()