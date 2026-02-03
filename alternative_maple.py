#MAPLE: Memory Augmented Policy that Learns from Errors at test time

from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, BaseBuffer
from alternative_maple_buffers import MapleRolloutBuffer, DynamicReplayBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from consolidator_class import Consolidator
from typing import Any, Optional, Union
from stable_baselines3 import PPO
from gymnasium import spaces
import torch as th
import numpy as np



class SyrupWithMapleMakesMeHappy(PPO):

    consolidator:Consolidator
    dynamic_buffer_class:DynamicReplayBuffer
    rollout_buffer:MapleRolloutBuffer

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
            self.observation_space,  # type: ignore[arg-type]
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

    def _update_dynamic_buffer(self, infos, dones):
        pass

    def update_prefixes(self, infos, dones):
        pass

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self.consolidator.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        #==========up to this point it should be ok==========

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            #=============================
            #self.policy.reset_noise(env.num_envs)
            raise NotImplementedError

        #TODO: make maple callback
        callback.on_rollout_start()

        #Make observation prefixes
        prefixes = self.consolidator._init_empty(self.n_envs)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                #=============================
                #self.policy.reset_noise(env.num_envs)
                raise NotImplementedError

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                obs_tensor = th.cat([obs_tensor, prefixes], dim=1) # Should work for flat and channel first
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds // We are not in continuos space
                    #                                         // Although this is kept for future works
                    # if they were previously squashed (scaled in [-1, 1])
                    #=============================
                    #clipped_actions = self.policy.unscale_action(clipped_actions)
                    raise NotImplementedError
                #else:
                #    # Otherwise, clip the actions to avoid out of bound error
                #    # as we are sampling from an unbounded Gaussian distribution
                #    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                #====================== Directly do:
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

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
            #Check with dones if end of play check with infos idx of the play/replay
            prefixes = self.consolidator(infos, dones, self.dynamic_buffer) #Check if this is play 0 then update prefix with consolidator
            prefixes = self.update_prefixes(infos, dones) # The rest are updated with simplified gradient descent

            # if "retries_left" in infos == 0 then eliminate from dynamic buffer and add in batch to 
            # rollout buffer, in extact terms, given that a dynamic buffer holds all retries of the same 
            # level, then to the buffer add:    
            #self.p1_observations  (play one observations with prefix = zeroes)
            #self.pn_observations (last play with prefix = zeroes)
            #self.pn_observations_with_prefix (last play with prefix = repeatedly supervised prefix)
            self._update_dynamic_buffer(infos, dones) 
            
            #Instead of this traditional rollout buffer add we store in dynamic util replays are overK
            #rollout_buffer.add(
            #    self._last_obs,  # type: ignore[arg-type]
            #    actions,
            #    rewards,
            #    self._last_episode_starts,  # type: ignore[arg-type]
            #    values,
            #    log_probs,
            #)
            #self._last_obs = new_obs  # type: ignore[assignment]
            #self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        #Not yet working TODO: CALLBACK 

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def train(self):
        return super().train()