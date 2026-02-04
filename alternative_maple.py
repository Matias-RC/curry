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
        # current_prefix is (Actor_Tensor_Batch, Critic_Tensor_Batch)
        a_prefix_batch, c_prefix_batch = current_prefix
        current_obs, actions, rewards, episode_starts, values, log_probs = contents
        
        for idx, _ in enumerate(past_infos):
            # Slice out the specific prefix for this environment
            # We use detach() because we don't want to store the whole computation graph
            env_prefix = (
                a_prefix_batch[idx].detach(), 
                c_prefix_batch[idx].detach()
            )
            
            self.dynamic_buffer.append_step(
                env_idx=idx, 
                obs=current_obs[idx], 
                actions=actions[idx],
                rewards=rewards[idx],
                episode_starts=episode_starts[idx],
                values=values[idx],
                log_probs=log_probs[idx],
                prefix=env_prefix
            )

    def maybe_update_prefixes(self, infos, dones):
            indices_of_generated = []
            to_be_generated = []
            prefix_lr = 0.05 

            for idx, (info, done) in enumerate(zip(infos, dones)):
                if not done:
                    continue

                # --- PATH A: GENERATION (Cold Start) ---
                if info.get("retry_count") == 0:
                    traj_last = self.dynamic_buffer.get_last_attempt(env_idx=idx)
                    obs_traj = th.as_tensor(traj_last['obs'], device=self.device)
                    if obs_traj.ndim == 4: # Handle single image vs sequence
                        obs_traj = obs_traj.unsqueeze(0)

                    to_be_generated.append(obs_traj)
                    indices_of_generated.append(idx)

                # --- PATH B: OPTIMIZATION (Actor-Critic Split Refinement) ---
                elif info.get("retries_left") > 0:
                    traj = self.dynamic_buffer.get_last_attempt(env_idx=idx)
                    obs = th.as_tensor(traj['obs'], device=self.device)
                    rewards = th.as_tensor(traj['rewards'], device=self.device)

                    # 1. Extract and Unpack the Tuple
                    # We fetch the specific env's prefix from the tuple of batches
                    a_prefix = self.dynamic_buffer.current_prefixes[0][idx].detach().clone().requires_grad_(True)
                    c_prefix = self.dynamic_buffer.current_prefixes[1][idx].detach().clone().requires_grad_(True)

                    T = obs.shape[0]
                    # Expand to (T, C, H, W) to match the trajectory length
                    a_prefix_exp = a_prefix.unsqueeze(0).expand(T, *a_prefix.shape)
                    c_prefix_exp = c_prefix.unsqueeze(0).expand(T, *c_prefix.shape)

                    # 2. Forward Pass with Tuple Prefix
                    with th.enable_grad():
                        # Policy now takes a tuple (actor_prefix, critic_prefix)
                        _, values, log_probs = self.policy(obs, (a_prefix_exp, c_prefix_exp))
                        values, log_probs = values.squeeze(), log_probs.squeeze()

                        # 3. Value Loss (Targets Critic Prefix)
                        returns = th.zeros(T, device=self.device)
                        prev_return = 0.0
                        for t in reversed(range(T)):
                            prev_return = rewards[t] + self.gamma * prev_return
                            returns[t] = prev_return
                        v_loss = 0.5 * (returns - values).pow(2).mean()

                        # 4. Policy Loss (Targets Actor Prefix)
                        with th.no_grad():
                            next_values = th.zeros_like(values)
                            next_values[:-1] = values[1:]
                            td_targets = rewards + self.gamma * next_values

                        td0_advantage = td_targets - values.detach()
                        p_loss = -(log_probs * td0_advantage).mean()

                        # Joint loss for simultaneous backprop
                        total_loss = p_loss + v_loss

                    # 5. Split Gradient Update
                    # We calculate gradients for both tensors in the tuple simultaneously
                    grads = th.autograd.grad(total_loss, [a_prefix, c_prefix], allow_unused=True)

                    if grads[0] is not None: # Actor Update
                        new_a = a_prefix - prefix_lr * grads[0]
                        self.dynamic_buffer.current_prefixes[0][idx] = new_a.detach()

                    if grads[1] is not None: # Critic Update
                        new_c = c_prefix - prefix_lr * grads[1]
                        self.dynamic_buffer.current_prefixes[1][idx] = new_c.detach()

            # Execute Batch Generation (Ensure Consolidator returns a tuple)
            if to_be_generated:
                with th.no_grad():
                    batch_obs = th.cat(to_be_generated, dim=0)
                    # Consolidator output should be (batch_actor_prefix, batch_critic_prefix)
                    gen_a, gen_c = self.consolidator(batch_obs)
                    for i, env_idx in enumerate(indices_of_generated):
                        self.dynamic_buffer.current_prefixes[0][env_idx] = gen_a[i].detach()
                        self.dynamic_buffer.current_prefixes[1][env_idx] = gen_c[i].detach()
    
    def maybe_flush_buffer(self, contents, past_infos):
        _, _, _, episode_starts, _, _ = contents
        for idx, info in enumerate(past_infos):
            if info.get("retries_left") == 0 and episode_starts[idx]:
                # Get the trajectory of the FIRST attempt (Prefix = 0)
                # This is the input for the Consolidator training
                traj_first = self.dynamic_buffer.get_first_attempt(env_idx=idx)
                traj_last = self.dynamic_buffer.get_last_attempt(env_idx=idx)
                self.rollout_buffer.add_trajectory(traj_first)
                self.rollout_buffer.add_trajectory(traj_last)
                
                # Clear Dynamic Buffer for this env to start fresh on next level
                self.dynamic_buffer.reset_env(env_idx=idx)

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
        actor_init, critic_init = self.consolidator._init_empty(self.n_envs) 
        
        assert isinstance(actor_init, th.Tensor), "prefixes must be tensors"
        assert isinstance(critic_init, th.Tensor), "prefixes must be tensors"
        assert actor_init.shape[0] == self.n_envs == critic_init.shape[0]

        self.dynamic_buffer.initialize_self((actor_init, critic_init))
        self._past_infos = [{"retries_left":1, "retry_count":0} for _ in range(self.n_envs)]

        while not self.rollout_buffer.full:

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor, self.dynamic_buffer.current_prefixes)
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
                        terminal_value = self.policy.predict_values(terminal_obs, self.dynamic_buffer.current_prefixes[1])[0]
                    rewards[idx] += self.gamma * terminal_value

            self._update_dynamic_buffer((self._last_obs, actions, rewards,self._last_episode_starts,
                                            values, log_probs),
                                            self._past_infos) 
            
            #Check with dones if end of play check with infos idx of the play/replay
            self.maybe_update_prefixes(self._past_infos, self._last_episode_starts) # The rest are updated with simplified gradient descent

            self.maybe_flush_buffer((self._last_obs, actions, rewards,self._last_episode_starts,
                                            values, log_probs),
                                            self._past_infos) 

            
            self._last_obs = new_obs
            self._past_infos = infos
            self._last_episode_starts = dones
            n_steps = rollout_buffer.get_current_size()

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), self.dynamic_buffer.current_prefixes[1])

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        #Not yet working TODO: CALLBACK 

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def train(self):
        return super().train()