#MAPLE: Memory Augmented Policy that Learns from Errors at test time

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




class Maple(PPO):

    consolidator: Consolidator
    dynamic_buffer_class: type
    rollout_buffer: RolloutBuffer

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
        dynamic_buffer_class: type = None,
        dynamic_buffer_kwargs: Optional[dict[str, Any]] = None,
        consolidator_class: type = None,
        consolidator_kwargs: Optional[dict[str, Any]] = None,
        num_retries: int = 10,
    ):
        # ---- MAPLE state ----
        self.num_retries = num_retries

        self.dynamic_buffer_class = dynamic_buffer_class
        self.dynamic_buffer_kwargs = dynamic_buffer_kwargs or {}

        self.consolidator_class = consolidator_class
        self.consolidator_kwargs = consolidator_kwargs or {}

        # episodic trajectories for consolidation
        self.consolidator_buffer = []

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
            max_grad_norm = 0.5

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
                        g_actor = grads[0]
                        actor_norm = g_actor.norm()
                        if actor_norm > max_grad_norm:
                            g_actor = g_actor * (max_grad_norm / (actor_norm + 1e-6))
                        new_a = a_prefix - prefix_lr*g_actor
                        self.dynamic_buffer.current_prefixes[0][idx] = new_a.detach()

                    if grads[1] is not None: # Critic Update
                        g_critic = grads[1]
                        critic_norm = g_critic.norm()
                        if critic_norm > max_grad_norm:
                            g_critic = g_critic * (max_grad_norm / (critic_norm + 1e-6))
                        new_c = c_prefix - prefix_lr * g_critic
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
                target_actor_prefix = traj_last['actor_prefix'][0].detach().clone()
                target_critic_prefix = traj_last['critic_prefix'][0].detach().clone()
                
                self.consolidator_buffer.append({
                    "input_obs": traj_first['obs'], # The "Error" sequence
                    "target_prefixes": (target_actor_prefix, target_critic_prefix)
                })
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
        self._past_infos = [{"retries_left":self.num_retries, "retry_count":0} for _ in range(self.n_envs)]

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
                        terminal_value = self.policy.predict_values(terminal_obs, self.dynamic_buffer.current_prefixes)[0]
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
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), self.dynamic_buffer.current_prefixes)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        #Not yet working TODO: CALLBACK 

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer (PPO).
        Also updates the Consolidator using the First -> Last trajectory mapping (Supervised).
        """
        # Switch to train mode (affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.consolidator.set_training_mode(True)
        
        # Update learning rates
        self._update_learning_rate(self.policy.optimizer)
        self._update_learning_rate(self.consolidator.optimizer)
        
        # Compute current clip ranges
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        consolidator_losses = []
        ppo_losses, entropy_losses, value_losses = [], [], []

        # ==============================================================================
        # MAIN TRAINING LOOP (Epochs)
        # ==============================================================================
        for _ in range(self.n_epochs):

            # --------------------------------------------------------------------------
            # PART A: Consolidator Training (Run once per epoch over its buffer)
            # --------------------------------------------------------------------------
            # NOTE: We do NOT clear the buffer here. We need it for the next epoch!
            if len(self.consolidator_buffer) > 0:
                for sample in self.consolidator_buffer:
                    # 1. Get Input: Sequence of observations from the failed/first attempt
                    obs_traj = th.as_tensor(sample["input_obs"], device=self.device)
                    obs_traj = obs_traj.unsqueeze(0) # Add batch dim
                    
                    # 2. Get Targets
                    target_a, target_c = sample["target_prefixes"]
                    target_a = target_a.unsqueeze(0).to(self.device)
                    target_c = target_c.unsqueeze(0).to(self.device)

                    # 3. Forward Pass
                    pred_a, pred_c = self.consolidator(obs_traj)

                    # 4. Loss
                    loss_a = F.mse_loss(pred_a, target_a)
                    loss_c = F.mse_loss(pred_c, target_c)
                    total_loss = loss_a + loss_c

                    # 5. Optimize
                    self.consolidator.optimizer.zero_grad()
                    total_loss.backward()
                    self.consolidator.optimizer.step()

                    consolidator_losses.append(total_loss.item())

            # --------------------------------------------------------------------------
            # PART B: PPO Training (Standard Minibatch Iteration)
            # --------------------------------------------------------------------------
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                current_prefixes = (
                    rollout_data.actor_prefixes, 
                    rollout_data.critic_prefixes
                )
                
                # Forward Pass (With Injection)
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    current_prefixes 
                )
                
                values = values.flatten()
                
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss
                if self.ent_coef > 0:
                    entropy_loss = -th.mean(entropy)
                else:
                    entropy_loss = 0

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                # Logging
                ppo_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item() if isinstance(entropy_loss, th.Tensor) else entropy_loss)

        # ==============================================================================
        # CLEANUP: Clear buffer ONLY after all epochs are done
        # ==============================================================================
        self.consolidator_buffer = []
        self._n_updates += self.n_epochs
        
        # ==============================================================================
        # Logging
        # ==============================================================================
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(ppo_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        
        if len(consolidator_losses) > 0:
            self.logger.record("train/consolidator_loss", np.mean(consolidator_losses))
            
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")