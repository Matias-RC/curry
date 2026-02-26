# On Policy Algorithm that keeps track of prefix with env info and trains consolidator

import torch as th
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from alternative_maple_buffers import PrefixRolloutBuffer
import numpy as np
import torch as th
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union, Tuple
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from attpolicy import ExperiencerActorCritic
from alternative_maple_buffers import PrefixRolloutBuffer


class EPPO(PPO):
    def __init__(
        self,
        policy: Union[str, Type[th.nn.Module]], # Your custom ExperiencerActorCritic
        env: Union[VecEnv, str],
        learning_rate: float = 3e-4,
        prefix_learning_rate: float = 1e-2, # Distinct LR for the prefixes
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_buffer_class: Optional[Type] = None, # Should be your PrefixBuffer
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.prefix_learning_rate = prefix_learning_rate
        self._last_level_ids = None # Tracks the active level ID for each env

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
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        # This calls the parent setup, which initializes the policy and the buffer
        super()._setup_model()

        # Initialize a stateless optimizer (SGD) for the ParameterDict to avoid memory leaks
        # when IDs are hot-swapped or dropped.
        if hasattr(self.policy, "prefix_dict"):
            self.prefix_optimizer = th.optim.SGD(
                self.policy.prefix_dict.parameters(), 
                lr=self.prefix_learning_rate
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: Any, # Your custom PrefixBuffer
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        #======= Post Inster =======
        max_steps_seen = 0 #Set as private variable of this method
        #======= Post Insert =======

        n_steps = 0
        rollout_buffer.reset()

        # Initialize level tracking if not already done
        if self._last_level_ids is None:
            # Assuming your env has a way to expose the starting level IDs
            self._last_level_ids = np.array(env.get_attr("level_id"))

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                
                # 1. Fetch live prefixes for the current step's environments
                live_prefixes = th.stack([
                    self.policy.prefix_dict[str(uid)] for uid in self._last_level_ids
                ])
                
                # 2. Forward pass with injected prefixes
                actions, values, log_probs = self.policy.forward(obs_tensor, live_prefixes)
            
            actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # 3. Step the environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # 4. Handle Level ID transitions and Terminal Values


            # TODO: Fix this incorrect vibe coded logic please it is killing me!

            new_level_ids = self._last_level_ids.copy()
            for idx, done in enumerate(dones):
                if done:
                    max_steps_seen = max(max_steps_seen, infos[idx]["steps_taken"]) # TODO: make this useful

                    new_level_ids[idx] = infos[idx].get("level_id", new_level_ids[idx])
                    if infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            # We must predict the terminal value using the prefix of the level that just ended
                            term_prefix = self.policy.prefix_dict[str(self._last_level_ids[idx])].unsqueeze(0)
                            terminal_value = self.policy.predict_values(terminal_obs, term_prefix)[0]
                        rewards[idx] += self.gamma * terminal_value

            # 5. Add to Custom Buffer (passing the instance_ids)
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                instance_ids=self._last_level_ids # <-- Custom argument for your PrefixBuffer
            )
            
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_level_ids = new_level_ids # Rollover IDs for next step

        with th.no_grad():
            # Compute value for the last timestep
            final_prefixes = th.stack([
                self.policy.prefix_dict[str(uid)] for uid in self._last_level_ids
            ])
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), final_prefixes)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()

        return True
    
    def train(self) -> None:
            """
            Update policy using the currently gathered rollout buffer.
            Phase 1: Standard PPO + Prefix State Optimization.
            Phase 2: Thinker Meta-Learning (Supervised target on optimized prefixes).
            """
            # Switch to train mode (affects batch norm / dropout)
            self.policy.set_training_mode(True)
            # Update optimizer learning rate
            self._update_learning_rate(self.policy.optimizer)
            # Compute current clip range
            clip_range = self.clip_range(self._current_progress_remaining)
            # Optional: clip range for the value function
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []

            continue_training = True
            
            # =========================================================
            # PHASE 1: PPO & Prefix State Optimization
            # =========================================================
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                
                # Buffer must yield (RolloutBufferSamples, batch_prefixes)
                for rollout_data, batch_prefixes in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = rollout_data.actions.long().flatten()

                    # --- CUSTOM: Pass prefixes into evaluate_actions ---
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, 
                        actions,
                        prefixes=batch_prefixes
                    )
                    values = values.flatten()
                    
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Ratio between old and new policy
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # Clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                        
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Approximate KL Divergence for early stopping
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # --- Optimization step for Policy AND Prefixes ---
                    self.policy.optimizer.zero_grad()
                    if hasattr(self.rollout_buffer, "prefix_optimizer"):
                        self.rollout_buffer.prefix_optimizer.zero_grad()

                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    
                    self.policy.optimizer.step()
                    if hasattr(self.rollout_buffer, "prefix_optimizer"):
                        self.rollout_buffer.prefix_optimizer.step()

                self._n_updates += 1
                if not continue_training:
                    break

            # =========================================================
            # PHASE 2: Thinker Optimization (Meta-Learning)
            # =========================================================
            thinker_losses = []
            
            # We assume get_trajectory_batches yields:
            # traj_obs: (B, S, Hw, Ww, C, ws, ws)
            # traj_mask: (B, S)
            # orig_prefixes: (B, P, D) -> Prefixes at the start of this rollout
            # opt_prefixes: (B, P, D)  -> Prefixes after Phase 1 PPO optimization
            
            if hasattr(self.rollout_buffer, "get_trajectory_batches"):
                for traj_obs, traj_mask, orig_prefixes, opt_prefixes in self.rollout_buffer.get_trajectory_batches(self.batch_size):
                    
                    self.policy.optimizer.zero_grad()
                    
                    # Target is the PPO-optimized prefix state (detached to protect Phase 1 results)
                    target_prefix = opt_prefixes.detach()
                    
                    # --- 1. Train Initial Generator (make_initial_prefix) ---
                    # We want the Thinker to be able to guess the optimal prefix 
                    # just by looking at the very first frame of the sequence.
                    first_obs = traj_obs[:, 0, ...] 
                    initial_pred = self.policy.make_initial_prefix(first_obs)
                    loss_init = F.mse_loss(initial_pred, target_prefix)
                    
                    # --- 2. Train Temporal Thinker (upgrade_prefix_with_trajectory) ---
                    # We want the Thinker to refine the original prefix using 
                    # the trajectory data to reach the same optimal target.
                    upgrade_pred = self.policy.upgrade_prefix_with_trajectory(
                        orig_prefixes, 
                        traj_obs, 
                        traj_mask
                    )
                    loss_upgrade = F.mse_loss(upgrade_pred, target_prefix)
                    
                    # Combined Thinker Loss
                    # Both paths are optimized simultaneously for every batch.
                    thinker_loss = loss_init + loss_upgrade
                    
                    thinker_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    
                    thinker_losses.append(thinker_loss.item())

            # =========================================================
            # Logging
            # =========================================================
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            
            if thinker_losses:
                self.logger.record("train/thinker_loss", np.mean(thinker_losses))
                
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)



class EPPO2(OnPolicyAlgorithm):
    """
    Experiencing Proximal Policy Optimization. Meta learning to play in sokoban.
    """
    rollout_buffer:th.nn.Module
    policy: ExperiencerActorCritic

    def __init__(
            self,
            policy: Union[str, Type[ExperiencerActorCritic]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048, #Size of buffer
            batch_size: int = 256, #Leverage GPU strength
            n_epochs: int = 10,
            gamma: int = 0.99,
            gae_lambda: int = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            rollout_buffer_class: Optional[Type[PrefixRolloutBuffer]] = None,
            rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1, #Hard coded because it is not our objective to work in sde
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                raise NotImplementedError("Maybe for future research!")
            else:
                self.rollout_buffer_class = PrefixRolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **(self.rollout_buffer_kwargs or {}),
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        self.rollout_buffer.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf) 
    
    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: PrefixRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None
        assert self._current_prefixes is not None

        self.policy.set_training_mode(False)

        max_steps_seen = 0

        n_steps = 0
        rollout_buffer.reset() #Not a hard reset on the class variables

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor, self._current_prefixes)

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions) #Without clipping

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False
            
            self._update_info_buffer(infos, dones)
            n_steps += 1

            instance_ids = []

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    #{"plays": inst['plays'], "wins": inst['wins'], "instance_id":self.active_instance_id, "past_id":past_id, "forget": forget}
                    max_steps_seen = max(max_steps_seen, infos[idx]["steps_taken"])
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    tensorized_new_obs = self.policy.obs_to_tensor(new_obs)
                    current_id_idx = infos[idx]["instance_id"]
                    past_id_idx = infos[idx]["past_id"]
                    forget_idx = infos[idx]["forget"]
                    prefix = self.rollout_buffer.get_prefix(current_id_idx)
                    with th.no_grad():
                        if prefix is None:
                            prefix = self.policy.make_initial_prefix(tensorized_new_obs)
                            self.rollout_buffer.set_prefix(prefix, current_id_idx)
                        terminal_value = self.policy.predict_values(terminal_obs, self._current_prefixes[idx])[0]  # type: ignore[arg-type]
                    if forget_idx:
                        self.rollout_buffer.set_forget(past_id_idx)
                    instance_ids.append(past_id_idx)
                    self._current_prefixes[idx] = prefix
                    rewards[idx] += self.gamma * terminal_value
                else:
                    instance_ids.append(infos[idx]["instance_id"])
            
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                instance_ids
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
        
        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def train(self):
        self.policy.set_training_mode(True)
        #  Update the optimizer lr (Later on add for prefix lr)
        self._update_learning_rate(self.policy.optimizer)
        # Compute clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: Compute clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        prefix_optimizer = self.rollout_buffer.prepare_prefix_optimizer(self.prefix_learning_rate)

        # For every epoch do a complete pass on the rollout buffer
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, 
                    actions, self.rollout_buffer.prefixes[rollout_data.indices])
                values = values.flatten()
                #Normalize Advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                # Optimization step
                self.policy.optimizer.zero_grad()
                prefix_optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(self.rollout_buffer.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                prefix_optimizer.step()
            
            self._n_updates += 1
            if not continue_training:
                break
        
        prefix_optimizer.state.clear()
        prefix_optimizer.param_groups.clear()
        del prefix_optimizer
        
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Organize the second section of the buffer onto (B, max_steps, *obs_shape) and set targets to (B, *prefix_shape)
        self.rollout_buffer.generate_supervition_buffer()
        
        supervision_losses = []

        # After which do a complete pass on the reorganized buffer for training the thinker
        for epoch in range(self.n_epochs):
            for supervition_data in self.rollout_buffer.get_trajectory_batches(self.batch_size):
                in_data = supervition_data.inputs
                input_traj_mask = supervition_data.masks
                targets = supervition_data.targets

                prefixes = self.policy.make_initial_prefix(in_data)
                upgade_prefixes = self.policy.upgrade_prefix_with_trajectory(prefixes, in_data, input_traj_mask)

                loss = th.nn.functional.mse_loss(prefixes, targets) + th.nn.functional.mse_loss(upgade_prefixes, targets)
                supervision_losses.append(loss.item())
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()
        
        self.rollout_buffer.clean()

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("supervition/loss", np.mean(supervision_losses))

    def setup_prefix(self):
        obs = obs_as_tensor(self._last_obs, self.device)
        with th.no_grad():
            self._current_prefixes = self.policy.make_initial_prefix(obs)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "ExperiencerAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        self.setup_prefix()

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self