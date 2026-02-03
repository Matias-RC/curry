from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from typing import Optional, Union
from gymnasium import spaces
import torch as th
import numpy as np

class MapleRolloutBuffer(BaseBuffer):

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        obs_shape = self.obs_shape

        self.p1_observations = np.zeros((self.buffer_size, self.n_envs, *obs_shape), dtype=self.observation_space.dtype)
        self.pn_observations = np.zeros_like(self.p1_observations)
        self.pn_observations_with_prefix = np.zeros_like(self.p1_observations)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)

        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.rewards_p1 = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values_p1 = np.zeros_like(self.rewards_p1)
        self.log_probs_p1 = np.zeros_like(self.rewards_p1)
        self.advantages_p1 = np.zeros_like(self.rewards_p1)
        self.returns_p1 = np.zeros_like(self.rewards_p1)

        self.rewards_pn = np.zeros_like(self.rewards_p1)
        self.values_pn = np.zeros_like(self.rewards_p1)
        self.log_probs_pn = np.zeros_like(self.rewards_p1)
        self.advantages_pn = np.zeros_like(self.rewards_p1)
        self.returns_pn = np.zeros_like(self.rewards_p1)

        self.rewards_pn_prefix = np.zeros_like(self.rewards_p1)
        self.values_pn_prefix = np.zeros_like(self.rewards_p1)
        self.log_probs_pn_prefix = np.zeros_like(self.rewards_p1)
        self.advantages_pn_prefix = np.zeros_like(self.rewards_p1)
        self.returns_pn_prefix = np.zeros_like(self.rewards_p1)

        self.generator_ready = False
        super().reset()

    def add(
        self,
        p1_obs,
        pn_obs,
        pn_obs_prefix,
        action,
        reward_p1,
        reward_pn,
        reward_pn_prefix,
        episode_start,
        value_p1,
        value_pn,
        value_pn_prefix,
        log_prob_p1,
        log_prob_pn,
        log_prob_pn_prefix,
    ) -> None:

        self.p1_observations[self.pos] = p1_obs
        self.pn_observations[self.pos] = pn_obs
        self.pn_observations_with_prefix[self.pos] = pn_obs_prefix

        self.actions[self.pos] = action
        self.episode_starts[self.pos] = episode_start

        self.rewards_p1[self.pos] = reward_p1
        self.rewards_pn[self.pos] = reward_pn
        self.rewards_pn_prefix[self.pos] = reward_pn_prefix

        self.values_p1[self.pos] = value_p1.cpu().numpy().flatten()
        self.values_pn[self.pos] = value_pn.cpu().numpy().flatten()
        self.values_pn_prefix[self.pos] = value_pn_prefix.cpu().numpy().flatten()

        self.log_probs_p1[self.pos] = log_prob_p1.cpu().numpy().flatten()
        self.log_probs_pn[self.pos] = log_prob_pn.cpu().numpy().flatten()
        self.log_probs_pn_prefix[self.pos] = log_prob_pn_prefix.cpu().numpy().flatten()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _compute_returns_and_advantage(self, rewards, values, advantages, returns, last_values, dones):
        last_values = last_values.cpu().numpy().flatten()
        last_gae_lam = 0.0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = values[step + 1]

            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns[:] = advantages + values

    def compute_returns_and_advantage(self, last_values_p1, last_values_pn, last_values_pn_prefix, dones):
        self._compute_returns_and_advantage(
            self.rewards_p1, self.values_p1, self.advantages_p1, self.returns_p1, last_values_p1, dones
        )
        self._compute_returns_and_advantage(
            self.rewards_pn, self.values_pn, self.advantages_pn, self.returns_pn, last_values_pn, dones
        )
        self._compute_returns_and_advantage(
            self.rewards_pn_prefix, self.values_pn_prefix, self.advantages_pn_prefix, self.returns_pn_prefix,
            last_values_pn_prefix, dones
        )

    def _prepare_generator(self):
        if self.generator_ready:
            return

        for name in [
            "p1_observations", "pn_observations", "pn_observations_with_prefix",
            "actions", "values_p1", "values_pn", "values_pn_prefix",
            "log_probs_p1", "log_probs_pn", "log_probs_pn_prefix",
            "advantages_p1", "advantages_pn", "advantages_pn_prefix",
            "returns_p1", "returns_pn", "returns_pn_prefix",
        ]:
            setattr(self, name, self.swap_and_flatten(getattr(self, name)))

        self.generator_ready = True

    def _get(self, obs, values, log_probs, advantages, returns, batch_size):
        self._prepare_generator()
        n_samples = obs.shape[0]
        indices = np.random.permutation(n_samples)
        if batch_size is None:
            batch_size = n_samples

        start = 0
        while start < n_samples:
            batch = indices[start:start + batch_size]
            yield RolloutBufferSamples(
                observations=self.to_torch(obs[batch]),
                actions=self.to_torch(self.actions[batch]),
                old_values=self.to_torch(values[batch]),
                old_log_prob=self.to_torch(log_probs[batch]),
                advantages=self.to_torch(advantages[batch]),
                returns=self.to_torch(returns[batch]),
            )
            start += batch_size

    def get_p1(self, batch_size: Optional[int] = None):
        yield from self._get(
            self.p1_observations,
            self.values_p1,
            self.log_probs_p1,
            self.advantages_p1,
            self.returns_p1,
            batch_size,
        )

    def get_pn(self, batch_size: Optional[int] = None):
        yield from self._get(
            self.pn_observations,
            self.values_pn,
            self.log_probs_pn,
            self.advantages_pn,
            self.returns_pn,
            batch_size,
        )

    def get_pn_prefix(self, batch_size: Optional[int] = None):
        yield from self._get(
            self.pn_observations_with_prefix,
            self.values_pn_prefix,
            self.log_probs_pn_prefix,
            self.advantages_pn_prefix,
            self.returns_pn_prefix,
            batch_size,
        )

class DynamicReplayBuffer:
    pass
