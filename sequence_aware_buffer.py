"""
Custom RecurrentRolloutBuffer that properly samples hidden states per observation.

This fixes the dimension mismatch that occurs during evaluate_actions where:
- features: (minibatch_size, C, H, W) e.g. (84, 32, 10, 10)
- states:   (n_layers, n_envs, state_dim) e.g. (1, 2, 9600)

The key insight is that when we sample observations at indices [5, 12, 23, 45, ...],
we should also sample the corresponding hidden states at those same indices.
"""

import numpy as np
import torch
from typing import NamedTuple, Optional
from gymnasium import spaces

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class RecurrentRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: RNNStates
    episode_starts: torch.Tensor
    mask: torch.Tensor


class SequenceAwareRolloutBuffer(RecurrentRolloutBuffer):
    """
    Custom RecurrentRolloutBuffer that properly samples LSTM states per observation.

    The standard RecurrentRolloutBuffer stores states with shape (n_steps, n_layers, n_envs, state_dim)
    and samples them only at sequence start indices. This works for standard LSTM evaluation but
    fails when evaluate_actions needs to process arbitrary minibatches where each observation
    needs its corresponding hidden state.

    This buffer:
    1. Stores states indexed per observation (same as observations, actions, etc.)
    2. When sampling, provides states for each sampled observation
    3. Maintains sequence awareness for proper episode masking
    """

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[np.ndarray] = None,
    ) -> RecurrentRolloutBufferSamples:
        """
        Sample observations and their corresponding LSTM states.

        Key difference from parent class:
        - Parent: Only samples states at sequence start indices
        - This: Samples states for EVERY observation in the batch

        Args:
            batch_inds: Indices of observations to sample from the flattened buffer
            env: Optional environment change indicators (not used in base implementation)

        Returns:
            RecurrentRolloutBufferSamples with properly indexed states
        """
        # Create episode_starts if not already done
        if not hasattr(self, 'episode_starts') or self.episode_starts is None:
            self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Import here to avoid circular dependency
        from sb3_contrib.common.recurrent.buffers import create_sequencers

        # Get episode starts for these batch indices
        episode_starts = self.episode_starts[batch_inds]

        # Index env_change using batch_inds (parent passes full array)
        if env is not None:
            env_change = env[batch_inds]
        else:
            # If not provided, compute from batch_inds
            # env_change[i] = True if batch_inds[i] and batch_inds[i-1] are not consecutive
            env_change = np.zeros_like(episode_starts, dtype=np.float32)
            if len(batch_inds) > 1:
                env_change[1:] = ((batch_inds[1:] - batch_inds[:-1]) != 1).reshape(-1, 1)
            env_change[0] = 1.0  # First element always starts a new sequence

        # Ensure consistent shapes (batch_size, 1)
        if episode_starts.ndim == 1:
            episode_starts = episode_starts.reshape(-1, 1)
        if env_change.ndim == 1:
            env_change = env_change.reshape(-1, 1)

        # Create sequence utilities for padding and chunking
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            episode_starts,
            env_change,
            self.device,
        )

        # ====================================================================
        # KEY MODIFICATION: Sample states for ALL observations, not just seq starts
        # ====================================================================

        # Standard approach (from parent class):
        # lstm_states_pi = (
        #     self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        #     self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        # )

        # New approach: Get states for EVERY observation in batch_inds
        # Shape before: hidden_states_pi is (n_total_samples, n_layers, state_dim) after flattening
        # After indexing: (batch_size, n_layers, state_dim)
        # After swapaxes: (n_layers, batch_size, state_dim) <- what policy expects

        lstm_states_pi = (
            torch.tensor(
                self.hidden_states_pi[batch_inds].swapaxes(0, 1),
                dtype=torch.float32,
                device=self.device
            ),
            torch.tensor(
                self.cell_states_pi[batch_inds].swapaxes(0, 1),
                dtype=torch.float32,
                device=self.device
            ),
        )

        lstm_states_vf = (
            torch.tensor(
                self.hidden_states_vf[batch_inds].swapaxes(0, 1),
                dtype=torch.float32,
                device=self.device
            ),
            torch.tensor(
                self.cell_states_vf[batch_inds].swapaxes(0, 1),
                dtype=torch.float32,
                device=self.device
            ),
        )

        lstm_states = RNNStates(pi=lstm_states_pi, vf=lstm_states_vf)

        # ====================================================================
        # Standard observation and advantage sampling (unchanged)
        # ====================================================================

        # Pad sequences if needed (for sequential processing)
        # Note: For evaluate_actions, we often process single timesteps, so padding may be minimal
        observations = self.pad(self.observations[batch_inds])
        actions = self.pad(self.actions[batch_inds])

        # Handle dict observations if needed
        if isinstance(self.observations, dict):
            observations = {key: self.pad(obs[batch_inds]) for key, obs in self.observations.items()}

        # Create mask for padded timesteps
        mask = self.pad(np.ones_like(self.returns[batch_inds]))

        return RecurrentRolloutBufferSamples(
            observations=observations,
            actions=actions,
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=lstm_states,
            episode_starts=self.pad_and_flatten(episode_starts),
            mask=self.pad_and_flatten(mask),
        )


class SequenceAwareDictRolloutBuffer(SequenceAwareRolloutBuffer):
    """
    Version of SequenceAwareRolloutBuffer for dictionary observation spaces.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        hidden_state_shape: tuple,
        device: str = "cpu",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        # Store observation keys before calling super
        self.obs_keys = list(observation_space.spaces.keys())
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            hidden_state_shape,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

    def reset(self) -> None:
        """Reset buffer - handle dict observations."""
        super().reset()
        # Initialize dict observations
        if isinstance(self.observation_space, spaces.Dict):
            self.observations = {}
            for key, obs_space in self.observation_space.spaces.items():
                self.observations[key] = np.zeros(
                    (self.buffer_size, self.n_envs, *obs_space.shape),
                    dtype=obs_space.dtype
                )

    def add(self, *args, **kwargs) -> None:
        """Add samples - handle dict observations."""
        # The parent's parent (RolloutBuffer) handles dict observations
        super().add(*args, **kwargs)
