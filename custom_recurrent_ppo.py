"""
Custom RecurrentPPO that uses SequenceAwareRolloutBuffer by default.

This subclass overrides the rollout_buffer_class to use our custom buffer
that properly samples hidden states per observation instead of per sequence start.
"""

from typing import Type
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, RecurrentDictRolloutBuffer
from sequence_aware_buffer import SequenceAwareRolloutBuffer, SequenceAwareDictRolloutBuffer
from gymnasium import spaces


class CustomRecurrentPPO(RecurrentPPO):
    """
    RecurrentPPO with SequenceAwareRolloutBuffer for spatial recurrent policies.

    This custom class uses a buffer that samples hidden states for every observation
    in the minibatch, which is necessary for spatial recurrent architectures like
    ConvLSTM and ConvAtt where each observation needs its corresponding hidden state.
    """

    def _setup_model(self) -> None:
        """
        Setup the model with our custom buffer class.

        This is a copy of the parent's _setup_model but uses SequenceAwareRolloutBuffer
        instead of RecurrentRolloutBuffer.
        """
        import torch as th
        from stable_baselines3.common.utils import get_schedule_fn
        from sb3_contrib.common.recurrent.type_aliases import RNNStates
        from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Determine which buffer class to use (our custom one)
        buffer_cls: Type[RecurrentRolloutBuffer] = SequenceAwareDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else SequenceAwareRolloutBuffer

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        # USE OUR CUSTOM BUFFER CLASS HERE
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
