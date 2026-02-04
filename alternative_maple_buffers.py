from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer
from typing import Optional, Union, Tuple
from gymnasium import spaces
import torch as th
import numpy as np

class MapleRolloutBuffer(RolloutBuffer):
    """
    Modified rollout buffer that forces n_envs=1 and adds support for adding entire trajectories.
    It also makes sure the size of the buffer matches correctly with the size of a natural buffer.
    """
    prefixes: np.ndarray
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        prefix_size: Tuple,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size*n_envs,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs=1,  # Force n_envs=1, ignoring passed value
        )
        self.prefix_size = prefix_size

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.prefixes = np.zeros((self.buffer_size, self.n_envs, *self.prefix_size))
        self.generator_ready = False
        super().reset()

    def add_trajectory(
        self,
        traj_obj
    ) -> None:
        obs: np.ndarray = traj_obj['obs']
        action: np.ndarray = traj_obj['actions']
        reward: np.ndarray = traj_obj['rewards']
        episode_start: np.ndarray = traj_obj['episode_starts']
        value: th.Tensor = traj_obj['values']
        log_prob: th.Tensor = traj_obj['log_probs']
        prefix: th.Tensor = traj_obj['prefix']
        """
        Add a trajectory (sequence of steps) to the buffer.
        Assumes inputs do not include the env dimension (since n_envs=1).
        :param obs: Observations (traj_len, *obs_shape)
        :param action: Actions (traj_len, action_dim)
        :param reward: Rewards (traj_len,)
        :param episode_start: Episode start signals (traj_len,)
        :param value: Estimated values (traj_len,)
        :param log_prob: Log probabilities (traj_len,)
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error (though unlikely for trajectory)
            log_prob = log_prob.reshape(-1, 1)

        traj_len = obs.shape[0]
        if self.pos + traj_len > self.buffer_size:
            raise ValueError(f"Trajectory of length {traj_len} exceeds remaining buffer space ({self.buffer_size - self.pos})")

        end = self.pos + traj_len if self.pos + traj_len <self.buffer_size else self.buffer_size -1

        selection = traj_len + (self.pos + traj_len - end)

        # Reshape needed when using discrete observations
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((traj_len, self.n_envs) + self.obs_shape)
        
        prefix = prefix.clone().cpu().numpy().reshape((traj_len, self.n_envs)+self.prefix_size)

        # Reshape to handle multi-dim and discrete action spaces
        action = action.reshape((traj_len, self.n_envs, self.action_dim))

        # Convert tensors to numpy and reshape
        values = value.clone().cpu().numpy().reshape((traj_len, self.n_envs))
        log_probs = log_prob.clone().cpu().numpy().reshape((traj_len, self.n_envs))
        rewards = reward.reshape((traj_len, self.n_envs))
        episode_starts = episode_start.reshape((traj_len, self.n_envs))

        # Assign to buffer slices
        self.observations[self.pos:end] = np.array(obs)[:selection]
        self.actions[self.pos:end] = np.array(action)[:selection]
        self.rewards[self.pos:end] = rewards[:selection]
        self.episode_starts[self.pos:end] = episode_starts[:selection]
        self.values[self.pos:end] = values[:selection]
        self.log_probs[self.pos:end] = log_probs[:selection]
        self.prefixes[self.pos:end] = prefix[:selection]

        self.pos += traj_len
        if self.pos == self.buffer_size:
            self.full = True

class DynamicReplayBuffer:
    def __init__(self, n_envs: int, device: str = "cpu"):
        self.n_envs = n_envs
        self.device = device
        # Storage: List of lists (one list per env containing its retry history)
        self.history = [[] for _ in range(n_envs)]
        # This stores the optimized prefix currently being used for each env
        self.current_prefixes = None 

    def initialize_self(self, initial_prefixes: th.Tensor):
        # initial_prefixes: (n_envs, C, H, W)
        self.current_prefixes = initial_prefixes.detach().clone()

    def append_step(self, env_idx, obs, actions, rewards, episode_starts, values, log_probs, prefix):
        # We store the data for the CURRENT active retry
        if episode_starts:
            self._start_new_retry(env_idx)
            
        step_data = {
            'obs': obs, 'actions': actions, 'rewards': rewards,
            'episode_starts': episode_starts, 'values': values, 
            'log_probs': log_probs, 'prefix': prefix
        }
        self.history[env_idx][-1].append(step_data)

    def _start_new_retry(self, env_idx):
        self.history[env_idx].append([])

    def get_first_attempt(self, env_idx):
        return self._format_traj(self.history[env_idx][0])

    def get_last_attempt(self, env_idx):
        return self._format_traj(self.history[env_idx][-1])

    def _format_traj(self, list_of_dicts):
        """Converts list of step dicts into a dict of numpy arrays."""
        my_return_dict = {}
        for k in list_of_dicts[0].keys():
            temporary_list = []
            for step in list_of_dicts:
                temporary_list.append(step[k])
            if isinstance(temporary_list[0], th.Tensor):
                my_return_dict[k] = th.Tensor(temporary_list)
            else:
                my_return_dict[k] = np.array(temporary_list)
        return my_return_dict

    def reset_env(self, env_idx):
        self.history[env_idx] = []
