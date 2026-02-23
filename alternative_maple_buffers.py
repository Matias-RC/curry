from typing import Optional, Union, Tuple, NamedTuple, Generator
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
import torch.nn as nn
import torch as th
import numpy as np

class MapleRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    actor_prefixes: th.Tensor   
    critic_prefixes: th.Tensor 

class MapleRolloutBuffer(RolloutBuffer):
    """
    Modified rollout buffer that forces n_envs=1 and adds support for adding entire trajectories.
    It also makes sure the size of the buffer matches correctly with the size of a natural buffer.
    """
    actor_prefixes: np.ndarray
    critic_prefixes: np.ndarray
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
        self.prefix_size = prefix_size
        super().__init__(
            buffer_size*n_envs,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs=1,  # Force n_envs=1, ignoring passed value
        )
        

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.actor_prefixes = np.zeros((self.buffer_size, self.n_envs, self.prefix_size[0], *self.obs_shape[1:]))
        self.critic_prefixes = np.zeros((self.buffer_size, self.n_envs, self.prefix_size[1], *self.obs_shape[1:]))
        self.generator_ready = False
        super().reset()

    def expand_by(self, delta: int) -> None:
        def grow_array(arr):
            padding_shape = (delta, self.n_envs) + arr.shape[2:]
            padding = np.zeros(padding_shape, dtype=arr.dtype)
            return np.concatenate((arr, padding), axis=0)

        self.observations = grow_array(self.observations)
        self.actions = grow_array(self.actions)
        self.rewards = grow_array(self.rewards)
        self.returns = grow_array(self.returns)
        self.episode_starts = grow_array(self.episode_starts)
        self.values = grow_array(self.values)
        self.log_probs = grow_array(self.log_probs)
        self.advantages = grow_array(self.advantages)

        self.actor_prefixes = grow_array(self.actor_prefixes)
        self.critic_prefixes = grow_array(self.critic_prefixes)


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
        actor_prefix: th.Tensor = traj_obj['actor_prefix']
        critic_prefix: th.Tensor = traj_obj['critic_prefix']
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

        end = self.pos + traj_len

        if end >= self.buffer_size:
            delta = end + 1 - self.buffer_size
            self.expand_by(delta)

        obs = obs.reshape((traj_len, self.n_envs) + self.obs_shape)    
        
        actor_prefix = actor_prefix.clone().cpu().numpy().reshape((traj_len, self.n_envs)+(self.prefix_size[0], self.obs_shape[1],self.obs_shape[2]))
        critic_prefix = critic_prefix.clone().cpu().numpy().reshape((traj_len, self.n_envs)+(self.prefix_size[1], self.obs_shape[1],self.obs_shape[2]))

        # Reshape to handle multi-dim and discrete action spaces
        action = action.reshape((traj_len, self.n_envs, self.action_dim))

        # Convert tensors to numpy and reshape
        values = value.clone().cpu().numpy().reshape((traj_len, self.n_envs))
        log_probs = log_prob.clone().cpu().numpy().reshape((traj_len, self.n_envs))
        rewards = reward.reshape((traj_len, self.n_envs))
        episode_starts = episode_start.reshape((traj_len, self.n_envs))

        # Assign to buffer slices
        self.observations[self.pos:end] = np.array(obs)
        self.actions[self.pos:end] = np.array(action)
        self.rewards[self.pos:end] = rewards
        self.episode_starts[self.pos:end] = episode_starts
        self.values[self.pos:end] = values
        self.log_probs[self.pos:end] = log_probs
        self.actor_prefixes[self.pos:end] = actor_prefix
        self.critic_prefixes[self.pos:end] = critic_prefix

        self.pos += traj_len
        if self.pos >= self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[MapleRolloutBufferSamples, None, None]:
            assert self.full, ""
            indices = np.random.permutation(self.buffer_size * self.n_envs)
            
            # Prepare the data
            if not self.generator_ready:
                # We add your custom prefix buffers to this list so they get 
                # swapped (time, env) -> (env, time) and flattened -> (batch_size, ...)
                _tensor_names = [
                    "observations",
                    "actions",
                    "values",
                    "log_probs",
                    "advantages",
                    "returns",
                    "actor_prefixes",   # <--- ADDED
                    "critic_prefixes",  # <--- ADDED
                ]

                for tensor in _tensor_names:
                    # This reshapes the storage from (n_steps, n_envs, ...) to (n_steps * n_envs, ...)
                    self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
                self.generator_ready = True

            # Return everything, don't create minibatches
            if batch_size is None:
                batch_size = self.buffer_size * self.n_envs

            start_idx = 0
            while start_idx < self.buffer_size * self.n_envs:
                yield self._get_samples(indices[start_idx : start_idx + batch_size])
                start_idx += batch_size
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MapleRolloutBufferSamples:
            data = (
                self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten(),
                self.actor_prefixes[batch_inds],   # <--- Retrieve specific batch of prefixes
                self.critic_prefixes[batch_inds],  # <--- Retrieve specific batch of prefixes
            )
            
            # Convert to torch tensors
            return MapleRolloutBufferSamples(*tuple(map(self.to_torch, data)))



class DynamicReplayBuffer:
    current_prefixes:Tuple[th.Tensor, th.Tensor]
    def __init__(self, n_envs: int, device: str = "cpu"):
        self.n_envs = n_envs
        self.device = device
        # Storage: List of lists (one list per env containing its retry history)
        self.history = [[[],] for _ in range(n_envs)]
        # This stores the optimized prefix currently being used for each env
        self.current_prefixes = None 

    def initialize_self(self, initial_prefixes: Tuple[th.Tensor, th.Tensor]):
            # Store as a tuple of tensors: ( (n_envs, ...), (n_envs, ...) )
            self.current_prefixes = (
                initial_prefixes[0].detach().clone(), 
                initial_prefixes[1].detach().clone()
            )

    def set_env_prefix(self, env_idx: int, actor_p: th.Tensor, critic_p: th.Tensor):
            """Helper to update a specific environment's prefixes"""
            self.current_prefixes[0][env_idx] = actor_p
            self.current_prefixes[1][env_idx] = critic_p 

    def append_step(self, env_idx, obs, actions, rewards, episode_starts, values, log_probs):
        # We store the data for the CURRENT active retry
        #if episode_starts:
        #    self._start_new_retry(env_idx)
            
        step_data = {
            'obs': obs, 'actions': actions, 'rewards': rewards,
            'episode_starts': episode_starts, 'values': values, 
            'log_probs': log_probs,  'actor_prefix': self.current_prefixes[0], 
            'critic_prefix': self.current_prefixes[1],
        }
        self.history[env_idx][-1].append(step_data)

    def _start_new_retry(self, env_idx):
        self.history[env_idx].append([])

    def get_first_attempt(self, env_idx):
        return self._format_traj(self.history[env_idx][0])

    def get_last_attempt(self, env_idx):
        return self._format_traj(self.history[env_idx][-1])
    
    def get_all_attempts(self, env_idx):
         hist = []
         for i in self.history[env_idx]:
              hist += i
         return self._format_traj(hist)

    def _format_traj(self, list_of_dicts):
        """Converts list of step dicts into a dict of numpy arrays."""
        my_return_dict = {}
        for k in list_of_dicts[0].keys():
            temporary_list = []
            for step in list_of_dicts:
                temporary_list.append(step[k])
            if isinstance(temporary_list[0], th.Tensor):
                my_return_dict[k] = th.stack(temporary_list)
            else:
                my_return_dict[k] = np.array(temporary_list)
        return my_return_dict

    def reset_env(self, env_idx):
        self.history[env_idx] = [[],]
        self.current_prefixes[0][env_idx].zero_()
        self.current_prefixes[1][env_idx].zero_()


import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Generator, Union, List, Tuple, Dict
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from gymnasium import spaces

class PrefixRolloutBuffer(RolloutBuffer, nn.Module):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        # 1. Initialize PyTorch Module to register ParameterDict
        nn.Module.__init__(self)
        
        # 2. Initialize SB3 Base Buffer
        RolloutBuffer.__init__(
            self, 
            buffer_size, 
            observation_space, 
            action_space, 
            device, 
            gae_lambda=gae_lambda, 
            gamma=gamma, 
            n_envs=n_envs
        )
        
        # --- Custom Architectural State Variables ---
        self.prefixes = nn.ParameterDict()
        self.trajectories: List[Tuple[str, List[np.ndarray]]] = []
        self.max_trajectory_size: int = 0
        self.current_prefix = [None for _ in range(self.n_envs)]
        
        # Parallel array to track instance_ids alongside standard SB3 variables
        self.instance_ids = np.empty((self.buffer_size, self.n_envs), dtype=object)

    def reset(self) -> None:
        """Overrides reset to clear custom trajectory tracking."""
        super().reset()
        self.instance_ids = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.trajectories = []
        self.max_trajectory_size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        instance_ids: Optional[List[str]] = None
    ) -> None:
        """Overrides standard add to inject instance_id tracking."""
        # Capture buffer position before super().add() increments it
        idx = self.pos 
        
        super().add(obs, action, reward, episode_start, value, log_prob)
        
        # Track the environment IDs for this specific step
        if instance_ids is not None:
            self.instance_ids[idx] = np.array(instance_ids)

    def add_to_prefix_tensor(self, instance_id: str, candidate: Optional[torch.Tensor] = None) -> None:
        """Registers a new prefix memory vector for a previously unseen level."""
        if instance_id not in self.prefixes:
            if candidate is None:
                raise ValueError("A candidate tensor must be provided for new prefixes.")
            
            # .clone().detach() ensures it acts as a leaf node, preventing graph history bleed.
            param = nn.Parameter(candidate.clone().detach().to(self.device).requires_grad_(True))
            self.prefixes[instance_id] = param

    def flush_dict(self, ids_to_be_flushed: List[str], prefix_optimizer: torch.optim.Optimizer) -> None:
        """
        Safely destroys memory vectors for forgotten levels and repairs the optimizer.
        """
        for inst_id in ids_to_be_flushed:
            if inst_id in self.prefixes:
                del self.prefixes[inst_id]
                
        # Rebuild optimizer param groups. This prevents PyTorch from crashing 
        # when trying to backpropagate into a deleted memory address.
        if len(prefix_optimizer.param_groups) > 0:
            prefix_optimizer.param_groups[0]['params'] = list(self.prefixes.values())

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        """Phase 1: Standard PPO Yield."""
        if not self.generator_ready:
            # SB3 will flatten standard tensors via super().get(). 
            # We must manually flatten our parallel ID array so it stays in sync.
            self.instance_ids = self.swap_and_flatten(self.instance_ids)
            
        yield from super().get(batch_size)

    def get_trajectories(self, batch_size: int) -> Generator[Tuple[List[str], torch.Tensor], None, None]:
        """
        Phase 2: Meta-Training Yield. 
        Groups contiguous rollout chunks by episode_starts, pads them, and yields batches.
        """
        assert self.full, "Buffer must be full to extract complete trajectories."
        
        # Ensure arrays are flattened (in case Phase 1 wasn't run)
        if not self.generator_ready:
            self.instance_ids = self.swap_and_flatten(self.instance_ids)
            _tensor_names = ["observations", "actions", "values", "log_probs", "advantages", "returns", "episode_starts"]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # 1. Parse flattened buffer into discrete trajectories via episode_starts
        flat_obs = self.observations
        flat_starts = self.episode_starts
        flat_ids = self.instance_ids
        
        self.trajectories = [] # Reset for this extraction
        self.max_trajectory_size = 0
        
        current_traj = []
        current_id = None
        
        for i in range(len(flat_obs)):
            # If an episode starts and we have existing data, seal the previous trajectory
            if flat_starts[i] == 1.0 and len(current_traj) > 0:
                self.trajectories.append((current_id, current_traj))
                self.max_trajectory_size = max(self.max_trajectory_size, len(current_traj))
                current_traj = []
                
            current_id = flat_ids[i]
            current_traj.append(flat_obs[i])
            
        # Seal the final trajectory at the end of the buffer
        if len(current_traj) > 0:
            self.trajectories.append((current_id, current_traj))
            self.max_trajectory_size = max(self.max_trajectory_size, len(current_traj))

        # 2. Shuffle and Yield Padded Batches
        indices = np.random.permutation(len(self.trajectories))
        
        for start_idx in range(0, len(indices), batch_size):
            batch_inds = indices[start_idx : start_idx + batch_size]
            
            batch_ids = []
            batch_tensors = []
            
            for idx in batch_inds:
                inst_id, traj_steps = self.trajectories[idx]
                batch_ids.append(inst_id)
                
                # Convert list of observation arrays to a PyTorch Tensor
                traj_tensor = torch.as_tensor(np.array(traj_steps), dtype=torch.float32, device=self.device)
                
                # Apply Zero-Padding if trajectory is shorter than max length
                pad_len = self.max_trajectory_size - len(traj_steps)
                if pad_len > 0:
                    padding = torch.zeros((pad_len, *self.obs_shape), dtype=torch.float32, device=self.device)
                    traj_tensor = torch.cat([traj_tensor, padding], dim=0)
                    
                batch_tensors.append(traj_tensor)
                
            # Yields: (List of Instance IDs length Batch), (Tensor shape: Batch x Max_Traj_Len x *obs_shape)
            yield batch_ids, torch.stack(batch_tensors)