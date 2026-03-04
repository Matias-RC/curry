from typing import Optional, Union, Tuple, NamedTuple, Generator, Type, Set
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer, BaseBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback, Any
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.utils import get_device
from gymnasium import spaces
import torch.nn as nn
import torch as th
import numpy as np
import warnings

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


import torch as th
import numpy as np
from torch import nn
from gymnasium import spaces
from typing import Union, Optional, Generator, Tuple, List, Dict
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples

class PrefixRolloutBuffer(RolloutBuffer, nn.Module):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        nn.Module.__init__(self)
        RolloutBuffer.__init__(self, buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

        # Learnable parameters (The memory)
        self.prefixes = nn.ParameterDict()
        self.prefix_snapshot: Dict[str, th.Tensor] = {}

        # Parallel state tracking
        self.instance_ids = np.empty((self.buffer_size, self.n_envs), dtype=object)
        
        # Pre-computed trajectory metadata (built in compute_returns_and_advantage)
        self.trajectory_map: List[Dict] = []
        self.max_traj_len = 0

    def reset(self) -> None:
        super().reset()
        self.instance_ids = np.empty((self.buffer_size, self.n_envs), dtype=object)
        self.trajectory_map = []
        self.max_traj_len = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        instance_ids: List[str] # Added this argument
    ) -> None:
        # Standard SB3 logic
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        
        # Custom logic: Store the instance IDs for this step
        self.instance_ids[self.pos] = np.array(instance_ids)
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        GAE Calculation + Trajectory Mapping.
        Reconstructs episodes per-env to avoid parsing during the training phase.
        """
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = np.zeros(self.n_envs)
        
        # Temporary storage to build trajectories for each env
        env_trajectories = [[] for _ in range(self.n_envs)]
        self.trajectory_map = []
        self.max_traj_len = 0

        # Reversed loop for GAE
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

            # --- Trajectory Mapping (Forward Logic in a Reverse Loop) ---
            # We record indices. Since we are going backwards, we insert at the start.
            for env_idx in range(self.n_envs):
                env_trajectories[env_idx].insert(0, step)
                
                # If this step was the START of an episode (or the start of the buffer), 
                # we "seal" the trajectory segment.
                if self.episode_starts[step, env_idx] == 1.0:
                    self._seal_trajectory(env_trajectories, env_idx)

        # Final seal for segments that reached the end of the buffer without a new episode start
        for env_idx in range(self.n_envs):
            self._seal_trajectory(env_trajectories, env_idx)

        self.returns = self.advantages + self.values

    def _seal_trajectory(self, env_trajectories, env_idx):
        if len(env_trajectories[env_idx]) > 0:
            indices = env_trajectories[env_idx]
            # Map the instance ID (taken from the first step of this segment)
            inst_id = self.instance_ids[indices[0], env_idx]
            
            self.trajectory_map.append({
                "id": str(inst_id),
                "indices": indices, # Indices in (buffer_size, n_envs)
                "env_idx": env_idx
            })
            self.max_traj_len = max(self.max_traj_len, len(indices))
            env_trajectories[env_idx] = []

    def save_prefix_snapshot(self):
        """Phase 2 Setup: Stores detached copies of current prefixes."""
        self.prefix_snapshot = {k: v.detach().clone() for k, v in self.prefixes.items()}

    def get(self, batch_size: Optional[int] = None) -> Generator[Tuple[RolloutBufferSamples, th.Tensor], None, None]:
        """Phase 1: Yields (samples, learnable_prefixes)."""
        assert self.full, "Buffer must be full."
        
        if not self.generator_ready:
            # Flatten everything including our IDs
            _tensor_names = ["observations", "actions", "values", "log_probs", "advantages", "returns"]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.instance_ids = self.swap_and_flatten(self.instance_ids)
            self.generator_ready = True

        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            
            # 1. Standard Samples
            samples = self._get_samples(batch_inds)
            
            # 2. Extract learnable prefix parameters for these specific steps
            batch_ids = self.instance_ids[batch_inds]
            batch_prefixes = th.stack([self.prefixes[str(uid)] for uid in batch_ids])
            
            yield samples, batch_prefixes
            start_idx += batch_size
            
    def get_trajectory_batches(self, batch_size: int) -> Generator[Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor], None, None]:
        """Phase 2: Yields (traj_obs, traj_mask, orig_prefixes, opt_prefixes)."""
        # Shuffle our pre-computed map
        indices = np.random.permutation(len(self.trajectory_map))
        
        for start_idx in range(0, len(indices), batch_size):
            batch_meta_inds = indices[start_idx : start_idx + batch_size]
            
            obs_list, mask_list, orig_list, opt_list = [], [], [], []
            
            for m_idx in batch_meta_inds:
                meta = self.trajectory_map[m_idx]
                inst_id = meta["id"]
                env_idx = meta["env_idx"]
                step_indices = meta["indices"]
                
                # observations is still (buffer_size, n_envs, ...) here 
                # because we use the meta indices before flattening or from the raw array
                traj_obs = th.as_tensor(self.observations[step_indices, env_idx], device=self.device)
                
                # Padding & Masking
                seq_len = len(step_indices)
                pad_len = self.max_traj_len - seq_len
                mask = th.zeros(self.max_traj_len, dtype=th.bool, device=self.device)
                
                if pad_len > 0:
                    padding = th.zeros((pad_len, *self.obs_shape), device=self.device)
                    traj_obs = th.cat([traj_obs, padding], dim=0)
                    mask[-pad_len:] = True
                
                obs_list.append(traj_obs)
                mask_list.append(mask)
                orig_list.append(self.prefix_snapshot[inst_id])
                opt_list.append(self.prefixes[inst_id])

            yield (
                th.stack(obs_list), 
                th.stack(mask_list), 
                th.stack(orig_list), 
                th.stack(opt_list)
            )

class PRBSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    indices: th.Tensor

class PRBSupervisionSamples(NamedTuple):
    inputs: th.Tensor  # (B, T, *Obs_Shape)
    masks: th.Tensor   # (B, T) - Padding mask for the temporal transformer
    targets: th.Tensor # (B, *Prefix_Shape) - The "ground truth" prefix reached


class PrefixRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer used in EPPO algorithm. 
    Experiences are discarded after policy update.
    Prefixes are kept if not explicitly shown to "forget" method.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advatages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    prefix_register: Dict[str, int]
    forget_registry: Set[str]

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            prefix_shape: Tuple[int, int],
            prefix_lr: int,
            prefix_optimizer: Type[nn.Module] = th.optim.SGD,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        if prefix_optimizer != th.optim.SGD:
            # Warn the user
            warnings.warn("Using optimizers other than SGD may lead to unexpected behavior.\
                           Ensure that optimizer state is properly managed across iterations.\
                           And coinsider sticking to optimizers without adaptive states like\
                           momentum or Adam.")
            
        self.prefix_shape = prefix_shape
        self.prefixes = th.empty((0, *self.prefix_shape))
        self.optimizer_class = prefix_optimizer

        self.prefix_register = {}
        self.forget_registry = set()
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs
        )


    def incorporate(self, vector):
        # Detach and clone to ensure no graph history is preserved
        vector = vector.detach().view(1, *self.prefix_shape)
        if self.prefixes.shape[0] == 0:
            self.prefixes = vector.clone()
        else:
            self.prefixes = th.cat([self.prefixes.data, vector], dim=0)

    def erase(self, index):
        if self.prefixes.shape[0] == 0: return
        indices = th.arange(self.prefixes.shape[0], device=self.prefixes.device)
        self.prefixes = self.prefixes.data[indices != index].clone()

    def reset(self) -> None:
        super().reset()
        self.indices = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

    def set_prefix(self, prefix, _id):
        self.incorporate(prefix)
        idx = self.prefixes.shape[0]-1
        self.prefix_register[_id] = idx

    def get_prefix(self, _id):
        # Go from id of level to idx in tensor.
        idx = self.prefix_register.get(_id)
        if idx is None:
            return None
        return self.prefixes[idx]
        
    def set_forget(self, _id):
        self.forget_registry.add(_id)
    
    def prepare_prefix_optimizer(self, learning_rate, device):
        self.prefixes = nn.Parameter(self.prefixes)
        optimizer = self.optimizer_class([self.prefixes], lr=learning_rate)
        self.prefixes.to(device)
        return optimizer
    def get(self, batch_size: Optional[int] = None) -> Generator[PRBSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
            

        
    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> PRBSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.indices[batch_inds].flatten()
        )
        return PRBSamples(*tuple(map(self.to_torch, data)))

    def generate_supervision_buffer(self):
        """
        Unlike get() which can be affordable.
        This method has to itrate through the buffer before even yielding.
        This is why we load in ram the batched form of trajectories before training.
        This way it is only called once per training iteration, not once per epoch.
        Has to be called before get() such that trajectories are in original order.
        """
        _tensor_names = [
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
            "episode_starts",
            "indices"
        ]
        for tensor in _tensor_names:
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])

        trajectories = []
        trajectory = []
        max_traj_len = 0
        current_traj_len = 0
        # Iterate
        for idx in range(self.n_envs*self.buffer_size):
            current_traj_len += 1
            trajectory.append({
                "obs": self.observations[idx],
                "index": self.indices[idx]
            })
            if self.episode_starts[idx] == 1.0:
                trajectories.append(trajectory)
                trajectory = []
                max_traj_len = max(max_traj_len, current_traj_len)
                current_traj_len = 0
        # Ensure that each trayectory is padded with valid observations and make masks
        padding_masks = []
        for item in trajectories:
            #repeat the last item until max len is reached
            mask_len = 0
            while len(item) < max_traj_len:
                item.append(item[-1])
                mask_len += 1
            padding_masks.append([False]*(max_traj_len-mask_len) + [True]*mask_len)

        # Make tensors
        tensor_inputs = []
        tensor_masks = []
        tensor_targets = []
        
        for item, mask in zip(trajectories, padding_masks):
            # Note: Wrap in th.as_tensor() if i["obs"] is a numpy array, 
            # since th.stack expects a list of tensors.
            tensor_inputs.append(th.stack([th.as_tensor(i["obs"]) for i in item]))
            tensor_masks.append(th.tensor(mask))
            tensor_targets.append(self.prefixes[item[0]["index"]])
            
        return PRBSupervisionSamples(
            inputs=th.stack(tensor_inputs),
            masks=th.stack(tensor_masks),
            targets=th.stack(tensor_targets)
        )


    def get_trajectory_batches(self, batch_size: int, supervision_samples: PRBSupervisionSamples) -> Generator[Tuple[th.Tensor, th.Tensor, th.Tensor], None, None]:
        """Yields (traj_obs, traj_mask, orig_prefixes)."""
        indices = np.random.permutation(len(supervision_samples.inputs))
        
        for start_idx in range(0, len(indices), batch_size):
            batch_inds = indices[start_idx : start_idx + batch_size]
            
            yield PRBSupervisionSamples(
                inputs=supervision_samples.inputs[batch_inds], 
                masks=supervision_samples.masks[batch_inds], 
                targets=supervision_samples.targets[batch_inds].detach()
            )
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        instance_ids: List[str]
    ) -> None:
        index = []
        for _id in instance_ids:
            index.append(self.prefix_register[_id])
        self.indices[self.pos] = np.array(index)
        super().add(
            obs,
            action,
            reward,
            episode_start,
            value,
            log_prob
        )




    def clean(self):
        """
        This method removes prefixes that won't be used anymore.
        """
        forget_indices = sorted(
            [self.prefix_register[_id] for _id in self.forget_registry if _id in self.prefix_register],
            reverse=True
        )

        for idx in forget_indices:
            self.erase(idx)

        temporary_register = [
            (_id, idx) for _id, idx in self.prefix_register.items() if _id not in self.forget_registry
        ]

        current_idx = 0
        for _id, idx in temporary_register:
            delta = sum(1 for fidx in forget_indices if fidx < idx)
            new_idx = idx - delta
            self.prefix_register[_id] = new_idx
            current_idx += 1
        
        self.forget_registry.clear()