import torch as th
from pathlib import Path
import gymnasium as gym

from gym_sokoban.envs import SokobanEnv
from stable_baselines3.common.monitor import Monitor

from sokoban_wrapper import (
    SokobanCompactWrapper,
    SokobanRetriesWrapper,
)

from gymnasium import spaces

import time
import numpy as np
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

import pygame.surfarray as surfarray
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from consolidator_class import ChannelStackedSpatialAccumulation, Consolidator
from alternative_maple_policy import MaplePolicy, PrefixCombinator, ConvFeatureExtractor
from alternative_maple import Maple
from alternative_maple_buffers import MapleRolloutBuffer, DynamicReplayBuffer
from alternative_maple_callback import MapleCallback

import time

# ============================================================
# Config (MUST MATCH TRAINING)
# ============================================================
SEED = 67
DIM_ROOM = (10, 10)
MAX_STEPS = 30
NUM_BOXES = 1

num_beam_search_steps = 10
beam_size = 64

NUM_RETRIES = num_beam_search_steps*beam_size # This way we dont face retries problems

HIDDEN_SIZE_CHANNELS = 64
POOL_SHAPE = 4

RUN_ID = 0  # <-- change this to the run you want to load
BASE_DIR = Path("./models") / str(RUN_ID)

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

CONSOLIDATOR_LR=1e-3


# ============================================================
# Environment builder
# ============================================================
def make_env(seed):
    env = SokobanRetriesWrapper(
        max_retries=NUM_RETRIES,
        dim_room=DIM_ROOM,
        max_steps=MAX_STEPS,
        num_boxes=NUM_BOXES,
        render_mode="rgb_array",
    )
    env = SokobanCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ============================================================
# Rebuild model (same as training)
# ============================================================
def build_model(env):
    config = {
        "consolidator":{
            "net_arch":{
                "n_shared_layers":2,
                "n_policy_layers":2,
                "n_vf_layers":2,
                "shared_class":[ChannelStackedSpatialAccumulation, ChannelStackedSpatialAccumulation],
                "shared_kwargs":[
                    {
                        "C_in":4,
                        "C_out":512
                    },
                    {
                        "C_in": 32,  
                        "C_out": 128
                    }
                ],

                "policy_layers":[nn.Conv2d, nn.Conv2d],
                "policy_kwargs":[
                    dict(in_channels=128, out_channels=64, kernel_size=3,stride=1, padding=1),
                    dict(in_channels=64, out_channels=HIDDEN_SIZE_CHANNELS, kernel_size=3,stride=1, padding=1)
                ],
                "vf_layers":[nn.Conv2d, nn.Conv2d],
                "vf_kwargs":[
                    dict(in_channels=128, out_channels=64, kernel_size=3,stride=1, padding=1),
                    dict(in_channels=64, out_channels=HIDDEN_SIZE_CHANNELS, kernel_size=3,stride=1, padding=1)
                ]
            },
            "actor_prefix_channels":HIDDEN_SIZE_CHANNELS,
            "critic_prefix_channels":HIDDEN_SIZE_CHANNELS,
            "lr_schedule":lambda _: CONSOLIDATOR_LR
        },
        "policy":{
            "net_arch":{
                "pi": [512, 256], 
                "vf": [512, 256]
            },
            "features_extractor_class":ConvFeatureExtractor,
            "features_extractor_kwargs":{
                "config": {
                    "conv_configs": [
                        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                        {'out_channels': HIDDEN_SIZE_CHANNELS, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                    ],
                    "features_dim":HIDDEN_SIZE_CHANNELS*POOL_SHAPE*POOL_SHAPE
                }
            },
            "prefix_combinator_class":PrefixCombinator,
            "prefix_combinator_kwargs":{
                "policy":{
                    "in_channels":2*HIDDEN_SIZE_CHANNELS,
                    "hidden_channels":HIDDEN_SIZE_CHANNELS,
                    "out_channels":HIDDEN_SIZE_CHANNELS,
                    "n_layers":2,
                    "out_shape":(POOL_SHAPE,POOL_SHAPE)
                },
                "vf":{
                    "in_channels":2*HIDDEN_SIZE_CHANNELS,
                    "hidden_channels":HIDDEN_SIZE_CHANNELS,
                    "out_channels":HIDDEN_SIZE_CHANNELS,
                    "n_layers":2,
                    "out_shape":(POOL_SHAPE,POOL_SHAPE)  
                }
            }
        }
    }

    model = Maple(
        policy=MaplePolicy,
        env=env,
        policy_kwargs=config["policy"],
        verbose=1,
        seed=SEED,
        learning_rate=1e-3,
        rollout_buffer_class=MapleRolloutBuffer,
        rollout_buffer_kwargs={
            "prefix_size": (HIDDEN_SIZE_CHANNELS, HIDDEN_SIZE_CHANNELS)
        },
        dynamic_buffer_class=DynamicReplayBuffer,
        dynamic_buffer_kwargs={"n_envs": 1},
        consolidator_class=Consolidator,
        consolidator_kwargs=config["consolidator"],
        num_retries=NUM_RETRIES,
        device=DEVICE
    )
    return model


# ============================================================
# Load state dicts
# ============================================================
def load_checkpoint(model, run_dir: Path):
    print(f"[MAPLE] Loading checkpoint from {run_dir}")

    # ---- Policy ----
    policy_sd = th.load(
        run_dir / "policy" / "policy_state_dict.pt",
        map_location=DEVICE,
    )
    model.policy.load_state_dict(policy_sd)

    opt_path = run_dir / "policy" / "policy_optimizer_state_dict.pt"
    if opt_path.exists() and hasattr(model.policy, "optimizer"):
        model.policy.optimizer.load_state_dict(
            th.load(opt_path, map_location=DEVICE)
        )

    # ---- Consolidator ----
    if hasattr(model, "consolidator"):
        cons_sd = th.load(
            run_dir / "consolidator" / "consolidator_state_dict.pt",
            map_location=DEVICE,
        )
        model.consolidator.load_state_dict(cons_sd)

        opt_path = run_dir / "consolidator" / "consolidator_optimizer_state_dict.pt"
        if opt_path.exists():
            model.consolidator.optimizer.load_state_dict(
                th.load(opt_path, map_location=DEVICE)
            )

    model.policy.to(DEVICE)
    model.policy.eval()
    model.consolidator.to(DEVICE)
    model.consolidator.eval()

    print("[MAPLE] Checkpoint loaded successfully")


# ============================================================
# Sanity test rollout
# ============================================================

"""
For succesfull beam search implementation I have to make the scripts flush to finish upon truncation of retries
also craft a dynamic buffer that keeps multiple env instances but understood as their own.
also currently the dynamic buffer expects parallel steps but for simplicyti these  steps will all be taken
sequentially.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Constants provided
INNER_EPOCHS = 5
BATCH_SIZE = 20
PREFIX_LR = 0.008
MAX_GRAD_NORM = 0.5 

class Prefix(nn.Module):
    def __init__(self, n_envs: int, initial_prefixes=None, prefix_shape=None, device='cpu', lr: float = 1e-3):
        """
        Args:
            initial_prefixes: Tuple (Actor_Tensor, Critic_Tensor). 
                              Shape of each: (n_envs, C, H, W).
                              If provided, these values are used as the starting point.
            prefix_shape: Used only if initial_prefixes is None (initializes to zeros).
        """
        super().__init__()
        self.n_envs = n_envs
        self.device = device
        
        self.params = nn.ParameterList()
        
        if initial_prefixes is not None:
            # Init from provided tensors (Consolidator output)
            init_actor, init_critic = initial_prefixes
            for i in range(n_envs):
                # Stack actor/critic for this specific env: Shape (2, C, H, W)
                # Clone and detach to make it a leaf variable for the new optimizer
                env_param = th.stack([init_actor[i], init_critic[i]], dim=0).clone().detach()
                self.params.append(nn.Parameter(env_param.to(device)))
        else:
            # Init from zeros (Fallback)
            for _ in range(n_envs):
                self.params.append(nn.Parameter(th.zeros((2, *prefix_shape), device=device)))
        
        # Independent Optimizer for each environment
        self.optimizers = [
            th.optim.Adam([p], lr=lr) for p in self.params
        ]

    def get_prefix_for_optim(self, idx: int):
        return self.params[idx]

    def get_current_batch_prefixes(self):
        stacked = th.stack(list(self.params), dim=0)
        return stacked[:, 0], stacked[:, 1]

    def optimize_step(self, idx: int, loss: th.Tensor, max_grad_norm: float = None):
        opt = self.optimizers[idx]
        opt.zero_grad()
        loss.backward()
        if max_grad_norm:
            th.nn.utils.clip_grad_norm_([self.params[idx]], max_grad_norm)
        opt.step()

def compute_returns_and_advantages(rewards, values, episode_starts, gamma=0.99, gae_lambda=0.95):
    """
    Computes GAE (Generalized Advantage Estimation) while respecting episode boundaries.
    Args:
        rewards: Tensor (T,)
        values: Tensor (T,)
        episode_starts: Tensor (T,) - Boolean or 0/1, where 1 indicates the start of a NEW episode.
    """
    returns = th.zeros_like(rewards)
    advantages = th.zeros_like(rewards)
    
    last_gae_lam = 0
    next_value = 0 
    
    # Iterate backwards through the trajectory
    for t in reversed(range(len(rewards))):
        # If t+1 is the start of a new episode, then step t was terminal.
        # We should NOT look at values[t+1] (mask it out).
        if t == len(rewards) - 1:
            next_non_terminal = 0.0 # Assume trajectory ends cleanly or bootstrap is handled externally
        else:
            # If the NEXT step starts a new episode, current step is terminal
            next_non_terminal = 1.0 - episode_starts[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        advantages[t] = last_gae_lam
        returns[t] = advantages[t] + values[t]
        
        next_value = values[t]
        
    return returns, advantages

def update_prefixes(
    dynamic_buffer, 
    idx: int, 
    prefix_module: Prefix, 
    model
):
    """
    Calculates loss from history and calls prefix_module.optimize_step().
    """
    
    # 1. Retrieve Data
    traj = dynamic_buffer.get_all_attempts(idx)
    if len(traj) == 0: return

    # Helper to convert/cast
    def to_tensor(x):
        if isinstance(x, np.ndarray): x = th.from_numpy(x)
        return x.to(device=model.device, dtype=th.float32)

    obs = to_tensor(traj['obs'])
    actions = to_tensor(traj['actions'])
    rewards = to_tensor(traj['rewards'])
    old_values = to_tensor(traj['values'])
    episode_starts = to_tensor(traj['episode_starts'])

    # 2. Compute Targets (GAE)
    with th.no_grad():
        returns, advantages = compute_returns_and_advantages(
            rewards, old_values, episode_starts,
            gamma=model.gamma, gae_lambda=model.gae_lambda
        )
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 3. Create Loader
    dataset = TensorDataset(obs, actions, returns, advantages)
    curr_batch_size = min(BATCH_SIZE, len(obs))
    dataloader = DataLoader(dataset, batch_size=curr_batch_size, shuffle=True)

    # 4. Optimization Loop
    model.policy.set_training_mode(True)
    
    # We get the parameter ONCE to expand it for batching inside the loop
    # (Note: we don't detach here because we need the graph for the optimizer step)
    target_param = prefix_module.get_prefix_for_optim(idx) 

    for _ in range(INNER_EPOCHS):
        for batch_obs, batch_actions, batch_returns, batch_adv in dataloader:
            
            # Expand for Batch: The policy needs (Batch_Size, ...)
            # target_param is (2, C, H, W) -> need (Batch, C, H, W)
            T_batch = batch_obs.shape[0]
            
            # [0] is Actor, [1] is Critic
            a_prefix_exp = target_param[0].unsqueeze(0).expand(T_batch, *target_param[0].shape)
            c_prefix_exp = target_param[1].unsqueeze(0).expand(T_batch, *target_param[1].shape)

            # Forward pass
            values, log_probs, _ = model.policy.evaluate_actions(
                batch_obs, 
                batch_actions, 
                (a_prefix_exp, c_prefix_exp)
            )
            
            # Loss Calculation
            values = values.flatten()
            v_loss = 0.5 * F.mse_loss(values, batch_returns)
            p_loss = -(log_probs * batch_adv).mean()
            total_loss = p_loss + v_loss
            
            # 5. Delegate Update to Module
            # This handles zero_grad, backward, clip, and step internally
            prefix_module.optimize_step(idx, total_loss, max_grad_norm=MAX_GRAD_NORM)
            
    # 6. Update Buffer Pointers
    # Since prefix_module manages the "Master" params, we just ensure 
    # the dynamic buffer points to the updated tensors for the next rollout.
    # We re-fetch the batch to get the updated values.
    current_a_batch, current_c_batch = prefix_module.get_current_batch_prefixes()

    dynamic_buffer.reset_env(env_idx=idx)
    
    # Update the legacy buffer format
    with th.no_grad():
        dynamic_buffer.current_prefixes[0][idx] = current_a_batch[idx].detach()
        dynamic_buffer.current_prefixes[1][idx] = current_c_batch[idx].detach()



def flush_beamer(dynamic_buffer, contents, past_infos, model: Maple, prefix_module: Prefix, beam_stats: dict):
    """
    Returns: The updated (or newly created) prefix_module.
    """
    obs, actions, rewards, dones, values, log_probs = contents
    
    # --- 1. Store Step ---
    for i in range(model.n_envs):
        dynamic_buffer.append_step(
            i, 
            obs[i], 
            actions[i], 
            rewards[i], 
            model._last_episode_starts[i], 
            values[i], 
            log_probs[i]
        )

    # --- 2. Check Logic ---
    for idx, info in enumerate(past_infos):
        if not dones[idx]:
            continue

        # --- A. Calculate Episode Return ---
        # Get the trajectory of the attempt that just finished
        traj_last = dynamic_buffer.get_last_attempt(idx)
        episode_return = np.sum(traj_last["rewards"])
        
        # Add to beam statistics
        beam_stats[idx]["sum_rewards"] += episode_return
        beam_stats[idx]["count"] += 1

        retry_count = info.get("retry_count", 0)

        # --- B. Handle Logic ---
        
        # Case 1: First Failure -> Initialize Prefix
        if retry_count == 0:
            print(f"[Env {idx}] First failure. Running Consolidator...")
            with th.no_grad():
                # Input shape: (Batch, T, C, H, W) -> unsqueeze batch dim
                obs_input = obs_as_tensor(traj_last["obs"], model.device).unsqueeze(0)
                gen_a, gen_c = model.consolidator(obs_input) 
                
            if prefix_module is None:
                prefix_module = Prefix(
                    n_envs=model.n_envs,
                    initial_prefixes=(gen_a, gen_c),
                    device=model.device,
                    lr=PREFIX_LR
                )
            
            with th.no_grad():
                dynamic_buffer.current_prefixes[0][idx] = gen_a[idx].detach()
                dynamic_buffer.current_prefixes[1][idx] = gen_c[idx].detach()
                
            dynamic_buffer.reset_env(env_idx=idx)
            
            # Reset stats for the upcoming first beam block
            beam_stats[idx] = {"sum_rewards": 0.0, "count": 0}

        # Case 2: Beam Block Finished -> Optimize & Report
        elif retry_count % beam_size == 0:
            # --- Report Stats ---
            avg_reward = beam_stats[idx]["sum_rewards"] / max(beam_stats[idx]["count"], 1)
            print(f"--------------------------------------------------")
            print(f"[Env {idx}] Beam Block Completed (Retries {retry_count - beam_size} to {retry_count})")
            print(f"        Average Reward: {avg_reward:.4f}")
            print(f"--------------------------------------------------")
            
            # Reset stats for the *next* beam block
            beam_stats[idx] = {"sum_rewards": 0.0, "count": 0}

            # Optimization
            print(f"[Env {idx}] Optimizing Prefix...")
            if prefix_module is not None:
                update_prefixes(dynamic_buffer, idx, prefix_module, model)
            else:
                print("Warning: Prefix module is None during beam step.")

        # Case 3: Standard Retry
        else:
            dynamic_buffer._start_new_retry(idx)

    return prefix_module


def test_rollout(model):
    # Setup
    model._last_obs = model.env.reset()

    actor_init, critic_init = model.consolidator._init_empty(model.n_envs, model.env.observation_space.shape) 
    model.dynamic_buffer.initialize_self((actor_init, critic_init))
    
    model._last_episode_starts = np.ones((model.env.num_envs,), dtype=bool)
    model._past_infos = [{"retries_left": NUM_RETRIES, "retry_count": 0} for _ in range(model.n_envs)]

    prefix_module = None 
    
    # Force reset retry counter on inner env if needed (hacky but sometimes necessary with wrappers)
    if hasattr(model.env.envs[0], "env") and hasattr(model.env.envs[0].env, "env"):
         try: model.env.envs[0].env.env.current_retry = 0
         except: pass

    # --- Initialize Statistics Tracker ---
    # Stores {env_idx: {'sum_rewards': 0.0, 'count': 0}}
    beam_stats = {i: {"sum_rewards": 0.0, "count": 0} for i in range(model.n_envs)}

    print("Starting rollout...")

    for step in range(MAX_STEPS * NUM_RETRIES):
        
        # 1. Action
        with th.no_grad():
            obs_tensor = obs_as_tensor(model._last_obs, model.device)
            actions, values, log_probs = model.policy(obs_tensor, model.dynamic_buffer.current_prefixes)
        
        actions_np = actions.cpu().numpy()

        # 2. Step
        new_obs, rewards, dones, infos = model.env.step(actions_np)

        # 3. Flush / Update
        prefix_module = flush_beamer(
            model.dynamic_buffer, 
            (model._last_obs, actions, rewards, dones, values, log_probs),
            model._past_infos,
            model, 
            prefix_module,
            beam_stats  # <--- Pass the stats dict
        )

        # 4. Advance
        model._last_obs = new_obs
        model._past_infos = infos
        model._last_episode_starts = dones
        
        # 5. Global Stop
        if all(info.get("retries_left", 0) == 0 for info in infos) and all(dones):
            break
            
    print("[MAPLE] Rollouts passed successfully")
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    env = make_env(SEED)
    model = build_model(env)
    load_checkpoint(model, BASE_DIR)
    test_rollout(model)
