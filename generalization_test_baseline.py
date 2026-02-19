import torch as th
from pathlib import Path
import gymnasium as gym

from gym_sokoban.envs import SokobanEnv
from stable_baselines3.common.monitor import Monitor

from sokoban_wrapper import (
    SokobanCompactWrapper,
    SokobanRetriesWrapper,
    SokobanCanonicalCompactWrapper,
)

from gymnasium import spaces

import time
import numpy as np
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

import pygame.surfarray as surfarray
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from consolidator_class import ChannelStackedSpatialAccumulation, Consolidator
from alternative_maple_policy import MaplePolicy, PrefixCombinator, ConvFeatureExtractor
from stable_baselines3 import PPO
from alternative_maple_buffers import MapleRolloutBuffer, DynamicReplayBuffer
from alternative_maple_callback import MapleCallback
from train4 import BaselineComparableCNN
import time

import pygame

class  DynamicBufferForBaseline:
    def __init__(self, device:str): #Single env enforced
        self.device = device

        self.history = [[],]

        self.current_layers = None

    def append_step(self, env_idx, obs, actions, rewards, episode_starts, values, log_probs):
        step_data = {
            'obs': obs, 'actions': actions, 'rewards': rewards,
            'episode_starts': episode_starts, 'values': values, 
            'log_probs': log_probs,  'actor_layer': self.current_layers[0], 
            'critic_layer': self.current_layers[1],
        }
        self.history[env_idx][-1].append(step_data)
    
    def set_env_layer(self, env_idx, actor_l, critic_l):
        self.current_layers[0][env_idx] = actor_l
        self.current_layers[1][env_idx] = critic_l


class LastLayerInstert(nn.Module):
    def __init__(self, value_dims, policy_dims, bias=True, device="cpu"):# Not taking into coinsideration the n_envs
        self.value_weights = nn.Parameter(th.eye(value_dims[0], value_dims[1], device=device))
        self.value_bias = nn.Parameter(th.zeros(value_dims[1]), device=device)

        self.policy_weights = nn.Parameter(th.eye(policy_dims[0], policy_dims[1], device=device))
        self.policy_bias = nn.Parameter(th.zeros(policy_dims[1], device=device))
    
    

SEED = 67
DIM_ROOM = (9, 9)
MAX_STEPS = 60
NUM_BOXES = 1

num_beam_search_steps = 30
beam_size = 5

NUM_RETRIES = 1 # This way we dont face retries problems

HIDDEN_SIZE_CHANNELS = 64
POOL_SHAPE = 4

RUN_ID = 2 # <-- change this to the run you want to load
ITER = 180
BASE_DIR = Path("./models_baseline") / str(RUN_ID) / f"iter_{ITER}"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

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
    env = SokobanCanonicalCompactWrapper(env, canonical_shape=(10,10))
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def build_model(env):
    config = {
        # Same visual backbone config as MAPLE
        "features_extractor_kwargs": {
            "config": {
                "conv_configs": [
                    {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                    {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                    {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                    {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                    {'out_channels': HIDDEN_SIZE_CHANNELS, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                ]
            },
            # We pass the combinator settings here so the Baseline class can replicate the depth
            "combinator_kwargs": {
                "hidden_channels": HIDDEN_SIZE_CHANNELS,
                "out_channels": HIDDEN_SIZE_CHANNELS,
                "n_layers": 2, 
                "out_shape": (POOL_SHAPE, POOL_SHAPE)
            }
        },
        # Same MLP Head config as MAPLE
        "net_arch": {
            "pi": [512, 256],
            "vf": [512, 256]
        }
    }
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=2e-4,
        verbose=1,
        tensorboard_log="./tensorboard_baseline/",
        seed=SEED,
        policy_kwargs={
            "features_extractor_class": BaselineComparableCNN,
            "features_extractor_kwargs": config["features_extractor_kwargs"],
            "net_arch": config["net_arch"], # Ensure MLP heads match
            "activation_fn": nn.Tanh,
            "optimizer_class": th.optim.Adam
        }
    )
    return model



def load_checkpoint(model, run_dir: Path, device=None):
    """
    Loads a Baseline PPO checkpoint (Policy + Optimizer).
    
    Args:
        model: The Stable Baselines3 PPO model instance.
        run_dir (Path): Path to the specific iteration folder (e.g., 'models/1/iter_100').
        device (torch.device, optional): Device to load tensors onto. 
                                         Defaults to global 'DEVICE' if defined, else 'cpu'.
    """
    
    # Handle device selection if not explicitly passed
    if device is None:
        device = globals().get("DEVICE", th.device("cpu"))

    print(f"[Baseline] Loading checkpoint from {run_dir}")

    # Ensure the path exists
    if not run_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")

    # ---- 1. Load Policy (ActorCritic) ----
    policy_path = run_dir / "policy" / "policy_state_dict.pt"
    if policy_path.exists():
        policy_sd = th.load(policy_path, map_location=device)
        model.policy.load_state_dict(policy_sd)
        print(f"  -> Policy weights loaded.")
    else:
        print(f"  [Warning] Policy state dict not found at {policy_path}")

    # ---- 2. Load Optimizer (Optional) ----
    # Useful if you plan to resume training, not strictly needed for inference
    opt_path = run_dir / "policy" / "policy_optimizer_state_dict.pt"
    if opt_path.exists() and hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
        try:
            model.policy.optimizer.load_state_dict(
                th.load(opt_path, map_location=device)
            )
            print(f"  -> Optimizer state loaded.")
        except Exception as e:
            print(f"  [Warning] Failed to load optimizer state: {e}")

    # ---- 3. Set Device & Mode ----
    model.policy.to(device)
    
    # Switch to eval mode by default (safer for inference/testing)
    model.policy.eval()

    print("[Baseline] Checkpoint loaded successfully")

if __name__ == "__main__":
    env = make_env(SEED)
    model = build_model(env)
    load_checkpoint(model, BASE_DIR, DEVICE)
    pygame.init()

    SCALE = 3
    H, W = 144, 144

    screen = pygame.display.set_mode((W*SCALE, H*SCALE))
    clock = pygame.time.Clock()

    # create surface ONCE
    surface = pygame.Surface((W, H))
    terminated  = True
    truncated = True
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if terminated or truncated:
            obs, _ = env.reset()
            terminated = False
            truncated = False
        else:
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(int(action[0]))
        rgb = env.render()
        surfarray.blit_array(surface, rgb.swapaxes(0, 1))
        surface_scaled = pygame.transform.scale(surface, (W * SCALE, H * SCALE))
        screen.blit(surface_scaled, (0, 0))
        pygame.display.flip()
        clock.tick(10)

    pygame.display.quit()
    pygame.quit()   