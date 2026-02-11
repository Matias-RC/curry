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
from stable_baselines3 import PPO
from alternative_maple_buffers import MapleRolloutBuffer, DynamicReplayBuffer
from alternative_maple_callback import MapleCallback
from train4 import BaselineComparableCNN
import time


SEED = 67
DIM_ROOM = (10, 10)
MAX_STEPS = 30
NUM_BOXES = 1

num_beam_search_steps = 30
beam_size = 5

NUM_RETRIES = num_beam_search_steps*beam_size # This way we dont face retries problems

HIDDEN_SIZE_CHANNELS = 64
POOL_SHAPE = 4

RUN_ID = 2  # <-- change this to the run you want to load
ITER = 50
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
    env = SokobanCompactWrapper(env)
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