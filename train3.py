import time
import numpy as np
import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from gym_sokoban.envs import SokobanEnv
from sokoban_wrapper import SokobanCompactWrapper, SokobanRetriesWrapper, SokobanCanonicalCompactWrapper

import numpy as np
import pygame
import pygame.surfarray as surfarray

from consolidator_class import ChannelStackedSpatialAccumulation, Consolidator
from alternative_maple_policy import MaplePolicy, PrefixCombinator, ConvFeatureExtractor
from alternative_maple import Maple
from alternative_maple_buffers import MapleRolloutBuffer, DynamicReplayBuffer
from alternative_maple_callback import MapleCallback

next_maple_img = pygame.image.load("next.png").convert_alpha()
stop_maple_img = pygame.image.load("stop_maple.png").convert_alpha()

SHARE_PREFIX_COMBINATOR=False #Currently implemented consolidator only allows for
SHARE_FEATURES_EXTRACTOR=True
HIDDEN_SIZE_CHANNELS = 64
POOL_SHAPE = 4

CONSOLIDATOR_LR=1e-3

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
            "in_channels":2*HIDDEN_SIZE_CHANNELS,
            "hidden_channels":HIDDEN_SIZE_CHANNELS,
            "out_channels":HIDDEN_SIZE_CHANNELS,
            "n_layers":2,
            "out_shape":(POOL_SHAPE,POOL_SHAPE)
        }
    }
}

# ============================================================
# 2. Environment Builders
# ============================================================



def make_env(dim_room, max_steps, num_boxes, num_retries, seed=42):
    env = SokobanRetriesWrapper(
        max_retries=num_retries,
        dim_room=dim_room,
        max_steps=max_steps,
        num_boxes=num_boxes,
        render_mode="rgb_array",
    )
    env = SokobanCanonicalCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

if __name__ == "__main__":
    SEED = 42
    DIM_ROOM = (5,5)
    MAX_STEPS = 12
    MAX_STEPS_EVAL = 20
    NUM_BOXES = 1
    NUM_RETRIES = 5

    train_env = make_env(dim_room=DIM_ROOM,
                               max_steps=MAX_STEPS,
                               num_boxes=NUM_BOXES,
                               num_retries=NUM_RETRIES,   
                               seed=SEED)
    
    eval_env = make_env(dim_room=DIM_ROOM,
                               max_steps=MAX_STEPS_EVAL,
                               num_boxes=NUM_BOXES,
                               num_retries=NUM_RETRIES,  
                               seed=SEED+1)
    
    model = Maple(
        policy=MaplePolicy,
        env=train_env,
        policy_kwargs=config["policy"],
        verbose=1,
        tensorboard_log="./tensorboard/",
        seed=SEED,
        learning_rate=1e-3,
        dynamic_buffer_class=DynamicReplayBuffer,
        dynamic_buffer_kwargs={"n_envs":1}

    )