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
"""
next_maple_img = pygame.image.load("next.png").convert_alpha()
stop_maple_img = pygame.image.load("stop_maple.png").convert_alpha()
"""
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
    env = SokobanCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

if __name__ == "__main__":
    SEED = 43
    DIM_ROOM = (6,6)
    MAX_STEPS = 22
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
        rollout_buffer_class=MapleRolloutBuffer,
        rollout_buffer_kwargs={"prefix_size":(HIDDEN_SIZE_CHANNELS, HIDDEN_SIZE_CHANNELS)},
        dynamic_buffer_class=DynamicReplayBuffer,
        dynamic_buffer_kwargs={"n_envs":1},
        consolidator_class=Consolidator,
        consolidator_kwargs=config["consolidator"],
        num_retries=NUM_RETRIES,
    )

    model.learn(total_timesteps=1000)
    # ============================================================
    # Save all MAPLE components as Torch state_dicts (versioned)
    # ============================================================
    import os
    from pathlib import Path
    import torch as th

    def next_run_id(base_dir: Path) -> int:
        if not base_dir.exists():
            return 0
        ids = []
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                ids.append(int(d.name))
        return max(ids) + 1 if ids else 0


    # Parent directory
    BASE_DIR = Path("./models")
    BASE_DIR.mkdir(exist_ok=True)

    # Auto-increment run id
    RUN_ID = next_run_id(BASE_DIR)
    RUN_DIR = BASE_DIR / str(RUN_ID)
    RUN_DIR.mkdir()

    print(f"[MAPLE] Saving model components to: {RUN_DIR}")

    # ------------------------------------------------------------
    # 1. Policy
    # ------------------------------------------------------------
    policy_dir = RUN_DIR / "policy"
    policy_dir.mkdir()

    th.save(model.policy.state_dict(), policy_dir / "policy_state_dict.pt")

    if hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
        th.save(
            model.policy.optimizer.state_dict(),
            policy_dir / "policy_optimizer_state_dict.pt"
        )

    # ------------------------------------------------------------
    # 2. Consolidator
    # ------------------------------------------------------------
    if hasattr(model, "consolidator") and model.consolidator is not None:
        consolidator_dir = RUN_DIR / "consolidator"
        consolidator_dir.mkdir()

        th.save(
            model.consolidator.state_dict(),
            consolidator_dir / "consolidator_state_dict.pt"
        )

        if hasattr(model.consolidator, "optimizer") and model.consolidator.optimizer is not None:
            th.save(
                model.consolidator.optimizer.state_dict(),
                consolidator_dir / "consolidator_optimizer_state_dict.pt"
            )

    # ------------------------------------------------------------
    # 3. Rollout Buffer (optional, if present)
    # ------------------------------------------------------------
    if hasattr(model, "rollout_buffer") and model.rollout_buffer is not None:
        buffer_dir = RUN_DIR / "rollout_buffer"
        buffer_dir.mkdir()

        th.save(
            model.rollout_buffer.__dict__,
            buffer_dir / "rollout_buffer_state.pt"
        )

    # ------------------------------------------------------------
    # 4. Dynamic Replay Buffer (optional)
    # ------------------------------------------------------------
    if hasattr(model, "dynamic_buffer") and model.dynamic_buffer is not None:
        dyn_buffer_dir = RUN_DIR / "dynamic_buffer"
        dyn_buffer_dir.mkdir()

        th.save(
            model.dynamic_buffer.__dict__,
            dyn_buffer_dir / "dynamic_buffer_state.pt"
        )

    # ------------------------------------------------------------
    # 5. Metadata (human sanity saver)
    # ------------------------------------------------------------
    meta = {
        "run_id": RUN_ID,
        "seed": SEED,
        "total_timesteps": 100_000,
        "share_prefix_combinator": SHARE_PREFIX_COMBINATOR,
        "share_features_extractor": SHARE_FEATURES_EXTRACTOR,
        "hidden_size_channels": HIDDEN_SIZE_CHANNELS,
        "pool_shape": POOL_SHAPE,
    }

    th.save(meta, RUN_DIR / "meta.pt")

    print("[MAPLE] All components saved successfully.")
