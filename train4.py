import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
import numpy as np
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# --- IMPORTS FROM YOUR PROJECT ---
from sokoban_wrapper import (
    SokoRetriesCurriculum, 
    SokobanCanonicalCompactWrapper  # <--- YOUR NEW WRAPPER
)

# ==============================================================================
# 1. The Baseline Comparable CNN
# ==============================================================================
class BaselineComparableCNN(BaseFeaturesExtractor):
    """
    Identical architecture to previous steps.
    Automatically adapts to 4-channel input from CanonicalWrapper.
    """
    def __init__(self, observation_space: spaces.Box, config: dict, combinator_kwargs: dict):
        pool_shape = combinator_kwargs["out_shape"] 
        out_channels = combinator_kwargs["out_channels"]
        features_dim = out_channels * pool_shape[0] * pool_shape[1]
        
        super().__init__(observation_space, features_dim=features_dim)

        self.conv_configs = config.get("conv_configs")
        
        # KEY CHANGE: This now automatically picks up '4' from your wrapper
        in_channels = observation_space.shape[0] 
        conv_layers = []
        
        for conv_conf in self.conv_configs:
            c_out = conv_conf['out_channels']
            k = conv_conf.get('kernel_size', 3)
            s = conv_conf.get('stride', 1)
            p = conv_conf.get('padding', 1)
            
            conv_layers.append(nn.Conv2d(in_channels, c_out, k, s, p))
            conv_layers.append(nn.ReLU())
            in_channels = c_out
            
        self.backbone = nn.Sequential(*conv_layers)

        c_in = in_channels 
        c_hidden = combinator_kwargs["hidden_channels"]
        c_out_head = combinator_kwargs["out_channels"]
        n_layers = combinator_kwargs["n_layers"]
        
        head_layers = []
        past_out = c_in
        
        for _ in range(n_layers):
            head_layers.append(nn.Conv2d(past_out, c_hidden, 3, 1, 1))
            head_layers.append(nn.ReLU())
            past_out = c_hidden
            
        head_layers.append(nn.Conv2d(past_out, c_out_head, 3, 1, 1))
        head_layers.append(nn.ReLU())
        
        self.head = nn.Sequential(*head_layers)
        self.pool = nn.AdaptiveAvgPool2d(pool_shape)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.backbone(observations)
        x = self.head(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

# ==============================================================================
# 2. Callbacks (Curriculum & Checkpoint)
# ==============================================================================
class VectorCurriculumCallback(BaseCallback):
    def __init__(self, schedule_plan, verbose=0):
        super().__init__(verbose)
        self.schedule_plan = schedule_plan
        self.thresholds = sorted(schedule_plan.keys())
        self.current_stage_idx = -1

    def _on_step(self) -> bool:
        next_stage_idx = self.current_stage_idx + 1
        if next_stage_idx < len(self.thresholds):
            threshold = self.thresholds[next_stage_idx]
            if self.num_timesteps >= threshold:
                new_schedule = self.schedule_plan[threshold]
                self.current_stage_idx = next_stage_idx
                
                if self.verbose > 0:
                    print(f"\n[Curriculum] UPGRADE TRIGGERED at step {self.num_timesteps}")
                    print(f"[Curriculum] New Schedule: {new_schedule}")

                self.training_env.env_method("update_schedule", new_schedule)
        return True

class BaselineCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.base_save_path = Path(save_path)
        self.run_dir = None
        self.iteration = 0

    def _init_callback(self) -> None:
        self.base_save_path.mkdir(parents=True, exist_ok=True)
        ids = [int(d.name) for d in self.base_save_path.iterdir() if d.is_dir() and d.name.isdigit()]
        run_id = max(ids) + 1 if ids else 1
        self.run_dir = self.base_save_path / str(run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose > 0:
            print(f"[Baseline Callback] Logging checkpoints to: {self.run_dir}")
        
    def _on_step(self) -> bool:
        return True    
    
    def _on_rollout_end(self) -> None:
        self.iteration += 1
        if self.iteration % self.save_freq == 0:
            self.save_checkpoint()

    def save_checkpoint(self):
        ckpt_dir = self.run_dir / f"iter_{self.iteration}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        policy_dir = ckpt_dir / "policy"
        policy_dir.mkdir(exist_ok=True)
        th.save(self.model.policy.state_dict(), policy_dir / "policy_state_dict.pt")
        
        if hasattr(self.model.policy, "optimizer") and self.model.policy.optimizer is not None:
            th.save(self.model.policy.optimizer.state_dict(), policy_dir / "policy_optimizer_state_dict.pt")

        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer is not None:
            buffer_dir = ckpt_dir / "rollout_buffer"
            buffer_dir.mkdir(exist_ok=True)
            th.save(self.model.rollout_buffer, buffer_dir / "rollout_buffer_obj.pt")

        meta = {"iteration": self.iteration, "timesteps": self.num_timesteps}
        th.save(meta, ckpt_dir / "meta.pt")

# ==============================================================================
# 3. Configuration & Main
# ==============================================================================

# GLOBAL CONFIG
HIDDEN_SIZE_CHANNELS = 64
POOL_SHAPE = 4
SEED = 43
NUM_RETRIES = 2
NUM_ENVS = 32
TOTAL_TIMESTEPS = 1_500_000

# Defines the MAX size your curriculum will ever reach.
# If your curriculum goes up to 10x10, set this to (10,10) or (12,12).
CANONICAL_SHAPE = (10, 10) 

# Same schedules as before
STAGE_1 = {
    (("dim_room", (6,6)), ("max_steps", 22), ("num_boxes", 1), ("num_gen_steps", int(1.7*12))): 0.8,
    (("dim_room", (7,7)), ("max_steps", 45), ("num_boxes", 2), ("num_gen_steps", int(1.7*14))): 0.2,
}
STAGE_2 = {
    (("dim_room", (7,7)), ("max_steps", 30), ("num_boxes", 1), ("num_gen_steps", int(1.7*14))): 0.6,
    (("dim_room", (7,7)), ("max_steps", 45), ("num_boxes", 2), ("num_gen_steps", int(1.7*14))): 0.4,
}
STAGE_3 = {
    (("dim_room", (7,7)), ("max_steps", 30), ("num_boxes", 1), ("num_gen_steps", int(1.7*14))): 0.5,
    (("dim_room", (8,8)), ("max_steps", 50), ("num_boxes", 2), ("num_gen_steps", int(1.7*16))): 0.5,
}
CURRICULUM_PLAN = {0: STAGE_1, 500_000: STAGE_2, 900_000: STAGE_3}

net_config = {
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
        "combinator_kwargs": {
            "hidden_channels": HIDDEN_SIZE_CHANNELS,
            "out_channels": HIDDEN_SIZE_CHANNELS,
            "n_layers": 2, 
            "out_shape": (POOL_SHAPE, POOL_SHAPE)
        }
    },
    "net_arch": {"pi": [512, 256], "vf": [512, 256]}
}


from typing import Callable

def linear_then_constant_schedule(initial_value: float, final_value: float, total_timesteps: int, endpoint_step: int) -> Callable[[float], float]:
    """
    Linear decay from initial_value to final_value until endpoint_step, 
    then constant at final_value.
    """
    def func(progress_remaining: float) -> float:
        # 1. Calculate current timestep (progress_remaining goes 1.0 -> 0.0)
        current_step = (1.0 - progress_remaining) * total_timesteps

        # 2. If we passed the endpoint, stay constant
        if current_step >= endpoint_step:
            return final_value

        # 3. Otherwise, interpolate linearly
        decay_progress = current_step / endpoint_step
        return initial_value + decay_progress * (final_value - initial_value)

    return func

if __name__ == "__main__":
    lr_schedule = linear_then_constant_schedule(
    initial_value=1e-3,      # Start here
    final_value=2e-4,        # End here
    total_timesteps=TOTAL_TIMESTEPS,
    endpoint_step=800_000    # Reach final value here
    )
    # 1. Env Args
    env_kwargs = dict(
        max_retries=NUM_RETRIES,
        dim_room=(6,6), 
        max_steps=22,
        num_boxes=1,
        curriculum=True,
        schedule_dic=STAGE_1
    )

    # 2. Wrapper Args for SokobanCanonicalCompactWrapper
    wrapper_kwargs = dict(
        canonical_shape=CANONICAL_SHAPE
    )
    
    print(f"Creating {NUM_ENVS} vectorized environments with Canonical Wrapper...")
    
    vec_env = make_vec_env(
        env_id=SokoRetriesCurriculum,
        n_envs=NUM_ENVS,
        seed=SEED,
        # Pass your NEW canonical wrapper here
        wrapper_class=SokobanCanonicalCompactWrapper,
        # Pass the args for that wrapper here
        wrapper_kwargs=wrapper_kwargs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv
    )

    # 3. Model
    print("Initializing PPO with BaselineComparableCNN (4-Channel)...")
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=256,
        batch_size=1024,
        verbose=1,
        tensorboard_log="./tensorboard_baseline/",
        seed=SEED,
        policy_kwargs={
            "features_extractor_class": BaselineComparableCNN,
            "features_extractor_kwargs": net_config["features_extractor_kwargs"],
            "net_arch": net_config["net_arch"],
            "activation_fn": nn.Tanh,
            "optimizer_class": th.optim.Adam
        }
    )

    # 4. Callbacks & Train
    curriculum_cb = VectorCurriculumCallback(CURRICULUM_PLAN, verbose=1)
    checkpoint_cb = BaselineCheckpointCallback(save_freq=20, save_path="./models_baseline/", verbose=1)
    callbacks = CallbackList([curriculum_cb, checkpoint_cb])

    print("Starting Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    
    model.save("final_model_baseline_canonical")
    print("Done.")