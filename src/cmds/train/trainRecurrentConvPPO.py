# train_with_human_render.py

import time
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

from gym_sokoban.envs import SokobanEnv
from src.envs.envs import SokobanCompactWrapper, SokobanCanonicalCompactWrapper
from src.policies.convpolicy import ConvFeatureExtractor, CustomConvLSTMPolicy

import pygame
import pygame.surfarray as surfarray

# ============================================================
# 1. Validation Callback (Recurrent-safe, Human Render)
# ============================================================

class HumanRenderCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        render_every_steps=25_000,
        max_steps=200,
        deterministic=True,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.render_every_steps = render_every_steps
        self.max_steps = max_steps
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.num_timesteps % self.render_every_steps != 0:
            return True
        
        done = False
        truncated = False
        SCALE = 4
        H, W = 80,80

        screen = pygame.display.set_mode((W*SCALE, H*SCALE))
        clock = pygame.time.Clock()

        # Recurrent state handling
        lstm_states = None
        surface = pygame.Surface((W, H))
        step = 0
        episode_starts = np.array([True])
        obs, _ = self.eval_env.reset()
        while step < self.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
            if done or truncated:
                obs, _ = self.eval_env.reset()
                lstm_states = None
                episode_starts = np.array([True])
                done = False
                truncated = False
            else:
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=self.deterministic,
                )

                obs, reward, done, truncated, info = self.eval_env.step(int(action))

                episode_starts = np.array([False])
            rgb = self.eval_env.render()

            surfarray.blit_array(surface, rgb.swapaxes(0, 1))
            surface_scaled = pygame.transform.scale(surface, (W * SCALE, H * SCALE))

            screen.blit(surface_scaled, (0, 0))
            pygame.display.flip()
            clock.tick(10)

            step += 1
        pygame.display.quit()
        pygame.quit()   
        return True
    
def curriculum(step):
    if step < 10_000:
        return dict(dim_room=(5,5), num_boxes=1, max_steps=8)
    elif step < 15_000:
        return dict(dim_room=(5,5), num_boxes=1, max_steps=12)
    elif step < 30_000:
        return dict(dim_room=(6,6), num_boxes=1, max_steps=16)
    else:
        return dict(dim_room=(6,6), num_boxes=2, max_steps=16)
    
class CurriculumCallback(BaseCallback):
    def __init__(self, schedule_fn=curriculum):
        super().__init__()
        self.schedule_fn = schedule_fn
        self.last_params = schedule_fn(0)

    def _on_step(self):
        if self.locals["dones"][0]:
            params = self.schedule_fn(self.num_timesteps)

            if params != self.last_params:
                env = self.training_env.envs[0]
                env.unwrapped.configure(**params)
                self.last_params = params
        return True


# ============================================================
# 2. Environment Builders
# ============================================================

def make_env(seed=42):
    env = SokobanEnv(
        dim_room=(6, 6),
        max_steps=32,
        num_boxes=1,
        render_mode="rgb_array",
    )
    env = SokobanCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

def make_curriculized_env(seed=123):
    env = SokobanEnv(**curriculum(0), render_mode="rgb_array")
    env = SokobanCanonicalCompactWrapper(env, (6,6))
    env = Monitor(env)
    env.reset(seed=seed)
    return env
# ============================================================
# 3. Make Hyperparameters
# ============================================================

# Feature Extractor Settings
EXTRACTOR_POOL_SIZE = (5, 5)  # Force spatial dims to 8x8
LAST_CONV_CHANNELS = 64
LSTM_HIDDEN_CHANNELS = 64
   # How many channels the LSTM maintains internally

# PoolReduce (Post-LSTM) Settings
# We will pool the LSTM output (8x8) down to (2,2) before the final heads
FINAL_POOL_SIZE = (3, 3)
FINAL_EMBEDDING_SIZE = 128    # Size of the vector entering the Actor/Critic MLP

# --- 2. Create the Config Dictionary ---

policy_kwargs = {
    # A. Feature Extractor Configuration
    # ----------------------------------
    "features_extractor_class": ConvFeatureExtractor,
    "features_extractor_kwargs": {
        "config": {
            "adaptive_pool_size": EXTRACTOR_POOL_SIZE,
            "conv_configs": [
                # Example Architecture:
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': LAST_CONV_CHANNELS, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ]
        }
    },

    # B. Policy Head Configuration
    # ----------------------------
    # This tells the parent class (RecurrentActorCriticPolicy) how large the 
    # input to the final MLP is. It must match the output of PoolReduce (defined below).
    "lstm_hidden_size": FINAL_EMBEDDING_SIZE, 

    # C. ConvLSTM Wrapper Configuration
    # ---------------------------------
    "conv_lstm_kwargs": {
        # Input Dimensions (Must match Extractor Output)
        "input_size": LAST_CONV_CHANNELS,      # 64
        "feature_shape": EXTRACTOR_POOL_SIZE,  # (8, 8)
        
        # Internal LSTM Dimensions
        "hidden_size": LSTM_HIDDEN_CHANNELS,   # 64
        "num_layers": 1,
        
        # Post-Processing (PoolReduce)
        # This sits between the LSTM and the final Actor/Critic heads
        "net_arch": {
            "mode": "pool_reduce",
            "shapes": (
                # Input to PoolReduce (Output of ConvLSTM)
                (LSTM_HIDDEN_CHANNELS, *EXTRACTOR_POOL_SIZE), # (64, 8, 8)
                
                # Target Pooling Size (AdaptiveAvgPool2d target)
                (LSTM_HIDDEN_CHANNELS, *FINAL_POOL_SIZE)      # (64, 2, 2)
            ),
            "config": [
                # Linear layer after pooling: 
                # Input: 64 channels * 2 * 2 = 256
                # Output: FINAL_EMBEDDING_SIZE = 256
                (LSTM_HIDDEN_CHANNELS * FINAL_POOL_SIZE[0] * FINAL_POOL_SIZE[1], FINAL_EMBEDDING_SIZE),
                (FINAL_EMBEDDING_SIZE, FINAL_EMBEDDING_SIZE),
                (FINAL_EMBEDDING_SIZE, FINAL_EMBEDDING_SIZE),
            ]
        }
    },

    # Standard SB3 Flags
    "shared_lstm": False,
    "enable_critic_lstm": True,
}

# ============================================================
# 4. Training Entry Point
# ============================================================

if __name__ == "__main__":
    SEED = 42

    train_env = make_curriculized_env(seed=SEED)
    eval_env = make_env(seed=SEED + 1)
#
    #render_callback = HumanRenderCallback(
    #    eval_env=eval_env,
    #    render_every_steps=500,
    #    max_steps=200,
    #    deterministic=False,
    #)
    curriculum_callback = CurriculumCallback(curriculum)

    model = RecurrentPPO(
        CustomConvLSTMPolicy,
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard/",
        seed=SEED,
        learning_rate=1e-3
    )

    #model.learn(
    #    total_timesteps=2_000,
    #    callback=render_callback,
    #)
    model.learn(
        total_timesteps=40_000,
        callback=curriculum_callback
    )
    print("===Training Finished===")
    pygame.init()


    SCALE = 4
    H, W = 96,96

    screen = pygame.display.set_mode((W*SCALE, H*SCALE))
    clock = pygame.time.Clock()

    # create surface ONCE
    surface = pygame.Surface((W, H))
    terminated  = True
    truncated = True
    running = True
    lstm_states = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if terminated or truncated:
            obs, _ = eval_env.reset()
            terminated = False
            truncated = False
            lstm_states = None
            episode_starts = np.array([True])
        else:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=False,
            )
            obs, reward, terminated, truncated, info = eval_env.step(int(action))


        episode_starts = np.array([False])
        rgb = eval_env.render()
        surfarray.blit_array(surface, rgb.swapaxes(0, 1))
        surface_scaled = pygame.transform.scale(surface, (W * SCALE, H * SCALE))
        screen.blit(surface_scaled, (0, 0))
        pygame.display.flip()
        clock.tick(10)

    pygame.display.quit()
    pygame.quit()   