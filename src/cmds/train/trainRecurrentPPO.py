import time
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from gym_sokoban.envs import SokobanEnv
from src.envs.envs import SokobanCompactWrapper, SokobanRetriesWrapper

import numpy as np
import pygame
import pygame.surfarray as surfarray

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
        H, W = 80, 80

        screen = pygame.display.set_mode((W*SCALE, H*SCALE))
        clock = pygame.time.Clock()

        # Recurrent state handling
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        surface = pygame.Surface((W, H))
        step = 0
        episode_starts = np.array([True])
        obs, _ = self.eval_env.reset(
                    seed=np.random.randint(0, 10_000)
                )
        while step < self.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
            if done or truncated:
                obs, _ = self.eval_env.reset(
                    seed=np.random.randint(0, 10_000)
                )
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

def make_env(dim_room, max_steps, num_boxes,seed=42):
    env = SokobanRetriesWrapper(
        dim_room=dim_room,
        max_steps=max_steps,
        num_boxes=num_boxes,
        render_mode="rgb_array",
    )
    env = SokobanCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ============================================================
# 3. Policy Args
# ============================================================

policy_kwargs = dict(
    # MLP BEFORE the LSTM
    net_arch=dict(
        pi=[128, 128],
        vf=[128, 128],
    ),

    # LSTM config
    lstm_hidden_size=256,
    n_lstm_layers=3,

    # Sharing choices
    shared_lstm=False,          # usually more stable
    enable_critic_lstm=True,    # critic also gets memory

    # Optional but recommended
    ortho_init=True,
)


# ============================================================
# 4. Training Entry Point
# ============================================================

if __name__ == "__main__":
    SEED = 42
    DIM_ROOM = (6,6)
    MAX_STEPS = 25
    MAX_STEPS_EVAL = 20
    NUM_BOXES = 1

    train_env = make_env(dim_room=DIM_ROOM,
                               max_steps=MAX_STEPS,
                               num_boxes=NUM_BOXES,   
                               seed=SEED)
    eval_env = make_env(dim_room=DIM_ROOM,
                               max_steps=MAX_STEPS_EVAL,
                               num_boxes=NUM_BOXES,   
                               seed=SEED+1)

# Currently not at use
    #render_callback = HumanRenderCallback(
    #    eval_env=eval_env,
    #    render_every_steps=500,
    #    max_steps=200,
    #    deterministic=False,
    #)


    model = RecurrentPPO(
        RecurrentActorCriticPolicy,
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
        total_timesteps=2000,
    )
    print("===Training Finished===")
    pygame.init()


    SCALE = 4
    H, W = 96, 96

    screen = pygame.display.set_mode((W*SCALE, H*SCALE))
    clock = pygame.time.Clock()

    # create surface ONCE
    surface = pygame.Surface((W, H))
    terminated  = True
    truncated = True
    running = True
    lstm_states = None
    episode_starts = np.array([True]) 
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