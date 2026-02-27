import gymnasium as gym
import torch as th
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# --- IMPORTS FROM YOUR PROJECT ---
# Ensure these match your actual file names
from sokoban_wrapper import SokoPoolCurriculumEnv, SokoCanonicalWithAttPadding
from eppo import EPPO
from attpolicy import ExperiencerActorCritic
from alternative_maple_buffers import PrefixRolloutBuffer


# ==============================================================================
# 1. Environment Configurations (For EXP3 Pool)
# ==============================================================================

config_dicts = [
    {'dim_room': (6, 6), 'max_steps': 18, 'num_boxes': 1, 'num_gen_steps': int(1.7*(6+6))},
    {'dim_room': (7, 7), 'max_steps': 25, 'num_boxes': 1, 'num_gen_steps': int(1.7*(7+7))},
    {'dim_room': (7, 7), 'max_steps': 28, 'num_boxes': 2, 'num_gen_steps': int(1.9*(7+7))},
    {'dim_room': (8, 8), 'max_steps': 30, 'num_boxes': 1, 'num_gen_steps': int(1.7*(8+8))},
    {'dim_room': (8, 8), 'max_steps': 34, 'num_boxes': 2, 'num_gen_steps': int(1.9*(8+8))}
]


# ==============================================================================
# 2. Main Execution Block
# ==============================================================================

if __name__ == "__main__":

    # --- Global Configuration ---
    NUM_ENVS = 1
    SEED = 123
    TOTAL_TIMESTEPS = 4_000

    # --- 1. Environment Arguments ---
    env_kwargs = dict(
        configurations=config_dicts,
        pool_size=8,
        min_plays_to_eval=3,
        max_plays_to_eval=25,
        gamma=0.2,
        dim_room=(6, 6),     # Initial fallback dimension
        max_steps=18,        # Initial fallback steps
        num_boxes=1
    )

    # --- 2. Wrapper Arguments ---
    wrapper_kwargs = dict(
        canonical_shape=(9, 9),
        window_size=3,
    )

    # --- 3. Create Vectorized Environment ---
    print(f"Creating {NUM_ENVS} vectorized environments...")

    vec_env = make_vec_env(
        env_id=SokoPoolCurriculumEnv,
        n_envs=NUM_ENVS,
        seed=SEED,
        wrapper_class=SokoCanonicalWithAttPadding,
        env_kwargs=env_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=DummyVecEnv  # Use DummyVecEnv for simplicity; switch to SubprocVecEnv if needed,
    )

    # --- 4. Instantiate EPPO Model ---
    print("Initializing EPPO with ExperiencerActorCritic...")

    # Define Prefix Constraints
    NUM_ACTOR_PREFIXES = 4
    ENABLE_CRITIC_PREFIX = True
    FEATURES_DIM = 256
    
    # If critic prefix is enabled, the thinker outputs 2x the prefixes
    TOTAL_PREFIXES = NUM_ACTOR_PREFIXES * 2 if ENABLE_CRITIC_PREFIX else NUM_ACTOR_PREFIXES

    model = EPPO(
        policy=ExperiencerActorCritic,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=512,
        n_epochs=10,
        rollout_buffer_class=PrefixRolloutBuffer,
        rollout_buffer_kwargs=dict(
            prefix_shape=(TOTAL_PREFIXES, FEATURES_DIM), 
            prefix_lr=1e-3
        ),
        verbose=1,
        tensorboard_log="./tensorboard/",
        policy_kwargs=dict(
            features_dim=FEATURES_DIM,
            length_prefix=TOTAL_PREFIXES,
            enable_critic_prefix=ENABLE_CRITIC_PREFIX,
            backbone_extractor_kwargs=dict(
                hidden_size=256, 
                num_heads=4, 
                layers=6
            ),
            limb_extractor_kwargs=dict(
                hidden_size=256, 
                num_heads=4, 
                layers=2
            ),
            temporal_extractor_kwargs=dict(
                hidden_size=256, 
                num_heads=4, 
                layers=2
            ),
            net_arch=dict(
                pi=[256, 256], 
                vf=[256, 256], 
                thinker=[256, 256]
            )
        ),
    )

    # --- 5. Train ---
    print("Starting Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # --- 6. Save Final Model ---
    model.save("final_eppo_soko_model")
    print("Training Complete. Model saved.")