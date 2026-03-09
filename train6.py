import gymnasium as gym
import torch as th
import numpy as np
from gymnasium import spaces
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
    NUM_ENVS = 32
    SEED = 123
    TOTAL_TIMESTEPS = 16_000_000

    # --- 1. Environment Arguments ---
    env_kwargs = dict(
        configurations=config_dicts,
        pool_size=4,
        min_plays_to_eval=6,
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
        vec_env_cls=SubprocVecEnv # Use DummyVecEnv for simplicity; switch to SubprocVecEnv if needed,
    )

    # --- 4. Instantiate EPPO Model ---
    print("Initializing EPPO with ExperiencerActorCritic...")

    # Define Prefix Constraints
    NUM_PREFIXES = 2
    ENABLE_CRITIC_PREFIX = True
    FEATURES_DIM = 256
    # TODO: make the boxoban env work, and make a vectorized env that supports custom call to do hard reset of env (separating between normal reset and
    # hard) \

    model = EPPO(
        policy=ExperiencerActorCritic,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=512,
        n_epochs=10,
        sup_epochs=3,
        rollout_buffer_class=PrefixRolloutBuffer,
        rollout_buffer_kwargs=dict(
            prefix_shape=(NUM_PREFIXES*2, FEATURES_DIM), 
            prefix_lr=5e-3
        ),
        verbose=1,
        tensorboard_log="./tensorboard/",
        policy_kwargs=dict(
            features_dim=FEATURES_DIM,
            length_prefix=NUM_PREFIXES,
            enable_critic_prefix=ENABLE_CRITIC_PREFIX,
            backbone_extractor_kwargs=dict(
                hidden_size=256, 
                num_heads=4, 
                layers=2
            ),
            limb_extractor_kwargs=dict(
                hidden_size=256, 
                num_heads=4, 
                layers=6
            ),
            temporal_extractor_kwargs=dict(
                hidden_size=256, 
                num_heads=4, 
                layers=4
            ),
            net_arch=dict(
                pi=[256, 256], 
                vf=[256, 256], 
                thinker=[256, 256]
            )
        ),
        device="cuda"
    )

    # --- 5. Train ---
    print("Starting Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # --- 6. Save Final Model ---
    model.save("final_eppo_soko_model")
    print("Training Complete. Model saved.")