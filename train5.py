import gymnasium as gym
import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# --- IMPORTS FROM YOUR PROJECT ---
# Ensure these files are in your directory or python path
from sokoban_wrapper import (
    SokoRetriesCurriculum, 
    SokoCanonicalWithAttPadding,
    SokobanRetriesWrapper
)
from attpolicy import SokoPlayerCentricAtt

# ==============================================================================
# 1. The Vector-Ready Curriculum Callback
# ==============================================================================
class VectorCurriculumCallback(BaseCallback):
    """
    Updates the environment's task distribution based on total timesteps.
    Designed to work with Vectorized Environments (VecEnv) by using env_method.
    """
    def __init__(self, schedule_plan, verbose=0):
        super().__init__(verbose)
        self.schedule_plan = schedule_plan
        # Sort thresholds to ensure we check them in order
        self.thresholds = sorted(schedule_plan.keys())
        self.current_stage_idx = -1

    def _on_step(self) -> bool:
        # Calculate which stage we should be in based on total steps
        next_stage_idx = self.current_stage_idx + 1
        
        # If there are stages left to advance to...
        if next_stage_idx < len(self.thresholds):
            threshold = self.thresholds[next_stage_idx]
            
            # If we passed the timestep threshold for the next stage
            if self.num_timesteps >= threshold:
                new_schedule = self.schedule_plan[threshold]
                self.current_stage_idx = next_stage_idx
                
                if self.verbose > 0:
                    print(f"\n[Curriculum] --------------------------------------------------")
                    print(f"[Curriculum] UPGRADE TRIGGERED at step {self.num_timesteps}")
                    print(f"[Curriculum] Broadcasting new schedule to {self.training_env.num_envs} environments.")
                    print(f"[Curriculum] New Schedule: {new_schedule}")
                    print(f"[Curriculum] --------------------------------------------------")

                # BROADCAST UPDATE:
                # 'update_schedule' must be a method in your SokoRetriesCurriculum class.
                # env_method automatically tunnels through Monitor/DummyVecEnv layers.
                self.training_env.env_method("update_schedule", new_schedule)
                
        return True

# ==============================================================================
# 2. Define The Curriculum Schedules
# ==============================================================================

# STAGE 1: The starting mix (Your specific 8x8 and 9x9 setup)
# 50% chance of 8x8 (1 Box), 50% chance of 9x9 (2 Boxes)
STAGE_1 = {
    (("dim_room", (6,6)), ("max_steps", 22), ("num_boxes", 1), ("num_gen_steps", int(1.7*(6+6)))): 0.8,
    (("dim_room", (7,7)), ("max_steps", 45), ("num_boxes", 2), ("num_gen_steps", int(1.7*(7+7)))): 0.2,
}

STAGE_2 = {
    (("dim_room", (7,7)), ("max_steps", 30), ("num_boxes", 1), ("num_gen_steps", int(1.7*(6+6)))): 0.6,
    (("dim_room", (7,7)), ("max_steps", 45), ("num_boxes", 2), ("num_gen_steps", int(1.7*(7+7)))): 0.4,
}


STAGE_3 = {
    (("dim_room", (7,7)), ("max_steps", 30), ("num_boxes", 1), ("num_gen_steps", int(1.7*16))): 0.5,
    (("dim_room", (8,8)), ("max_steps", 50), ("num_boxes", 2), ("num_gen_steps", int(1.7*20))): 0.5,
}

"""STAGE_3 = {
    (("dim_room", (8,8)), ("max_steps", 30), ("num_boxes", 1), ("num_gen_steps", int(1.7*16))): 0.2,
    (("dim_room", (10,10)), ("max_steps", 60), ("num_boxes", 2), ("num_gen_steps", int(1.7*20))): 0.8,
}
"""
# The Master Plan: Map Timesteps -> Schedule
CURRICULUM_PLAN = {
    0: STAGE_1,       # Active from step 0
    200_000: STAGE_2,  # Switch at 200k steps
    400_000: STAGE_3
}

# ==============================================================================
# 3. Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    
    # --- Configuration ---
    NUM_ENVS = 8
    SEED = 123
    TOTAL_TIMESTEPS = 1_000_000
    
    # 1. Environment Arguments
    # These are passed to SokoRetriesCurriculum.__init__
    env_kwargs = dict(
        max_retries=2, 
        dim_room=(8,8),  # Initial default size
        max_steps=30, 
        num_boxes=2, 
        curriculum=True, 
        schedule_dic=STAGE_1 # Important: Start with Stage 1
    )

    # 2. Wrapper Arguments
    # These are passed to SokoCanonicalWithAttPadding.__init__
    # CRITICAL: canonical_shape must be large enough to hold the largest level you ever plan to train on.
    # If using grid coordinates (10x10 tiles), use (10, 10).
    # If using pixels (10x10 tiles * 16px), use (160, 160).
    # Assuming grid coords based on window_size=3:
    wrapper_kwargs = dict(
        canonical_shape=(9, 9), 
        window_size=3 
    )

    # 3. Create Vectorized Environment
    # make_vec_env handles the creation of 8 parallel processes/threads
    print(f"Creating {NUM_ENVS} vectorized environments...")
    vec_env = make_vec_env(
        env_id=SokoRetriesCurriculum, 
        n_envs=NUM_ENVS, 
        seed=SEED, 
        wrapper_class=SokoCanonicalWithAttPadding,
        env_kwargs=env_kwargs, 
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=SubprocVecEnv# Change to SubprocVecEnv for true multiprocessing
    )

    # 4. Instantiate PPO Model
    print("Initializing PPO with Player-Centric Attention Policy...")
    model = PPO(
        policy="MlpPolicy",  # MlpPolicy because our custom extractor outputs a flat vector
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,        # 2048 * 8 envs = 16384 steps per update
        batch_size=64,
        verbose=1,
        tensorboard_log="./tensorboard/",
        policy_kwargs={
            "features_extractor_class": SokoPlayerCentricAtt,
            "share_features_extractor": False,
            "features_extractor_kwargs": {
                "features_dim": 256, 
                "hidden_size": 128,
                "num_heads": 4,
                "layers": 3
            },
            # Ensure the MLP head on top of the extractor matches your preference
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        }
    )

    # 5. Setup Callbacks
    curriculum_cb = VectorCurriculumCallback(CURRICULUM_PLAN, verbose=1)
    checkpoint_cb = CheckpointCallback(
        save_freq=50000 // NUM_ENVS, # Adjust freq for vec envs
        save_path="./models_vec/", 
        name_prefix="soko_att_vec"
    )
    
    callbacks = CallbackList([curriculum_cb, checkpoint_cb])

    # 6. Train
    print("Starting Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    
    # 7. Save Final Model
    model.save("final_soko_model_vec")
    print("Training Complete. Model saved.")