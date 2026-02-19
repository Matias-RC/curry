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
    SokobanRetriesWrapper,
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
                    print("\n[Curriculum] --------------------------------------------------")
                    print(f"[Curriculum] UPGRADE TRIGGERED at step {self.num_timesteps}")
                    print(
                        f"[Curriculum] Broadcasting new schedule to "
                        f"{self.training_env.num_envs} environments."
                    )
                    print(f"[Curriculum] New Schedule: {new_schedule}")
                    print("[Curriculum] --------------------------------------------------")

                # Broadcast update to all envs
                self.training_env.env_method("update_schedule", new_schedule)

        return True


# ==============================================================================
# 2. Define The Curriculum Schedules
# ==============================================================================

# STAGE 1: The starting mix
STAGE_1 = {
    (("dim_room", (7, 7)), ("max_steps", 25), ("num_boxes", 1), ("num_gen_steps", int(1.7 * (6 + 6)))): 1,
}

# The Master Plan: Map Timesteps -> Schedule
CURRICULUM_PLAN = {
    0: STAGE_1,
}


# ==============================================================================
# 3. Main Execution Block
# ==============================================================================

if __name__ == "__main__":

    # --- Configuration ---
    NUM_ENVS = 64
    SEED = 123
    TOTAL_TIMESTEPS = 1_000_000

    # 1. Environment Arguments
    env_kwargs = dict(
        max_retries=12,
        dim_room=(7, 7),
        max_steps=25,
        num_boxes=1,
        curriculum=True,
        schedule_dic=STAGE_1,
    )

    # 2. Wrapper Arguments
    wrapper_kwargs = dict(
        canonical_shape=(9, 9),
        window_size=3,
    )

    # 3. Create Vectorized Environment
    print(f"Creating {NUM_ENVS} vectorized environments...")

    vec_env = make_vec_env(
        env_id=SokoRetriesCurriculum,
        n_envs=NUM_ENVS,
        seed=SEED,
        wrapper_class=SokoCanonicalWithAttPadding,
        env_kwargs=env_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    # 4. Instantiate PPO Model
    print("Initializing PPO with Player-Centric Attention Policy...")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=1024,
        verbose=1,
        tensorboard_log="./tensorboard/",
        policy_kwargs=dict(
            features_extractor_class=SokoPlayerCentricAtt,
            share_features_extractor=False,
            features_extractor_kwargs=dict(
                features_dim=256,
                hidden_size=256,
                num_heads=4,
                layers=4,
            ),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )

    # 5. Setup Callbacks
    curriculum_cb = VectorCurriculumCallback(CURRICULUM_PLAN, verbose=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="./models_vec/",
        name_prefix="soko_att_vec",
    )

    callbacks = CallbackList([curriculum_cb])

    # 6. Train
    print("Starting Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    # 7. Save Final Model
    model.save("final_soko_model_vec")
    print("Training Complete. Model saved.")
