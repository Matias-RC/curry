import argparse
import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList

# --- PROJECT IMPORTS ---
from sokoban_wrapper import SokoRetriesCurriculum, SokoCanonicalWithAttPadding
from attpolicy import SokoPlayerCentricAtt
from curriculum_utils import VectorCurriculumCallback # Moved your class here for cleanliness

def main():
    parser = argparse.ArgumentParser(description="Train Sokoban with Attention and Curriculum")
    
    # Environment & Training Args
    parser.add_argument("--exp_name", type=str, default="exp_default", help="Name for Tensorboard/ID")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=123)
    
    # Model Architecture Args
    parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--layers", type=int, default=3, help="Attention layers")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    
    # PPO Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    # Define Stage 1 (You can also parameterize this if needed)
    STAGE_1 = {
        (("dim_room", (6,6)), ("max_steps", 22), ("num_boxes", 1), ("num_gen_steps", 20)): 0.8,
        (("dim_room", (7,7)), ("max_steps", 26), ("num_boxes", 1), ("num_gen_steps", 23)): 0.2,
    }
    CURRICULUM_PLAN = {0: STAGE_1}

    # 1. Create Vectorized Environment
    vec_env = make_vec_env(
        env_id=SokoRetriesCurriculum, 
        n_envs=args.num_envs, 
        seed=args.seed, 
        wrapper_class=SokoCanonicalWithAttPadding,
        env_kwargs=dict(max_retries=12, curriculum=True, schedule_dic=STAGE_1), 
        wrapper_kwargs=dict(canonical_shape=(9, 9), window_size=3),
        vec_env_cls=SubprocVecEnv
    )

    # 2. Instantiate PPO (Model Saving Removed)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        tensorboard_log=f"./tensorboard/{args.exp_name}",
        policy_kwargs={
            "features_extractor_class": SokoPlayerCentricAtt,
            "features_extractor_kwargs": {
                "features_dim": 256, 
                "hidden_size": args.hidden,
                "num_heads": args.heads,
                "layers": args.layers
            },
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        }
    )

    # 3. Training with Curriculum only (CheckpointCallback removed)
    curriculum_cb = VectorCurriculumCallback(CURRICULUM_PLAN, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, callback=curriculum_cb)
    
    print(f"Experiment {args.exp_name} finished. No model files saved.")

if __name__ == "__main__":
    main()