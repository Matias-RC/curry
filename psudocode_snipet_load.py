import torch as th
from pathlib import Path
import gymnasium as gym

from gym_sokoban.envs import SokobanEnv
from stable_baselines3.common.monitor import Monitor

from sokoban_wrapper import (
    SokobanCompactWrapper,
    SokobanRetriesWrapper,
)

from consolidator_class import Consolidator
from alternative_maple_policy import MaplePolicy
from alternative_maple import Maple
from alternative_maple_buffers import MapleRolloutBuffer, DynamicReplayBuffer

# ============================================================
# Config (MUST MATCH TRAINING)
# ============================================================
SEED = 43
DIM_ROOM = (6, 6)
MAX_STEPS = 22
NUM_BOXES = 1
NUM_RETRIES = 5

HIDDEN_SIZE_CHANNELS = 64
POOL_SHAPE = 4

RUN_ID = 0  # <-- change this to the run you want to load
BASE_DIR = Path("./models") / str(RUN_ID)

DEVICE = "cuda" if th.cuda.is_available() else "cpu"


# ============================================================
# Environment builder
# ============================================================
def make_env(seed):
    env = SokobanRetriesWrapper(
        max_retries=NUM_RETRIES,
        dim_room=DIM_ROOM,
        max_steps=MAX_STEPS,
        num_boxes=NUM_BOXES,
        render_mode="rgb_array",
    )
    env = SokobanCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ============================================================
# Rebuild model (same as training)
# ============================================================
def build_model(env):
    from config_file import config  # OR inline the same config dict

    model = Maple(
        policy=MaplePolicy,
        env=env,
        policy_kwargs=config["policy"],
        verbose=1,
        seed=SEED,
        learning_rate=1e-3,
        rollout_buffer_class=MapleRolloutBuffer,
        rollout_buffer_kwargs={
            "prefix_size": (HIDDEN_SIZE_CHANNELS, HIDDEN_SIZE_CHANNELS)
        },
        dynamic_buffer_class=DynamicReplayBuffer,
        dynamic_buffer_kwargs={"n_envs": 1},
        consolidator_class=Consolidator,
        consolidator_kwargs=config["consolidator"],
        num_retries=NUM_RETRIES,
        device=DEVICE,
    )
    return model


# ============================================================
# Load state dicts
# ============================================================
def load_checkpoint(model, run_dir: Path):
    print(f"[MAPLE] Loading checkpoint from {run_dir}")

    # ---- Policy ----
    policy_sd = th.load(
        run_dir / "policy" / "policy_state_dict.pt",
        map_location=DEVICE,
    )
    model.policy.load_state_dict(policy_sd)

    opt_path = run_dir / "policy" / "policy_optimizer_state_dict.pt"
    if opt_path.exists() and hasattr(model.policy, "optimizer"):
        model.policy.optimizer.load_state_dict(
            th.load(opt_path, map_location=DEVICE)
        )

    # ---- Consolidator ----
    if hasattr(model, "consolidator"):
        cons_sd = th.load(
            run_dir / "consolidator" / "consolidator_state_dict.pt",
            map_location=DEVICE,
        )
        model.consolidator.load_state_dict(cons_sd)

        opt_path = run_dir / "consolidator" / "consolidator_optimizer_state_dict.pt"
        if opt_path.exists():
            model.consolidator.optimizer.load_state_dict(
                th.load(opt_path, map_location=DEVICE)
            )

    model.policy.to(DEVICE)
    model.policy.eval()
    model.consolidator.to(DEVICE)
    model.consolidator.eval()

    print("[MAPLE] Checkpoint loaded successfully")


# ============================================================
# Sanity test rollout
# ============================================================
def test_rollout(model, env, n_steps=50):
    obs, _ = env.reset()
    prefix = model.dynamic_buffer.get_prefix(0)

    for step in range(n_steps):
        action, _ = model.predict(
            obs,
            prefix=prefix,
            deterministic=True,
        )
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
            prefix = model.dynamic_buffer.get_prefix(0)

    print("[MAPLE] Rollout test completed without crashing")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    env = make_env(SEED)
    model = build_model(env)
    load_checkpoint(model, BASE_DIR)
    test_rollout(model, env)
