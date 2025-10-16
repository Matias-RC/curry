import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

try:
    register(
        id="Sokoban-v0",
        entry_point="gym_sokoban.envs:SokobanEnv",
        max_episode_steps=1000
    )
except Exception:
    # Ignore "already registered" error
    pass

env = gym.make("Sokoban-v0")  # or "rgb_array" / "human" depending on support

obs, info = env.reset(seed=42)
print("obs shape/type:", type(obs), getattr(obs, "shape", None))

for t in range(20):
    action = env.action_space.sample()  # random action for smoke test
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"t={t} reward={reward:.3f} done={done}")
    if done:
        obs, info = env.reset()
env.close()
