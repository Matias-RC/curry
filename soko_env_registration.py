# register_sokoban.py
import importlib
import sys
import traceback
import gymnasium as gym
from gymnasium.envs.registration import register

def safe_register(id, entry_point, max_episode_steps=1000, force=False):
    """Register only if the id isn't already present (or if force=True)."""
    if id in gym.envs.registry and not force:
        print(f"SKIP: {id} already registered")
        return False
    try:
        register(id=id, entry_point=entry_point, max_episode_steps=max_episode_steps)
        print(f"REGISTERED: {id} -> {entry_point}")
        return True
    except Exception as e:
        print(f"FAILED to register {id} -> {entry_point}: {e}")
        traceback.print_exc()
        return False

def main():
    print("Python:", sys.version.splitlines()[0])
    print("gymnasium version:", getattr(gym, "__version__", "unknown"))

    # try importing the package and its envs submodule
    try:
        pkg = importlib.import_module("gym_sokoban")
        print("gym_sokoban imported from:", pkg.__file__)
    except Exception as e:
        print("ERROR importing gym_sokoban:", e)
        raise

    try:
        envs_mod = importlib.import_module("gym_sokoban.envs")
    except Exception as e:
        print("ERROR importing gym_sokoban.envs:", e)
        raise

    # list discovered Sokoban-like classes
    candidates = [n for n in dir(envs_mod) if "Sokoban" in n or "Sokoban" in n]
    print("Discovered classes in gym_sokoban.envs:", candidates)

    # Mapping from desired gym ids -> module:Class entry_point
    mapping = {
        "Sokoban-v0": "gym_sokoban.envs:SokobanEnv",
        "Sokoban-v1": "gym_sokoban.envs:SokobanEnv1",
        "Sokoban-v2": "gym_sokoban.envs:SokobanEnv2",
        "Sokoban-Small-v0": "gym_sokoban.envs:SokobanEnv_Small0",
        "Sokoban-Small-v1": "gym_sokoban.envs:SokobanEnv_Small1",
        "Sokoban-Large-v0": "gym_sokoban.envs:SokobanEnv_Large0",
        "Sokoban-Large-v1": "gym_sokoban.envs:SokobanEnv_Large1",
        "Sokoban-Huge-v0": "gym_sokoban.envs:SokobanEnv_Huge0",
        "Sokoban-FixedTargets-v0": "gym_sokoban.envs:FixedTargetsSokobanEnv",
        "Sokoban-PushPull-v0": "gym_sokoban.envs:PushAndPullSokobanEnv",
        "Sokoban-TwoPlayer-v0": "gym_sokoban.envs:TwoPlayerSokobanEnv",
    }

    # Only attempt registration for entries that point to discovered names,
    # otherwise skip (helps avoid typos / missing classes).
    for env_id, entry in mapping.items():
        cls_name = entry.split(":")[1]
        if cls_name in candidates:
            safe_register(env_id, entry, max_episode_steps=1000)
        else:
            print(f"SKIP mapping {env_id} -> {entry} (class {cls_name} not found)")

    # Print resulting registry keys that include "Sokoban"
    sok_keys = [k for k in gym.envs.registry.keys() if "Sokoban" in k]
    print("\nFinal registered Gym IDs containing 'Sokoban':")
    for k in sok_keys:
        print(" ", k)

    # Quick smoke test: try creating Sokoban-v0 (if registered)
    test_id = "Sokoban-v0"
    if test_id in gym.envs.registry:
        try:
            print(f"\nAttempting gym.make('{test_id}', render_mode='tiny_rgb_array') ...")
            env = gym.make(test_id, render_mode="tiny_rgb_array")
            obs, info = env.reset(seed=0)
            print("Reset OK, obs type/shape:", type(obs), getattr(obs, "shape", None))
            env.close()
        except Exception as e:
            print("Error creating or resetting env:", e)
            traceback.print_exc()
    else:
        print(f"\n{test_id} not registered; skip smoke test. Use the printed registry above to pick an ID.")

if __name__ == "__main__":
    main()
