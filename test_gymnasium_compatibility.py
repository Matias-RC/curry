"""
Test script to verify gym_sokoban compatibility with gymnasium.
"""
import gymnasium as gym
import numpy as np
from gym_sokoban.envs import SokobanEnv


def test_direct_env_creation():
    """Test direct environment creation."""
    print("Testing direct environment creation...")
    env = SokobanEnv(dim_room=(7, 7), max_steps=50, num_boxes=2, render_mode='rgb_array')

    # Test reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    assert isinstance(info, dict), "Info should be a dictionary"
    assert obs.shape == (7 * 16, 7 * 16, 3), f"Expected observation shape (112, 112, 3), got {obs.shape}"
    print(f"✓ Reset returns (observation, info) - observation shape: {obs.shape}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    assert isinstance(reward, (int, float, np.number)), "Reward should be numeric"
    assert isinstance(terminated, (bool, np.bool_)), "Terminated should be boolean"
    assert isinstance(truncated, (bool, np.bool_)), "Truncated should be boolean"
    assert isinstance(info, dict), "Info should be a dictionary"
    print(f"✓ Step returns (obs, reward, terminated, truncated, info)")
    print(f"  - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    # Test multiple steps
    print("\nTesting multiple steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"✓ Episode ended at step {i+1}")
            print(f"  - Terminated (goal reached): {terminated}")
            print(f"  - Truncated (max steps): {truncated}")
            break

    env.close()
    print("✓ Direct environment creation test passed!\n")


def test_gymnasium_make():
    """Test environment creation through gymnasium.make()."""
    print("Testing gymnasium.make() registration...")
    try:
        env = gym.make('Sokoban-v0', render_mode='rgb_array')
        print("✓ Environment registered and created via gym.make()")

        # Test reset
        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
        assert isinstance(info, dict), "Info should be a dictionary"
        print(f"✓ Reset works - observation shape: {obs.shape}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step works - reward: {reward}")

        env.close()
        print("✓ gymnasium.make() test passed!\n")
    except Exception as e:
        print(f"✗ gymnasium.make() test failed: {e}\n")


def test_action_space():
    """Test action space compatibility."""
    print("Testing action space...")
    env = SokobanEnv(dim_room=(7, 7), max_steps=50, num_boxes=2)

    assert hasattr(env, 'action_space'), "Environment should have action_space"
    assert isinstance(env.action_space, gym.spaces.Discrete), "Action space should be Discrete"
    print(f"✓ Action space: {env.action_space} (9 actions)")

    # Test all actions
    obs, _ = env.reset(seed=42)
    for action in range(env.action_space.n):
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = info.get("action.name", "unknown")
        # print(f"  Action {action} ({action_name}): valid")

    print(f"✓ All {env.action_space.n} actions are valid\n")
    env.close()


def test_observation_space():
    """Test observation space compatibility."""
    print("Testing observation space...")
    env = SokobanEnv(dim_room=(10, 10), max_steps=100, num_boxes=3, render_mode='rgb_array')

    assert hasattr(env, 'observation_space'), "Environment should have observation_space"
    assert isinstance(env.observation_space, gym.spaces.Box), "Observation space should be Box"

    obs, _ = env.reset(seed=99)
    assert env.observation_space.contains(obs), "Observation should be in observation space"
    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Observation within space bounds\n")
    env.close()


def test_seeding():
    """Test seeding reproducibility."""
    print("Testing seeding reproducibility...")

    env1 = SokobanEnv(dim_room=(7, 7), max_steps=50, num_boxes=2)
    env2 = SokobanEnv(dim_room=(7, 7), max_steps=50, num_boxes=2)

    obs1, _ = env1.reset(seed=12345)
    obs2, _ = env2.reset(seed=12345)

    assert np.array_equal(obs1, obs2), "Same seed should produce same initial observation"
    print("✓ Seeding produces reproducible results")

    # Test different seeds produce different results
    obs3, _ = env1.reset(seed=54321)
    assert not np.array_equal(obs1, obs3), "Different seeds should produce different observations"
    print("✓ Different seeds produce different results\n")

    env1.close()
    env2.close()


def test_full_episode():
    """Test a full episode."""
    print("Testing full episode...")
    env = SokobanEnv(dim_room=(7, 7), max_steps=100, num_boxes=2, render_mode='rgb_array')

    obs, info = env.reset(seed=777)
    total_reward = 0
    steps = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"✓ Episode completed in {steps} steps")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Terminated (goal): {info.get('all_boxes_on_target', False)}")
    print(f"  - Truncated (max steps): {info.get('maxsteps_used', False)}")
    print()

    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing gym_sokoban compatibility with gymnasium")
    print("=" * 60)
    print()

    try:
        test_direct_env_creation()
        test_gymnasium_make()
        test_action_space()
        test_observation_space()
        test_seeding()
        test_full_episode()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise
