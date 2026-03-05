import numpy as np
import matplotlib.pyplot as plt
from gym_sokoban.envs.boxoban_env import BoxobanEnv
import random

def test_environment():
    print("--- Initializing Boxoban Environment ---")
    # 1. Instantiate the environment
    # split='train' and difficulty='unfiltered' are defaults
    env = BoxobanEnv(max_steps=50, difficulty='unfiltered', split='train')
    
    print("--- Resetting Environment (This may download data if first time) ---")
    # 2. Reset the environment
    # This triggers the cache check and level selection
    observation = env.reset()
    
    print(f"Observation shape: {observation.shape}")
    
    # 3. Basic Validation
    assert observation is not None, "Observation should not be None"
    assert isinstance(observation, np.ndarray), "Observation should be a numpy array"
    
    print("--- Running 5 Random Steps ---")
    for i in range(5):
        # Sample a random action (Sokoban typically has 0-8 or 0-4 actions)
        # 0: noop, 1: up, 2: down, 3: left, 4: right (standard Gym-Sokoban)
        action = random.randint(0, 4) 
        
        # In a real gym env, we'd use env.step(action)
        # Since BoxobanEnv inherits from SokobanEnv, make sure SokobanEnv is available
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}")
            
            if done:
                print("Environment finished early. Resetting...")
                env.reset()
        except Exception as e:
            print(f"Step failed: {e}")
            break

    # 4. Visualize the result
    plt.imshow(observation)
    plt.title("Boxoban - Initial State")
    plt.axis('off')
    plt.show()
    
    print("--- Test Complete ---")

if __name__ == "__main__":
    # Ensure you have the necessary dependencies installed for your class
    # pip install requests tqdm numpy matplotlib
    test_environment()