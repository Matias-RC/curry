import sys
import numpy

# Aggressive redirection to catch the internal cloudpickle calls
if not hasattr(numpy, "_core"):
    import numpy.core as _core
    sys.modules["numpy._core"] = _core
    sys.modules["numpy._core.numeric"] = _core.numeric
    sys.modules["numpy._core.multiarray"] = _core.multiarray
    sys.modules["numpy._core.umath"] = _core.umath
    sys.modules["numpy._core._multiarray_umath"] = _core.multiarray
    import numpy.core.numeric as _numeric
    sys.modules["numpy._core.numeric"] = _numeric

# ==========================================================
# 2. NOW DO YOUR REGULAR IMPORTS
# ==========================================================
import torch as th
import pygame
import pygame.surfarray as surfarray
from pathlib import Path
from stable_baselines3 import PPO

# --- PROJECT IMPORTS ---
from src.envs.envs import SokoPoolCurriculumEnv, SokoCanonicalWithAttPadding, SokobanEnv
from src.policies.attpolicy import SokoPlayerCentricAtt
# --- CONFIGURATION ---
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
CANONICAL_SHAPE = (9, 9)
WINDOW_SIZE = 3
DIM_ROOM = 9


def make_test_env():
    # Use the same base environment used in training
    # We use a single config for testing
    config_dicts = [{'dim_room': (DIM_ROOM, DIM_ROOM), 'max_steps': 25, 'num_boxes': 1, 'num_gen_steps': 20}]
    
    # Instantiate the base curriculum env
    env = SokobanEnv(
        dim_room=config_dicts[0]['dim_room'],
        max_steps=config_dicts[0]['max_steps'],
        num_boxes=config_dicts[0]['num_boxes'],
        render_mode="rgb_array"
    )
    
    # Apply your specific Attention-based Wrapper
    env = SokoCanonicalWithAttPadding(
        env, 
        canonical_shape=CANONICAL_SHAPE, 
        window_size=WINDOW_SIZE
    )
    return env

if __name__ == "__main__":
    # 1. Path Check
    model_path = Path("final_soko_model_vec.zip")
    if not model_path.exists():
        print(f"Model file {model_path} not found.")
        exit(1)
    # 2. Environment Setup
    test_env = make_test_env()

    # 3. Load the Model
    print(f"Loading Attention-based model from {model_path}...")
    
    # NEW: Define custom_objects to bypass the broken NumPy random state
    custom_objects = {
        "lr_schedule": lambda _: 0.0,  # Dummies for parameters that might be unpickled incorrectly
        "clip_range": lambda _: 0.0,
        "observation_space": test_env.observation_space,
        "action_space": test_env.action_space
    }

    try:
        model = PPO.load(
            model_path, 
            env=test_env, 
            device=DEVICE, 
            custom_objects=custom_objects
        )
    except ValueError as e:
        if "BitGenerator" in str(e):
            print("Detected NumPy version mismatch in saved seeds. Attempting weight injection...")
            # FALLBACK: If the above still fails, we load the weights manually
            from src.policies.attpolicy import SokoPlayerCentricAtt
            
            # Re-initialize the model architecture fresh
            model = PPO(
                SokoPlayerCentricAtt, 
                test_env, 
                verbose=1, 
                device=DEVICE
            )
            # Inject only the weights (this bypasses all pickle/numpy version issues)
            model.set_parameters(model_path)
        else:
            raise e

    # 4. Pygame Display Setup
    pygame.init()
    SCALE = 4
    H_PX, W_PX = DIM_ROOM * 16, DIM_ROOM * 16  
    screen = pygame.display.set_mode((W_PX * SCALE, H_PX * SCALE))
    pygame.display.set_caption("Sokoban Attention Policy Eval")
    clock = pygame.time.Clock()
    
    surface = pygame.Surface((W_PX, H_PX))
    obs, _ = test_env.reset()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 5. Inference
        # 'obs' is now a windowed tensor of shape (Hw, Ww, C, ws, ws)
        action, _ = model.predict(obs, deterministic=False)
        
        # Step the environment
        # Note: action[0] because SB3 returns an array, but we are in a single env
        obs, reward, terminated, truncated, info = test_env.step(action.item())
        
        # 6. Render
        # We call the BASE env's render for the visual RGB array
        # test_env.unwrapped allows us to bypass the windowed observation wrapper for rendering
        rgb = test_env.unwrapped.render() 
        
        surfarray.blit_array(surface, rgb.swapaxes(0, 1))
        scaled_surf = pygame.transform.scale(surface, (W_PX * SCALE, H_PX * SCALE))
        screen.blit(scaled_surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print("Episode complete. Resetting...")
            obs, _ = test_env.reset()

        clock.tick(8)

    pygame.quit()