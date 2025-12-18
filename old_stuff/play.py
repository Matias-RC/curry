# test_agent.py
import os
import argparse
import numpy as np
from PIL import Image
import torch

from train_sokoban_PPO import *
from sokoban_gym_env import SokobanEnv  # assume available in PATH
def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

def load_model(checkpoint_path: str, input_channels: int, num_actions: int, device='cpu'):
    """
    Build model with the exact architecture used in training and load checkpoint.
    This function assumes shapes match exactly and will load state dict directly.
    """
    block_configs = [
        (3, 32, 16),
        (None, 64, 32),
    ]
    final_conv_out = 128
    actor_hidden = 128
    critic_hidden = 128

    model = ActorCritic(
        input_channels=input_channels,
        block_configs=block_configs,
        num_actions=num_actions,
        final_conv_out=final_conv_out,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden
    )
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)   # direct load; assumes exact match
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/ppo_resnet_concat_update_500.pt')
    parser.add_argument('--outdir', default='test_frames')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--generator-module', default=None, help='Optional python module path that defines the generator function')
    parser.add_argument('--generator-func', default='generate_level', help='function name in generator module that returns level')
    parser.add_argument('--env-grid', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)

    level = generate_simple_random_easy(args.env_grid, args.env_grid, 5)

    # create env
    init_type = "matrix" if isinstance(level, np.ndarray) else "str"
    env = SokobanEnv(grid_size=args.env_grid, render_mode="rgb_array", init_type=init_type, level_map=level, max_steps=200)
    obs, _ = env.reset()
    H, W, C = obs.shape
    num_actions = env.action_space.n
    print(f"Env obs shape: {obs.shape}, num_actions: {num_actions}")

    # load model (must match training architecture exactly)
    model = load_model(args.checkpoint, input_channels=C, num_actions=num_actions, device=device)

    # Run greedy rollout (argmax) and save frames
    obs, _ = env.reset()
    total_reward = 0.0
    print_model_params(model)
    for i in range(400):  # safety upper bound
        img = env.render()
        fname = os.path.join(args.outdir, f"frame_{i:04d}.png")
        Image.fromarray(img).save(fname)

        # preprocess obs -> tensor (B, C, H, W), normalized [0,1]
        x = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            logits, value = model(x)   # model expects (B, C, H, W)
            action = int(torch.argmax(logits, dim=-1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: action={action}, reward={reward:.3f}, done={terminated or truncated}")
        total_reward += float(reward)
        obs = next_obs
        if terminated or truncated:
            break

    print("Episode finished. Total reward:", total_reward)
    print("Saved frames to:", os.path.abspath(args.outdir))
