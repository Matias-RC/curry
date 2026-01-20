import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from tqdm import tqdm
import random
import numpy as np


levels = [
    "##########\n"
    "#@ #######\n"
    "#.$#######\n"
    "#   ######\n"
    "#.$ ######\n"
    "#  #######\n"
    "# $ ######\n"
    "#$ . #####\n"
    "# .  #####\n"
    "##########",

    "##########\n"
    "##########\n"
    "# ###@####\n"
    "#    $   #\n"
    "#   $    #\n"
    "# ##  ####\n"
    ".##  ####\n"
    "# ###$$.##\n"
    "#  .   . #\n"
    "##########",

    "##########\n"
    "#####   ##\n"
    "#####   ##\n"
    "####.    #\n"
    "# .  $@ ##\n"
    "# $ $ $ ##\n"
    "#   .   ##\n"
    "#####.   #\n"
    "######   #\n"
    "##########",

    "##########\n"
    "######## #\n"
    "###      #\n"
    "### $ . .#\n"
    "### $ #  #\n"
    "#@$  ##  #\n"
    "##  . #  #\n"
    "## $ ##  #\n"
    "## .     #\n"
    "##########",
]


def set_seed_for_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    sys.path.append("./src")

    from envs.sokoban import SokobanEnv
    from models.maple import MAPLE

    # Set seed for reproducibility
    seed = 42
    set_seed_for_reproducibility(seed)

    # Configuration
    config_sokoban_env = {
        "action_padding_value": -100,
        "channels": ['boxes', 'goals', 'player', 'walls'],
        "action_map": [(-1,0),(0,1),(1,0),(0,-1)],
    }
    sokoban_env = SokobanEnv(config_sokoban_env)

    # MAPLE model configuration
    maple_config = {
        "thinker_learning_rate": 1e-4,
        "max_solution_length": 120,
        "discount": 0.99,
        "learning_rate": 0.01,  # For target forward gradient descent
    }

    thinker_config = {
        "config_visual_encoder": {
            "in_channels": 4,
            "latent_dim": 64,
            "grid_shape_x": 10,
            "grid_shape_y": 10,
            "channels": [16, 32, 64],
            "kernel_size": 3,
            "padding": 1,
            "stride": 2,
        },
        "config_action_decoder": {
            "model_name": "qwen2",
            "args": {
                "num_layers": 2,
                "hidden_size": 64,
                "num_attention_heads": 4,
            },
            "hidden_size": 64,
            "vocab_size": 4,
            "num_think_steps": 2,
        },
    }

    model_config = {
        "maple_config": maple_config,
        "thinker_config": thinker_config,
        "env": sokoban_env,
    }

    # Initialize MAPLE model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = MAPLE(model_config)
    model = model.to(device)
    model.device = device
    model.train()

    # Training parameters
    num_epochs = 100

    print(f"Training MAPLE with TD(0) on {len(levels)} levels")
    print(f"Number of epochs: {num_epochs}")
    print("=" * 50)

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0.0

        # Train on each level in the batch
        batch_losses = []
        for level_str in levels:
            str_batch = [level_str]
            loss = model.simplified_maple_loop(str_batch)
            batch_losses.append(loss)

        total_loss = sum(batch_losses) / len(batch_losses)

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}, Avg Loss: {total_loss:.6f}")

            # Test generation on first level
            model.eval()
            with torch.no_grad():
                test_level = [levels[0]]
                dynamic_batch = model.env.get_dynamic_batch(test_level)
                _, _, _, rewards_seq, _, _ = model.generate(
                    dynamic_batch,
                    max_solution_length=100
                )

                # Check if solved
                final_reward = rewards_seq[0, -1].item()
                print(f"Test level final reward: {final_reward}")

            model.train()

    print("\n" + "=" * 50)
    print("Training completed!")

    # Final evaluation on all levels
    print("\nFinal Evaluation:")
    model.eval()
    with torch.no_grad():
        for i, level_str in enumerate(levels):
            test_batch = [level_str]
            dynamic_batch = model.env.get_dynamic_batch(test_batch)
            _, _, _, rewards_seq, _, _ = model.generate(
                dynamic_batch,
                max_solution_length=100
            )

            final_reward = rewards_seq[0, -1].item()
            total_reward = rewards_seq.sum().item()
            print(f"Level {i+1}: Final reward = {final_reward:.2f}, Total reward = {total_reward:.2f}")


if __name__ == "__main__":
    main()

