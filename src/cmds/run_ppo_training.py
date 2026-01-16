from argparse import ArgumentParser
import random
import numpy as np
import torch
import os
import sys
from itertools import islice
import copy

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # A/
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

def parser_args():

    parser = ArgumentParser(description="Train Thinker And Consolidator Models with PPO for multiple replays")

    # Global arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Training arguments
    parser.add_argument("--batch_size_train", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--learning_rate_reduction_coef", type=float, default=0.99, help="After each epoch learning rate gets multiplied by this")
    parser.add_argument("--min_learning_rate", type=float, default=1e-4, help="Minimum learning rate for optimizer")

    # Dataset arguments
    parser.add_argument("--max_num_levels", type=int, default=8192, help="Number of levels in pool")

    # Thinker arguments
    parser.add_argument("--num_think_steps", type=int, default=2, help="Number of thinking steps")

    # Consolidator arguments
    parser.add_argument("--num_plays", type=int, default=2, help="Number of times an environment is attempted")
    parser.add_argument("--memory_size", type=int, default=4, help="Number of past states stored in memory")
    

    # PPO with GAE arguments
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for PPO")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="Clipping epsilon for PPO")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--max_steps_per_play", type=int, default=120, help="Maximum steps per playthrough")

    # Curriculum arguments
    parser.add_argument("--filter_by", type=str,default="shortest_first" ,help="Options: shortest_first, longest_first, no_filter")


    return parser.parse_args()

def set_seed_for_reproducibility(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_batches(iterable, k):
    """
    This function is used to replace torch dataloader. since now we do not have direct loss dinamics
    """
    data = list(iterable)          # needed for backward access
    N = len(data)
    it = iter(data)
    full_blocks = N // k
    chunks = [list(islice(it, k)) for _ in range(full_blocks)]
    remainder = N % k
    if remainder:
        chunks.append(data[-k:])   # backward extension
    return chunks


def main():
    args = parser_args()
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed_for_reproducibility(seed)
    from curriculum_utils.environment import SokobanEnvironment
    config_sokoban_env = {
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "device": device,
        "max_steps_per_play": args.max_steps_per_play,
        "batch_size": args.batch_size_train,
    }
    sokoban_env = SokobanEnvironment(config_sokoban_env)


    from data.dataset import SokobanDataset, collate_fn
    config_total_dataset = {
        "mode": "ppo",
        "difficulty": "unfiltered",
        "subset_name": "train",
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "max_num_levels": args.max_num_levels,
        "filter_by": args.filter_by,
    }
    total_dataset = SokobanDataset(config_total_dataset)

    # Train/Val fraction
    train_fraction = 0.9
    num_total_levels = len(total_dataset)
    num_train_levels = int(train_fraction * num_total_levels)
    num_eval_levels = num_total_levels - num_train_levels

    
    print(f"Total levels: {num_total_levels}, Train levels: {num_train_levels}, Eval levels: {num_eval_levels}")

    batch_size_train = batch_size_eval = args.batch_size_train

    from models.thinker import Thinker
    from models.consolidator import Consolidator
    visual_encoder_config = {
        "in_channels": 4,
        "latent_dim": 64,
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "channels": [16, 32, 64],   # out channels for each conv layer
        "kernel_size": 3,
        "padding": 1,
        "stride": 2,                # applied only to last conv
    }
    action_decoder_config = {
        "model_name": "qwen2",
        "args": {
            "num_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
        },
        "hidden_size": 64,
        "vocab_size": 4,
        "num_think_steps": args.num_think_steps,
    }
    thinker_config = {
        "config_visual_encoder": visual_encoder_config,
        "config_action_decoder": action_decoder_config,
    }
    thinker = Thinker(thinker_config)
    thinker.train()

    consolidator_config = {
        "config_two_tower": {
            "model_name": "qwen2",
            "hidden_size": 64,
            "prefix_size": args.memory_size,
            "args": {
                "num_attention_heads": 4,
                "consolidator_tower_1_num_layers": 1,
                "consolidator_tower_2_num_layers": 2,
                "hidden_size": 64,
                "num_attention_heads": 4
            }
        },
    }
    consolidator = Consolidator(consolidator_config)

    consolidator.train()

    if torch.cuda.is_available():
        thinker = thinker.to("cuda")
        consolidator = consolidator.to("cuda")
        print("Using GPU for training")
    else:
        print("Using CPU for training")
    # Optimzer
    learning_rate = args.learning_rate
    thinker_optimizer = torch.optim.Adam(thinker.parameters(), lr=learning_rate)
    consolidator_optimizer = torch.optim.Adam(consolidator.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_dataset = copy.deepcopy(total_dataset)
    train_dataset.data = train_dataset.data[:num_train_levels]
    eval_dataset = total_dataset
    eval_dataset.data = eval_dataset.data[num_train_levels:]
    

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        train_dataset.shuffle_pool()
        # Make batches
        for batch_idx, batch in enumerate(make_batches(train_dataset.data, args.batch_size_train)):
            thinker_optimizer.zero_grad()
            consolidator_optimizer.zero_grad()
            sokoban_env.load_levels([item["initial_state_tensor"] for item in batch])
            #  Generate the trayectory autoregressively
            trajectory_hidden_values = []
            trajectory_
            kv_caches = None

            while not sokoban_env.all_levels_done():
                states_tensors = sokoban_env._get_obs() # shape [B, H, W, C] ((self.batch_size, self.size_y, self.size_x, 4),dtype=torch.float32,device=self.device)
                visual_latent_states = thinker.visual_encoder(states_tensors.unsqueeze(1))  # shape [B, 1, D]
                out = thinker.forward(visual_latent_states, "autoregressive", kv_caches)
                trajectory
            #  These vectors are used with normalized advantages to train Thinker
            #  The same vectors are used to generate memory
            #   Generated memory minimizes the loss of the advanages for fully paralelized predictions
            #   Since new memory minimizes past regret this should be mathematically sound
            #pass
        break



if __name__ == "__main__":
    main()
