from argparse import ArgumentParser
import random
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath("../"))

def parser_args():

    parser = ArgumentParser(description="Train Thinker Model")

    # Global arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()

def set_seed_for_reproducibility(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parser_args()
    print("Training with args:", args)

    seed = args.seed
    set_seed_for_reproducibility(seed)


    from data.dataset import SokobanDataset, collate_fn
    from torch.utils.data import DataLoader

    config_dataset = {
        "difficulty": "medium",
        "subset_name": "valid",
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "max_num_levels": 512,
    }
    train_dataset = SokobanDataset(config_dataset)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    from models.thinker import Thinker

    visual_encoder_config = {
        "in_channels": 4,
        "latent_dim": 32,
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "channels": [16, 32, 32],   # out channels for each conv layer
        "kernel_size": 3,
        "padding": 1,
        "stride": 2,                # applied only to last conv
    }
    action_decoder_config = {
        "model_name": "qwen2",
        "args": {
            "num_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 1,
        },
        "hidden_size": 32,
        "vocab_size": 4,
    }
    model_config = {
        "config_visual_encoder": visual_encoder_config,
        "config_action_decoder": action_decoder_config,
    }
    model = Thinker(model_config)
    model.train()

    # Optimzer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    num_epochs = 100
    for epoch in range(num_epochs):
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(batch)
            decoder_output = output["decoder_output"]  # shape [B, T, num_actions]
            logits = decoder_output["logits"]
            actions_ids = batch["actions_ids"]         # shape [B, T]
            
            B, T, num_actions = logits.shape
            loss = loss_fn(logits.view(B * T, num_actions), actions_ids.view(B * T))

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()