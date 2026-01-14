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

    # Training arguments
    parser.add_argument("--batch_size_train", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")

    # Dataset arguments
    parser.add_argument("--max_num_levels", type=int, default=2048, help="Maximum number of levels to use from the dataset")

    # Thinker arguments
    parser.add_argument("--num_think_steps", type=int, default=2, help="Number of thinking steps")

    return parser.parse_args()

def set_seed_for_reproducibility(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_eval(model, eval_loader, loss_fn):
    model.eval()
    eval_loss, acc_loss = 0.0, 0.0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to("cuda") if torch.cuda.is_available() else v for k, v in batch.items()}
            output = model(batch)
            decoder_output = output["decoder_output"]  # shape [B, T, num_actions]
            logits = decoder_output["logits"]
            actions_ids = batch["actions_ids"]         # shape [B, T]
            
            B, T, num_actions = logits.shape
            loss = loss_fn(logits.view(B * T, num_actions), actions_ids.view(B * T))
            eval_loss += loss.item() * B

            preds = logits.argmax(dim=-1)  # shape [B, T]
            correct = (preds == actions_ids).float()
            mask = (actions_ids != -100).float()
            acc = (correct * mask).sum() / mask.sum()
            acc_loss += acc.item() * B
            
    eval_loss /= len(eval_loader.dataset)
    acc_loss /= len(eval_loader.dataset)
    model.train()
    return eval_loss, acc_loss

def main():
    args = parser_args()
    print("Training with args:", args)

    seed = args.seed
    set_seed_for_reproducibility(seed)


    from data.dataset import SokobanDataset, collate_fn
    from torch.utils.data import DataLoader

    config_total_dataset = {
        "difficulty": "medium",
        "subset_name": "valid",
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "max_num_levels": args.max_num_levels,
    }
    total_dataset = SokobanDataset(config_total_dataset)

    # Train/Val fraction
    train_fraction = 0.8
    num_total_levels = len(total_dataset)
    num_train_levels = int(train_fraction * num_total_levels)
    num_eval_levels = num_total_levels - num_train_levels
    train_dataset, eval_dataset = torch.utils.data.random_split(
        total_dataset,
        [num_train_levels, num_eval_levels],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"Total levels: {num_total_levels}, Train levels: {num_train_levels}, Eval levels: {num_eval_levels}")

    batch_size_train = args.batch_size_train
    batch_size_eval = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=collate_fn)

    from models.thinker import Thinker

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
    model_config = {
        "config_visual_encoder": visual_encoder_config,
        "config_action_decoder": action_decoder_config,
    }
    model = Thinker(model_config)
    model.train()

    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Using GPU for training")
    else:
        print("Using CPU for training")

    # Optimzer
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {k: v.to("cuda") if torch.cuda.is_available() else v for k, v in batch.items()}

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

        eval_loss, eval_acc = model_eval(model, eval_loader, loss_fn)
        print(f"Epoch [{epoch+1}/{num_epochs}], Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")

if __name__ == "__main__":
    main()