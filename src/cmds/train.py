from argparse import ArgumentParser
import random
import uuid
import numpy as np
import torch
import os
import sys
import torch.nn as nn
from datetime import datetime
import boto3
import json
import os
import tempfile
from tqdm import tqdm

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
    parser.add_argument("--train_fraction", type=float, default=0.8, help="Fraction of data to use for training")
    parser.add_argument("--difficulty", type=str, default="medium", help="Difficulty level of the Sokoban levels: easy, medium, hard")
    parser.add_argument("--split", type=str, default="valid", help="Subset name of the dataset to use")
    parser.add_argument("--order_by", type=str,default="no_filter" ,help="Options: shortest_first, longest_first, no_filter")
    parser.add_argument("--solution_length_min", type=int, default=-1, help="Minimum solution length to filter levels")
    parser.add_argument("--solution_length_max", type=int, default=1000, help="Maximum solution length to filter levels")   

    # Thinker arguments
    parser.add_argument("--num_think_steps", type=int, default=2, help="Number of thinking steps")

    # MAPLE arguments
    parser.add_argument("--num_supervision_steps", type=int, default=5, help="Number of supervision steps in MAPLE")
    
    # Output arguments
    parser.add_argument("--where_to_save", type=str, default="s3://rl6-reinforcement-learning-01", help="Where to save outputs: local or s3")
    parser.add_argument("--output_dir", type=str, default="behavioral-cloning/test", help="Directory to save outputs")
    parser.add_argument("--metrics_save_epoch_rate", type=int, default=5, help="Interval (in epochs) to save metrics")

    # Log arguments
    parser.add_argument("--verbose", type=int, default=1, help="If set, print training logs")

    # Device 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training: cuda or cpu")

    return parser.parse_args()

def set_seed_for_reproducibility(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_bc_eval(model, eval_loader, loss_fn):
    model.eval()
    eval_loss = {step: 0.0 for step in range(model.num_supervision_steps)}
    eval_acc  = {step: 0.0 for step in range(model.num_supervision_steps)}
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(model.device) if torch.cuda.is_available() and isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(batch)
            actions_ids = batch["actions_ids"]         # shape [B, T]

            for step in range(model.num_supervision_steps):
                decoder_output = output[step]["decoder_output"]  # shape [B, T, num_actions]
                logits = decoder_output["logits"]
                B, T, num_actions = logits.shape
                loss = loss_fn(logits.view(B * T, num_actions), actions_ids.view(B * T))
                eval_loss[step] += loss.item() * B
                preds = logits.argmax(dim=-1)  # shape [B, T]
                correct = (preds == actions_ids).float()
                mask = (actions_ids != -100).float()
                acc = (correct * mask).sum() / mask.sum()
                eval_acc[step] += acc.item() * B
            
    eval_loss = {step: eval_loss[step] / len(eval_loader.dataset) for step in eval_loss}
    eval_acc  = {step: eval_acc[step] / len(eval_loader.dataset) for step in eval_acc}
    model.train()
    return eval_loss, eval_acc

def model_gen_eval(model, eval_loader, max_solution_length=100):
   
    model.eval()
    eval_acc = {step: 0.0 for step in range(model.num_supervision_steps)}
    with torch.no_grad():
        for batch in eval_loader:
            output = model.generate(batch, max_solution_length)
            states_0 = [model.env.parse_sokoban_level(level) for level in batch["level_strs"]] 
            for step in range(model.num_supervision_steps):
                decoder_output = output[step]["decoder_output"]  # shape [B, T, num_actions]
                logits = decoder_output["logits"]
                preds = logits.argmax(dim=-1)
                for b in range(len(states_0)):
                    state_0 = states_0[b]
                    actions_str = "".join([str(a.item()) for a in preds[b]])
                    _, status = model.env.play(state_0, actions_str, early_stop=True)
                    if status == "solved":
                        eval_acc[step] += 1.0
            
    eval_acc  = {step: eval_acc[step]  for step in eval_acc}
    model.train()

    return eval_acc

def create_experiment(args, verbose=True): # Get experiment name from date and time. Also save args in json file.

    # Create experiment name based on date and time
    now = datetime.now()
    exp_name = now.strftime("%Y%m%d_%H%M%S")
    exp_name += f"_{uuid.uuid4().hex[:8]}"  # Add random suffix to avoid overwriting

    # Save args to json file (s3 or local
    if args.where_to_save.startswith("s3://"):
        s3 = boto3.client('s3')
        bucket_name = args.where_to_save.replace("s3://", "")
        args_path = os.path.join(args.output_dir, exp_name, "args.json")
        # Save args to a temporary local file then upload to s3, and remove the local file. Use tempfile module
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
            json.dump(vars(args), tmpfile)
            tmpfile_path = tmpfile.name

        s3_key = args_path
        s3.upload_file(tmpfile_path, bucket_name, s3_key)
        if verbose:
            print(f"Experiment args saved to s3://{bucket_name}/{s3_key}")
        os.remove(tmpfile_path)

        return {
            "output_dir": args.output_dir,
            "exp_name": exp_name,
            "s3_client": s3,
            "bucket_name": bucket_name,
        }
    
    else:
        # TODO: Local saving
        return {
            "exp_name": exp_name,
        }


# save_metrics(args.where_to_save, args.output_dir, metrics_per_epoch, epoch + 1):
def save_metrics(experiment_details, metrics_dict, epoch=None, verbose=True): # If overwrite, then replace existing file. If epoch=None then save as metrics.json

    metrics_filename = f"{epoch}.json" if epoch != None else "metrics.json"
    metrics_path = os.path.join(experiment_details["exp_name"], metrics_filename)

    if "s3_client" in experiment_details:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
            json.dump(metrics_dict, tmpfile)
            tmpfile_path = tmpfile.name

        s3 = experiment_details["s3_client"]
        bucket_name = experiment_details["bucket_name"]
        s3_key = os.path.join(experiment_details["output_dir"], experiment_details["exp_name"], metrics_filename)
        s3.upload_file(tmpfile_path, bucket_name, s3_key)
        if verbose:
            print(f"Metrics saved to s3://{bucket_name}/{s3_key}")

        os.remove(tmpfile_path)

    else:
        if verbose:
            print(f"Metrics saved locally to {metrics_path}")


def main():
    args = parser_args()
    print("Training with args:", args)

    experiment_details = create_experiment(args)

    seed = args.seed
    set_seed_for_reproducibility(seed)

    from envs.sokoban import SokobanEnv
    from data.dataset import SokobanDataset
    from torch.utils.data import DataLoader

    config_sokoban_env = {
        "action_padding_value": -100,
        "channels": ['boxes', 'goals', 'player', 'walls'],
        "action_map": [(-1,0),(0,1),(1,0),(0,-1)],
    }
    sokoban_env = SokobanEnv(config_sokoban_env)

    config_total_dataset = {
        "source_levels": {
            "github": "google-deepmind/boxoban-levels",
            "cache_dir": "~/scratch/curry/",
        },
        "source_solutions": {
            "huggingface": "AlignmentResearch/boxoban-astar-solutions",
            "cache_dir": "~/scratch/curry/",
        },
        "difficulty": args.difficulty,
        "split": args.split,
        "grid_shape": {
            "x": 10,
            "y": 10,
        },
        "max_num_levels": args.max_num_levels,
        "order_by": args.order_by,
        "solution_length": {
            "min": args.solution_length_min,
            "max": args.solution_length_max,
        },
        "seed": seed,
        "env": sokoban_env,
    }
    total_dataset = SokobanDataset(config_total_dataset)

    # Train/Val fraction
    train_fraction = args.train_fraction
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=sokoban_env.collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=sokoban_env.collate_fn)

    from models.maple import MAPLE

    model_config = {
        "thinker_config": {
            "config_visual_encoder": {
                "in_channels": 4,
                "latent_dim": 64,
                "grid_shape_x": 10,
                "grid_shape_y": 10,
                "channels": [16, 32, 64],   # out channels for each conv layer
                "kernel_size": 3,
                "padding": 1,
                "stride": 2,                # applied only to last conv
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
                "num_think_steps": args.num_think_steps,
            },
        },
        "consolidator_config": {
            "model_name": "t5",
            "args": {
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "hidden_size": 64,
                "num_attention_heads": 2,
            }
        },
        "memory_config": {
            "hidden_size": 64,
            "memory_size": 2,
        },
        "num_supervision_steps": args.num_supervision_steps,
        "env": sokoban_env,
    }
    model = MAPLE(model_config)
    model.train()

    if args.device == "cuda":
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("Using GPU for training")
        else:
            model = model.to("cpu")
            print("Using CPU for training")
    else:
        model = model.to("cpu")
        print("Using CPU for training")

    device = next(model.parameters()).device
    model.device = device

    # Optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    metrics_per_epoch = {}

    num_epochs = args.num_epochs
    tqdm_epochs = tqdm(
        range(num_epochs),
        desc="Training Epochs",
        unit="epoch",
        disable=(args.verbose == 0)
    )
    for epoch in tqdm_epochs:

        total_loss = 0.0
        total_count = 0

        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {k: v.to(model.device) if torch.cuda.is_available() and isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output = model(batch)
            decoder_output = output[-1]["decoder_output"]  # shape [B, T, num_actions]
            logits = decoder_output["logits"]
            actions_ids = batch["actions_ids"]         # shape [B, T]
            
            B, T, num_actions = logits.shape
            loss = loss_fn(logits.view(B * T, num_actions), actions_ids.view(B * T))

            total_loss += loss.item() * B
            total_count += B

            loss.backward()
            optimizer.step()
        
        train_loss = total_loss / total_count
        bc_eval_loss, bc_eval_acc = model_bc_eval(model, eval_loader, loss_fn)
        gen_eval_acc = model_gen_eval(model, eval_loader, max_solution_length=100)

        metrics_per_epoch[epoch] = {
            "train_loss": train_loss,
            "eval_loss": bc_eval_loss,
            "eval_acc": {"bc": bc_eval_acc, "gen": gen_eval_acc},
        }
        if args.verbose == 1: # Print using tqdm
            tqdm_epochs.set_postfix({
                "Eval Loss": {step: f"{bc_eval_loss[step]:.4f}" for step in bc_eval_loss},
                "Eval (BC) Acc": {step: f"{bc_eval_acc[step]:.4f}" for step in bc_eval_acc},
                "Eval (Gen) Acc": {step: f"{gen_eval_acc[step]:.4f}" for step in gen_eval_acc},
                "Train Loss": f"{train_loss:.4f}",
            })
        # Save metric every 10 epochs in the s3 (args.where_to_save)
        if (epoch + 1) % args.metrics_save_epoch_rate == 0:
            save_metrics(experiment_details, metrics_per_epoch, None, verbose=(args.verbose == 1))
            

if __name__ == "__main__":
    main()