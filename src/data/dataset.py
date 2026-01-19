from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from datasets import load_dataset, Features, Value

from pathlib import Path
import requests

import pandas as pd
import numpy as np
import os

from collections import deque


def load_level_by_id(config_dataset, folder_name: str, level_name: str) -> str:
    repo = config_dataset["source_levels"]["github"]
    split = config_dataset['split']
    split = split if split != "validation" else "valid"

    split_path = f"{config_dataset['difficulty']}/{split}"
    cache_dir = Path(config_dataset["source_levels"]["cache_dir"]).expanduser()
    root_path = cache_dir / Path(repo) / Path(split_path)
    if not root_path.exists():
        root_path.mkdir(parents=True, exist_ok=True)
        api_url = f"https://api.github.com/repos/{repo}/contents/{split_path}?ref=master"

        response = requests.get(api_url)
        response.raise_for_status()

        files = response.json()

        unexisting_files = [file for file in files if file["type"] == "file" and file["name"].endswith(".txt") and not (root_path / file["name"]).exists()]
        tqdm_iterator = tqdm(unexisting_files, desc="Downloading files", unit="file")
        for file in tqdm_iterator:
            download_url = file["download_url"]
            filename = root_path / file["name"]

            r = requests.get(download_url)
            r.raise_for_status()

            filename.write_bytes(r.content)

    target = int(level_name)  # "000" -> 0
    current_id = None
    collecting = False
    level_lines = []
    folder_name = str(folder_name).zfill(3)  # 0 - > 000
    file_path = root_path / Path(f"{folder_name}.txt")
    with open(file_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            # Level header
            if line.startswith(";"):
                # Stop if we were collecting and hit next level
                if collecting:
                    break

                # Parse level number
                try:
                    current_id = int(line[1:].strip())
                except ValueError:
                    current_id = None

                collecting = (current_id == target)
                continue

            # Collect level lines
            if collecting:
                level_lines.append(line)

    if not level_lines:
        raise ValueError(f"Level {level_name} not found in file")

    return "\n".join(level_lines)




def get_solution_subset(config_dataset):
    source_solutions = load_dataset(
            config_dataset["source_solutions"]["huggingface"],
            cache_dir=config_dataset["source_solutions"]["cache_dir"],
            split=config_dataset["split"],
            features=Features({
                "File": Value("int64"),
                "Level": Value("int64"),
                "Actions": Value("string"),
                "Steps": Value("string"),
                "SearchSteps": Value("int64"),
            })
        )
    
    source_solutions = source_solutions.filter(
            lambda example: example["Steps"] != "INCORRECT_SOLUTION_FOUND" and
                            example["Actions"] != "SEARCH_STATE_FAILED" and
                            example["Actions"] != "NOT_FOUND"
        )
        
    source_solutions = source_solutions.map(lambda example: {"Steps": int(example["Steps"])})
    source_solutions = source_solutions.filter(
        lambda example: example["Steps"] == len(example["Actions"])
    )

    return source_solutions

class SokobanDataset(Dataset):
    def __init__(self, config):

        self.env = config["env"] 
        self.data = self.load_data(config)

    def load_data(self, config):

        grid_shape_x = config["grid_shape"]["x"]
        grid_shape_y = config["grid_shape"]["y"]

        max_num_levels = config["max_num_levels"]
        order_by = config["order_by"]  # "shortest_first", "longest_first", "no_filter"

        source_solutions = get_solution_subset(config)

        min_length = config["solution_length"]["min"]
        max_length = config["solution_length"]["max"]
        filtered = source_solutions.filter(
                lambda example: (example["Steps"] >= min_length) and (example["Steps"] <= max_length)
            )
        # Sort deterministically
        if order_by == "shortest_first":
            filtered = filtered.sort("Steps", reverse=False)
        elif order_by == "longest_first":
            filtered = filtered.sort("Steps", reverse=True)
        elif order_by == "shuffle":
            filtered = filtered.shuffle(seed=config["seed"])
        elif order_by == "no_filter":
            pass
        else:
            raise ValueError(f"Unknown order_by: {order_by}")

        data = []
        count_levels = 0
        ix = 0
        stats_levels = {"solved": {"boxes": [], "sol_len": [], "walls": []}, "not solved": {}}
        while (count_levels < max_num_levels) and (ix < len(filtered)):
            row = filtered[ix]
            folder_name = row["File"]
            level_name = row["Level"]

            level = load_level_by_id(
                config,
                folder_name,
                level_name,
            )
            total_boxes = level.count("$") + level.count("*")
            total_walls = level.count("#")

            actions_str = row["Actions"]
            solution_length = len(actions_str)

            state_0 = self.env.parse_sokoban_level(level)
            states, status = self.env.play(state_0, actions_str) 

            if status == "solved":
                states_tensor = [
                    self.env.symbolic_state_to_tensor(state, grid_shape_x, grid_shape_y)
                    for state in states[:-1]
                ]
                states_tensor = torch.stack(states_tensor, dim=0)

                actions_id = torch.tensor([int(a) for a in actions_str], dtype=torch.long)

                data.append({
                    "level_str": level,
                    "actions_str": actions_str,
                    "folder_name": folder_name,
                    "level_name": level_name,
                    "states_tensor": states_tensor,
                    "actions_id": actions_id,
                })
                count_levels += 1

                stats_levels[status]["boxes"].append(total_boxes)
                stats_levels[status]["sol_len"].append(solution_length)
                stats_levels[status]["walls"].append(total_walls)
                
            else:
                if status not in stats_levels["not solved"]:
                    stats_levels["not solved"][status] = 0
                stats_levels["not solved"][status] += 1

            ix += 1
        
        print(pd.DataFrame(stats_levels["solved"]).describe())
        print("Not solved stats:", stats_levels["not solved"])

        return data
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



# Example usage
if __name__ == "__main__":
    config_dataset = {
        "source_levels": {
            "github": "google-deepmind/boxoban-levels",
            "cache_dir": "~/scratch/curry/",
        },
        "source_solutions": {
            "huggingface": "AlignmentResearch/boxoban-astar-solutions",
            "cache_dir": "~/scratch/curry/",
        },

        "difficulty": "unfiltered",
        "split": "validation",
        "grid_shape": {
            "x": 10,
            "y": 10
        },
        "max_num_levels": 16,
        "solution_length":{
            "min": 1,
            "max": 50,
        },
        "order_by": "shortest_first",
    }
    dataset = SokobanDataset(config_dataset)
    print(next(iter(dataset)))

    """{'level_str': '##########\n##########\n##########\n##########\n##@#######\n# $   ####\n# ..$$ ###\n#  $.  ###\n#. #    ##\n##########\n',
    'actions_str': '211123233112110303033322',
    'folder_name': 0,
    'level_name': 2,
    'states_tensor': tensor([[[[0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            ...,
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.]],
    
            [[0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            ...,
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.]],
    
            [[0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            ...,
            [0., 0., 0., 1.],
    ...
            ...,
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.]]]]),
    'actions_id': tensor([2, 1, 1, 1, 2, 3, 2, 3, 3, 1, 1, 2, 1, 1, 0, 3, 0, 3, 0, 3, 3, 3, 2..."""