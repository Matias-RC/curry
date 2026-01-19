from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from datasets import load_dataset, Features, Value
from typing import Set, Tuple
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



def parse_sokoban_level(level_str: str):
    """
    Parses a 10x10 Sokoban level string.

    Returns:
        walls  : Set[(r, c)]
        boxes  : Set[(r, c)]
        goals  : Set[(r, c)]
        player : (r, c)
    """
    walls: Set[Tuple[int, int]] = set()
    boxes: Set[Tuple[int, int]] = set()
    goals: Set[Tuple[int, int]] = set()
    player = None

    rows = level_str.split("\n")

    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))

            elif ch == "$":
                boxes.add((r, c))

            elif ch == ".":
                goals.add((r, c))

            elif ch == "@":
                player = (r, c)

            elif ch == "*":          # box on goal
                boxes.add((r, c))
                goals.add((r, c))

            elif ch == "+":          # player on goal
                player = (r, c)
                goals.add((r, c))

    if player is None:
        raise ValueError("No player found in level")

    return {
        "walls": walls,
        "boxes": boxes,
        "goals": goals,
        "player": player
    }   

def play(state, actions_str):
    status = "incomplete"
    action_map = [(-1,0),(0,1),(1,0),(0,-1)]
    states = [state]
    for action in actions_str:
        walls, boxes, goals, player = state["walls"], state["boxes"], state["goals"], state["player"]
        dy, dx = action_map[int(action)]
        new_pos_player = (player[0]+dy, player[1]+dx)
        if new_pos_player in walls:
            status = "hit wall"
            break
        if new_pos_player in boxes:
            new_pos_box = (new_pos_player[0]+dy, new_pos_player[1]+dx)
            if new_pos_box in boxes:
                status = "box hit box"
                break
            boxes.remove(new_pos_player)
            boxes.add(new_pos_box)

        player = new_pos_player
        state = {"walls": walls, "boxes": boxes, "goals": goals, "player": player}
        states.append(state)

    if tuple(sorted(boxes)) == tuple(sorted(goals)):
        status = "solved"
    
    return states, status

def symbolic_state_to_tensor(state, grid_shape_x, grid_shape_y, channels):
    num_channels = len(channels)
    tensor = torch.zeros((grid_shape_x, grid_shape_y, num_channels), dtype=torch.float32)

    for c, key in enumerate(channels):
        values = state[key] if key != "player" else {state[key]}
        idx = torch.tensor(list(values), dtype=torch.long)  # shape [N, 2]
        tensor[idx[:, 0], idx[:, 1], c] = 1.0

    return tensor

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

        self.data = self.load_data(config)


    def load_data(self, config):

        grid_shape_x = config["grid_shape"]["x"]
        grid_shape_y = config["grid_shape"]["y"]

        max_num_levels = config["max_num_levels"]
        order_by = config["order_by"]  # "shortest_first", "longest_first", "no_filter"

        channels = ['boxes', 'goals', 'player', 'walls']
        
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

            state_0 = parse_sokoban_level(level)
            states, status = play(state_0, actions_str) 

            if status == "solved":
                states_tensor = [
                    symbolic_state_to_tensor(state, grid_shape_x, grid_shape_y, channels)
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


def collate_fn(batch):
    batch_states = [item["states_tensor"] for item in batch]
    batch_actions = [item["actions_id"] for item in batch]

    batch_states_padded = torch.nn.utils.rnn.pad_sequence(batch_states, batch_first=True, padding_value=0.0)
    batch_actions_padded = torch.nn.utils.rnn.pad_sequence(batch_actions, batch_first=True, padding_value=-100)

    level_strs = [item["level_str"] for item in batch]
    action_strs = [item["actions_str"] for item in batch]
    folder_names = [item["folder_name"] for item in batch]
    level_names = [item["level_name"] for item in batch]

    return {
        "level_strs": level_strs,
        "action_strs": action_strs,
        "folder_names": folder_names,
        "level_names": level_names,
        "states_tensors": batch_states_padded,
        "actions_ids": batch_actions_padded
    }


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