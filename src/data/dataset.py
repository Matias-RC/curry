from torch.utils.data import Dataset
import torch
from tqdm import tqdm

import pandas as pd
import numpy as np
import os

from collections import deque

def load_level_by_id(path: str, level_id: str) -> str:
    """
    path: path to the .txt level file
    level_id: stringified triple, e.g. "000", "014"

    Returns:
        Level grid as a single string (with newlines)
    """
    target = int(level_id)  # "000" -> 0
    current_id = None
    collecting = False
    level_lines = []

    with open(path, "r") as f:
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
        raise ValueError(f"Level {level_id} not found in file")

    return "\n".join(level_lines)

def fill_name(string, length=3):
    while len(string) < length:
        string = "0" + string
    return string

from typing import Set, Tuple

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
    action_map = [(-1,0),(0,1),(1,0),(0,-1)]
    states = [state]
    for action in actions_str:
        walls, boxes, goals, player = state["walls"], state["boxes"], state["goals"], state["player"]
        if tuple(sorted(boxes)) == tuple(sorted(goals)):
            print("solved")
            break
        dy, dx = action_map[int(action)]
        new_pos_player = (player[0]+dy, player[1]+dx)
        if new_pos_player in walls:
            continue
        if new_pos_player in boxes:
            new_pos_box = (new_pos_player[0]+dy, new_pos_player[1]+dx)
            if new_pos_box in boxes:
                continue
            boxes.remove(new_pos_player)
            boxes.add(new_pos_box)

        player = new_pos_player
        state = {"walls": walls, "boxes": boxes, "goals": goals, "player": player}
        states.append(state)

    return states

def symbolic_state_to_tensor(state, grid_shape_x, grid_shape_y, channels):
    num_channels = len(channels)
    tensor = torch.zeros((grid_shape_x, grid_shape_y, num_channels), dtype=torch.float32)

    for c, key in enumerate(channels):
        values = state[key] if key != "player" else {state[key]}
        idx = torch.tensor(list(values), dtype=torch.long)  # shape [N, 2]
        tensor[idx[:, 0], idx[:, 1], c] = 1.0

    return tensor

class SokobanDataset(Dataset):
    def __init__(self, config):
        self.config = config
        assert config["mode"] in ["supervised", "ppo"], "mode must be either 'supervised' or 'ppo'"
        if config["mode"] == "supervised":
            self.data = self.load_data(config)
        else:
            self.data = self.load_data_for_ppo(config)

    def load_data(self, config):

        difficulty = config["difficulty"]
        subset_name = config["subset_name"]

        grid_shape_x = config["grid_shape_x"]
        grid_shape_y = config["grid_shape_y"]

        max_num_levels = config["max_num_levels"]
        filter_by = config.get("filter_by", "no_filter")

        channels = ['boxes', 'goals', 'player', 'walls']

        df = pd.read_csv(f"../../boxoban-astar-solutions/{difficulty}_{subset_name}.csv")
        filtered = df[
            (df["Steps"] != "INCORRECT_SOLUTION_FOUND") &
            (df["Actions"] != "SEARCH_STATE_FAILED") &
            (df["Actions"] != "NOT_FOUND") &
            (df["Steps"] != -1)
        ].copy()

        # Convert Steps to numeric
        filtered["Steps"] = pd.to_numeric(filtered["Steps"], errors="coerce")
        filtered = filtered.dropna(subset=["Steps"])

        # Sort deterministically
        if filter_by == "shortest_first":
            filtered = filtered.sort_values("Steps", ascending=True)
        elif filter_by == "longest_first":
            filtered = filtered.sort_values("Steps", ascending=False)
        # else: no_filter → keep CSV order

        # Truncate to max_num_levels (no randomness)
        if max_num_levels is not None:
            filtered = filtered.iloc[:max_num_levels]

        data = []

        tqdm_levels = tqdm(filtered.iterrows(), total=len(filtered), desc="Loading dataset")
        for _, row in tqdm_levels:
            folder_name = fill_name(str(int(row["File"])))
            level_id_filled = fill_name(str(int(row["Level"])))

            level = load_level_by_id(
                f"../../boxoban-levels/{difficulty}/{subset_name}/{folder_name}.txt",
                level_id_filled
            )

            actions_str = row["Actions"]

            state_0 = parse_sokoban_level(level)
            states = play(state_0, actions_str)

            states_tensor = [
                symbolic_state_to_tensor(state, grid_shape_x, grid_shape_y, channels)
                for state in states[:-1]
            ]
            states_tensor = torch.stack(states_tensor, dim=0)

            actions_id = torch.tensor([int(a) for a in actions_str], dtype=torch.long)

            data.append({
                "states_tensor": states_tensor,
                "actions_id": actions_id,
            })

        return data
    def load_data_for_ppo(self, config):
        """
        Main load data works for supervised learning. This: for PPO training.
        """
        difficulty = config["difficulty"]
        subset_name = config["subset_name"]
        grid_shape_x = config["grid_shape_x"]
        grid_shape_y = config["grid_shape_y"]
        max_num_levels = config["max_num_levels"]
        filter_by = config.get("filter_by", "no_filter")
        channels = ['boxes', 'goals', 'player', 'walls']

        df = pd.read_csv(f"../../boxoban-astar-solutions/{difficulty}_{subset_name}.csv")

        filtered = df[
            (df["Steps"] != "INCORRECT_SOLUTION_FOUND") &
            (df["Actions"] != "SEARCH_STATE_FAILED") & 
            (df["Actions"] != "NOT_FOUND") &
            (df["Steps"] != -1)
        ].copy()
        
        # Convert Steps to numeric
        filtered["Steps"] = pd.to_numeric(filtered["Steps"], errors="coerce")
        filtered = filtered.dropna(subset=["Steps"])
        # Sort deterministically
        if filter_by == "shortest_first":
            filtered = filtered.sort_values("Steps", ascending=True)
        elif filter_by == "longest_first":
            filtered = filtered.sort_values("Steps", ascending=False)
        # Truncate to max_num_levels (no randomness)
        if max_num_levels is not None:
            filtered = filtered.iloc[:max_num_levels]
        data = []
        tqdm_levels = tqdm(filtered.iterrows(), total=len(filtered), desc="Loading dataset for PPO")
        for _, row in tqdm_levels:
            folder_name = fill_name(str(int(row["File"])))
            level_id_filled = fill_name(str(int(row["Level"])))
            level = load_level_by_id(
                f"../../boxoban-levels/{difficulty}/{subset_name}/{folder_name}.txt",
                level_id_filled
            )
            state = parse_sokoban_level(level)
            tensor = symbolic_state_to_tensor(state, grid_shape_x, grid_shape_y, channels)
            data.append({
                "initial_state_tensor": tensor,
                "level_id": level_id_filled,
                "folder_name": folder_name,
            })
        return data
    def shuffle_pool(self):
        np.random.shuffle(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    batch_states = [item["states_tensor"] for item in batch]
    batch_actions = [item["actions_id"] for item in batch]

    batch_states_padded = torch.nn.utils.rnn.pad_sequence(batch_states, batch_first=True, padding_value=0.0)
    batch_actions_padded = torch.nn.utils.rnn.pad_sequence(batch_actions, batch_first=True, padding_value=-100)

    return {
        "states_tensors": batch_states_padded,
        "actions_ids": batch_actions_padded
    }


# Example usage
if __name__ == "__main__":
    config_dataset = {
        "difficulty": "medium",
        "subset_name": "valid",
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "max_num_levels": 10,
        "filter_by": "shortest_first",  # Options: "shortest_first", "longest_first", "no_filter"
    }

    dataset = SokobanDataset(config_dataset)
    print(next(iter(dataset)))
    print(len(dataset[0]["actions_id"]))

    """{'states_tensor': tensor([[[[0., 0., 0., 1.],
           [0., 0., 0., 1.],
           [0., 0., 0., 1.],
           ...,
           [0., 0., 0., 1.],
           [0., 0., 0., 1.],
           [0., 0., 0., 1.]],
 
          [[0., 0., 0., 1.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           ...,
           [0., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 0., 1.]],
 
          [[0., 0., 0., 1.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           ...,
           [0., 0., 0., 1.],
           [0., 0., 0., 1.],
           [0., 0., 0., 1.]],
 
          ...,
...
           [0., 0., 0., 1.],
           [0., 0., 0., 1.]]]]),
 'actions_id': tensor([3, 2, 3, 0, 2, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 1, 2, 3, 2, 2, 1, 1, 2, 2,
         2, 3, 3, 0, 0, 1, 0, 1, 2, 2, 0, 3, 3, 0, 0, 0, 1, 0, 0, 3, 2, 1, 1, 1,
         1, 0, 1, 2, 2, 1, 2, 2, 3, 0, 2, 2, 2, 3, 2, 3, 3])}"""