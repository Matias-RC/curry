"""
This would return torch data set that is probably too big; let's for now limit the size of the amount of batches for pre-training

States [B,T,C,H,W]
Actions [B, T, 4]

Length of a batch: say 5
we order sets of five within csv and ensure lengths are same. Check how many sets of five we can do: Load them to  data with a limit of 1 000 sets
the go ahead loading sequentially (not to overflow ram)
"""

import pandas as pd
import argparse

from collections import deque, defaultdict
import random
import math
import copy
import time
from typing import Tuple, List, Dict, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import csv
import matplotlib.pyplot as plt

import shutil
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

import pandas as pd

df_medium_valid = pd.read_csv("boxoban-astar-solutions/medium_valid.csv")

# Remove invalid rows
df_medium_valid = df_medium_valid[
    (df_medium_valid["Steps"] != "INCORRECT_SOLUTION_FOUND") &
    (df_medium_valid["Actions"] != "SEARCH_STATE_FAILED") &
    (df_medium_valid["Steps"] != "-1")
].copy()

df_medium_valid["Steps"] = pd.to_numeric(df_medium_valid["Steps"], errors="coerce")

df_medium_valid = df_medium_valid[
    (df_medium_valid["Steps"] > 21) &
    (df_medium_valid["Steps"] < 81)
]

df_medium_valid = df_medium_valid.sort_values("Steps").reset_index(drop=True)

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

    return walls, boxes, goals, player

def render_state(walls, boxes, goals, agent_pos):
    H = 10
    W = 10
    obs = torch.zeros((3, H, W), dtype=torch.float32)
    ay, ax = agent_pos
    if agent_pos not in goals:
        obs[:,ay, ax] = torch.tensor([0.75, 0.5, 0.25])
    else:
        obs[:,ay, ax] = torch.tensor([0.25, 0.5, 0.75])
    for by, bx in (boxes - goals):
        obs[:,by, bx] = torch.tensor([0.25,0.75,0.5])
    for gy, gx in (goals-boxes):
        obs[:,gy, gx] = torch.tensor([0.5, 0.25, 0.75])
    for py, px in boxes & goals:
        obs[:,py, px] = torch.tensor([0.5, 0.75, 0.25])
    for wy, wx in walls:
        obs[:,wy, wx] = torch.tensor([1.0, 1.0, 1.0])
    return obs


for i in df_medium_valid.head(4).iterrows():
    level_name = str(i[1]["File"])
    while len(level_name) < 3:
        level_name = "0"+level_name
    level_id = str(i[1]["Level"])
    while len(level_id) < 3:
        level_id = "0"+level_id
    

    level_str = load_level_by_id(f"boxoban-levels/medium/valid/{level_name}.txt", f"{level_id}")

    walls, boxes, goals, player = parse_sokoban_level(level_str)
    print(render_state(walls, boxes, goals, player))
