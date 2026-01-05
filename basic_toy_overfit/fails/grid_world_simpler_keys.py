import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Set, Tuple
from PIL import Image, ImageDraw, ImageFont
import os
import time

"""
What are the characteristics of the environment?:
FLoor
Player
Walls
key
Lock
Test 0ne: Simple MLP method for small levels
Test Two: For the same levels, use CNN RGB encodig
Test Three: CNN RBG encoding for bigger representation of levels
Test Four: more key colours
"""

class ApplePlacerEnvironment:
    def __init__(self, size, max_steps, max_no_touch_steps=6, failure_distance=6):
        self.size_y, self.size_x = size
        self.curriculum_stage = (1,1)

        self.bounds = set()
        for i in range(self.size_y):
            self.bounds.add((i,0))
            self.bounds.add((i, self.size_x-1))
        for i in range(self.size_x):
            self.bounds.add((0, i))
            self.bounds.add((self.size_y-1, i))

        self.apples = set()
        self.key_pos = None
        self.player_pos = None
        self.last_time_key_was_touch = 0
        self.prev_min_manhattan = None

        self.action_map = [(-1,0),(1,0),(0,-1),(0,1)]
        self.expanded_map = self.action_map.copy() + [
            (-1,-1), (1,1), (0,0), (-1,1), (1,-1)
        ]
        self.left_steps = max_steps
        self.max_steps = max_steps
        self.finished = False

        # parameters for shaping / termination
        self.max_no_touch_steps = max_no_touch_steps
        self.failure_distance = failure_distance

    def initialize(self):
        self.finished = False
        self.left_steps = self.max_steps
        self.apples = set()
        self.key_pos = None
        self.player_pos = None
        self.last_time_key_was_touch = 0
        self.prev_min_manhattan = None

        center = (int(self.size_y/2-1), int(self.size_x/2-1))
        dy, dx = random.choice(self.expanded_map)
        self.player_pos = (center[0] + dy, center[1] + dx)
        candidates = set()
        candidates.add(self.player_pos)
        for _ in range(self.curriculum_stage[0]):
            for cand in list(candidates.copy()):
                for dyp, dxp in self.expanded_map:
                    valid = True
                    new_cand = (cand[0] + dyp, cand[1] + dxp)
                    if new_cand in candidates:
                        pass
                    else:
                        for dypp, dxpp in self.action_map:
                            if (new_cand[0] + dypp, new_cand[1] + dxpp) in self.bounds:
                                valid = False
                        if valid:
                            candidates.add(new_cand)
        candidates.remove(self.player_pos)
        self.key_pos = random.choice(list(candidates))
        candidates.remove(self.key_pos)
        candidates = list(candidates)
        random.shuffle(candidates)
        for cand in candidates[:self.curriculum_stage[1]]:
            self.apples.add(cand)

        # initialize prev_min_manhattan
        if self.apples:
            self.prev_min_manhattan = min(
                self.manhattan_distance(self.key_pos, a) for a in self.apples
            )
        else:
            self.prev_min_manhattan = 0

    def manhattan_distance(self, pos_1, pos_2):
        dy = abs(pos_1[0]-pos_2[0])
        dx = abs(pos_1[1]-pos_2[1])
        return dy + dx

    def update(self, action: int):
        """
        Returns float reward. May set self.finished.
        Conservative shaping: ONLY apply a small shaping reward when the key is pushed,
        based on whether the key->nearest-apple distance decreased.
        """
        # time/step bookkeeping
        self.left_steps -= 1
        if self.left_steps < 0:
            self.finished = True
            return 0.0

        # Soft termination if key hasn't been touched for too long
        if self.last_time_key_was_touch > self.max_no_touch_steps:
            self.finished = True
            return 0.0

        dy, dx = self.action_map[action]
        new_pos = (self.player_pos[0] + dy, self.player_pos[1] + dx)

        def in_bounds(pos):
            return 0 <= pos[0] < self.size_y and 0 <= pos[1] < self.size_x

        # If player attempts to move into key -> push key
        if new_pos == self.key_pos:
            key_new_pos = (self.key_pos[0] + dy, self.key_pos[1] + dx)
            # update "last touched" because the key was acted upon
            self.last_time_key_was_touch = 0

            if in_bounds(key_new_pos):
                # compute previous min distance (safe-guard)
                if self.apples:
                    prev_min = min(self.manhattan_distance(self.key_pos, a) for a in self.apples)
                else:
                    prev_min = 0

                # perform push
                self.player_pos = new_pos
                self.key_pos = key_new_pos

                # compute new min distance and conservative shaping reward
                if self.apples:
                    new_min = min(self.manhattan_distance(self.key_pos, a) for a in self.apples)
                else:
                    new_min = 0

                shaping = 0.0
                # only small shaping on pushes
                if new_min < prev_min:
                    shaping = 1.0   # push moved key closer to some apple
                elif new_min > prev_min:
                    shaping = -0.5  # push moved key away from apples

                # If key got to a boundary (e.g. lock)
                if self.key_pos in self.bounds:
                    self.finished = True
                    # hitting the bounds: treat as failure (no success reward)
                    return -1.0 + shaping

                # If the key landed on an apple
                if self.key_pos in self.apples:
                    self.apples.remove(self.key_pos)
                    if len(self.apples) == 0:
                        self.finished = True
                        return 40.0 + shaping
                    # partial success (one apple)
                    return 10.0 + shaping

                # check if the push made the key hopelessly far from all apples -> fail
                if self.apples:
                    min_manhattan = min(self.manhattan_distance(self.key_pos, a) for a in self.apples)
                    if min_manhattan > 3:   # tunable threshold
                        self.finished = True
                        return -2.0 + shaping

                return shaping  # no other reward except shaping for pushes
            else:
                # pushing key out of bounds -> immediate finish as failure
                self.finished = True
                return -1.0

        # Normal move (player moves without touching the key)
        elif in_bounds(new_pos):
            self.player_pos = new_pos
            # increment "time since key was touched" (we didn't touch it now)
            self.last_time_key_was_touch += 1

            # if player gets too far from key, consider it a failure (keeps episodes short)
            if self.manhattan_distance(self.player_pos, self.key_pos) > self.failure_distance:
                self.finished = True
                return -1.0

            return 0.0
        else:
            # illegal move off-map: treat as failure if it's sign of bad policy
            self.last_time_key_was_touch += 1
            if self.manhattan_distance(self.player_pos, self.key_pos) > self.failure_distance:
                self.finished = True
                return -1.0
            return 0.0

    def render(self):
        H = self.size_y
        W = self.size_x
        board = torch.ones((H, W, 3), dtype=torch.float32)
        for (y,x) in self.apples:
            board[y, x] = torch.tensor((0.0, 0.0, 1.0))
        py, px = self.player_pos
        if self.player_pos in self.apples:
            board[py, px] = torch.tensor((1.0, 0.0, 1.0))
        else:
            board[py, px] = torch.tensor((1.0, 0.0, 0.0))
        ky, kx = self.key_pos
        board[ky, kx] = torch.tensor((0.0, 1.0, 0.0))
        return torch.flatten(board)

    def render_for_human(self, filename="env_render.png", cell_size=36, show_grid=True, grid_line_width=1):
        # unchanged from your original
        width = self.size_x * cell_size
        height = self.size_y * cell_size
        img = Image.new("RGB", (width, height), (255,255,255))
        draw = ImageDraw.Draw(img)
        for (ay, ax) in self.apples:
            y0 = ay*cell_size
            x0 = ax*cell_size
            inset = cell_size // 6
            draw.ellipse([x0 + inset, y0 + inset, x0 + cell_size - inset - 1, y0 + cell_size - inset - 1], fill=(0,255,0))
        py, px = self.player_pos
        x0 = px * cell_size
        y0 = py * cell_size
        inset = cell_size // 8
        draw.polygon([(x0+(cell_size)/2, y0 + inset),(x0+inset, y0 + cell_size - inset - 1), (x0+cell_size-inset,y0 + cell_size - inset - 1) ], fill=(255,0,0))
        ky, kx = self.key_pos
        x0 = kx * cell_size
        y0 = ky * cell_size
        inset = cell_size // 8
        draw.rectangle([x0+inset, y0+inset, x0 + cell_size - inset, y0 + cell_size - inset], fill=(255,255, 0))
        if show_grid:
            for cx in range(self.size_x + 1):
                x = cx * cell_size
                draw.line([(x, 0), (x, height)], fill=(150,150,150), width=grid_line_width)
            for cy in range(self.size_y + 1):
                y = cy * cell_size
                draw.line([(0, y), (width, y)], fill=(150,150,150), width=grid_line_width)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename)
        return filename

    def load(self, player_pos, key_pos, apples):
        self.finished = False
        self.left_steps = self.max_steps
        self.player_pos = player_pos
        self.key_pos = key_pos
        self.apples = apples.copy()
        self.last_time_key_was_touch = 0
        if self.apples:
            self.prev_min_manhattan = min(self.manhattan_distance(self.key_pos, a) for a in self.apples)
        else:
            self.prev_min_manhattan = 0



if __name__ == "__main__":
    env = ApplePlacerEnvironment((12,12), 80)
    #                      (distance, apples)
    env.curriculum_stage = (1,7)
    env.initialize()
    i = 0
    while True:
        env.render_for_human(filename=f"images/step{i}.png")
        #print(env.render().shape) : 432
        print("w=1 s=2 a=3 d=4")
        cmd = input(">> ").strip()
        if cmd == "q":
            break
        if cmd in ["1","2","3","4"]:
            if not env.finished:
                r = env.update(int(cmd)-1)
                print("reward:", r, "finished:", env.finished)
            else:
                break
            i += 1
            continue
        print("unknown command")