import argparse

from collections import deque, defaultdict
import random
import math
import copy
import time
from typing import Tuple, List, Dict, Optional

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

# Heavy curriculum manipulation.


class SokobanCurriculumEnvironment:
    def __init__(self, config= None):
        assert config is not None, "Config must be provided"

        self.config = config

        self.size_x, self.size_y = config.grid_size #Not partial obs yet
        self.action_map = [(-1,0),(1,0),(0,-1),(0,1)]

        self.agent_pos = (0,0)
        self.boxes = set()
        self.goals =  set()
        self.walls = set()
        self.outer_shell = set()

        for i in range(-1,self.size_y+1):
            self.outer_shell.add((i, 0))
            self.outer_shell.add((i, self.size_x))
        for i in range(self.size_x):
            self.outer_shell.add((-1, i))
            self.outer_shell.add((self.size_y, i))

        self.left_steps = config.initial_max_steps
        self.max_steps = config.initial_max_steps

        self.finished = False

        self.replay_pool =deque(maxlen=self.config.replay_pool_capacity)
        self.replay_sample_prob = self.config.replay_sample_prob

        self.stage = 0
        self.sub_stage = 0
    
    def _is_push(self, agent_to, boxes):
        return agent_to in boxes
    
    def in_bounds(self, pos):
        return 0<=pos[0]<self.size_y and 0<=pos[1]<self.size_x
    
    def _state_succes(self, goals, boxes):
        return tuple(sorted(boxes)) == tuple(sorted(goals))
    
    def _is_dead_lock(self, boxes, walls, goals):
        walls = walls | self.outer_shell

        rotatePattern = [[0,1,2,3,4,5,6,7,8],
                         [2,5,8,1,4,7,0,3,6],
                         [0,1,2,3,4,5,6,7,8][::-1],
                         [2,5,8,1,4,7,0,3,6][::-1]]
        flipPattern = [[2,1,0,5,4,3,8,7,6],
                        [0,3,6,1,4,7,2,5,8],
                        [2,1,0,5,4,3,8,7,6][::-1],
                        [0,3,6,1,4,7,2,5,8][::-1]]

        allPattern = rotatePattern + flipPattern
        for box in boxes:
            if box not in goals:
                board = [(box[0]-1, box[1]-1), (box[0]-1, box[1]), (box[0]-1,box[1]+1),
                         (box[0],box[1]-1), (box[0],box[1]), (box[0],box[1]+1),
                         (box[0]+1,box[1]-1), (box[0]+1, box[1]), (box[0]+1, box[1]+1)]
                for pattern in allPattern:
                    newBoard = [board[i] for i in pattern]
                    if newBoard[1] in walls and newBoard[5] in walls: return True
                    elif newBoard[1] in boxes and newBoard[2] in walls and newBoard[5] in walls: return True
                    elif newBoard[1] in boxes and newBoard[2] in walls and newBoard[5] in boxes: return True
                    elif newBoard[1] in boxes and newBoard[2] in boxes and newBoard[5] in boxes: return True
                    elif newBoard[1] in boxes and newBoard[2] in walls and newBoard[3] in walls: return True
        return False 
    
    def _apply_action(self, action):
        dy, dx = self.action_map[action]
        a_old = self.agent_pos
        boxes_old = self.boxes.copy()

        a_new = (a_old[0]+ dy, a_old[1]+ dx)

        if not self.in_bounds(a_new):
            return a_old, boxes_old, (False, False, False), self._state_succes(self.goals, boxes_old)
        
        if self._is_push(a_new, boxes_old):
            b_target = (a_new[0]+dy,a_new[1]+dx)
            if not self.in_bounds(b_target) or b_target in boxes_old or b_target in self.walls:
                return a_old, boxes_old, (False, False, False), self._state_succes(self.goals, boxes_old)
            
            #Valid push: Remove new agent pos from box set and aggregate b_target

            boxes_new = boxes_old.copy()
            boxes_new.remove(a_new)
            boxes_new.add(b_target)

            return a_new, boxes_new, (True, b_target in self.goals, a_new in self.goals, self._is_dead_lock(boxes_new, self.walls, self.goals)), self._state_succes(self.goals, boxes_new)
        return a_new, boxes_old, (False,False,False), self._state_succes(self.goals, boxes_old)
    
    def render_state(self):
        H = self.size_y
        W = self.size_x

        obs = torch.zeros((3, H, W), dtype=torch.float32)

        ay, ax = self.agent_pos
        if self.agent_pos not in self.goals:
            obs[:,ay, ax] = torch.tensor([0.75, 0.5, 0.25])
        else:
            obs[:,ay, ax] = torch.tensor([0.25, 0.5, 0.75])
        for by, bx in (self.boxes - self.goals):
            obs[:,by, bx] = torch.tensor([0.25,0.75,0.5])
        for gy, gx in (self.goals-self.boxes):
            obs[:,gy, gx] = torch.tensor([0.5, 0.25, 0.75])
        for py, px in self.boxes & self.goals:
            obs[:,py, px] = torch.tensor([0.5, 0.75, 0.25])
        for wy, wx in self.walls:
            obs[:,wy, wx] = torch.tensor([1.0, 1.0, 1.0])

        return obs

    def load(self, tup):
        self.agent_pos, self.boxes, self.goals, self.walls = tup

    def export(self):
        return (self.agent_pos, self.boxes, self.goals, self.walls)
    
    def new_size(self, size_y, size_x):
        self.size_y = size_y
        self.size_x = size_x
        self.outer_shell.clear()

        for i in range(-1,self.size_y+1):
            self.outer_shell.add((i, 0))
            self.outer_shell.add((i, self.size_x))
        for i in range(self.size_x):
            self.outer_shell.add((-1, i))
            self.outer_shell.add((self.size_y, i))

    def add_to_replay(self, initial_state):
        self.replay_pool.append(initial_state)

    def reset_buffer(self):
        self.replay_pool.clear()
    
    def maybe_advance_stage(self, success_rate: float, avg_entropy: float, reset) -> bool:
        if success_rate >= self.config.advance_success_rate: # and avg_entropy <= self.config.advance_entropy_proportion * math.log(4):
            self.reset_buffer()
            for _ in range(self.config.replay_pool_capacity):
                reset(self, use_replay=False)
                self.add_to_replay((self.agent_pos, self.box_pos, self.goal_pos, self.push_pos))
            return True
        return False
    
    def render_for_human(self, filename="env_render.png", cell_size=36, show_grid=True, grid_line_width=1):
        # unchanged from your original
        width = self.size_x * cell_size
        height = self.size_y * cell_size
        img = Image.new("RGB", (width, height), (255,255,255))
        draw = ImageDraw.Draw(img)

        for gy, gx in (self.goals-self.boxes):
            y0 = gy*cell_size
            x0 = gx*cell_size
            inset = cell_size // 8
            draw.rectangle([x0+inset, y0+inset, x0 + cell_size - inset, y0 + cell_size - inset], fill=(128,64,191))           
        for by, bx in (self.boxes-self.goals):
            y0 = by*cell_size
            x0 = bx*cell_size
            inset = cell_size // 8
            draw.rectangle([x0+inset, y0+inset, x0+cell_size-inset, y0+cell_size-inset], fill=(64,191,128))
        for py, px in self.boxes & self.goals:
            y0 = py*cell_size
            x0 = px*cell_size
            inset = cell_size // 8
            draw.rectangle([x0+inset, y0+inset, x0+cell_size-inset, y0+cell_size-inset], fill=(128,191,64))          
        ay, ax = self.agent_pos
        y0 = ay*cell_size
        x0 = ax*cell_size
        inset = cell_size // 4
        if self.agent_pos not in self.goals:
            draw.polygon([(x0+(cell_size)/2, y0 + inset),(x0+inset, y0 + cell_size - inset - 1), (x0+cell_size-inset,y0 + cell_size - inset - 1) ], fill=(191,128,64))
        else:
            draw.rectangle([x0+inset, y0+inset, x0 + cell_size - inset, y0 + cell_size - inset], fill=(255,255,255))  
            draw.polygon([(x0+(cell_size)/2, y0 + inset),(x0+inset, y0 + cell_size - inset - 1), (x0+cell_size-inset,y0 + cell_size - inset - 1) ], fill=(64,128,191))

        for wy, wx in self.walls:
            y0 = wy*cell_size
            x0 = wx*cell_size

            draw.rectangle([x0,y0,x0+cell_size,y0+cell_size])

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename)
        return filename
    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def reset_stage_zero(self, replay=False):
        if replay and len(self.replay_pool)>0 and random.random() < self.replay_sample_prob:
            p, b, g = self.replay_pool.popleft()
        else:
            goal_pos = (random.randint(0, self.size_y-1), random.randint(0, self.size_x-1))
            g = set((goal_pos,))
            box_candidates = []
            for dy, dx in self.action_map:
                if 0 <= goal_pos[0]+2*dy < self.size_y and 0<= goal_pos[1]+2*dx<self.size_x:
                    pass
        pass
    def update_zero(self):
        pass