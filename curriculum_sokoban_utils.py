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
class BoardEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        conv_layers = []
        in_channels = 3

        for out_channels, kernel_size in config.conv_layers:
            conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2
                )
            )
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = in_channels

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # -> [1, C, H, W]

        h = self.conv(x)                 # -> [B, C, H', W']
        h = self.global_pool(h)          # -> [B, C, 1, 1]
        h = h.view(h.size(0), -1)        # -> [B, C]

        return h

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

#Load the agent
from architecture import RotaryEmbedding, MLP, CausalAttentionBlock, CausalQueryAttention

class Config:
    n_embd = 64
    n_head = 4
    n_layers = 2
    block_size = 500
    dropout = 0.0
    bias = False
    conv_layers = [[32, 3], [64, 3]]
    grid_size = (6,6)
    initial_max_steps = 2
    replay_pool_capacity = 500
    replay_sample_prob = 0.2
    alpha = 0.9


class ActorCritic(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.encoder = BoardEncoder(self.config)
        self.layers = nn.ModuleList()
        for _ in range(self.config.n_layers):
            self.layers.append(CausalAttentionBlock(self.config))

        self.policy = nn.Sequential(nn.Linear(self.config.n_embd, self.config.n_embd*2), nn.ReLU(), nn.Linear(self.config.n_embd*2, 4))
        self.value = nn.Sequential(nn.Linear(self.config.n_embd, self.config.n_embd*2), nn.ReLU(), nn.Linear(self.config.n_embd*2, 1))

    def forward(self, x, kv_cache=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        h = self.encoder(x)
        for idx, layer in enumerate(self.layers):
            h, cache = layer(h, layer_past=kv_cache[idx])
            kv_cache[idx] = cache

        return self.policy(h), self.value(h), kv_cache
    
    def fast_causal_forward(self, B:torch.Tensor, f):
        """
        Takes advantage of compounded plans
        """
        T = self.encoder(B) # (B, Ch, H, W) -> (B, T, d)
        i  =  1
        kv_cache = None
        outputs = []
        while i<T.size(1):
            k = f(i)
            if i == 1:
                kv_cache = []
                for layer in self.layers:
                    out, cache = layer(T[:,i-1:i-1+k,:], k=None)
                    kv_cache.append(cache)
                    outputs.append(out)
            else:
                for idx, layer in enumerate(self.layers):
                    out, cache = layer(T[:,i-1:i-1+k,:], k=kv_cache[idx])
                    kv_cache[idx] = cache
                    outputs.append(out)
            i += k
        

agent = ActorCritic(Config())
env_helper = SokobanCurriculumEnvironment(Config())

class Loose_curr:
    push_pos = (0,0)

def sample_goal_box(env):
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    while True:
        d = random.choice(dirs)
        gy = random.randint(0, 5)
        gx = random.randint(0, 5)

        by, bx = gy - d[0], gx - d[1]
        py, px = by - d[0], bx - d[1]

        G = (gy, gx)
        B = (by, bx)
        P = (py, px)

        if env.in_bounds(B) and env.in_bounds(P):
            return G, B, P
        
def sample_agent_path(env, P, B, N):
    visited = {P}
    path = [P]
    for _ in range(N):
        y, x = path[-1]
        candidates = []
        for dy, dx in env.action_map:
            ny, nx = y + dy, x + dx
            np = (ny, nx)
            if not env.in_bounds(np):
                continue
            if np in visited:
                continue
            if np == B:
                continue
            candidates.append(np)
        if not candidates:
            return None
        nxt = random.choice(candidates)
        visited.add(nxt)
        path.append(nxt)
    return path[::-1]

def sample_phase_N(env:SokobanCurriculumEnvironment, N):
    while True:
        G, B, P = sample_goal_box(env)
        path = sample_agent_path(env, P, B, N)
        if path is None:
            continue
        
        A = path[0]
        
        return A, B, G, P

def reset_zero(env:SokobanCurriculumEnvironment, use_replay:bool, stage:int, curriculum_obj:Loose_curr):
    # possibly sample from replay pool
    steps = [2, 4, 8, 12]
    if use_replay and len(env.replay_pool) > 0 and random.random() < env.replay_sample_prob:
        a, b, g, p = env.replay_pool.popleft()
    else:
        a, b, g, p = sample_phase_N(env, stage)
    env.load((a, set((b,)), set((g,)), set(())))
    curriculum_obj.push_pos = p
    env.max_steps = steps[stage]
    env.left_steps = env.max_steps
    env.finished = False

    return env.render_state()
def update_zero(env:SokobanCurriculumEnvironment, curriculum_obj:Loose_curr,a:int):
    def compute_phi(agent, box, goal, alpha):
        by, bx = box
        gy, gx = goal
        dy = int(math.copysign(1, gy - by)) if gy-by != 0 else 0
        dx = int(math.copysign(1, gx - bx)) if gx - bx != 0 else 0
        #these are the directions in which to push the box.
        candidates = []
        if dy != 0:
            if env.in_bounds((by-dy,bx)):
                candidates.append((by-dy,bx))
        if dx != 0:
            if env.in_bounds((by,bx-dx)):
                candidates.append((by,bx-dx))
        if not candidates:
            return None
        i, _ = min(enumerate(candidates), key=lambda x:env.manhattan(x[1], agent))
        push_pos = candidates[i]
        d_box_goal = env.manhattan(box, goal)
        d_agent_push = env.manhattan(agent, push_pos)
        phi = -( (1 - alpha) * d_box_goal + alpha * d_agent_push )
        return float(phi)
    a_old = env.agent_pos
    boxes_old = env.boxes
    phi_prev = compute_phi(env.agent_pos, env.boxes.copy().pop(),env.goals.copy().pop(),env.config.alpha)
    if phi_prev == None:
        phi_prev = 0
    a_pos, boxes, tup, is_finished = env_helper._apply_action(a)
    env.agent_pos = a_pos
    env.boxes  = boxes
    
    phi_post = compute_phi(env.agent_pos, env.boxes.copy().pop(),env.goals.copy().pop(),env.config.alpha)
    
    reward = (is_finished)*-0.1 +is_finished*5
    if is_finished or phi_post == None:
        phi_post = 0
        env.finished = True
    shaped_reward = reward + env.config.gamma * phi_post - phi_prev

    


