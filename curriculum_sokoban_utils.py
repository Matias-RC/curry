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

        self.left_steps = config.initial_max_steps
        self.max_steps = config.initial_max_steps

        self.finished = False

        self.replay_pool =deque(maxlen=self.config.replay_pool_capacity)
        self.replay_sample_prob = self.config.replay_sample_prob
    
    def render_state(self):
        H = self.size_y
        W = self.size_x

        obs = torch.zeros((3, H, W), dtype=torch.float32)

        ay, ax = self.agent_pos
        if self.agent_pos not in self.goals:
            obs[0:3, ay, ax] = torch.tensor([0.75, 0.5, 0.25])
        else:
            obs[0:3, ay, ax] = torch.tensor([0.25, 0.5, 0.75])
        for by, bx in (self.boxes - self.goals):
            obs[0:3, by, bx] = torch.tensor([0.25,0.75,0.5])
        for gy, gx in (self.goals-self.boxes):
            obs[0:3, gy, gx] = torch.tensor([0.5, 0.25, 0.75])
        for py, px in self.boxes & self.goals:
            obs[0:3, py, px] = torch.tensor([0.5, 0.75, 0.25])
        for wy, wx in self.walls:
            obs[0:3, wy, wx] = torch.tensor([1.0, 1.0, 1.0])

        return obs
    
    def _is_push(self, agent_to, boxes):
        return agent_to in boxes

    def load(self, tup):
        self.agent_pos, self.boxes, self.goals, self.walls = tup
    def export(self):
        return (self.agent_pos, self.boxes, self.goals, self.walls)
    def add_to_replay(self, initial_state):
        self.replay_pool.append(initial_state)
    def reset_buffer(self):
        self.replay_pool.clear()
    
    def maybe_advance_stage(self, success_rate: float, avg_entropy: float) -> bool:
        if success_rate >= self.config.advance_success_rate: # and avg_entropy <= self.config.advance_entropy_proportion * math.log(4):
            self.reset_buffer()
            for _ in range(self.config.replay_pool_capacity):
                self.reset(use_replay=False)
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

class CausalQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Register a large buffer for the causal mask (lower triangular)
        # We assume max_block_size is the maximum T ever seen
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, k=None):
        B, T, C = x.size() # Batch, Time (Sequence Length), Channels (Embed Dim)
        if k is None:
            k = T        
        assert k <= T


        qkv = self.c_attn(x)
        q, k_vec, v = qkv.split(self.n_embd, dim=2)
        
        # Q shape: [B, k, C]
        q = q[:, -k:, :] 

        # k_vec and v remain length T
        k_vec = k_vec.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, k, self.n_head, C // self.n_head).transpose(1, 2)         # (B, nh, k, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)         # (B, nh, T, hs)

        # 4. Compute Attention Scores
        # (B, nh, k, hs) x (B, nh, hs, T) -> (B, nh, k, T)
        att = (q @ k_vec.transpose(-2, -1)) * (1.0 / math.sqrt(k_vec.size(-1)))

        # 5. Apply Causal Mask
        # We need the mask to align such that the last query sees everything, 
        # and previous queries see their respective pasts.
        # We take the bottom k rows of the T x T causal mask.
        mask_slice = self.bias[:, :, -k:, :T]
        att = att.masked_fill(mask_slice == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 6. Aggregate Values
        # (B, nh, k, T) x (B, nh, T, hs) -> (B, nh, k, hs)
        y = att @ v 
        
        # 7. Reassemble Heads
        y = y.transpose(1, 2).contiguous().view(B, k, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y
