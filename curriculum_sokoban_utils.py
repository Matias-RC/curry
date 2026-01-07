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

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d,
        H,
        T,
        bias=False,
        dropout=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0

        # key, query, value projections for all heads, but in a batch
        # output is 3X the dimension because it includes key, query and value
        self.c_attn = nn.Linear(d, 3*d, bias=bias)

        # projection of concatenated attention head outputs
        self.c_proj = nn.Linear(d, d, bias=bias)

        # dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d

        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(T, T))
                                    .view(1, 1, T, T))

    def forward(self, x):
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality

        # compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v  = self.c_attn(x).split(self.d, dim=2) # [B, T, d]

        # reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2) # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        # compute the attention matrix, perform masking, and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # [B, H, T, T]
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # compute output vectors for each token
        y = att @ v # [B, H, T, d // H]

        # concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)
        y = self.resid_dropout(self.c_proj(y))
        return y

