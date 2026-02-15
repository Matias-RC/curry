import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

import numpy as np
from gym_sokoban.envs import SokobanEnv
from sokoban_wrapper import SokobanCompactWrapper, SokobanRetriesWrapper, SokoRetriesCurriculum, SokoCanonicalWithAttPadding
import os
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

level = th.Tensor([
    [
        [1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,1,1,1,1],
        [1,1,0,0,1,1,1,1,1],
        [1,1,0,1,1,1,1,1,1],
        [1,0,0,0,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,0]
    ],
    [
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0]
    ],
    [
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0]
    ],
    [
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0]
    ]   
])

registered_level_space = set()


for i in range(0, len(level[0][:,0].tolist()), 3):
    for  j in range(0, len(level[0][0].tolist()), 3):
        if (level[0][i:i+3,j:j+3] == 0).any():
            registered_level_space.add((i,j))



#===============other test


def make_env(dim_room, max_steps, num_boxes, num_retries, seed=42):
    env = SokobanRetriesWrapper(
        max_retries=num_retries,
        dim_room=dim_room,
        max_steps=max_steps,
        num_boxes=num_boxes,
        render_mode="rgb_array",
    )
    env = SokobanCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

my_schedule_dic = {(("dim_room", (8,8)),("max_steps", 30), ("num_boxes", 1),("num_gen_steps", int(1.7*(8+8)))):0.5,
                   (("dim_room", (9,9)),("max_steps", 45), ("num_boxes", 2),("num_gen_steps", int(1.7*(9+9)))):0.5,
}

env_kwargs = dict(max_retries=2, dim_room=(8,8), max_steps=30, num_boxes=2, curriculum=True, schedule_dic=my_schedule_dic)

wrapper_kwargs = dict(canonical_shape=(9,9), window_size=3)


env = make_vec_env(SokoRetriesCurriculum, 8, 123, 100, wrapper_class=SokoCanonicalWithAttPadding,
             env_kwargs=env_kwargs, wrapper_kwargs=wrapper_kwargs)


obs_example = env.reset()


import torch as th
import torch.nn as nn
import math

class RoPE2D(nn.Module):
    def __init__(self, dim, base=1000):
        super().__init__()
        assert dim % 4 == 0, "embedding dim must be divisible by 4"
        self.dim = dim
        self.base = base

        half = dim // 2
        freq_dim = half // 2

        inv_freq = 1.0 / (base ** (th.arange(freq_dim) / freq_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (B, H, W, D)
        B, H, W, D = x.shape
        device = x.device

        half = D // 2
        x_x = x[..., :half]
        x_y = x[..., half:]

        pos_x = th.arange(W, device=device)
        pos_y = th.arange(H, device=device)

        freqs_x = th.einsum("w,f->wf", pos_x, self.inv_freq)
        freqs_y = th.einsum("h,f->hf", pos_y, self.inv_freq)

        cos_x = th.cos(freqs_x)[None, None, :, :]
        sin_x = th.sin(freqs_x)[None, None, :, :]

        cos_y = th.cos(freqs_y)[None, :, None, :]
        sin_y = th.sin(freqs_y)[None, :, None, :]

        x_x = self.rotate(x_x, cos_x, sin_x)
        x_y = self.rotate(x_y, cos_y, sin_y)

        return th.cat([x_x, x_y], dim=-1)

    def rotate(self, x, cos, sin):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos

        return th.stack((xr1, xr2), dim=-1).flatten(-2)

class MyAttPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_dim = 64

        self.features_extractor = nn.Linear(36, self.embed_dim)

        self.rope = RoPE2D(self.embed_dim)

        self.retriever = nn.Parameter(th.randn(self.embed_dim))

        self.wxk = nn.Linear(self.embed_dim, self.embed_dim)
        self.wxv = nn.Linear(self.embed_dim, self.embed_dim)
        self.wrq = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, obs):
        if not isinstance(obs, th.Tensor):
            obs = th.as_tensor(obs, dtype=th.float32)

        B, Hc, Wc, C, hw, ww = obs.shape
        N = Hc * Wc

        valid_mask = ~(obs == -1).all(dim=(-1, -2, -3))
        print(valid_mask[0])
        valid_mask = valid_mask.view(B, N)

        clean = obs.clone()
        clean[obs == -1] = 0


        x = clean.view(B * N, C * hw * ww)
        x = self.features_extractor(x)
        x = x.view(B, Hc, Wc, self.embed_dim)

        x = self.rope(x)

        x = x.view(B, N, self.embed_dim)

        K = self.wxk(x)
        V = self.wxv(x)

        q = self.wrq(self.retriever).expand(B, -1).unsqueeze(1)

        att = (q @ K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        att = att.squeeze(1)

        att = att.masked_fill(~valid_mask, -1e9)

        weights = th.softmax(att, dim=-1)

        out = (weights.unsqueeze(-1) * V).sum(dim=1)

        return out
    
policy = MyAttPolicy()

policy(obs_example)


new_obs, rewards, dones, infos = env.step(np.array([0,0,0,0,0,0,0,0]))
