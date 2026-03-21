"""
The Policy has a initial memory that is parametric that tries to acomodate itself to the necesities of all envs rest is trivial
"""
from attpolicy import RotaryEmbedding2D, RotaryEmbedding1D

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gymnasium import spaces

import torch as th
import torch.nn as nn


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_h, max_w, base=10000):
        super().__init__()
        dim_x = dim // 2
        dim_y = dim - dim_x
        
        inv_freq_x = 1.0 / (base ** (th.arange(0, dim_x, 2).float() / dim_x))
        inv_freq_y = 1.0 / (base ** (th.arange(0, dim_y, 2).float() / dim_y))
        
        self.register_buffer("inv_freq_x", inv_freq_x)
        self.register_buffer("inv_freq_y", inv_freq_y)
        self.max_h = max_h
        self.max_w = max_w
        self.cache = None

    def update_cache(self, device, dtype):
        if self.cache is not None and self.cos_cached.device == device: 
            return

        y = th.arange(self.max_h, device=device, dtype=self.inv_freq_y.dtype)
        x = th.arange(self.max_w, device=device, dtype=self.inv_freq_x.dtype)
        
        freqs_x = th.einsum("i, j -> ij", x, self.inv_freq_x) 
        freqs_y = th.einsum("i, j -> ij", y, self.inv_freq_y) 
        
        emb_x = freqs_x.unsqueeze(0).repeat(self.max_h, 1, 1)
        emb_y = freqs_y.unsqueeze(1).repeat(1, self.max_w, 1)
        
        emb = th.cat((emb_x, emb_y), dim=-1)
        emb = emb.view(-1, emb.shape[-1]) # (L, D/2)
        
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
        self.cache = True

    def apply_rope(self, x):
        # x: (B, Heads, L, D_head)
        # We slice the cache to current seq_len (in case of dynamic sizes, though usually fixed here)
        seq_len = x.shape[2]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        cos = th.cat((cos, cos), dim=-1)
        sin = th.cat((sin, sin), dim=-1)
        
        x1, x2 = x.chunk(2, dim=-1)
        rotate_half = th.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (rotate_half * sin)

class RotaryEmbedding1D(nn.Module):
    """1D RoPE with caching to match the policy's update_cache calls."""
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (th.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cache = None
        self.cos_cached = None
        self.sin_cached = None

    def update_cache(self, seq_len, device, dtype):
        if self.cache is not None and self.cache >= seq_len and self.cos_cached.device == device:
            return
        t = th.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = th.einsum("i, j -> ij", t, self.inv_freq)
        emb = th.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
        self.cache = seq_len

    def apply_rope(self, x):
        # x: (B, Heads, L, D_head)
        seq_len = x.shape[2]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        rotate_half = th.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotate_half * sin)

class AttBlockSelectivePE(nn.Module):
    """
    Only aplies the rope module to the first n*m items in seq, which correspond to the grid tokens, leaving the special tokens and prefix unaffected by positional encoding.
    """
    def __init__(self, hidden_size, num_heads, rope_module, grid_shape, full_length):
        super().__init__()
        # Num not positionally encoded tokens = full_length - grid_shape[0] * grid_shape[1]
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope_module
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        self.rope_module = rope_module
        self.grid_len = grid_shape[0] * grid_shape[1]
        self.full_length = full_length
    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        shortcut = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_spatial, q_other = q[:, :, :self.grid_len, :], q[:, :, self.grid_len:, :]
        k_spatial, k_other = k[:, :, :self.grid_len, :], k[:, :, self.grid_len:, :]

        #Apply rope only to spatial tokens
        q_spatial = self.rope.apply_rope(q_spatial)
        k_spatial = self.rope.apply_rope(k_spatial)

        #Recombine
        q = th.cat((q_spatial, q_other), dim=2)
        k = th.cat((k_spatial, k_other), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2) 
            attn = attn.masked_fill(mask_expanded, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class FullAtt(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, 
                 hidden_size: int = 128, num_heads: int = 4, layers: int = 3,
                 num_special_tokens: int = 1, length_prefix: int = 4):
        super().__init__(observation_space, features_dim)
        
        self.H_grid, self.W_grid, self.C, self.wx, self.wy = observation_space.shape
        self.hidden_size = hidden_size
        self.seq_len = self.H_grid * self.W_grid + 2*num_special_tokens + length_prefix
        self.prefix_length = length_prefix
        self.pi_length = num_special_tokens
        self.vf_length = num_special_tokens
        # Input Projection
        input_dim = self.C * self.wx * self.wy
        self.mlp_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        # Policy/Vf special tokens
        self.policy_token = nn.Parameter(th.randn(num_special_tokens, hidden_size))
        self.vf_token = nn.Parameter(th.randn(num_special_tokens, hidden_size))

        # Prefix
        self.prefix = nn.Parameter(th.randn(length_prefix, hidden_size))

        # RoPE
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding2D(head_dim, self.H_grid, self.W_grid)

        self.attBlocks = nn.ModulesList(*[
            AttBlockSelectivePE(hidden_size, num_heads, self.rope, (self.H_grid, self.W_grid), self.seq_len + 2*num_special_tokens + length_prefix)
            for _ in range(layers)
        ])
    
    def forward(self, observations):
        if observations.dim() == 5:
            observations = observations.unsqueeze(0)
        B = observations.shape[0]
        
        is_padding = (observations[:,:,:,0,0,0] == -1)
        padding_mask = is_padding.reshape(B, -1)

        # Input Clening and proj
        clean_obs = observations.clone()
        clean_obs[clean_obs == -1] = 0

        obs_flat = clean_obs.reshape(B, self.H_grid * self.W_grid, -1)
        x = self.mlp_extractor(obs_flat)
        # Append special tokens and prefix
        policy_tokens = self.policy_token.unsqueeze(0).expand(B, -1, -1)
        vf_tokens = self.vf_token.unsqueeze(0).expand(B, -1, -1)
        prefix_tokens = self.prefix.unsqueeze(0).expand(B, -1, -1)
        x = th.cat((x, policy_tokens, vf_tokens, prefix_tokens), dim=1)
        # Extend mask for special tokens and prefix (not masked)
        extended_padding_mask = th.cat((padding_mask, th.zeros(B, 2 + self.prefix.shape[0], device=padding_mask.device, dtype=padding_mask.dtype)), dim=1)

        # Pass through attention blocks
        for block in self.attBlocks:
            x = block(x, padding_mask=extended_padding_mask)
        # Extract prefix, policy and vf
        l = self.H_grid * self.W_grid
        policy_out = x[:, l:l+self.pi_length, :]
        vf_out = x[:, l+self.pi_length:l+self.vf_length+self.pi_length, :]
        prefix_out = x[:, l+self.pi_length+self.vf_length:, :]
        return policy_out, vf_out, prefix_out
    
class AttBasedPolicyClass(ActorCriticPolicy):
    features_extractor: th.nn.Module
    mlp_extractor: th.nn.Module

    def __init__(self):
        super().__init__()