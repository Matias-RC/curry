import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
        if self.cache is not None: return

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

class MaskedAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_module):
        super().__init__()
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

    def forward(self, x, padding_mask=None):
        """
        x: (B, L, D)
        padding_mask: (B, L) where True indicates PADDED (ignored) value.
        """
        B, L, D = x.shape
        shortcut = x
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        q = self.rope.apply_rope(q)
        k = self.rope.apply_rope(k)
        
        # Attention Score
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, H, L, L)
        
        # --- MASKING LOGIC ---
        if padding_mask is not None:
            # padding_mask is (B, L). We need (B, 1, 1, L) to broadcast across Heads and Query-dim
            # We mask positions where padding_mask is True
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2) 
            attn = attn.masked_fill(mask_expanded, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = shortcut + x
        
        x = x + self.mlp(self.norm2(x))
        return x

class SokoPlayerCentricAtt(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, 
                 hidden_size: int = 128, num_heads: int = 4, layers: int = 3):
        
        # The wrapper returns: (Hw, Ww, C, ws, ws)
        # SB3 inputs: (B, Hw, Ww, C, ws, ws)
        
        super().__init__(observation_space, features_dim)
        
        self.H_grid, self.W_grid, self.C, self.wx, self.wy = observation_space.shape
        self.hidden_size = hidden_size
        self.seq_len = self.H_grid * self.W_grid
        
        # Input Projection
        input_dim = self.C * self.wx * self.wy
        self.mlp_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # RoPE
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding2D(head_dim, self.H_grid, self.W_grid)
        
        # Transformer Stack
        self.blocks = nn.ModuleList([
            MaskedAttentionLayer(hidden_size, num_heads, self.rope) 
            for _ in range(layers)
        ])
        
        # Final Projection (combines Raw + Context)
        self.final_proj = nn.Linear(hidden_size , features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
            if observations.dim() == 5: 
                observations = observations.unsqueeze(0)
                
            B = observations.shape[0]
            
            # 1. Padding Mask Generation
            is_padding = (observations[:, :, :, 0, 0, 0] == -1) 
            padding_mask = is_padding.reshape(B, -1) 
            
            # 2. Input Cleaning & Projection
            clean_obs = observations.clone()
            clean_obs[clean_obs == -1] = 0
            
            # FIX: .reshape() handles non-contiguous memory from VecEnvs
            obs_flat = clean_obs.reshape(B, self.H_grid, self.W_grid, -1)
            
            embeddings = self.mlp_extractor(obs_flat)
            x = embeddings.reshape(B, self.seq_len, self.hidden_size)
            
            # 3. Player Localization
            player_map = observations[..., 3, :, :].sum(dim=(-1, -2)) 
            player_map_flat = player_map.reshape(B, -1)
            player_indices = player_map_flat.argmax(dim=1) 
            
            gather_indices = player_indices.view(B, 1, 1).expand(-1, -1, self.hidden_size)
            raw_player_token = x.gather(1, gather_indices).squeeze(1) 
            
            # 4. Masked Transformer Stack
            self.rope.update_cache(x.device, x.dtype)
            
            for block in self.blocks:
                x = block(x, padding_mask=padding_mask)
                
            # 5. Extract Contextualized Player Token
            attended_player_token = x.gather(1, gather_indices).squeeze(1) 
            
            # 6. Final Readout
            # -combined = th.cat([attended_player_token, raw_player_token], dim=1)
            return self.final_proj(attended_player_token)