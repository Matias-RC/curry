import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, Union, Type
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    make_proba_distribution, 
    CategoricalDistribution, 
    DiagGaussianDistribution,
    BernoulliDistribution,
    MultiCategoricalDistribution
)

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

class RotaryEmbedding1D(nn.Module):
    """1D RoPE with caching to match the policy's update_cache calls."""
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (th.arange(0, dim, 2).float() / dim))
        self.cache = None
        self.cos_cached = None
        self.sin_cached = None

    def update_cache(self, seq_len, device, dtype):
        if self.cache is not None and self.cache >= seq_len:
            return
        t = th.arange(seq_len, device=device).type_as(self.inv_freq)
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


class PrefixMaskedAttentionLayer(nn.Module):
    """
    A modified attention layer that strictly prevents RoPE from being applied 
    to the learned prefix tokens, applying it ONLY to the spatial sequence.
    """
    def __init__(self, hidden_size, num_heads, rope_module, num_prefixes):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope_module
        self.num_prefixes = num_prefixes
        
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
        B, L, D = x.shape
        shortcut = x
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # --- ISOLATE ROPE ---
        # Split Q and K into [Prefixes, Spatial]
        q_prefix, q_spatial = q[:, :, :self.num_prefixes, :], q[:, :, self.num_prefixes:, :]
        k_prefix, k_spatial = k[:, :, :self.num_prefixes, :], k[:, :, self.num_prefixes:, :]
        
        # Apply RoPE ONLY to the spatial tokens
        q_spatial = self.rope.apply_rope(q_spatial)
        k_spatial = self.rope.apply_rope(k_spatial)
        
        # Recombine
        q = th.cat([q_prefix, q_spatial], dim=2)
        k = th.cat([k_prefix, k_spatial], dim=2)
        
        # Standard Attention
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

class TemporalAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_1d_module):
        super().__init__()
        self.layer = PrefixMaskedAttentionLayer(
            hidden_size, num_heads, rope_1d_module, num_prefixes=0 # 1D RoPE applies to whole sequence 
        )
        
    def forward(self, x, padding_mask=None):
        return self.layer(x, padding_mask)

# --- Main Policy Class ---

class ExperiencerActorCritic(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        hidden_size: int = 128,
        num_prefixes: int = 4,
        num_heads: int = 4,
        attn_layers: int = 2,
        *args,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_prefixes = num_prefixes
        self.num_heads = num_heads
        
        super().__init__(
            observation_space, 
            action_space, 
            lr_schedule,
            features_extractor_class=SokoPlayerCentricAtt,
            features_extractor_kwargs=dict(
                hidden_size=hidden_size, 
                num_heads=num_heads,
                features_dim=hidden_size
            ),
            *args, 
            **kwargs
        )

        # 2. Initial Prefix Generator
        self.prefix_init_net = nn.Linear(self.features_dim, num_prefixes * hidden_size)

        # 3. Standard Forward Pass Modules (Separated Pi and Vf)
        self.pi_attn_layers = nn.ModuleList([
            PrefixMaskedAttentionLayer(hidden_size, num_heads, self.features_extractor.rope, num_prefixes)
            for _ in range(attn_layers)
        ])
        
        self.vf_attn_layers = nn.ModuleList([
            PrefixMaskedAttentionLayer(hidden_size, num_heads, self.features_extractor.rope, num_prefixes)
            for _ in range(attn_layers)
        ])

        # 4. The Thinker Modules (Used exclusively in upgrade_prefix_with_trajectory)
        self.thinker_spatial= nn.ModuleList([
            PrefixMaskedAttentionLayer(hidden_size, num_heads, self.features_extractor.rope, num_prefixes)
            for _ in range(attn_layers)
        ])
        
        self.rope1d = RotaryEmbedding1D(hidden_size // num_heads)
        self.thinker_temporal = TemporalAttentionLayer(hidden_size, num_heads, self.rope1d)

    def _build_mlp_extractor(self) -> None:
        """Override to prevent standard SB3 MLP construction. We route manually."""
        self.mlp_extractor = nn.Module()
        self.mlp_extractor.latent_dim_pi = self.hidden_size
        self.mlp_extractor.latent_dim_vf = self.hidden_size

    def make_initial_prefix(self, obs: th.Tensor) -> th.Tensor:
        """Step 2: Generate initial prefixes from isolated $x_{(i,j)}$."""
        # backbone isolates the player token natively
        player_centric_features = self.features_extractor(obs) 
        prefixes_flat = self.prefix_init_net(player_centric_features)
        return prefixes_flat.view(-1, self.num_prefixes, self.hidden_size)

    def _get_backbone_outputs(self, obs: th.Tensor):
        """Helper to run the backbone and isolate spatial tokens + player coords."""
        B = obs.shape[0]
        
        # Generate Padding Mask
        is_padding = (obs[:, :, :, 0, 0, 0] == -1) 
        padding_mask = is_padding.reshape(B, -1) 
        
        # Clean and embed
        clean_obs = obs.clone()
        clean_obs[clean_obs == -1] = 0
        obs_flat = clean_obs.reshape(B, self.features_extractor.H_grid, self.features_extractor.W_grid, -1)
        embeddings = self.features_extractor.mlp_extractor(obs_flat)
        x = embeddings.reshape(B, self.features_extractor.seq_len, self.hidden_size)
        
        # Backbone Transformer Blocks (Full Attention over grid)
        self.features_extractor.rope.update_cache(x.device, x.dtype)
        for block in self.features_extractor.blocks:
            x = block(x, padding_mask=padding_mask)
            
        # Player Localization
        player_map = obs[..., 3, :, :].sum(dim=(-1, -2)).reshape(B, -1)
        player_indices = player_map.argmax(dim=1)
        
        return x, padding_mask, player_indices

    def forward(self, obs: th.Tensor, prefix: th.Tensor, deterministic: bool = False):
        """Step 3: Standard Forward Pass using separated Pi and Vf attention layers."""
        B = obs.shape[0]
        
        # Get raw spatial tokens from backbone
        spatial_tokens, spatial_mask, player_indices = self._get_backbone_outputs(obs)
        
        # Combine [Prefix, Spatial]
        combined_seq = th.cat([prefix, spatial_tokens], dim=1)
        
        # Mask setup: Prefixes are never masked
        prefix_mask = th.zeros((B, self.num_prefixes), device=obs.device, dtype=th.bool)
        combined_mask = th.cat([prefix_mask, spatial_mask], dim=1)
        
        # --- Actor Pass (Pi) ---
        pi_seq = combined_seq
        for layer in self.pi_attn_layers:
            pi_seq = layer(pi_seq, padding_mask=combined_mask)
            
        # --- Critic Pass (Vf) ---
        vf_seq = combined_seq
        for layer in self.vf_attn_layers:
            vf_seq = layer(vf_seq, padding_mask=combined_mask)
            
        # Isolate attended player token at $(i,j)$ for both
        gather_indices = player_indices.view(B, 1, 1).expand(-1, -1, self.hidden_size)
        
        pi_attended_spatial = pi_seq[:, self.num_prefixes:, :]
        player_token_pi = pi_attended_spatial.gather(1, gather_indices).squeeze(1)
        
        vf_attended_spatial = vf_seq[:, self.num_prefixes:, :]
        player_token_vf = vf_attended_spatial.gather(1, gather_indices).squeeze(1)
        
        # Generate Actions and Values
        distribution = self._get_action_dist_from_latent(player_token_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(player_token_vf)
        
        return actions, values, log_prob

    def upgrade_prefix_with_trajectory(self, prev_prefix, trajectory_of_obs, trajectory_mask):
            """
            Inputs:
                prev_prefix: (B, P, D)
                trajectory_of_obs: (B, S, ...) repeated at the end if S < max_len
                trajectory_mask: (B, S) where True means PAD (ignore), False means KEEP
            """
            B, S = trajectory_of_obs.shape[:2]
            
            # 1. Spatial Processing
            flat_obs = trajectory_of_obs.reshape(B * S, *trajectory_of_obs.shape[2:])
            spatial_tokens, spatial_mask, _ = self._get_backbone_outputs(flat_obs)
            
            exp_prefix = prev_prefix.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, self.num_prefixes, self.hidden_size)
            combined_seq = th.cat([exp_prefix, spatial_tokens], dim=1)
            
            # Spatial prefix mask (prefixes are always valid)
            prefix_mask_spat = th.zeros((B * S, self.num_prefixes), device=flat_obs.device, dtype=th.bool)
            combined_mask_spat = th.cat([prefix_mask_spat, spatial_mask], dim=1)
            
            for layer in self.thinker_spatial:
                combined_seq = layer(combined_seq, padding_mask=combined_mask_spat)
                
            mod_prefixes = combined_seq[:, :self.num_prefixes, :].reshape(B, S, self.num_prefixes, self.hidden_size)

            # 2. Temporal Alignment
            # Construct temporal sequence: [Original, Step1, Step2, ...]
            temporal_stack = th.cat([prev_prefix.unsqueeze(1), mod_prefixes], dim=1) # (B, 1+S, P, D)
            
            # Reshape to treat each prefix slot as an independent sequence
            temporal_seq = temporal_stack.transpose(1, 2).reshape(B * self.num_prefixes, 1 + S, self.hidden_size)
            
            # Construct Temporal Mask
            # Index 0 (original prefix) is always False (keep). 
            # Indices 1..S follow trajectory_mask.
            orig_mask = th.zeros((B, 1), device=trajectory_mask.device, dtype=th.bool)
            full_traj_mask = th.cat([orig_mask, trajectory_mask], dim=1) # (B, 1+S)
            
            # Repeat mask for all prefix slots
            temporal_mask = full_traj_mask.unsqueeze(1).expand(-1, self.num_prefixes, -1).reshape(B * self.num_prefixes, 1 + S)
            
            # 3. Apply 1D Thinker
            self.rope1d.update_cache(1 + S, temporal_seq.device, temporal_seq.dtype)
            upgraded_seq = self.thinker_temporal(temporal_seq, padding_mask=temporal_mask)
            
            # 4. Extract at index 0 (the anchor position)
            upgraded_flat = upgraded_seq[:, 0, :] 
            return upgraded_flat.reshape(B, self.num_prefixes, self.hidden_size)




if __name__ == "__main__":
    def test_overfit_drive():
        from gym_sokoban.envs import SokobanEnv
        from sokoban_wrapper import SokoCanonicalWithAttPadding
        my_env = SokoCanonicalWithAttPadding(SokobanEnv(), (12,12), 3)

        obs_space = my_env.observation_space
        action_space = my_env.action_space

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        policy = ExperiencerActorCritic(
            obs_space, action_space, lambda _: 3e-4,
            hidden_size=128, num_prefixes=4
        ).to(device)
        print("--- Starting Overfit Stress Test ---")
        # 1. Setup
        B, S, P, D = 1, 10, 4, 128
        device = next(policy.parameters()).device
        
        # Static observation and a "target" trajectory
        obs, _ = my_env.reset()
        obs_tensor = th.as_tensor(obs).unsqueeze(0).to(device)
        traj = obs_tensor.unsqueeze(1).repeat(1, S, 1, 1, 1, 1, 1) # Static trajectory
        mask = th.zeros((1, S), dtype=th.bool, device=device)
        
        # We want the model to learn that for this obs, Action 2 is the 'correct' one
        target_action = th.tensor([2], device=device)
        
        # 2. Optimization Setup
        # We optimize the generator and the thinker to agree on Action 2
        optimizer = th.optim.Adam(policy.parameters(), lr=1e-4)
        
        print(f"Goal: Force policy to pick Action {target_action.item()} via prefix upgrade.")
        
        for i in range(50):
            optimizer.zero_grad()
            
            # Step A: Get Initial Prefix
            prefix = policy.make_initial_prefix(obs_tensor)
            
            # Step B: Upgrade it via the Thinker
            upgraded_prefix = policy.upgrade_prefix_with_trajectory(prefix, traj, mask)
            
            # Step C: Get Action Distribution from the UPGRADED prefix
            # We manually call the forward logic with the upgraded prefix
            spatial_tokens, spatial_mask, player_indices = policy._get_backbone_outputs(obs_tensor)
            combined = th.cat([upgraded_prefix, spatial_tokens], dim=1)
            
            # Setup mask for combined seq
            p_mask = th.zeros((B, P), device=device, dtype=th.bool)
            c_mask = th.cat([p_mask, spatial_mask], dim=1)
            
            # Actor Pass
            pi_seq = combined
            for layer in policy.pi_attn_layers:
                pi_seq = layer(pi_seq, padding_mask=c_mask)
            
            # Extract player token and distribution
            gather_idx = player_indices.view(B, 1, 1).expand(-1, -1, D)
            player_token = pi_seq[:, P:, :].gather(1, gather_idx).squeeze(1)
            dist = policy._get_action_dist_from_latent(player_token)
            
            # Loss: Negative Log Prob of the target action
            loss = -dist.log_prob(target_action).mean()
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                prob = th.exp(-loss).item()
                print(f"Iteration {i:02d} | Target Action Prob: {prob:.4f} | Loss: {loss.item():.4f}")

        print("--- Overfit Test Complete ---")
    test_overfit_drive()
