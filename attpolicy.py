import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

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
        # Split Q and K into [Spatial Tokens | Prefix Tokens]

        q_spatial, q_prefix = q[:, :, :-self.num_prefixes, :], q[:, :, -self.num_prefixes:, :]
        k_spatial, k_prefix = k[:, :, :-self.num_prefixes, :], k[:, :, -self.num_prefixes:, :]
        
        # Apply RoPE ONLY to the spatial tokens
        q_spatial = self.rope.apply_rope(q_spatial)
        k_spatial = self.rope.apply_rope(k_spatial)
        
        # Recombine
        q = th.cat((q_spatial, q_prefix), dim=2)
        k = th.cat((k_spatial, k_prefix), dim=2) 
        
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
        
        # 4. Masked Transformer Stack
        self.rope.update_cache(x.device, x.dtype)
        
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
            
        # 5. Extract Contextualized Player Token
        attended_player_token = x.gather(1, gather_indices).squeeze(1) 
        
        # 6. Final Readout
        # -combined = th.cat([attended_player_token, raw_player_token], dim=1)
        return self.final_proj(attended_player_token)

class SokoAtt(BaseFeaturesExtractor):
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
        
        # 4. Masked Transformer Stack
        self.rope.update_cache(x.device, x.dtype)
        
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        return (x, padding_mask, gather_indices)

class  PlayerCentric(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, 
                 hidden_size: int = 128, num_heads: int = 4, layers: int = 3, num_prefixes: int = 4):
        super().__init__()
        self.H_grid, self.W_grid, self.C, self.wx, self.wy = observation_space.shape
        self.hidden_size = hidden_size
        self.seq_len = self.H_grid * self.W_grid
        
        # RoPE
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding2D(head_dim, self.H_grid, self.W_grid)
        
        # Transformer Stack
        self.blocks = nn.ModuleList([
            PrefixMaskedAttentionLayer(hidden_size, num_heads, self.rope, num_prefixes) 
            for _ in range(layers)
        ])
        # Final Projection (combines Raw + Context)
        self.final_proj = nn.Linear(hidden_size , features_dim)
        self.features_dim = features_dim

    def forward(self, x: th.Tensor, padding_mask: th.Tensor, gather_indices) -> th.Tensor:
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
    def __init__(self, hidden_size, num_heads, layers, num_prefixes):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        self.rope_1d_module = RotaryEmbedding1D(hidden_size//num_heads)
        self.blocks = nn.Sequential(*[
            PrefixMaskedAttentionLayer(hidden_size, num_heads, self.rope_1d_module, num_prefixes) 
            for _ in range(layers)
        ])
    def forward(self, x, padding_mask=None):
        return self.blocks(x, padding_mask)

class PrefixMLPExtractor(nn.Module):
    def __init__(
            self, 
            features_dim: int, 
            net_arch: List[int], 
            num_prefix: int, 
            enable_critic_prefix: bool, 
            activation_fn: Type[nn.Module],
            device: Union[str, th.device] = "auto"
        ):
        super().__init__()
        if device == "auto":
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        else:
            self.device = th.device(device)
        self.out_dim = features_dim*num_prefix
        self.out_shape = (num_prefix, features_dim)
        if enable_critic_prefix:
            self.out_dim = self.out_dim*2
            self.out_shape = (2*num_prefix, features_dim)

        last_dim = features_dim
        layers = []
        
        for layer in net_arch:
            layers.append(nn.Linear(last_dim, layer))
            layers.append(activation_fn)
            last_dim = layer
        
        layers.append(nn.Linear(last_dim, self.out_dim))
        self.ffn = nn.Sequential(*layers)
    
    def forward(self, x: th.Tensor):
        if x.device != self.device:
            x = x.to(self.device)
        features = self.ffn(x)

        return th.reshape(features, (-1, *self.out_shape))

class ARMs(nn.Module):
    _policy: PlayerCentric
    _critic: PlayerCentric   
    thinker: PlayerCentric

    def __init__(
            self,
            obs_space: spaces.Space,
            backbone_extractor_class: Type[BaseFeaturesExtractor],
            limbs_extractor_class: Type[nn.Module],
            backbone_kwargs: Optional[Dict[str, Any]],
            limbs_kwargs: Optional[Dict[str, Any]],
            temporal_extractor_kwargs: Optional[Dict[str, Any]],
            num_prefixes: int = 4,
            enable_critic_prefix: bool = True
    ):
        super().__init__()
        self.num_prefixes = num_prefixes
        self.enable_critic_prefix = enable_critic_prefix
        self.backbone = backbone_extractor_class(obs_space, **backbone_kwargs)
        self._policy = limbs_extractor_class(obs_space, num_prefixes=num_prefixes, **limbs_kwargs)
        self._critic = limbs_extractor_class(obs_space, num_prefixes=num_prefixes if enable_critic_prefix else 0, **limbs_kwargs)
        self.temporal_extractor = TemporalAttentionLayer(num_prefixes=(num_prefixes*2 if enable_critic_prefix else num_prefixes), **temporal_extractor_kwargs)
    
    def actor_critic(self, obs, prefix):
        # Distinguish between actor and critic prefixes if needed
        if self.enable_critic_prefix:
            actor_prefix = prefix[:, :self.num_prefixes]
            critic_prefix = prefix[:, self.num_prefixes:]
        else:
            actor_prefix = prefix
            critic_prefix = None
        backbone_out = self.backbone(obs)
        x, padding_mask, gather_indices = backbone_out
        x = th.cat([x, actor_prefix], dim=1)
        new_mask = F.pad(padding_mask, (0, actor_prefix.shape[1]), value=False)
        policy_out = self._policy(x, new_mask, gather_indices)   
        if critic_prefix is not None:
            critic_out = self._critic(th.cat([x, critic_prefix], dim=1), new_mask, gather_indices)
        else:
            critic_out = self._critic(x, padding_mask, gather_indices)
        return policy_out, critic_out
    def actor(self, obs, prefix):
        if self.enable_critic_prefix:
            actor_prefix = prefix[:, :self.num_prefixes]
        else:
            actor_prefix = prefix
        backbone_out = self.backbone(obs)
        x, padding_mask, gather_indices = backbone_out
        x = th.cat([x, actor_prefix], dim=1)
        new_mask = F.pad(padding_mask, (0, actor_prefix.shape[1]), value=False)
        policy_out = self._policy(x, new_mask, gather_indices)   
        return policy_out
    def critic(self, obs, prefix):
        if self.enable_critic_prefix:
            critic_prefix = prefix[:, self.num_prefixes:]
        else:
            critic_prefix = None
        backbone_out = self.backbone(obs)
        x, padding_mask, gather_indices = backbone_out
        if critic_prefix is not None:
            x = th.cat([x, critic_prefix], dim=1)
            new_mask = F.pad(padding_mask, (0, critic_prefix.shape[1]), value=False)
            critic_out = self._critic(x, new_mask, gather_indices)
        else:
            critic_out = self._critic(x, padding_mask, gather_indices)
        return critic_out
    def bot(self, obs):
        x, _, gather_indices = self.backbone(obs)
        player_token = x.gather(1, gather_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])).squeeze(1)
        return player_token

    def update_prefix(self, trajectory_of_obs, prev_prefix, trajectory_mask):
        # trajectory_of_obs: (B, T, Obs...)
        B, T = trajectory_of_obs.shape[:2]
        self.temporal_extractor.rope_1d_module.update_cache(T, trajectory_of_obs.device, trajectory_of_obs.dtype)
        flat_obs = trajectory_of_obs.view(B*T, *trajectory_of_obs.shape[2:])
        x, _, gather_indices = self.backbone(flat_obs)
        # Get player indices to go from (B*T, Seq, D) -> (B*T, D) using gather_indices
        player_tokens = x.gather(1, gather_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])).squeeze(1)
        player_tokens = player_tokens.view(B, T, -1)
        # trajectory mask accounts for T but we insert prev prefix at right of the tensor making it T + 2*num_prefixes or T + num_prefixes
        prefix_len = prev_prefix.shape[1]
        prefix_mask = th.zeros(
            B,
            prefix_len,
            dtype=trajectory_mask.dtype,
            device=trajectory_mask.device
        )

        combined_mask = th.cat([trajectory_mask, prefix_mask], dim=1)
        # Combine prev prefix with player tokens
        combined_input = th.cat([player_tokens, prev_prefix], dim=1)
        # Process through temporal extractor
        updated_prefix = self.temporal_extractor(combined_input, combined_mask)
        return updated_prefix[:, -prefix_len:, :]   
    
class ExperiencerActorCritic(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.GELU,
            ortho_init: bool = True,
            backbone_extractor_class: Type[BaseFeaturesExtractor] = SokoAtt,
            limb_extractor_class: Type[nn.Module] = PlayerCentric,
            features_extractor_coordinator_class: Type[ARMs] = ARMs,
            backbone_extractor_kwargs: Optional[Dict[str, Any]] = None,
            limb_extractor_kwargs: Optional[Dict[str, Any]] = None,
            temporal_extractor_kwargs: Optional[Dict[str, Any]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            features_dim: int = 256,
            length_prefix: int = 4,
            shared_prefix: bool = False,
            enable_critic_prefix: bool = True,
            use_sde: bool = False,
    ):
        self.thinker_output_shape = (length_prefix, features_dim)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            False, # use_sde
            0.0, # log_std_init
            True, # full_std
            False, # use_expln
            False, # squash_output
            BaseFeaturesExtractor, #FeaturesExtractor
            dict(features_dim=features_dim), #FeaturesExtractorKwargs
            True, # Share features extractor (handled in arms)
            True, # normalize_images
            optimizer_class,
            optimizer_kwargs,
        )

        self.temporal_extractor_kwargs = temporal_extractor_kwargs or {}
        self.shared_prefix = shared_prefix
        self.enable_critic_prefix = enable_critic_prefix
        self.features_extractor = self.orchestrator = features_extractor_coordinator_class(
            observation_space,
            backbone_extractor_class,
            limb_extractor_class,
            backbone_extractor_kwargs or {},
            limb_extractor_kwargs or {},
            temporal_extractor_kwargs or {}
        )

        self._build_prefix_extractor()

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_prefix_extractor(self) -> None:
        self.prefix_extractor = PrefixMLPExtractor(
            self.features_dim,
            net_arch=self.net_arch["thinker"],
            activation_fn=self.activation_fn,
            device=self.device
        )
    
    def forward(self, obs: th.Tensor, prefix: th.Tensor, deterministic: bool = False):
        assert prefix.shape[-2:] == self.thinker_output_shape
        pi_features, vf_features = self.orchestrator.actor_critic(obs, prefix)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob
    
    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor, prefix: th.Tensor) -> Tuple[th.Tensor, th.Tensor,  th.Tensor]:
        assert prefix.shape[-2:] == self.thinker_output_shape
        pi_features, vf_features = self.orchestrator.actor_critic(obs, prefix)
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs, prefix: th.Tensor) -> Distribution:
        assert prefix.shape[-2:] == self.thinker_output_shape
        features = self.orchestrator.actor(obs, prefix)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)
    
    def predict_values(self, obs: PyTorchObs, prefix: th.Tensor) -> th.Tensor:
        assert prefix.shape[-2:] == self.thinker_output_shape
        features = self.orchestrator.critic(obs, prefix)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
    
    def make_initial_prefix(self, obs: th.Tensor) -> th.Tensor:
        prefix_features = self.orchestrator.bot(obs) #BOT: Beginning Of Thinking
        latent_prefixes = self.prefix_extractor(prefix_features) # pipeline: obs -> att -> select player token -> prefix extractor MLP -> initial prefix
        return  latent_prefixes
    
    def upgrade_prefix_with_trajectory(self, prev_prefix: th.Tensor, trajectory_of_obs: th.Tensor, trajectory_mask: th.Tensor):
        return self.orchestrator.update_prefix(trajectory_of_obs, prev_prefix, trajectory_mask)
