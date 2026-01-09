import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple


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



# -----------------------------------------------------------------------------
# 1. Rotary Positional Embeddings (RoPE)
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.max_seq_len = max_seq_len
        self.update_cache(max_seq_len)

    def update_cache(self, max_seq_len: int, device=None):
        self.max_seq_len = max_seq_len
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Buffers are saved with state_dict but not trained
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_start_pos: int = 0):
        """
        q, k: [Batch, Head, SeqLen, HeadDim]
        seq_start_pos: integer offset for the start of this sequence (for caching)
        """
        # Slicing the precomputed cos/sin to match the current sequence positions
        seq_len = q.shape[2] 
        # Note: k usually has the same seq_len as q during rotation, 
        # or we rotate the new k-chunk before concatenating to cache.
        
        end_pos = seq_start_pos + seq_len
        
        # Ensure cache is big enough
        if end_pos > self.max_seq_len:
             # In a real scenario, you might want to trigger a resize here automatically
             pass 

        cos = self.cos[seq_start_pos:end_pos, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[seq_start_pos:end_pos, :].unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        
        return q_rot, k_rot

# -----------------------------------------------------------------------------
# 2. Multi-Layer Perceptron (MLP)
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# -----------------------------------------------------------------------------
# 3. Causal Query Attention with KV Caching
# -----------------------------------------------------------------------------

class CausalQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout
        
        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Rotary Embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)

        # Causal mask buffer
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def update_block_size(self, new_size):
        if new_size <= self.bias.shape[-1]: return
        new_bias = torch.tril(torch.ones(new_size, new_size)).view(1, 1, new_size, new_size)
        self.rotary_emb.update_cache(new_size, device=self.bias.device)
        self.register_buffer("bias", new_bias, persistent=False)

    def forward(
        self, 
        x: torch.Tensor, 
        k: Optional[int] = None, 
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        B, T, C = x.size()
        
        # 1. Project Q, K, V
        qkv = self.c_attn(x)
        q, k_vec, v = qkv.split(self.n_embd, dim=2)
        
        # 2. Reshape for heads: [B, H, T, HeadDim]
        k_vec = k_vec.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q     = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v     = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Handle Caching and RoPE Offset
        seq_start_pos = 0
        if layer_past is not None:
            past_k, past_v = layer_past
            seq_start_pos = past_k.shape[2] # The length of what's already cached
            
        # Apply RoPE (Rotates based on absolute position in sequence)
        q, k_vec = self.rotary_emb(q, k_vec, seq_start_pos=seq_start_pos)

        # Update Cache
        if layer_past is not None:
            k_vec = torch.cat([past_k, k_vec], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Save current state for next step return
        present = (k_vec, v) 

        # 4. Handle "Query Attention" Slicing (The 'k' parameter)
        # If we are inferencing (T=1), k is ignored (implicitly 1).
        # If we are training/prefilling (T is large), we slice Q to last k.
        total_len = k_vec.shape[2]
        
        if k is not None and k < T:
            # We only keep the last k queries from the INPUT x
            # Note: T is the length of x. q is length T. 
            q = q[:, :, -k:, :]
            q_len = k
        else:
            q_len = T

        # 5. Attention Calculation
        # q: [B, H, q_len, D]
        # k_vec: [B, H, total_len, D]
        # att: [B, H, q_len, total_len]
        att = (q @ k_vec.transpose(-2, -1)) * (1.0 / math.sqrt(k_vec.size(-1)))

        # 6. Apply Causal Mask
        # We need to map the queries (which are at the END of the sequence)
        # to the keys (which represent the WHOLE sequence).
        # The relevant part of the mask is the bottom-right rectangle.
        
        # Row indices for Q: range(total_len - q_len, total_len)
        # Col indices for K: range(0, total_len)
        
        if total_len > self.bias.shape[-1]:
            self.update_block_size(total_len)
            
        # Select the mask rows corresponding to the absolute positions of our queries
        mask_slice = self.bias[:, :, -q_len:, :total_len] 
        att = att.masked_fill(mask_slice == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 7. Weighted Sum
        y = att @ v # [B, H, q_len, D]
        
        # 8. Reassemble
        y = y.transpose(1, 2).contiguous().view(B, q_len, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, present

# -----------------------------------------------------------------------------
# 4. The Causal Attention Block (Orchestrator)
# -----------------------------------------------------------------------------

class CausalAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalQueryAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self, 
        x: torch.Tensor, 
        k: Optional[int] = None, 
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        x: Input tensor [B, T, C]
        k: (Optional) Int, number of queries to process (Query Attention)
        layer_past: (Optional) Tuple of (K, V) cache from previous step
        """
        # 1. Attention Branch
        attn_out, present = self.attn(self.ln_1(x), k=k, layer_past=layer_past)
        
        # 2. Residual Connection Logic
        # If k < T, attn_out is shorter than x. 
        # We must slice x to the same length (taking the end) to add the residual.
        # If using cache (T=1), usually k is None or 1, so shapes match naturally.
        
        output_len = attn_out.size(1)
        x_slipped = x[:, -output_len:, :]
        
        x = x_slipped + attn_out
        
        # 3. MLP Branch
        x = x + self.mlp(self.ln_2(x))
        
        return x, present
    

if __name__ == "__main__":
    # Dummy Config
    class Config:
        n_embd = 128
        n_head = 4
        block_size = 1024
        dropout = 0.0
        bias = False

    config = Config()
    block = CausalAttentionBlock(config)

    # --- 1. Prefill Step (Processing a prompt) ---
    # Input: "The cat sat on" (4 tokens)
    # We want to process all 4 to fill the cache, but maybe we only care about the last 1 output.
    x_input = torch.randn(1, 4, 128) 

    # Pass k=None to get full output, or k=1 to just get the last vector
    out, cache = block(x_input, k=None) 

    print(f"Prefill Output: {out.shape}") # [1, 4, 128]
    print(f"Cache Keys: {cache[0].shape}") # [1, 4, 4, 32] (B, H, T, D_head)

    # --- 2. Generation Step (Next token) ---
    # Input: " the" (1 token)
    x_next = torch.randn(1, 1, 128)

    # We pass the cache we got from step 1
    out_gen, new_cache = block(x_next, layer_past=cache)

    print(f"Gen Output: {out_gen.shape}") # [1, 1, 128]
    print(f"New Cache Keys: {new_cache[0].shape}") # [1, 5, 4, 32] (Length increased to 5)


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
            x = x.unsqueeze(0,1) # [B, T, C, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(0) #[B, T, C, H, W]

        h = self.encoder(x) # [B, T, d]
        if kv_cache == None:
            kv_cache = []
        for idx, layer in enumerate(self.layers):
            kv_cache.append(None)
            h, cache = layer(h, layer_past=kv_cache[idx])
            kv_cache[idx] = cache

        return self.policy(h), self.value(h), kv_cache