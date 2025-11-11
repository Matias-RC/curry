import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class Gate_Convertor(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(d, d, bias=True),
            nn.ReLU(),
            nn.Linear(d, 4 * d, bias=False)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.dense(h)  # (B, L, 4d)


class CustomMultiheadCrossAttention(nn.Module):
    def __init__(self, embed_dim, kv_dim, out_dim, num_heads, hidden_seq_len, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_head_dim = out_dim // num_heads
        self.hidden_seq_len = hidden_seq_len

        self.kv_proj = nn.Linear(kv_dim, embed_dim + out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.q_params = nn.Parameter(torch.randn(hidden_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _att_core(self, q, k, v):
        # q: (B, H, Lq, Dh)
        # k: (B, H, Lk, Dh)
        # v: (B, H, Lk, Dv)
        # It automatically scales by sqrt(Dh), applies softmax, and dropout if needed.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,          # or mask tensor if you need one
            dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0,
            is_causal=False
        )
        return out

    def forward(self, kv):
        # kv: (B, L, kv_dim)
        B, L, _ = kv.shape
        Lq = self.hidden_seq_len

        kv_proj = self.kv_proj(kv)                      # (B, L, embed_dim + out_dim)
        k, v = torch.split(kv_proj, [self.embed_dim, self.out_dim], dim=-1)

        # q_params: (Lq, embed_dim) -> expand to batch, then split into heads
        q = self.q_params.unsqueeze(0).expand(B, -1, -1).contiguous()
        # NOTE: view -> (B, Lq, num_heads, head_dim) then transpose -> (B, H, Lq, Dh)
        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # (B, H, Lq, Dh)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()   # (B, H, L, Dh)
        v = v.view(B, L, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous() # (B, H, L, Dv)

        out = self._att_core(q, k, v)                   # (B, H, Lq, Dv)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Lq, self.out_dim)  # (B, Lq, out_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class contextualizationModule(nn.Module):
    def __init__(self, embed_dim, kv_dim, num_heads, hidden_seq_len, dropout=0.0):
        """
        Implements:
          - Cross-attention from kv -> slots (NO residual; slots are overwritten)
          - LayerNorm on the produced slots
          - Self-attention with residual (pre-norm): x = x + self_att(LN(x))
          - FFN with residual (pre-norm): x = x + ffn(LN(x))
        """

        super().__init__()
        if isinstance(num_heads, (list, tuple)):
            cross_heads, self_heads = num_heads
        else:
            cross_heads = self_heads = int(num_heads)

        # Cross-attention produces the slots (no residual here by design).
        self.layer1_cross_att = CustomMultiheadCrossAttention(
            embed_dim=embed_dim,
            kv_dim=kv_dim,
            out_dim=kv_dim,         # slots have kv_dim
            num_heads=cross_heads,
            hidden_seq_len=hidden_seq_len,
            dropout=dropout
        )

        # Self-attention over the slots
        self.layer2_self_att = nn.MultiheadAttention(embed_dim=kv_dim, num_heads=self_heads,
                                                     dropout=dropout, batch_first=True)

        # Feed-forward / projection after self-att
        self.layer3_dense = nn.Sequential(
            nn.Linear(kv_dim, 128),
            nn.GELU(),
            nn.Linear(128, kv_dim)
        )

        # LayerNorms: pre-norm on self-att and ffn; also normalize slots after cross-attention
        self.ln_after_cross = nn.LayerNorm(kv_dim)  # normalize produced slots
        self.ln_self = nn.LayerNorm(kv_dim)         # pre-norm before self-att
        self.ln_ffn = nn.LayerNorm(kv_dim)          # pre-norm before ffn

    def forward(self, context):
        # context: (B, L_ctx, embed_dim)
        # 1) Cross-attention: produce slots from learned queries (no residual).
        x = self.layer1_cross_att(context)          # (B, Lq, kv_dim)
        # Normalize produced slots (we do not add the original queries back).
        x = self.ln_after_cross(x)

        # 2) Self-attention block (pre-norm + residual)
        x_res = x
        x_ln = self.ln_self(x)                      # pre-norm
        sa_out, _ = self.layer2_self_att(x_ln, x_ln, x_ln)
        x = x_res + sa_out                          # residual

        # 3) FFN block (pre-norm + residual)
        x_res = x
        x_ln = self.ln_ffn(x)                       # pre-norm
        ffn_out = self.layer3_dense(x_ln)
        x = x_res + ffn_out                         # residual

        return x


class AttLSTMCell(nn.Module):
    def __init__(self, gate_projector: Gate_Convertor, d: int, num_heads: int, h_length: int, dropout: float = 0.0):
        """
        Attentive LSTM cell that:
          - applies self-attention to h_cur with a pre-norm residual
          - computes gates via gate_projector(h_cur) and cross-attends those queries to input_tensor
          - computes next c and h
          - adds residual from previous h_cur to h_next (then normalizes)
        """
        super().__init__()
        assert d % num_heads == 0, "d must be divisible by num_heads for MultiheadAttention"

        self.gate_projector = gate_projector
        self.d = d
        self.h_length = h_length

        # self-attention over h_cur (batch_first=True)
        self.self_mha = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True, dropout=dropout)
        # cross-attention where queries are gate slices
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)

        # LayerNorms for pre-norm patterns and output normalization after residual
        self.ln_self = nn.LayerNorm(d)
        self.ln_out = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, cell_state: Tuple[torch.Tensor, torch.Tensor], input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cell_state: (h_cur, c_cur) each shaped (B, L, d)
        input_tensor: (B, input_len, d) used as keys/values for cross-attention
        """
        h_cur, c_cur = cell_state                                 # (B, L, d)

        # --- Self-attention on h_cur with pre-norm + residual ---
        h_res = h_cur
        h_ln = self.ln_self(h_cur)                               # pre-norm
        h_att, _ = self.self_mha(h_ln, h_ln, h_ln)               # (B, L, d)
        h_cur = h_res + self.dropout(h_att)                      # residual

        # --- Gate projection (compute 4d per token) ---
        gates_4d = self.gate_projector(h_cur)                    # (B, L, 4d)
        B, L, four_d = gates_4d.shape
        assert four_d % 4 == 0, "Gate dimension must be divisible by 4"
        d = four_d // 4

        # Re-chunk to perform cross-attention per gate slice in one batched call
        # x_split: (B, 4L, d) where queries are the 4 gate-query chunks concatenated
        x_split = gates_4d.view(B, L, 4, d).permute(0, 2, 1, 3).reshape(B, 4 * L, d)  # (B, 4L, d)

        # Cross-attend queries (gate queries) to input_tensor (keys/values)
        attn_out, _ = self.mha(x_split, input_tensor, input_tensor)           # (B, 4L, d)

        # Re-chunk to (B, L, 4d)
        attn_rechunk = attn_out.view(B, 4, L, d).permute(0, 2, 1, 3).reshape(B, L, 4 * d)  # (B, L, 4d)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(attn_rechunk, chunks=4, dim=-1)  # each (B, L, d)

        # LSTM-like gating
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # --- Output residual: add previous h_cur to h_next (stabilizes updates) ---
        h_next = h_next + h_cur
        # Normalize the resulting hidden state
        h_next = self.ln_out(h_next)

        return h_next, c_next


class StackedAttLSTM(nn.Module):
    def __init__(self, d: int, num_heads: int, h_length: int, num_cells: int, dropout: float = 0.0):
        super().__init__()
        # NOTE: Gate_Convertor is shared across layers by default; you can change to per-layer if you prefer
        self.gates = Gate_Convertor(d)
        layers = []
        for _ in range(num_cells):
            # each layer gets its own AttLSTMCell (and hence its own base states)
            layers.append(AttLSTMCell(self.gates, d=d, num_heads=num_heads, h_length=h_length, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.d = d
        self.num_cells = num_cells
        self.h_length = h_length

    def forward(self, input_tensor: torch.Tensor, hidden_state: List[Tuple[torch.Tensor, torch.Tensor]]):
        assert len(hidden_state) == self.num_cells, "hidden_state must have length == num_cells"
        current = input_tensor
        new_states = []
        for idx, layer in enumerate(self.layers):
            h_prev, c_prev = hidden_state[idx]
            h_next, c_next = layer((h_prev, c_prev), current)
            new_states.append((h_next, c_next))
            current = h_next
        top_h = new_states[-1][0]
        return top_h, new_states
