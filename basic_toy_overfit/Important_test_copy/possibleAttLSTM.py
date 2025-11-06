import torch
import torch.nn as nn
from typing import List, Tuple, Optional


# ============================================================
# GATE PROJECTOR
# ============================================================
class Gate_Convertor(nn.Module):
    def __init__(self, d: int, nonlinear: str = "relu"):
        """
        nonlinear: 'relu' (default, fast, robust) or 'gelu' (smoother, better for large models)
        """
        super().__init__()
        if nonlinear == "relu":
            act = nn.ReLU()
        # Optionally use GELU for large-scale setups (Transformer-like)
        # elif nonlinear == "gelu":
        #     act = nn.GELU()
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinear}")

        self.dense = nn.Sequential(
            nn.Linear(d, d, bias=True),
            act,
            nn.Linear(d, 4 * d, bias=False)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Output: (B, L, 4d)
        return self.dense(h)


# ============================================================
# ATTENTION-BASED LSTM CELL
# ============================================================
class AttLSTMCell(nn.Module):
    def __init__(
        self,
        gate_projector: Gate_Convertor,
        d: int,
        num_heads: int,
        h_length: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.gate_projector = gate_projector
        self.d = d
        self.h_length = h_length

        self.self_mha = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout
        )

        # Residual normalizations
        self.norm_self = nn.LayerNorm(d)
        self.norm_cross = nn.LayerNorm(d)

        # Per-gate learned biases (forget gate initialized to +1 for better memory retention)
        self.gate_bias = nn.Parameter(torch.zeros(4, d))
        nn.init.constant_(self.gate_bias[1], 1.0)  # forget gate bias

        # Learnable base states (per-layer)
        self.base_hidden_state = nn.Parameter(torch.randn(1, h_length, d))
        self.base_cell_state = nn.Parameter(torch.randn(1, h_length, d))

    # ------------------------------------------------------------
    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None):
        dev = device if device is not None else self.base_hidden_state.device
        h0 = self.base_hidden_state.to(dev).repeat(batch_size, 1, 1)
        c0 = self.base_cell_state.to(dev).repeat(batch_size, 1, 1)
        return h0, c0

    # ------------------------------------------------------------
    def forward(
        self,
        cell_state: Tuple[torch.Tensor, torch.Tensor],
        input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cell_state: (h_cur, c_cur), each (B, L, d)
        input_tensor: (B, input_len, d)
        """
        h_cur, c_cur = cell_state

        # --- SELF-ATTENTION + RESIDUAL ---
        h_self, _ = self.self_mha(h_cur, h_cur, h_cur)
        h_cur = self.norm_self(h_cur + h_self)

        # --- GATE PROJECTION ---
        gates_4d = self.gate_projector(h_cur)  # (B, L, 4d)
        B, L, four_d = gates_4d.shape
        d = four_d // 4

        # --- CROSS-ATTENTION FOR GATE FEATURES ---
        x_split = (
            gates_4d.view(B, L, 4, d)
            .permute(0, 2, 1, 3)
            .reshape(B, 4 * L, d)
        )
        attn_out, _ = self.mha(x_split, input_tensor, input_tensor)
        attn_out = self.norm_cross(attn_out + x_split)  # residual + norm
        attn_rechunk = (
            attn_out.view(B, 4, L, d)
            .permute(0, 2, 1, 3)
            .reshape(B, L, 4 * d)
        )

        # --- SPLIT GATES ---
        cc_i, cc_f, cc_o, cc_g = torch.chunk(attn_rechunk, 4, dim=-1)

        # Add learned gate biases
        cc_i = cc_i + self.gate_bias[0]
        cc_f = cc_f + self.gate_bias[1]
        cc_o = cc_o + self.gate_bias[2]
        cc_g = cc_g + self.gate_bias[3]

        # --- STANDARD LSTM UPDATE ---
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ============================================================
# STACKED VERSION
# ============================================================
class StackedAttLSTM(nn.Module):
    def __init__(self, d: int, num_heads: int, h_length: int, num_cells: int, dropout: float = 0.0):
        super().__init__()
        self.gates = Gate_Convertor(d, nonlinear="relu")  # or 'gelu'
        self.layers = nn.ModuleList([
            AttLSTMCell(self.gates, d=d, num_heads=num_heads, h_length=h_length, dropout=dropout)
            for _ in range(num_cells)
        ])
        self.d = d
        self.num_cells = num_cells
        self.h_length = h_length

    def forward(self, input_tensor: torch.Tensor, hidden_state: List[Tuple[torch.Tensor, torch.Tensor]]):
        current = input_tensor
        new_states = []
        for idx, layer in enumerate(self.layers):
            h_prev, c_prev = hidden_state[idx]
            h_next, c_next = layer((h_prev, c_prev), current)
            new_states.append((h_next, c_next))
            current = h_next
        top_h = new_states[-1][0]
        return top_h, new_states

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None):
        return [layer.init_hidden(batch_size, device) for layer in self.layers]


# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    B, L, d = 2, 32, 64
    num_cells = 3
    num_heads = 4
    model = StackedAttLSTM(d=d, num_heads=num_heads, h_length=L, num_cells=num_cells)

    x = torch.randn(B, L, d)
    hidden = model.init_hidden(B)
    top_h, new_states = model(x, hidden)

    print("top_h", top_h.shape)
    for i, (h, c) in enumerate(new_states):
        print(f"layer {i}: h={h.shape}, c={c.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
