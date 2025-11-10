import torch
import torch.nn as nn
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

class contextualizationModule(nn.Module):
    def __init__(self, d: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d = d
        """        
        self.base_hidden_state = nn.Parameter(torch.randn(1, h_length, d))
        self.base_cell_state = nn.Parameter(torch.randn(1, h_length, d))
        """
        #From now on manual customization (TODO: Actually make this class good)
        self.layer1_cross_att = None

class CustomMultiheadQueryAttention(nn.Module):
    def __init__(self, embed_dim, kv_dim, out_dim, num_heads, batch_first=True, dropout=0.0):
        pass


class AttLSTMCell(nn.Module):
    def __init__(self, gate_projector: Gate_Convertor, d: int, num_heads: int, h_length: int, dropout: float = 0.0):
        super().__init__()
        self.gate_projector = gate_projector
        self.d = d
        self.h_length = h_length

        self.self_mha = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)

        # Learnable base states saved per layer; shape (1, h_length, d) so we can repeat across batch easily.



    def forward(self, cell_state: Tuple[torch.Tensor, torch.Tensor], input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cell_state: (h_cur, c_cur) each shaped (B, L, d)
        input_tensor: (B, input_len, d) used as keys/values for cross-attention
        """
        h_cur, c_cur = cell_state                                 # (B, L, d)
        h_cur, _ = self.self_mha(h_cur, h_cur, h_cur)        # (B, L, d)
        gates_4d = self.gate_projector(h_cur)                     # (B, L, 4d)

        B, L, four_d = gates_4d.shape
        d = four_d // 4

        # Re-chunk to perform cross-attention per gate slice
        x_split = gates_4d.view(B, L, 4, d).permute(0, 2, 1, 3).reshape(B, 4 * L, d)  # (B, 4L, d)
        attn_out, _ = self.mha(x_split, input_tensor, input_tensor)           # (B, 4L, d)

        attn_rechunk = attn_out.view(B, 4, L, d).permute(0, 2, 1, 3).reshape(B, L, 4 * d)  # (B, L, 4d)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(attn_rechunk, chunks=4, dim=-1)  # each (B, L, d)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class StackedAttLSTM(nn.Module):
    def __init__(self, d: int, num_heads: int, h_length: int, num_cells: int, dropout: float = 0.0):
        super().__init__()
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
        current = input_tensor
        new_states = []
        for idx, layer in enumerate(self.layers):
            h_prev, c_prev = hidden_state[idx]
            h_next, c_next = layer((h_prev, c_prev), current)
            new_states.append((h_next, c_next))
            current = h_next
        top_h = new_states[-1][0]
        return top_h, new_states

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        states = []
        for layer in self.layers:
            h0, c0 = layer.init_hidden(batch_size, device)
            states.append((h0, c0))
        return states


if __name__ == "__main__":
    B, L, d = 2, 32, 64
    num_cells = 3
    num_heads = 4  # ensure divides d appropriately
    model = StackedAttLSTM(d=d, num_heads=num_heads, h_length=L, num_cells=num_cells)

    x = torch.randn(B, L, d)
    hidden = model.init_hidden(B) 
    top_h, new_states = model(x, hidden)
    print("top_h", top_h.shape)      # (B, L, d)
    print("num layers", len(new_states))
    for i, (h, c) in enumerate(new_states):
        print(f"layer {i} h shape {h.shape}, c shape {c.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
