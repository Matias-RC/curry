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

class BoardEncoderCNN(nn.Module):
    """
    Encodes Sokoban board into a (B, 64, 12, 12) feature map and flattened tokens (B, 144, 64).
    Layer spec (applied in order):
      - conv1: in=6, out=32, k=5, s=1, p=2  -> 25x25 -> 25x25
      - conv2: in=32, out=64, k=4, s=2, p=1 -> 25x25 -> 12x12
      - conv3: in=64, out=64, k=3, s=1, p=1 -> 12x12 -> 12x12
    """
    def __init__(self, in_channels: int = 6, base_channels: int = 32, use_bn: bool = False):
        super().__init__()
        self.use_bn = use_bn

        # conv1 -> (B, 32, 25, 25)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()

        # conv2 -> (B, 64, 12, 12)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2) if use_bn else nn.Identity()

        # conv3 -> (B, 64, 12, 12)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 2) if use_bn else nn.Identity()

        # small conv head optional (identity here)
        self.out_channels = base_channels * 2  # 64 by default

        # init
        self._init_weights()

    def _init_weights(self):
        # standard conv init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C=6, H=25, W=25)
        returns:
          feat_map: (B, 64, 12, 12)
          tokens:   (B, 144, 64)   # 12*12 = 144
        """
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        # conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)

        # final shape check
        B, C, H, W = x.shape
        assert C == self.out_channels, f"expected out channels {self.out_channels}, got {C}"
        assert H == 12 and W == 12, f"expected spatial 12x12, got {H}x{W}"

        # flatten to tokens: (B, 144, 64)
        # permute to (B, H, W, C) -> reshape (B, H*W, C)
        tokens = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return x, tokens
    
class AttLSTMCell(nn.Module):
    def __init__(self, d: int, num_heads: int, h_length: int, dropout: float = 0.0):
        """
        Attentive LSTM cell that:
          - applies self-attention to h_cur with a pre-norm residual
          - computes gates via gate_projector(h_cur) and cross-attends those queries to input_tensor
          - computes next c and h
          - adds residual from previous h_cur to h_next (then normalizes)
        """
        super().__init__()
        assert d % num_heads == 0, "d must be divisible by num_heads for MultiheadAttention"

        self.gate_projector = Gate_Convertor(d)
        self.d = d
        self.h_length = h_length

        # self-attention over h_cur (batch_first=True)
        self.self_mha = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True, dropout=dropout)
        # cross-attention where queries are gate slices
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)

        self.i_gate = nn.Linear(d,d,bias=False)
        self.f_gate = nn.Linear(d,d,bias=False)
        self.o_gate = nn.Linear(d,d,bias=False)
        self.g_gate = nn.Linear(d,d,bias=False)
        # LayerNorms for pre-norm patterns and output normalization after residual
        self.ln_self = nn.LayerNorm(d)
        self.ln_cross = nn.LayerNorm(d)
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
        attn_out = self.ln_cross(attn_out)

        # Re-chunk to (B, L, 4d)
        attn_rechunk = attn_out.view(B, 4, L, d).permute(0, 2, 1, 3).reshape(B, L, 4 * d)  # (B, L, 4d)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(attn_rechunk, chunks=4, dim=-1)  # each (B, L, d)
        cc_i = self.i_gate(cc_i)
        cc_f = self.f_gate(cc_f)
        cc_o = self.o_gate(cc_o)
        cc_g = self.g_gate(cc_g)
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
        # NOTE: Gate_Convertor was shared across layers by default;
        #self.gates = Gate_Convertor(d)
        layers = []
        for _ in range(num_cells):
            # each layer gets its own AttLSTMCell (and hence its own base states)
            layers.append(AttLSTMCell(d=d, num_heads=num_heads, h_length=h_length, dropout=dropout))
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

if __name__ == "__main__":
    import torch
    from collections import defaultdict
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # params
    B = 2
    in_channels = 6
    H = W = 25
    d = 128
    num_cells = 3
    h_length = 32
    num_heads = 4
    hidden_seq_len = num_cells * h_length * 2  # hidden + cell for each layer

    # instantiate modules
    encoder = BoardEncoderCNN(in_channels=in_channels, base_channels=d//2, use_bn=False).to(device)
    contextualizer = contextualizationModule(embed_dim=d, kv_dim=d, num_heads=num_heads,
                                            hidden_seq_len=hidden_seq_len, dropout=0.0).to(device)
    stacked = StackedAttLSTM(d=d, num_heads=num_heads, h_length=h_length, num_cells=num_cells, dropout=0.0).to(device)

    # random board batch
    board = torch.randn(B, in_channels, H, W, device=device)

    # forward encoder -> tokens
    feat_map, tokens = encoder(board)                         # feat_map: (B,64,12,12)  tokens: (B,144,64)
    assert tokens.shape == (B, 144, d)

    # produce slots and split into initial hidden/cell states
    slots = contextualizer(tokens)                            # (B, hidden_seq_len, d)
    assert slots.shape == (B, hidden_seq_len, d)

    hidden_slots, cell_slots = torch.chunk(slots, 2, dim=1)  # each (B, hidden_seq_len/2, d)
    # reshape into per-layer tuples
    hidden_slots = hidden_slots.view(B, num_cells, h_length, d)
    cell_slots = cell_slots.view(B, num_cells, h_length, d)

    hidden_state = []
    for i in range(num_cells):
        h_prev = hidden_slots[:, i, :, :].contiguous()
        c_prev = cell_slots[:, i, :, :].contiguous()
        hidden_state.append((h_prev, c_prev))

    # zero grads, forward through stacked attentive LSTM
    for p in list(contextualizer.parameters()):
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    top_h, new_states = stacked(tokens, hidden_state)        # top_h: (B, h_length, d)

    # simple loss and backward
    loss = top_h.mean()
    #loss = (top_h.mean() * 1000000000)

    loss.backward()

    # prints
    print("feat_map", feat_map.shape)
    print("tokens", tokens.shape)
    print("slots", slots.shape)
    print("top_h", top_h.shape)
    q_grad = contextualizer.layer1_cross_att.q_params.grad
    print("q_params.grad is None?", q_grad is None)
    if q_grad is not None:
        print("q_params.grad mean abs:", q_grad.abs().mean().item())
    else:
        print("No gradient found for q_params; check computation graph.")
    models = {"encoder": encoder, "contextualizer": contextualizer, "stacked": stacked}
    params = []
    for mname, m in models.items():
        for name, p in m.named_parameters():
            params.append((f"{mname}.{name}", p))

    rows = []
    total_abs = 0.0
    total_sum = 0.0
    total_elems = 0

    for name, p in params:
        g = p.grad
        if g is None:
            mean_grad = float("nan")
            mean_abs = float("nan")
            got = False
        else:
            mean_grad = g.mean().item()
            mean_abs = g.abs().mean().item()
            got = True
            total_abs += g.abs().sum().item()
            total_sum += g.sum().item()
            total_elems += p.numel()
        rows.append((name, tuple(p.shape), got, mean_grad, mean_abs))

    # Try to show with pandas if available (nicer)
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["param", "shape", "has_grad", "mean_grad", "mean_abs_grad"])
        # show rounded floats for readability
        df["mean_grad"] = df["mean_grad"].apply(lambda x: float(f"{x:.6e}") if not (isinstance(x, float) and math.isnan(x)) else float("nan"))
        df["mean_abs_grad"] = df["mean_abs_grad"].apply(lambda x: float(f"{x:.6e}") if not (isinstance(x, float) and math.isnan(x)) else float("nan"))
        print(df.to_string(index=False))
    except Exception:
        # fallback pretty print
        fmt = "{:60s} {:15s} {:6s} {:14s} {:14s}"
        print(fmt.format("param", "shape", "got", "mean_grad", "mean_abs"))
        for name, shape, got, mg, ma in rows:
            mg_s = f"{mg:.6e}" if not (isinstance(mg, float) and math.isnan(mg)) else "None"
            ma_s = f"{ma:.6e}" if not (isinstance(ma, float) and math.isnan(ma)) else "None"
            print(fmt.format(name, str(shape), str(got), mg_s, ma_s))

    # overall metrics (weighted by parameter count)
    if total_elems > 0:
        overall_mean_abs = total_abs / total_elems
        overall_mean = total_sum / total_elems
        print("\nOverall mean gradient (element-wise):", f"{overall_mean:.6e}")
        print("Overall mean absolute gradient (element-wise):", f"{overall_mean_abs:.6e}")
    else:
        print("\nNo gradients found on any parameters.")
    
    def model_summary(model: torch.nn.Module) -> dict:
        """Return counts: num parameter tensors, total scalars, trainable scalars,
        and params per top-level submodule."""
        num_tensors = sum(1 for _ in model.parameters())
        total_scalars = sum(p.numel() for p in model.parameters())
        trainable_scalars = sum(p.numel() for p in model.parameters() if p.requires_grad)
        per_module = defaultdict(int)
        for name, p in model.named_parameters():
            top = name.split(".")[0]  # e.g., "stack", "pool_proj", etc.
            per_module[top] += p.numel()
        return {
            "num_tensors": num_tensors,
            "total_scalars": total_scalars,
            "trainable_scalars": trainable_scalars,
            "per_module": dict(per_module)
        }
    print(model_summary(encoder))
    print(model_summary(contextualizer))
    print(model_summary(stacked))