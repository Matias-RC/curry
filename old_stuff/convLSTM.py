import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Sequence, Tuple

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dims, hidden_dims, kernel_size, stride, padding):
        super().__init__()
        self.in_channels, self.height, self.width = input_dims
        self.h_channels = hidden_dims 

        self.wi = nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.h_channels * 4,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.wh1 = nn.Conv2d(in_channels=self.h_channels,
                             out_channels=self.h_channels * 4,
                             kernel_size=kernel_size, stride=stride, padding=padding)
        self.wh2 = nn.Conv2d(in_channels=self.h_channels,
                             out_channels=self.h_channels * 4,
                             kernel_size=kernel_size, stride=stride, padding=padding)

        self.pool_and_inject = nn.Conv2d(in_channels=self.h_channels,
                                          out_channels=self.h_channels * 4,
                                          kernel_size=kernel_size, stride=stride, padding=padding)


        self.pool_proj = nn.Linear(in_features=self.h_channels * 2, out_features=self.h_channels)

    def forward(self, input, cell_state, hidden_state_1, hidden_state_2):
        B = input.shape[0]
        max_pooled = F.adaptive_max_pool2d(hidden_state_1, 1).view(B, self.h_channels)
        avg_pooled = F.adaptive_avg_pool2d(hidden_state_1, 1).view(B, self.h_channels)
        pooled_cat = torch.cat([max_pooled, avg_pooled], dim=1)  
        proj = self.pool_proj(pooled_cat) 
        proj_spatial = proj.view(B, self.h_channels, 1, 1).expand(-1, -1, self.height, self.width)

        wi_out = self.wi(input) # (B, 4*in_channels, H', W')
        wh1_out = self.wh1(hidden_state_1)
        wh2_out = self.wh2(hidden_state_2)
        pool_out = self.pool_and_inject(proj_spatial)

        combined = wi_out + wh1_out + wh2_out + pool_out
        i_t, f_t, g_t, o_t = torch.chunk(combined, chunks=4, dim=1)

        i_gate = torch.sigmoid(i_t)
        f_gate = torch.sigmoid(f_t)
        o_gate = torch.sigmoid(o_t)
        g_gate = torch.tanh(g_t)

        new_cell = f_gate * cell_state + i_gate * g_gate
        new_hidden = o_gate * torch.tanh(new_cell)
        return new_cell, new_hidden

class DRC(nn.Module):
    def __init__(self, d, n, spec):
        super().__init__()
        input_dims, hidden_dims, kernel_size, stride, padding = spec
        self.depth = d
        self.length = n
        self.stack = nn.ModuleList([])
        for _ in range(self.depth):
            self.stack.append(ConvLSTMCell(input_dims, hidden_dims, kernel_size, stride, padding))
        
    def forward(self, input, cell_states, hidden_states, skip_conection):
        for _ in range(self.length):
            for j in range(self.depth):
                new_cell, new_hidden = self.stack[j](input, 
                                                     cell_states[j], 
                                                     hidden_states[j], 
                                                     skip_conection)
                skip_conection = new_hidden
                hidden_states[j] = new_hidden
                cell_states[j] = new_cell
        return cell_states, hidden_states

class SPARSE_cell(nn.Module):
    def __init__(self, input_dims, hidden_dims, memory_dim, kernel_size, stride, padding):
        super().__init__()
        self.in_channels, self.height, self.width = input_dims
        self.h_channels = hidden_dims 
        self.m_channels, intermediary = memory_dim

        self.wi = nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.h_channels * 4,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.wm = nn.Conv2d(in_channels=self.m_channels,
                            out_channels=self.h_channels * 4,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.wh1 = nn.Conv2d(in_channels=self.h_channels,
                             out_channels=self.h_channels * 4,
                             kernel_size=kernel_size, stride=stride, padding=padding)
        self.wh2 = nn.Conv2d(in_channels=self.h_channels,
                             out_channels=self.h_channels * 4,
                             kernel_size=kernel_size, stride=stride, padding=padding)

        self.pool_and_inject = nn.Conv2d(in_channels=self.h_channels,
                                          out_channels=self.h_channels * 4,
                                          kernel_size=kernel_size, stride=stride, padding=padding)


        self.pool_proj = nn.Linear(in_features=self.h_channels * 2, out_features=self.h_channels)

        self.wms1 = nn.Conv2d(in_channels=int(self.h_channels + self.m_channels),
                            out_channels=intermediary,
                            kernel_size=(3,3), stride=1, padding=1, bias=True)
        self.norm1 = nn.BatchNorm2d(intermediary)
        self.wms2 = nn.Conv2d(in_channels=intermediary,
                            out_channels=self.m_channels,
                            kernel_size=(3,3), stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(self.m_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input, memory, cell_state, hidden_state_1, hidden_state_2):
        B = input.shape[0]

        max_pooled = F.adaptive_max_pool2d(hidden_state_1, 1).view(B, self.h_channels)
        avg_pooled = F.adaptive_avg_pool2d(hidden_state_1, 1).view(B, self.h_channels)
        pooled_cat = torch.cat([max_pooled, avg_pooled], dim=1)  
        proj = self.pool_proj(pooled_cat) 
        proj_spatial = proj.view(B, self.h_channels, 1, 1).expand(-1, -1, self.height, self.width)

        wi_out = self.wi(input) # (B, 4*in_channels, H', W')
        wm_out = self.wm(memory)
        wh1_out = self.wh1(hidden_state_1)
        wh2_out = self.wh2(hidden_state_2)
        pool_out = self.pool_and_inject(proj_spatial)

        combined = wi_out + wm_out + wh1_out + wh2_out + pool_out
        i_t, f_t, g_t, o_t = torch.chunk(combined, chunks=4, dim=1)

        i_gate = torch.sigmoid(i_t)
        f_gate = torch.sigmoid(f_t)
        o_gate = torch.sigmoid(o_t)
        g_gate = torch.tanh(g_t)

        new_cell = f_gate * cell_state + i_gate * g_gate
        new_hidden = o_gate * torch.tanh(new_cell)
        cat = torch.cat((memory, new_hidden), dim=1)


        out = self.wms1(cat)
        out = self.norm1(out)
        out = self.act(out)

        out = self.wms2(out)
        out = self.norm2(out)
        return new_cell, new_hidden, memory+out
    
class SparseMemory_DRC(nn.Module):
    def __init__(self, d, n, spec):
        super().__init__()
        input_dims, hidden_dims, memory_dim, kernel_size, stride, padding = spec
        self.depth = d
        self.length = n
        self.stack = nn.ModuleList([])
        for _ in range(self.depth):
            self.stack.append(SPARSE_cell(input_dims, hidden_dims, memory_dim, kernel_size, stride, padding))
        
    def forward(self, input, memory_slot, cell_states, hidden_states, skip_conection):
        for _ in range(self.length):
            for j in range(self.depth):
                new_cell, new_hidden, memory_slot = self.stack[j](input, 
                                                     memory_slot,
                                                     cell_states[j], 
                                                     hidden_states[j], 
                                                     skip_conection)
                skip_conection = new_hidden
                hidden_states[j] = new_hidden
                cell_states[j] = new_cell
        return cell_states, hidden_states, memory_slot
    

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    import pandas as pd
    from collections import defaultdict
    from typing import List

    torch.manual_seed(0)

    def param_grad_table(model: torch.nn.Module) -> pd.DataFrame:
        rows = []
        for name, p in model.named_parameters():
            g = p.grad
            has_grad = g is not None
            mean_grad = float(g.mean().item()) if has_grad else None
            mean_abs_grad = float(g.abs().mean().item()) if has_grad else None
            grad_norm = float(g.norm().item()) if has_grad else None
            rows.append({
                "param": name,
                "shape": tuple(p.shape),
                "numel": int(p.numel()),
                "requires_grad": bool(p.requires_grad),
                "has_grad": has_grad,
                "mean_grad": mean_grad,
                "mean_abs_grad": mean_abs_grad,
                "grad_norm": grad_norm
            })
        df = pd.DataFrame(rows).sort_values("param").reset_index(drop=True)
        return df

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

    # ---------------- Config ----------------
    B = 2
    in_ch = 32  
    H = W = 8
    h_ch = 32
    depth = 3
    length = 3
    m_ch = 32

    # ---------------- DRC ----------------
    drc_spec = ((in_ch, H, W), h_ch, 3, 1, 1)
    drc = DRC(depth, length, drc_spec)

    x = torch.randn(B, in_ch, H, W)
    cell_states: List[torch.Tensor] = [torch.zeros(B, h_ch, H, W) for _ in range(depth)]
    hidden_states: List[torch.Tensor] = [torch.zeros(B, h_ch, H, W) for _ in range(depth)]
    skip = torch.zeros(B, h_ch, H, W)

    # Forward + loss (MSE on last hidden)
    cell_states, hidden_states = drc(x, cell_states, hidden_states, skip)
    final = hidden_states[-1]
    target = torch.randn_like(final)
    loss_drc = F.mse_loss(final, target)

    # Backprop
    drc.zero_grad(set_to_none=True)
    loss_drc.backward()

    # Tables and summary
    drc_df = param_grad_table(drc)
    drc_summary = model_summary(drc)

    print("\n=== DRC per-parameter table ===")
    print(drc_df.to_string(index=False))
    print("\n--- DRC summary ---")
    print(f"parameter tensors: {drc_summary['num_tensors']}")
    print(f"total scalar parameters: {drc_summary['total_scalars']:,}")
    print(f"trainable scalar parameters: {drc_summary['trainable_scalars']:,}")
    print("params per top-level module (scalars):")
    for k, v in sorted(drc_summary["per_module"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")

    # ---------------- SparseMemory_DRC ----------------
    sparse_spec = ((in_ch, H, W), h_ch, (m_ch, 16), 3, 1, 1)
    sparse = SparseMemory_DRC(depth, length, sparse_spec)

    x2 = torch.randn(B, in_ch, H, W)
    memory = torch.zeros(B, m_ch, H, W)
    cell_states2: List[torch.Tensor] = [torch.zeros(B, h_ch, H, W) for _ in range(depth)]
    hidden_states2: List[torch.Tensor] = [torch.zeros(B, h_ch, H, W) for _ in range(depth)]
    skip2 = torch.zeros(B, h_ch, H, W)

    # Forward
    cell_states2, hidden_states2, memory_up = sparse(x2, memory, cell_states2, hidden_states2, skip2)

    # Loss includes memory so memory-update parameters get gradients
    final2 = hidden_states2[-1]
    target2 = torch.randn_like(final2)
    memory_target = torch.randn_like(memory_up)
    loss_sparse = F.mse_loss(final2, target2) + 0.5 * F.mse_loss(memory_up, memory_target)

    sparse.zero_grad(set_to_none=True)
    loss_sparse.backward()

    sparse_df = param_grad_table(sparse)
    sparse_summary = model_summary(sparse)

    print("\n=== SparseMemory_DRC per-parameter table ===")
    print(sparse_df.to_string(index=False))
    print("\n--- SparseMemory_DRC summary ---")
    print(f"parameter tensors: {sparse_summary['num_tensors']}")
    print(f"total scalar parameters: {sparse_summary['total_scalars']:,}")
    print(f"trainable scalar parameters: {sparse_summary['trainable_scalars']:,}")
    print("params per top-level module (scalars):")
    for k, v in sorted(sparse_summary["per_module"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")

    # ---------------- Overall compact summary ----------------
    summary_df = pd.DataFrame([
        {"model": "DRC", "loss": float(loss_drc.item()), "param_tensors": drc_summary["num_tensors"],
         "total_params": drc_summary["total_scalars"], "trainable_params": drc_summary["trainable_scalars"],
         "missing_grads": int((~drc_df["has_grad"]).sum())},
        {"model": "SparseMemory_DRC", "loss": float(loss_sparse.item()), "param_tensors": sparse_summary["num_tensors"],
         "total_params": sparse_summary["total_scalars"], "trainable_params": sparse_summary["trainable_scalars"],
         "missing_grads": int((~sparse_df["has_grad"]).sum())},
    ])
    print("\n=== Compact summary table ===")
    print(summary_df.to_string(index=False))