import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryAttentionLayer(nn.Module):
    def __init__(self, prefix_size, latent_dims):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(prefix_size, latent_dims))
        self.Key_proj = nn.Linear(latent_dims, latent_dims)
        self.Value_proj = nn.Linear(latent_dims, latent_dims)
    def forward(self, seq):
        B, T, D = seq.shape
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        keys = self.Key_proj(seq)
        values = self.Value_proj(seq)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, values)
        return attended



