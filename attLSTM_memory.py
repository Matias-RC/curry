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
    
"""
Attention LSTM with sparse memory: One should focus on parameter equivalence for same dimentionality, as well as
managing a way to make the parameters contextualize apropiately. That is why im going to begin by testing the
    - Full sized cell
    |   - Recives (input, hidden state, cell state, memory)
    |   - All inputs are tensors that have last dimention = d
    |   - contextualization module with two sets of cross attention with residual conections att(input, memory, memory) || att(hidden state, memory, memory)
    |   -SelfAtt(input)
    |   -then natural cell procedure. selfAtt(hidden)->projection matrix->Reshape->attention(gates, input,input)->projec the gates -> LSTM
    |   -With the resulting hidden crossAtt(input,new hidden, new hidden)
    |   -then we update memory crossAtt(memory, input,input)
""" 

class SPARSEAttLSTMCell_1p0(nn.Module):
    def __init__(self, d: int, num_heads: int, h_length, dropout: float=0.0):
        super().__init__()
        assert d%num_heads == 0, "d must be divisible by num_heads for MultiheadAttention"

        self.cross_input_memory_1 = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.cross_hidden_memory_1 = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.input_proj_1 = nn.Linear(d,d,bias=True)
        self.self_input_1 = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.hidden_proj_1 = nn.Linear(d,d,bias=True)
        self.self_hidden_1 = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.hidden_proj_2 = nn.Linear(d,4*d, bias=True)
        self.cross_gates_input = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.i_gate = nn.Linear(d,d,bias=False)
        self.f_gate = nn.Linear(d,d,bias=False)
        self.o_gate = nn.Linear(d,d,bias=False)
        self.g_gate = nn.Linear(d,d,bias=False)
        self.cross_input_newhidden = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.cross_memory_input = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)


"""
Although cell 1.0 might work it is vey expensive to execute all of these attention methods at every cell step of the stack it might be
in our best interest not to differentiate the memory from the inout that much this way we could organize a contextualization module such that
before entering the stack the input is fused via attention with the memory provided.
    - Contextualization module
    |   - recives (input,  memory)
    |   - then does co-attention() with memory and input basically crossAtt(memory, input, input) concat  crossAtt(input, memory, memory)
    |   - The result is residually conected and we exectue three layers of self attention in this then it is given to the cell
    - New Cell module
    |   - exact same procedure as original cell but at the end we contextualize input with hiddenstate vias cross att 
        (maybe contextualizing only the memory sub sequence)

"""

class sparse_AttLSTMCell(nn.Module):
    def __init__(self, d: int, num_heads: int, h_length: int, mem_length: int, input_length:int, dropout: float = 0.0):
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
        self.mem_length = mem_length
        self.input_length = input_length
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
        self.ln_out_h = nn.LayerNorm(d)
        self.ln_m_1 = nn.LayerNorm(d)
        self.ln_m_2 = nn.LayerNorm(d)        
        self.ln_m_3 = nn.LayerNorm(d)
        self.ln_m_4 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.mha_out_contextualization = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.ffn_1 = nn.Sequential(
            nn.Linear(d, 2*d, bias=True),
            nn.GELU(),
            nn.Linear(2*d, d, bias=True)
        )
        self.mha_self_contextualization = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.ffn_2 = nn.Sequential(
            nn.Linear(d, 2*d, bias=True),
            nn.GELU(),
            nn.Linear(2*d, d, bias=True)
        )

    def forward(self, cell_state: Tuple[torch.Tensor, torch.Tensor], input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_cur, c_cur = cell_state 
        h_res = h_cur
        h_ln = self.ln_self(h_cur)
        h_att, _ = self.self_mha(h_ln, h_ln, h_ln)
        h_cur = h_res + self.dropout(h_att)

        
        gates_4d = self.gate_projector(h_cur)
        B, L, four_d = gates_4d.shape
        d = four_d // 4

        x_split = gates_4d.view(B, L, 4, d).permute(0, 2, 1, 3).reshape(B, 4 * L, d)

        attn_out, _ = self.mha(x_split, input_tensor, input_tensor) 
        attn_out = self.ln_cross(attn_out)

        attn_rechunk = attn_out.view(B, 4, L, d).permute(0, 2, 1, 3).reshape(B, L, 4 * d) 
        cc_i, cc_f, cc_o, cc_g = torch.chunk(attn_rechunk, chunks=4, dim=-1)
        cc_i = self.i_gate(cc_i)
        cc_f = self.f_gate(cc_f)
        cc_o = self.o_gate(cc_o)
        cc_g = self.g_gate(cc_g)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        h_next = h_next + h_cur
        h_next = self.ln_out(h_next)

        input_data, x = torch.split(input_tensor, [self.input_length, self.mem_length], dim=1)
        x = x + self.mha_out_contextualization(x, h_next, h_next)
        x = self.ln_m_1(x)
        x = x + self.ffn_1(x)
        x = self.ln_m_2(x)
        x = x + self.mha_self_contextualization(x, x, x)
        x = self.ln_m_3(x)
        x = x + self.ffn_2(x)
        x = self.ln_m_4(x)

        return h_next, c_next, torch.cat([input_data, x], dim=1)

class att_block(nn.Module):
    def __init__(self, d, h, num_heads, dropout):
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.ln_1= nn.LayerNorm(d)      
        self.ffn = nn.Sequential(
            nn.Linear(d, h, bias=True),
            nn.GELU(),
            nn.Linear(h, d, bias=True)
        )      
        self.ln_2 = nn.LayerNorm(d)      

    def forward(self, x):
        x = self.ln_1(x+self.mha(x,x,x))
        x = self.ln_2(x + self.ffn(x))    
        return x
    
class FusionGate(nn.Module):
    def __init__(self, d: int, num_heads: int, num_att_blocks: int, hidden_dim: int, dropout: float = 0.0):
        self.mha_contextualize_memory = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.mha_contextualize_input = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, kdim=d, vdim=d, batch_first=True, dropout=dropout)
        self.ln_m_1 = nn.LayerNorm(d)
        self.ln_i_1= nn.LayerNorm(d)      
        self.ffn_m = nn.Sequential(
            nn.Linear(d, 2*d, bias=True),
            nn.GELU(),
            nn.Linear(2*d, d, bias=True)
        )      
        self.ffn_i = nn.Sequential(
            nn.Linear(d, 2*d, bias=True),
            nn.GELU(),
            nn.Linear(2*d, d, bias=True)
        )
        self.ln_m_2 = nn.LayerNorm(d)
        self.ln_i_2 = nn.LayerNorm(d)  

        self.att_blocks = nn.Sequential([att_block(d,hidden_dim, num_heads, dropout) for _ in range(num_att_blocks)])


    def forward(self, i, m):
        y = self.ln_i_1(i+self.mha_contextualize_input(i, m, m))
        y = self.ln_i_2(y + self.ffn_i(y))

        x = self.ln_m_1(m+self.mha_contextualize_memory(m, i, i))
        x = self.ln_i_2(x + self.ffn_i(x))      

        joint = torch.cat([y, x], dim=1)

        return self.att_blocks(joint)