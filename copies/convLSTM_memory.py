import torch
import torch.nn as nn
from typing import List, Dict, Optional, Sequence, Tuple


class MemoryCLC(nn.Module):
    """ConvLSTMCell that expects a conv_block which accepts a tuple (input, h_cur)."""
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True, conv_block: Optional[nn.Module] = None):
        super().__init__()
        if conv_block is None:
            raise ValueError("ConvLSTMCell requires a conv_block that accepts (input, h_cur) tuples")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.gate_conv = conv_block

    def forward(self, input_tensor: torch.Tensor, cur_state, mem):
        h_cur, c_cur = cur_state
        mem, gates = self.gate_conv((input_tensor, h_cur, mem))  # conv_block must accept tuple
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next, mem

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]):
        h, w = image_size
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device))


class MemoryConvLSTM(nn.Module):
    """Single ConvLSTM layer. Requires conv_block_characteristics (list of 'custom' specs)."""
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True,
                 conv_block_characteristics: Optional[List[Dict]] = None):
        super().__init__()
        if conv_block_characteristics is None:
            raise ValueError("ConvLSTM requires conv_block_characteristics (list of custom specs)")
        # build block assuming in_channels = input_dim + hidden_dim
        conv_block = self._build_conv_block_from_characteristics(conv_block_characteristics,
                                                                 in_channels=input_dim + hidden_dim)
        self.cell = MemoryCLC(input_dim=input_dim, hidden_dim=hidden_dim, bias=bias, conv_block=conv_block)

    def forward(self, input_tensor: torch.Tensor, hidden_state, memory):
        h_prev, c_prev = hidden_state
        return self.cell(input_tensor, (h_prev, c_prev), memory)

    def _init_hidden(self, batch_size: int, image_size: Tuple[int, int]):
        return self.cell.init_hidden(batch_size, image_size)

    def _build_conv_block_from_characteristics(self, specs: List[Dict], in_channels: int):
        """
        Only support 'custom' specs. Each spec must be:
           {'type': 'custom', 'layer': nn.Module, optionally 'out_channels': int}
        The returned nn.Sequential is the conv_block used by MemoryConvLSTMCell and must accept tuples.
        """
        layers = []
        cur_in = in_channels
        for idx, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise ValueError("Each spec must be a dict")
            typ = spec.get('type', 'custom').lower()
            if typ != 'custom':
                raise ValueError(f"Only 'custom' spec is supported (index {idx})")
            layer = spec.get('layer', None)
            if layer is None or not isinstance(layer, nn.Module):
                raise ValueError(f"'custom' spec requires a 'layer' (nn.Module) at index {idx}")
            layers.append(layer)
            if 'out_channels' in spec:
                cur_in = int(spec['out_channels'])
        return nn.Sequential(*layers)


class StackedMemoryConvLSTM(nn.Module):
    """
    Stacked MemoryConvLSTM. Only accepts per_layer_conv_block_characteristics: a sequence of length num_layers,
    where each entry is a list[dict] of custom specs (no defaults allowed).
    """
    def __init__(self, layer_input_dims: List[int], output_dim: int, num_layers: int = 2, bias: bool = True,
                 per_layer_conv_block_characteristics: Optional[Sequence[Optional[List[Dict]]]] = None):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if len(layer_input_dims) != num_layers:
            raise ValueError("layer_input_dims must have length num_layers")
        self.num_layers = int(num_layers)
        self.output_dim = int(output_dim)
        self.bias = bias
        self._layer_input_dims = [int(x) for x in layer_input_dims]
        # normalize per-layer conv chars
        if per_layer_conv_block_characteristics is None:
            raise ValueError("per_layer_conv_block_characteristics must be provided (no defaults allowed)")
        if len(per_layer_conv_block_characteristics) != self.num_layers:
            raise ValueError("per_layer_conv_block_characteristics length must equal num_layers")
        self._per_layer_conv_block_characteristics = list(per_layer_conv_block_characteristics)
        self._build_layers()

    def forward(self, input_tensor: torch.Tensor, hidden_state: Sequence[Tuple[torch.Tensor, torch.Tensor]]
                , selected_memory: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        current_input = input_tensor
        new_states = []
        new_memory = []
        for layer_idx, layer in enumerate(self.layers):
            h_prev, c_prev  = hidden_state[layer_idx]
            h_next, c_next, updated_memory = layer(current_input, hidden_state=(h_prev, c_prev), memory=selected_memory[layer_idx])
            new_states.append((h_next, c_next))
            new_memory.append(updated_memory)
            current_input = h_next
        top_h = new_states[-1][0]

        return top_h, new_states, new_memory

    def _init_hidden(self, batch_size: int, image_size: Tuple[int, int]):
        states = []
        for layer in self.layers:
            states.append(layer._init_hidden(batch_size=batch_size, image_size=image_size))
        return states

    def _build_layers(self):
        modules = []
        for layer_idx in range(self.num_layers):
            layer_chars = self._per_layer_conv_block_characteristics[layer_idx]
            if layer_chars is None:
                raise ValueError(f"conv_block_characteristics required for layer {layer_idx}")
            layer_in = self._layer_input_dims[layer_idx]
            if self.num_layers == layer_idx + 1:
                hidden_dim = self.output_dim
            else:
                hidden_dim = self._layer_input_dims[layer_idx + 1]
            layer_module = MemoryConvLSTM(input_dim=layer_in, hidden_dim=hidden_dim, bias=self.bias,
                                    conv_block_characteristics=layer_chars)
            modules.append(layer_module)
        self.layers = nn.ModuleList(modules)

    def layer_input_dims(self) -> List[int]:
        return list(self._layer_input_dims)
