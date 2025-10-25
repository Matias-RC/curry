import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional


class ConvLSTMCell(nn.Module):
    """
    ConvLSTMCell that accepts an optional conv_block (nn.Module) applied to concat([input, hidden]).
    If conv_block is None, a single conv maps concat -> 4*hidden_dim.
    If conv_block is provided, conv_block_out_channels must be provided (int) indicating the features
    produced by conv_block; the cell will use a 1x1 conv to map those features -> 4*hidden_dim.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 bias: bool = True,
                 conv_block: Optional[nn.Module] = None):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.gate_conv = conv_block
        if self.gate_conv is not None:
            pass
        else:
            # original single conv behavior: concat -> 4*hidden_dim
            self.gate_conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                       out_channels=4 * self.hidden_dim,
                                       kernel_size=(3,3),
                                       padding=1,
                                       bias=self.bias)

    def forward(self, input_tensor: torch.Tensor, cur_state):
        h_cur, c_cur = cur_state

        #combined = torch.cat([input_tensor, h_cur], dim=1)  # (B, input_dim + hidden_dim, H, W)
        gates = self.gate_conv((input_tensor,h_cur))

        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size):
        height, width = image_size
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTM(nn.Module):
    """
    Single-layer ConvLSTM with flexible conv_block builder.

    Args:
        input_dim: channels in input
        hidden_dim: channels in hidden state
        kernel_size: default kernel for simple behavior
        bias: whether convs have bias
        conv_block: optional pre-built nn.Module to use (alternative to conv_block_characteristics)
        conv_block_out_channels: required if conv_block is provided
        conv_block_characteristics: optional list[dict] describing layers to build.
                                   If provided, conv_block is built from it.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 bias: bool = True,
                 conv_block: Optional[nn.Module] = None,
                 conv_block_characteristics: Optional[List[Dict]] = None):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # If user provides conv_block_characteristics, we ignore conv_block input and build block here.
        if conv_block_characteristics is not None:
            conv_block = self._build_conv_block_from_characteristics(
                conv_block_characteristics, in_channels=self.input_dim + self.hidden_dim)
        else:
            # use provided conv_block or None
            if conv_block is not None and not isinstance(conv_block, nn.Module):
                raise ValueError("conv_block must be an nn.Module or None")

        # instantiate ConvLSTMCell with the prepared block (or defaults)
        self.cell = ConvLSTMCell(input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim,
                                 bias=bias,
                                 conv_block=conv_block)

    def forward(self, input_tensor: torch.Tensor, hidden_state=None):
        b, c_in, h, w = input_tensor.size()
        h_prev, c_prev = hidden_state
        h_next, c_next = self.cell(input_tensor, (h_prev, c_prev))
        return h_next, c_next

    def _init_hidden(self, batch_size: int, image_size):
        return self.cell.init_hidden(batch_size, image_size)

    # ---------------------------
    # conv_block builder helpers
    # ---------------------------
    @staticmethod
    def _parse_padding(padding, kernel_size):
        """Return integer padding given padding spec. Accepts 'same' or int/tuple."""
        if padding == 'same':
            if isinstance(kernel_size, int):
                k = kernel_size
            else:
                k = kernel_size[0]
            return k // 2
        if isinstance(padding, tuple):
            return padding
        return int(padding)

    @staticmethod
    def _maybe_activation(act_spec):
        """Return an nn.Module activation for common names or an nn.Module if already provided."""
        if act_spec is None:
            return None
        if isinstance(act_spec, nn.Module):
            return act_spec
        if isinstance(act_spec, str):
            n = act_spec.lower()
            if n in ('relu',):
                return nn.ReLU(inplace=True)
            if n in ('leakyrelu', 'leaky_relu'):
                return nn.LeakyReLU(inplace=True)
            if n in ('gelu',):
                return nn.GELU()
            if n in ('sigmoid',):
                return nn.Sigmoid()
            if n in ('tanh',):
                return nn.Tanh()
        raise ValueError("Unsupported activation spec: {}".format(act_spec))

    def _build_conv_block_from_characteristics(self, specs: List[Dict], in_channels: int):
        """
        Build an nn.Sequential from a list of spec dicts.
        Each spec dict can contain:
            - 'type': 'conv' | 'maxpool' | 'avgpool' | 'identity' | 'custom' (default 'conv')
            - for 'conv':
                - 'out_channels' (required)
                - 'kernel_size' (default 3)
                - 'stride' (default 1)
                - 'padding' (default 'same')
                - 'activation' (optional): 'relu', nn.Module, etc.
                - 'norm' (optional): 'bn' for BatchNorm2d
            - for 'maxpool'/'avgpool':
                - 'kernel_size' (default 2), 'stride' (default kernel), 'padding' (default 0)
                - optionally 'shape_transform' callable(in_ch) -> out_ch
            - for 'custom':
                - 'layer' (nn.Module) required
                - optionally 'out_channels' OR 'shape_transform' to update channel count
            - optionally 'shape_transform' callable(in_ch) -> out_ch for any spec to override channel tracking
        Returns:
            nn.Sequential(block)
        """
        layers = []
        cur_in = in_channels
        for idx, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise ValueError("Each conv_block_characteristics entry must be a dict")

            typ = spec.get('type', 'conv').lower()

            # allow user-provided shape_transform callable
            shape_transform = spec.get('shape_transform', None)
            if shape_transform is not None and not callable(shape_transform):
                raise ValueError("shape_transform, if provided, must be callable(in_channels) -> out_channels")

            if typ == 'conv':
                out_ch = spec.get('out_channels', None)
                if out_ch is None:
                    raise ValueError(f"'out_channels' must be provided for conv at index {idx}")
                k = spec.get('kernel_size', 3)
                s = spec.get('stride', 1)
                p = spec.get('padding', 'same')
                pad = self._parse_padding(p, k)
                conv = nn.Conv2d(in_channels=cur_in, out_channels=out_ch,
                                 kernel_size=k, stride=s, padding=pad, bias=spec.get('bias', True))
                layers.append(conv)

                # optional norm
                if spec.get('norm', None) == 'bn':
                    layers.append(nn.BatchNorm2d(out_ch))
                # optional activation
                act = self._maybe_activation(spec.get('activation', None))
                if act is not None:
                    layers.append(act)

                cur_in = out_ch

            elif typ in ('maxpool', 'avgpool'):
                k = spec.get('kernel_size', 2)
                s = spec.get('stride', k)
                p = spec.get('padding', 0)
                if typ == 'maxpool':
                    layers.append(nn.MaxPool2d(kernel_size=k, stride=s, padding=p))
                else:
                    layers.append(nn.AvgPool2d(kernel_size=k, stride=s, padding=p))

                # pools usually do not change channels, but allow shape_transform to indicate otherwise
                if shape_transform is not None:
                    cur_in = shape_transform(cur_in)

            elif typ == 'identity' or typ == 'noop':
                layers.append(nn.Identity())

            elif typ == 'custom':
                layer = spec.get('layer', None)
                if layer is None or not isinstance(layer, nn.Module):
                    raise ValueError(f"'custom' spec requires a 'layer' (nn.Module) at index {idx}")
                layers.append(layer)
                # update cur_in via out_channels or shape_transform if provided
                if 'out_channels' in spec:
                    cur_in = int(spec['out_channels'])
                elif shape_transform is not None:
                    cur_in = shape_transform(cur_in)
                # else assume the custom layer preserved channels

            else:
                raise ValueError(f"Unsupported layer type '{typ}' at index {idx}")

        block = nn.Sequential(*layers)
        return block

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Sequence, Tuple, Union


class StackedConvLSTM(nn.Module):
    """
    Stacked ConvLSTM where every layer is a ConvLSTM (single-layer module).
    Keeps ConvLSTM's expressability: each layer can be constructed either with
    a pre-built conv_block (nn.Module) + conv_block_out_channels OR with a
    conv_block_characteristics list (builder).

    Important:
      - All layers must share the same hidden_dim (enforced).
      - By default layer 0 input_dim == input_dim arg, and layers 1..D-1 input_dim == hidden_dim.
        Use set_inter_layer_input_dims(...) or set_layer_input_dims(...) to change that.
    """

    def __init__(self,
                 layer_input_dims: List[int],
                 output_dim: int,
                 num_layers: int = 2,
                 bias: bool = True,
                 # Global conv_block or conv_block_characteristics (applies to all layers unless per-layer provided)
                 conv_block: Optional[nn.Module] = None,
                 conv_block_characteristics: Optional[List[Dict]] = None,
                 # Optional per-layer overrides: must be list of length num_layers or None
                 per_layer_conv_block: Optional[Sequence[Optional[nn.Module]]] = None,
                 per_layer_conv_block_characteristics: Optional[Sequence[Optional[List[Dict]]]] = None):
        """
        Args:
            input_dim: channels of external input (x_t)
            hidden_dim: channels of hidden states (same for all layers)
            kernel_size, bias: forwarded to each ConvLSTM
            num_layers: number of stacked ConvLSTM layers
            conv_block / conv_block_out_channels / conv_block_characteristics:
                either pass a conv_block (nn.Module) + conv_block_out_channels (int)
                or pass conv_block_characteristics (list of dicts) to build block.
                That acts as a global default for all layers unless per-layer overrides are provided.
            per_layer_conv_block etc.: lists of length num_layers (can contain None)
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = int(num_layers)
        self.output_dim = output_dim
        self.bias = bias

        # store global/defaults
        self._global_conv_block = conv_block
        self._global_conv_block_characteristics = conv_block_characteristics

        # normalize per-layer overrides into lists of length num_layers
        def _normalize_arg(arg, name):
            if arg is None:
                return [None] * self.num_layers
            if isinstance(arg, (list, tuple)):
                if len(arg) != self.num_layers:
                    raise ValueError(f"'{name}' list length must equal num_layers ({self.num_layers}) if provided")
                return list(arg)
            # single non-list value -> replicate
            return [arg] * self.num_layers

        self._per_layer_conv_block = _normalize_arg(per_layer_conv_block, "per_layer_conv_block")
        self._per_layer_conv_block_characteristics = _normalize_arg(per_layer_conv_block_characteristics,
                                                                    "per_layer_conv_block_characteristics")

        # default layer input dims: first layer uses provided input_dim; further layers use hidden_dim
        self._layer_input_dims = layer_input_dims

        # build the actual layers
        self._build_layers()

    # -------------------
    # public API methods
    # -------------------
    def set_inter_layer_input_dims(self, inter_dims: Sequence[int]):
        """
        Set input dims for layers 1..num_layers-1.

        Args:
            inter_dims: sequence of length (num_layers - 1) specifying input_dim for layer 1,2,...,D-1.
                        Example: if input has 6 channels but hidden_dim is 4, and you want layer1 to accept 6,
                        pass inter_dims=[6, 4, 4] for a 4-layer stack (len=3).
        Rebuilds the underlying layer modules.
        """
        if len(inter_dims) != max(0, self.num_layers - 1):
            raise ValueError("inter_dims must have length num_layers-1")
        self._layer_input_dims = [self.input_dim] + [int(x) for x in inter_dims]
        self._build_layers()

    def set_layer_input_dims(self, layer_input_dims: Sequence[int]):
        """
        Set input dims for all layers. Provide sequence of length num_layers.
        Rebuilds the layer modules.
        """
        if len(layer_input_dims) != self.num_layers:
            raise ValueError("layer_input_dims must have length num_layers")
        self._layer_input_dims = [int(x) for x in layer_input_dims]
        self._build_layers()

    def forward(self,
                input_tensor: torch.Tensor,
                hidden_state: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        Single time-step forward.

        Args:
            input_tensor: (B, C_in, H, W) where C_in must match layer 0 input_dim
            hidden_state: optional list/tuple length num_layers of (h_prev, c_prev) pairs,
                          each of shape (B, hidden_dim, H, W)

        Returns:
            top_h: hidden state of top layer (B, hidden_dim, H, W)
            new_states: list[(h_t^0, c_t^0), ..., (h_t^{D-1}, c_t^{D-1})]
        """
        b, c_in, h, w = input_tensor.shape

        current_input = input_tensor
        new_states = []

        for layer_idx, layer in enumerate(self.layers):
            # choose previous states
            h_prev, c_prev = hidden_state[layer_idx]

            # forward through layer (layer expects its own input_dim)
            h_next, c_next = layer(current_input, hidden_state=(h_prev, c_prev))
            new_states.append((h_next, c_next))

            # next layer input is the h produced by this layer
            current_input = h_next

        top_h = new_states[-1][0]
        return top_h, new_states

    def _init_hidden(self, batch_size: int, image_size: Tuple[int, int]):
        """Return list of initial (h,c) for every layer."""
        states = []
        for layer in self.layers:
            states.append(layer._init_hidden(batch_size=batch_size, image_size=image_size))
        return states

    # -----------------------
    # internal build helpers
    # -----------------------
    def _build_layers(self):
        """(Re)build layer ModuleList according to self._layer_input_dims and conv_block specs."""
        modules = []
        for layer_idx in range(self.num_layers):
            # determine conv_block / characteristics for this layer: per-layer overrides win, otherwise global defaults
            layer_conv_block = self._per_layer_conv_block[layer_idx] if self._per_layer_conv_block[layer_idx] is not None else self._global_conv_block
            layer_conv_block_chars = (self._per_layer_conv_block_characteristics[layer_idx]
                                      if self._per_layer_conv_block_characteristics[layer_idx] is not None
                                      else self._global_conv_block_characteristics)

            layer_in = self._layer_input_dims[layer_idx]
            if self.num_layers == layer_idx+1:
                hidden_dim = self.output_dim
            else:
                hidden_dim = self._layer_input_dims[layer_idx+1]

            # build the ConvLSTM for this layer. (ConvLSTM must be in scope)
            layer_module = ConvLSTM(input_dim=layer_in,
                                    hidden_dim=hidden_dim,
                                    bias=self.bias,
                                    conv_block=layer_conv_block,
                                    conv_block_characteristics=layer_conv_block_chars)
            modules.append(layer_module)

        # register ModuleList (replaces previous)
        self.layers = nn.ModuleList(modules)

    # expose a small convenience for inspection
    def layer_input_dims(self) -> List[int]:
        """Return current configured input dims for each layer."""
        return list(self._layer_input_dims)
