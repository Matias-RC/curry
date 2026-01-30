import torch as th
import torch.nn as nn
import numbers
from dataclasses import dataclass
from typing import Tuple, List, Dict
import warnings
import math

class ConvRNNBase(nn.Module):
    r"""Base class for   Convolution based RNNmodules (ConvLSTM)

    Implements aspects of RNN's shared by RNN, LSTM, and GRU, such as module initialization
    and utility  methods for parameter management.

    .. note::
        The forward method is not implemented by the ConvRNNBase class.
    
    .. note::
        ConvLSTM is the only implemented subclass and it overrides some methods implemented by ConvRNNBase.
    """
    __constants__ = ['mode', 'feature_shape', 'input_size', 'hidden_size', 'num_layers', 
                     'bias', 'batch_first', 'dropout', 'proj_size']
    
    mode: str
    feature_shape: Tuple[int, int]
    input_size:int
    hidden_size:int
    num_layers:int
    bias:bool
    batch_first:bool
    dropout:float
    proj_size: int


    def __init__(self, mode: str, feature_shape: Tuple[int, int], input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool= True, batch_first: bool = False,
                 dropout: float = 0., proj_size:int = 0,
                 device=None, dtype=None)->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.mode = mode
        self.feature_shape = feature_shape
        self.input_channels = input_size
        self.hidden_channels = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.proj_size = proj_size

        if not isinstance(dropout, numbers.Number) or not 0 <=  dropout <= 1 or \
            isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          f"num_layers greater than 1, but got dropout={dropout} and "
                          f"num_layers={num_layers}")
            
        if not isinstance(hidden_size, int):
            raise TypeError(f"hidden_size should be of type int, got: {type(hidden_size).__name__}")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        if proj_size < 0:
            raise ValueError("proj_size should be a positive integer or zero to disable projections")
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")
        
        if not isinstance(feature_shape, tuple):
            raise(f"feature_shape should be of type tuple, got: {type(feature_shape).__name__}")
        if min(*feature_shape) < 0:
            raise ValueError("all items in feature_shape must be greater than zero")
        
        if mode == 'ConvLSTM':
            gate_size = 4 * hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)
        
        layers =  []

        for idx in range(num_layers):
            if idx == 0:
                layer = nn.Conv2d(in_channels=input_size+hidden_size, 
                          out_channels= gate_size,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=bias)
            else:
                layer = nn.Conv2d(in_channels=2*hidden_size,
                          out_channels=gate_size,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=bias)
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        stdv = 1.0/math.sqrt(self.hidden_channels) if self.hidden_channels  > 0  else 0
        for layer in self.layers:
            for weight in layer.parameters():
                nn.init.uniform_(weight, -stdv, stdv)


class ConvLSTM(ConvRNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__("ConvLSTM", *args, **kwargs)
        self.input_size = self.input_channels*self.feature_shape[0]*self.feature_shape[1]
        self.hidden_size  = self.hidden_channels*self.feature_shape[0]*self.feature_shape[1]

    def forward(self, input:th.Tensor, hx=None):
        if input.dim() not in (2, 3):
            raise ValueError(f"ConvLSTM: Expected input to be 2D or 3D, got {input.dim()}D instead")
        if input.shape[-1] != self.input_channels*self.feature_shape[0]*self.feature_shape[1]:
            raise ValueError(f"ConvLSTM: Expected flattened spatial features of size {self.input_size*self.feature_shape[0]*self.feature_shape[1]} but got {input.shape[-1]}")
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            input = input.unsqueeze(dim=batch_dim)
        
        h = th.zeros([len(self.layers), input.shape[batch_dim], self.hidden_channels*self.feature_shape[0]*self.feature_shape[1]],
                     dtype=input.dtype, device=input.device)
        c = th.zeros([len(self.layers), input.shape[batch_dim], self.hidden_channels*self.feature_shape[0]*self.feature_shape[1]],
                     dtype=input.dtype, device=input.device)
        if hx is not None:
            if not isinstance(hx, Tuple):
                raise ValueError(f"hx has to be a tuple (h, c), got: {type(hx).__name__}")
            if len(hx) != 2:
                raise ValueError(f"hx must contain exactly two items, got: {len(hx)}")
            if hx[0].shape != h.shape or hx[1].shape != c.shape:
                raise ValueError(f"shape mismatch between expected size of a hidden state and the actual size")
            h = hx[0]
            c = hx[1]

        input = input.view(input.shape[0], input.shape[1], self.input_channels, self.feature_shape[0], self.feature_shape[1])
        
        h = h.view(h.shape[0], h.shape[1], self.hidden_channels, self.feature_shape[0], self.feature_shape[1])
        c = c.view(c.shape[0], c.shape[1], self.hidden_channels, self.feature_shape[0], self.feature_shape[1])
    
        if not self.batch_first:
            out, state = self.lstm_sequential_method(th.unbind(input, axis=0), (h, c))
        else:
            out, state = self.lstm_sequential_method(th.unbind(input, axis=1), (h, c))
        h, c = state
        h = h.view(h.shape[0], h.shape[1], h.shape[2]*h.shape[3]*h.shape[4])
        c = c.view(c.shape[0], c.shape[1], c.shape[2]*c.shape[3]*c.shape[4])
        out = th.cat(out, dim=int(self.batch_first))
        out = out.view(out.shape[0], out.shape[1], out.shape[2]*out.shape[3]*out.shape[4])
        return out, (h, c,)


    def lstm_sequential_method(self, iterable: List[th.Tensor], states: Tuple[th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
            h, c = states

            # Unbind the stacked tensors into lists of tensors (one per layer).
            # This breaks the single-tensor dependency.
            h_list = list(h.unbind(0))
            c_list = list(c.unbind(0))

            lstm_output = []

            for idx, feature in enumerate(iterable):
                for jdx, layer in enumerate(self.layers):
                    if jdx == 0:
                        this_layer_input = feature
                    else:
                        # Use the hidden state from the previous layer
                        this_layer_input = h_list[jdx-1]

                    # Use current layer's hidden state
                    gates = layer(th.cat([this_layer_input, h_list[jdx]], dim=1))

                    i, f, g, o = gates.chunk(4, dim=1)
                    i = th.sigmoid(i)
                    f = th.sigmoid(f)
                    g = th.tanh(g)
                    o = th.sigmoid(o)

                    # Calculate NEW state tensors (out of place)
                    new_c = f * c_list[jdx] + i * g
                    new_h = o * th.tanh(new_c)

                    # Update the list to point to the NEW tensors.
                    # The OLD tensors are preserved in the graph for autograd.
                    c_list[jdx] = new_c
                    h_list[jdx] = new_h

                lstm_output.append(h_list[-1].unsqueeze(dim=int(self.batch_first)))

            # Stack the lists back into a single tensor for the return value
            h_out = th.stack(h_list, dim=0)
            c_out = th.stack(c_list, dim=0)

            return lstm_output, (h_out, c_out)

class PoolReduce(nn.Module):
    def __init__(self, shapes: Tuple[Tuple[int, int,int],Tuple[int, int, int]], config: List[Tuple[int,int]]):
        super().__init__()
        self.original_shape = shapes[0]
        self.new_shape = shapes[1]
        self.pool_layer = nn.AdaptiveAvgPool2d((self.new_shape[1], self.new_shape[2]))
        layers = []
        for idx in range(-1, len(config)):
            if idx == -1:
                layers.append(nn.Linear(self.new_shape[0]*self.new_shape[1]*self.new_shape[2], config[0][0]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(config[idx][0], config[idx][1]))
                layers.append(nn.ReLU())
        self.layers =nn.Sequential(*layers)
    
    def forward(self, features):
        original0 = features.shape[0]
        original1  = features.shape[1]
        features = features.view(original0*original1, self.original_shape[0], self.original_shape[1], self.original_shape[2])
        features  = self.pool_layer(features)
        features =  features.view(original0,original1, self.new_shape[0]*self.new_shape[1]*self.new_shape[2])
        return self.layers(features)


class ConvLSTMWrapper(ConvLSTM):
    def __init__(self, *args, **kwargs):
        self.net_arch = kwargs.pop("net_arch", None)
        super().__init__(*args, **kwargs)
        if self.net_arch is None:
            warnings.WarningMessage("Caution! Not including a post-process architecture to \
                                     the ConvLSTM is very resource intensive")
            self.postprocess_unit = nn.Identity()
        else:
            mode  = self.net_arch.get("mode", None)
            if mode == "pool_reduce":
                net_arch_kwargs = self.net_arch.copy()
                net_arch_kwargs.pop("mode")
                self.postprocess_unit = PoolReduce(**net_arch_kwargs)
            else:
                raise NotImplementedError(f"Wrapper mode \'{mode}\' not implemented")
    def forward(self, input, hx=None):
        out, states = super().forward(input, hx)
        out = self.postprocess_unit(out)
        return out, states
    

def make_model(
    batch_first=False,
    T=5,
    B=3,
    C=4,
    H=7,
    W=7,
    hidden=8,
    layers=2
):
    model = ConvLSTM(
        feature_shape=(H, W),
        input_size=C,
        hidden_size=hidden,
        num_layers=layers,
        batch_first=batch_first
    )
    F = C * H * W
    if batch_first:
        x = th.randn(B, T, F)
    else:
        x = th.randn(T, B, F)
    return model, x, (T, B, F, hidden, layers, H, W)

def test_output_and_state_shapes():
    model, x, meta = make_model()
    T, B, F, hidden, layers, H, W = meta

    out, (h, c) = model(x)

    # Output shape
    assert out.shape == (T, B, hidden * H * W)

    # Hidden state shapes
    # In test_output_and_state_shapes
    assert h.shape == (layers, B, hidden * H * W)
    assert c.shape == (layers, B, hidden * H * W)

def test_batch_first_equivalence():
    # 1. Create the first model (Time-first)
    model_1, x_1, meta = make_model(batch_first=False)
    
    # 2. Create the second model (Batch-first)
    model_2, _, _ = make_model(batch_first=True)

    # 3. Load weights so they are identical
    model_2.load_state_dict(model_1.state_dict())

    # 4. Create input for model_2 that matches x_1 numerically
    # x_1 is (T, B, F) -> x_2 must be (B, T, F)
    x_2 = x_1.transpose(0, 1).contiguous()

    # 5. Run both
    out_1, _ = model_1(x_1)
    out_2, _ = model_2(x_2)

    # 6. Align output shapes for comparison: (B, T, F) -> (T, B, F)
    out_2 = out_2.transpose(0, 1)

    assert th.allclose(out_1, out_2, atol=1e-5)

def test_state_carry_over():
    model, x, meta = make_model()
    T, B, F, hidden, layers, H, W = meta

    x1 = x[:2]
    x2 = x[2:]

    out1, state1 = model(x1)
    out2, state2 = model(x2, hx=state1)

    out_full, _ = model(x)

    out_cat = th.cat([out1, out2], dim=0)

    assert th.allclose(out_cat, out_full, atol=1e-5)

def test_backward_pass():
    model, x, _ = make_model()

    out, _ = model(x)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)

def test_determinism_given_same_state():
    model, x, _ = make_model()

    out1, state1 = model(x)
    out2, state2 = model(x, hx=state1)

    assert not th.allclose(out1, out2)


def run_tests():
    print("Running tests for ConvRNN...")

    try:
        print("Running test_output_and_state_shapes...")
        test_output_and_state_shapes()
        print("✅ test_output_and_state_shapes passed!")
    except AssertionError as e:
        print("❌ test_output_and_state_shapes failed!")
        print(e)

    try:
        print("Running test_batch_first_equivalence...")
        test_batch_first_equivalence()
        print("✅ test_batch_first_equivalence passed!")
    except AssertionError as e:
        print("❌ test_batch_first_equivalence failed!")
        print(e)

    try:
        print("Running test_state_carry_over...")
        test_state_carry_over()
        print("✅ test_state_carry_over passed!")
    except AssertionError as e:
        print("❌ test_state_carry_over failed!")
        print(e)

    try:
        print("Running test_backward_pass...")
        test_backward_pass()
        print("✅ test_backward_pass passed!")
    except AssertionError as e:
        print("❌ test_backward_pass failed!")
        print(e)

    try:
        print("Running test_determinism_given_same_state...")
        test_determinism_given_same_state()
        print("✅ test_determinism_given_same_state passed!")
    except AssertionError as e:
        print("❌ test_determinism_given_same_state failed!")
        print(e)

if __name__ == "__main__":
    run_tests()

