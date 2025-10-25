import torch.nn as nn
import torch
from convLSTM import ConvLSTM, ConvLSTMCell, StackedConvLSTM


class ApplyToInput(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, xy):
        return self.layer(xy[0]), xy[1]


class ConcatAndApply(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    def forward(self, xy):
        combined = torch.cat([xy[0], xy[1]], dim=1)
        return self.layer(combined)

class PrintInAndOut(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        if type(x) == tuple:
            print(x[0].shape, x[1].shape)
        else:
            print(x.shape)
        out = self.layer(x)
        if type(out) == tuple:
            print(out[0].shape, out[1].shape)
        else:
            print(out.shape)
        return out
convBlock1 = [
    {
        "type": "custom",
        "layer": ConcatAndApply(nn.Conv2d(in_channels=22, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)),
        "out_channels": 16
    },
    {
        "type": "custom",
        "layer": nn.ReLU()
    },
    {
        "type": "custom",
        "layer": nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        "out_channels": 64
    }
]

convBlock2 = [
    {
        "type": "custom",
        "layer": ApplyToInput(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=1, padding=1))
    },
    {
        "type": "custom",
        "layer": ApplyToInput(nn.ReLU())
    },
    {
        "type": "custom",
        "layer": ApplyToInput(nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0))
    },
    {
        "type": "custom",
        "layer": ConcatAndApply(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1))
    },
    {
        "type": "custom",
        "layer": nn.ReLU(),
        "out_channels": 128
    }
]

model = StackedConvLSTM(layer_input_dims=[6,16], output_dim=32, num_layers=2, bias=True, 
                        per_layer_conv_block_characteristics=[convBlock1,convBlock2])

B, C_in, H, W = 2, 6, 25, 25
x = torch.randn(B, C_in, H, W)
device = next(model.parameters()).device
hid = [(torch.zeros(2, 16, 25, 25, device=device),
                torch.zeros(2, 16, 25, 25, device=device)),
        (torch.zeros(2, 32, 12, 12, device=device),
                torch.zeros(2, 32, 12, 12, device=device))]
top_h, new_states = model(x, hid)

print(top_h.shape)
print(len(list(new_states)))
for i in list(new_states):
    print(i[0].shape)
    print(i[1].shape)
total_params = 0
for name, param in model.named_parameters():
    param_count = param.numel()  # Number of elements in the parameter tensor
    total_params += param_count
print(total_params)