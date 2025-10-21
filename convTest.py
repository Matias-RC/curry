import torch.nn as nn
import torch
from convLSTM import ConvLSTM, ConvLSTMCell

x = torch.rand((32, 64, 128, 128))
convlstm = ConvLSTM(input_dim=64, hidden_dim=16, kernel_size=(3,3), bias=True)
h, c = convlstm(x)
h, c = convlstm(x, (h, c))
print(h.shape,c.shape)