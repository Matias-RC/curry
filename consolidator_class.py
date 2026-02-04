import torch as th
import torch.nn as nn

class ChannelStackedSpatialAccumulation(nn.Module):
    def __init__(self, C_in, C_out, kernel):
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out

        self.conv = nn.Conv2d(
            C_in,
            C_out,
            kernel,
            padding=kernel // 2
        )

    def forward(self, x):
        B, CT, H, W = x.shape
        C = self.C_in
        assert CT%C == 0
        
        T = CT//C

        x = x.view(B, T, C, H, W)

        x = x.reshape(B*T, C, H, W)
        z = self.conv(x)

        z = z.view(B, T, self.C_out, H, W)

        z = th.relu(z.sum(dim=1))

        return z   # (B, C_out, H, W)


class Consolidator(nn.Module):
    def __init__(self, n_layers, net_arch):
        layers = []
        for i in range(n_layers):
            layers.append(net_arch["layer_class_list"][i](**net_arch["layer_kwargs_list"][i]))
        self.layers = nn.Sequential(layers)
    def set_training_mode(self, mode: bool) -> None:
        pass
    def _init_empty(self):
        pass