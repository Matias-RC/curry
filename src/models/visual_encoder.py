import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channels = config["in_channels"]
        latent_dim = config["latent_dim"]
        channels = config["channels"]
        kernel_size = config["kernel_size"]
        padding = config["padding"]
        stride = config["stride"]
        grid_shape_x = config["grid_shape_x"]
        grid_shape_y = config["grid_shape_y"]
        self.input_hw = (grid_shape_x, grid_shape_y)

        layers = []
        prev_c = self.in_channels

        for i, out_c in enumerate(channels):
            layers.append(
                nn.Conv2d(
                    prev_c,
                    out_c,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride if i == len(channels) - 1 else 1,
                )
            )
            layers.append(nn.ReLU())
            prev_c = out_c

        self.cnn = nn.Sequential(*layers)

        self._cnn_out_dim = self._infer_cnn_out_dim()

        self.fc = nn.Linear(self._cnn_out_dim, latent_dim)

    def _infer_cnn_out_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *self.input_hw)
            out = self.cnn(dummy)
            return out.numel()

    def forward(self, x):
        B, T, W, H, C = x.shape

        # (B, T, C, W, H) → (B*T, C, W, H)
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, W, H)

        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x.view(B, T, -1)

if __name__ == "__main__":
        
    config_model = {
        "in_channels": 4,
        "latent_dim": 32,
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "channels": [16, 32, 32],   # out channels for each conv layer
        "kernel_size": 3,
        "padding": 1,
        "stride": 2,                # applied only to last conv
    }
    model = VisualEncoder(config_model)
    model.eval()

    """
    states_tensors = batch["states_tensors"]  # shape [B, T, H, W, C]
    out = model(states_tensors)
    out.shape > torch.Size([8, 65, 32])
    """ 