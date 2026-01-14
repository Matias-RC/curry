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

class ActionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_name = config["model_name"]
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]

        args = config["args"]

        if self.model_name == "qwen2":
            from transformers import Qwen2Model, Qwen2Config
            args_model = {
                "num_hidden_layers": args["num_layers"],
                "hidden_size": args["hidden_size"],
                "num_attention_heads": args["num_attention_heads"],
                "num_key_value_heads": args["num_attention_heads"],
                "intermediate_size": args["hidden_size"] * 4,
            }
            config_model =  Qwen2Config(**args_model)
            self.backbone = Qwen2Model(config_model)

        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        latent_states = x["latent_states"]
        if self.model_name == "qwen2":
            outputs = self.backbone(inputs_embeds=latent_states)
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states[:, :, :])

            return {
                "logits": logits,
            }
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")


class Thinker(nn.Module):
    def __init__(self, config):
        super().__init__()

        config_visual_encoder = config["config_visual_encoder"]
        self.visual_encoder = VisualEncoder(config_visual_encoder)

        config_action_decoder = config["config_action_decoder"]
        self.action_decoder = ActionDecoder(config_action_decoder)
    
    def forward(self, batch):

        states_tensors = batch["states_tensors"]  # shape [B, T, H, W, C]
        latent_states = self.visual_encoder(states_tensors)  # shape [B, T, D]
        x = {
            "latent_states": latent_states,
        }
        decoder_output = self.action_decoder(x)  # shape [B, T, num_actions]

        return {
            "decoder_output": decoder_output
        }


if __name__ == "__main__":
        
    config_visual_encoder = {
        "in_channels": 4,
        "latent_dim": 32,
        "grid_shape_x": 10,
        "grid_shape_y": 10,
        "channels": [16, 32, 32],   # out channels for each conv layer
        "kernel_size": 3,
        "padding": 1,
        "stride": 2,                # applied only to last conv
    }

    visual_encoder = VisualEncoder(config_visual_encoder)
    visual_encoder.eval()

    """
    states_tensors = batch["states_tensors"]  # shape [B, T, H, W, C]
    out = model(states_tensors)
    out.shape > torch.Size([8, 65, 32])
    """ 

    config_action_decoder = {
        "model_name": "qwen2",
        "args": {
            "num_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 1,
        },
        "hidden_size": 32,
        "vocab_size": 4,
    }

    action_decoder = ActionDecoder(config_action_decoder)
    action_decoder.eval()

    config_thinker = {
        "config_visual_encoder": config_visual_encoder,
        "config_action_decoder": config_action_decoder,
    }

    thinker = Thinker(config_thinker)
    thinker.eval()