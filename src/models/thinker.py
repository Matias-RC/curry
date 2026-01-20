import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

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
        # Handle both (B, C, H, W) and (B, T, C, H, W)  
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            merge_time = True
        else:
            merge_time = False

        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        if merge_time:
            x = x.view(B, T, -1)
        return x

class ActionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_name = config["model_name"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.num_think_steps = config["num_think_steps"]

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
        
        # PPO Heads
        self.actor_head = nn.Linear(self.hidden_size, self.vocab_size)
        self.value_head = nn.Linear(self.hidden_size, 1)
    
    def detach_kv_cache(self, cache):
        my_new_cache = []
        for i in range(cache):
            my_new_cache.append((i[0].detach(), i[1].detach()))
        return tuple(my_new_cache)

    def forward_step(self, x):
        latent_states = x["latent_states"]
        past_key_values = x["kv_cache"]
        memory_states = x.get("memory_states", None)
        if self.model_name == "qwen2":
            if memory_states != None or past_key_values == None:
                outputs = self.backbone(
                    inputs_embeds=memory_states,
                    use_cache=False
                )          
                current_kv = self.detach_kv_cache(outputs.past_key_values.to_legacy_cache())
            else:
                current_kv = past_key_values

            h = latent_states
    
            for i in range(self.num_think_steps):
                outputs = self.backbone(
                    inputs_embeds=h,
                    past_key_values=DynamicCache.from_legacy_cache(current_kv),
                    use_cache=True
                )
                h = outputs.last_hidden_state
            h_out = h
            new_kv = outputs.past_key_values  # Tuple of KVs

            # Heads
            logits = self.actor_head(h_out)
            value = self.value_head(h_out)

            return {
                "logits": logits,      # (B, 1, Vocab)
                "value": value,        # (B, 1, 1)
                "last_hidden_state": h_out, # (B, 1, D)
                "past_key_values": self.detach_kv_cache(new_kv.to_legacy_cache())
            }
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def forward_parallel(self, x):
        latent_states = x["latent_states"]  # (B, T, D)    
        memory_states = x.get("memory_states", None)  # (B, M, D) or None
        if self.model_name == "qwen2":
            h = latent_states
    
            for _ in range(self.num_think_steps):
                inputs_embeds = h if memory_states is None else torch.cat([memory_states, h], dim=1)
                outputs = self.backbone(inputs_embeds=inputs_embeds)
                last_hidden_state = outputs.last_hidden_state
                h_new = last_hidden_state[:, -h.size(1):, :]  # Take the last T tokens
    
                h = h_new
    
            logits = self.actor_head(h)
            value = self.value_head(h)

    
            return {
                "logits": logits,
                "value": value,
                "last_hidden_state": h, #No need to pass the last keys and values since this is done all in parallel
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
    
    def forward(self, batch, memory_states=None):
        if batch["step_type"] == "parallel":
            latent_states = batch["states_tensors"]  # shape [B, T, D]
            x = {
                "latent_states": latent_states,
            }
            if memory_states is not None:
                x["memory_states"] = memory_states

            decoder_output = self.action_decoder.forward_parallel(x)  # shape [B, T, num_actions]

            return {
                "decoder_output": decoder_output,
                "attention_mask": batch["attention_mask"],
            }
        elif batch["step_type"] == "autorregressive":
            states_tensors = batch["states_tensors"]  # shape [B, H, W, C]
            latent_states = self.visual_encoder(states_tensors)
            x = {
                "latent_states": latent_states,
                "kv_cache":batch["kv_cache"]
            }
            if memory_states is not None and batch["step"] == 0:
                x["memory_states"] = memory_states
            decoder_output = self.action_decoder.forward_step(x)
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
        "num_think_steps": 1
    }

    action_decoder = ActionDecoder(config_action_decoder)
    action_decoder.eval()

    config_thinker = {
        "config_visual_encoder": config_visual_encoder,
        "config_action_decoder": config_action_decoder,
    }

    thinker = Thinker(config_thinker)
    thinker.eval()
