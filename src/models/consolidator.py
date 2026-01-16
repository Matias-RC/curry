import torch
import torch.nn as nn
import torch.nn.functional as F
from models.query_attention import QueryAttentionLayer
import os
import sys

sys.path.append(os.path.abspath("../"))

class TwoTowerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_name = config["model_name"]
        self.hidden_size = config["hidden_size"]
        self.prefix_size = config.get("prefix_size", 2)

        args = config["args"]

        self.connector_layer = QueryAttentionLayer(prefix_size=self.prefix_size, latent_dims=self.hidden_size)

        if self.model_name == "qwen2":
            from transformers import Qwen2Model, Qwen2Config
            args_model_1 = {
                "num_hidden_layers": args["consolidator_tower_1_num_layers"],
                "hidden_size": args["hidden_size"],
                "num_attention_heads": args["num_attention_heads"],
                "num_key_value_heads": args["num_attention_heads"],
                "intermediate_size": args["hidden_size"] * 4,
            }
            args_model_2 = {
                "num_hidden_layers": args["consolidator_tower_2_num_layers"],
                "hidden_size": args["hidden_size"],
                "num_attention_heads": args["num_attention_heads"],
                "num_key_value_heads": args["num_attention_heads"],
                "intermediate_size": args["hidden_size"] * 4,
            }
            config_model =  Qwen2Config(**args_model_1)
            self.encoder1 = Qwen2Model(config_model)
            config_model =  Qwen2Config(**args_model_2)
            self.encoder2 = Qwen2Model(config_model)

    def forward(self, x):
        x = self.encoder1(
            inputs_embeds=x,
            attention_mask=torch.ones(x.shape[:-1], device=x.device),
        ).last_hidden_state   # (B, T, D)

        prefix = self.connector_layer(x)  # (B, P, D)

        x = self.encoder2(
            inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:-1], device=prefix.device),
        ).last_hidden_state   # (B, P, D)

        return x


    
class Consolidator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_name = config["model_name"]
        args = config["args"]
        if self.model_name == "t5":
            from transformers import T5Model, T5Config
            args_model = {
                "num_layers": args["num_encoder_layers"],
                "num_decoder_layers": args["num_decoder_layers"],
                "d_model": args["hidden_size"],
                "num_heads": args["num_attention_heads"],
                "d_ff": args["hidden_size"] * 4,
            }
            config_model =  T5Config(**args_model)
            self.backbone = T5Model(config_model)

        else:
            config_two_tower = config["config_two_tower"]
            self.backbone = TwoTowerEncoder(config_two_tower)


    def forward(self, x):
        if self.model_name == "two_tower":
            return self.backbone(x)
        
        elif self.model_name == "t5":

            out = self.backbone(
                inputs_embeds=x["thinking_stream"],
                decoder_inputs_embeds=x["memory_states"],
            ).last_hidden_state
            
            return out
    

if __name__ == "__main__":
    config = {
        "config_two_tower": {
            "model_name": "qwen2",
            "hidden_size": 64,
            "prefix_size": 2,
            "args": {
                "num_attention_heads": 4,
                "consolidator_tower_1_num_layers": 2,
                "consolidator_tower_2_num_layers": 2,
                "hidden_size": 64,
                "num_attention_heads": 4
            }
        },
    }
    model = Consolidator(config)

    x = torch.randn((4, 10, 64))  # (B, T, D)
    print(x.shape)
    out = model(x)
    print(out.shape)  # Expected output shape: (4, prefix_size, hidden_size)