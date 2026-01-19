import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.abspath("../"))


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
            return NotImplementedError(f"Consolidator model {self.model_name} not implemented yet.")


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