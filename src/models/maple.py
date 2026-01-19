from torch import nn
import torch

from models.thinker import Thinker
from models.consolidator import Consolidator

class MAPLE(nn.Module):
    def __init__(self, config):
        super().__init__()

        thinker_config = config["thinker_config"]
        self.thinker = Thinker(thinker_config)
        consolidator_config = config["consolidator_config"]
        self.consolidator = Consolidator(consolidator_config)

        # Memory as nn.Parameter
        memory_config = config["memory_config"]
        self.memory = nn.Parameter(torch.randn(memory_config["memory_size"], memory_config["hidden_size"]))

        # Supervision
        self.num_supervision_steps = config["num_supervision_steps"] 
        
        # Enviroment
        self.env = config["env"] 

    def forward(self, batch):

        B = batch["states_tensors"].size(0)
        
        memory_states = self.memory.unsqueeze(0).expand(B, -1, -1)  # shape [B, memory_size, hidden_size]
        thinker_outputs = []
        for _ in range(self.num_supervision_steps):
            thinker_output = self.thinker(batch, memory_states)
            memory_states = self.consolidator({
                "memory_states": memory_states,
                "thinking_stream": thinker_output["decoder_output"]["last_hidden_state"],
                "thinking_stream_attention_mask": thinker_output["attention_mask"],
            })
            thinker_outputs.append(thinker_output)
        return thinker_outputs
    
    @torch.no_grad()
    def generate(self, batch, max_solution_length=50):

        dynamic_batch = self.env.get_dynamic_batch(batch["level_strs"])

        memory_states = self.memory.unsqueeze(0).expand(len(dynamic_batch["states"]), -1, -1)  # shape [B, memory_size, hidden_size]
        thinker_outputs = []
        for _ in range(self.num_supervision_steps):
            thinker_output = self.thinker.generate(self.env, dynamic_batch, memory_states, max_solution_length)
            memory_states = self.consolidator({
                "memory_states": memory_states,
                "thinking_stream": thinker_output["decoder_output"]["last_hidden_state"],
                "thinking_stream_attention_mask": thinker_output["attention_mask"],
            })
            thinker_outputs.append(thinker_output)

        return thinker_outputs