import torch.nn.functional as F
from torch import nn
import torch

from models.thinker import Thinker
from models.consolidator import Consolidator

class MAPLE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config["maple_config"]
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

    def target_forward(self, batch):
        z = batch["hidden_states"].detach().clone()
        z.requires_grad_(True)
    
        thinker_input = {
            "step_type": "parallel",
            "states_tensors": z
        }
    
        thinker_output = self.thinker(thinker_input)
    
        logits = thinker_output["decoder_output"]["logits"]
        values = thinker_output["decoder_output"]["value"]
    
        loss = (
            torch.mean(batch["advantages"] *
                       torch.log_softmax(logits, dim=-1))
            + F.mse_loss(values, batch["gae"])
        )
    
        grad, = torch.autograd.grad(
            loss, z,
            retain_graph=False,
            create_graph=False
        )
    
        z = z - self.config["learning_rate"] * grad
        z = z.detach()
    
        return z
    
    
    def generate(self, symbolic_batch, max_solution_length=120):
        """
        The idea is to generate autoregresively the solution to a given batch of levels.
        """
        obs = self.env.symbolic_batch_to_tensor(symbolic_batch)
        for i in range(max_solution_length):
            #use autoregressive forward from thinker to iteratively generate thes solution in  a batch of sokoban env.
            #if all levels in batch are solved and loop  still doesnt finish cut short
            pass

    def maple_loop(self, str_batch, num_supervision_steps=2):
        for i in range(num_supervision_steps):
            #if this is the first supervision step we dont have past trayectory yet we dont make with consolidator
            #else go ahead
            continue
        #TODO:  @MatiasRC
        pass