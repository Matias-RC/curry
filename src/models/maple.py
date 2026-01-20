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
    
    
    def generate(self, symbolic_batch, max_solution_length=120, memory=None):
        """
        The idea is to generate autoregresively the solution to a given batch of levels.
        Symbolic batch: List of Hash Maps that relate  keys like ["walls", "boxes", "goals", "player"] to sets or a tuple in the case of player
        """
        kv_cache = None
        obs = self.env.symbolic_batch_to_tensor(symbolic_batch, self.config["grid_shape_x"],  self.config["grid_shape_y"])
        thinker_input =  {
            "step_type":"autoregressive",
            "states_tensors":obs,
            "kv_cache":kv_cache,
            "step":0
        }
        thinker_output = self.thinker(thinker_input, memory) #Memory gets print into kv so it is only seen once
        decoder_output = thinker_output["decoder_output"]
        """
        thinker_output = {
            "decoder_output":{
                "logits": logits,      # (B, 1, Vocab)
                "value": value,        # (B, 1, 1)
                "last_hidden_state": h_out, # (B, 1, D)
                "past_key_values": self.detach_kv_cache(new_kv.to_legacy_cache()) #Already detached to avoid BPTT
            }
        }
        """
        logits_seq = decoder_output["logits"]
        logits = decoder_output["logits"].squeeze(1) # Keep this as break point cuz not sure if (B, 1, Vocab) is its true shape)
        values_seq =  decoder_output["value"]
        hidden_states_seq = decoder_output["last_hidden_state"]
        kv_cache = decoder_output["past_key_values"]
        #rewards tensor in the form (B,) wed like it to be (B, T) later on
        mask = self.env.get_att_mask(symbolic_batch, next(self.thinker.parameters()).device) #mask (B,1)
        step_batch_obj = {
            "states":symbolic_batch,
            "states_tensors":obs,
            "mask": mask
        }

        obs, attention_mask, rewards_tensor,  actions_tensor =  self.env.step_batch(step_batch_obj, logits) #maybe logits should pass through exp first?
        actions_seq = actions_tensor.unsqueeze(1)
        step_batch_obj["mask"] =  attention_mask
        step_batch_obj["states_tensors"] = torch.cat([step_batch_obj["states_tensors"], obs], dim=1) #B, T, H, W, C--- Thus making the seq
        #batch["states"] = new_states alredy done in step_batch

        rewards_seq = rewards_tensor.unsqueeze(1)

        for i in range(max_solution_length-1):
            if attention_mask.max() == 0:
                break
            thinker_input =  {
                "step_type":"autoregressive",
                "states_tensors":obs,
                "kv_cache":kv_cache,
                "step": i+1
            } 
            thinker_output = self.thinker(thinker_input)
            decoder_output = thinker_output["decoder_output"]

            hidden_states_seq = torch.cat([hidden_states_seq, decoder_output["last_hidden_state"]], dim=1)
            logits_seq  = torch.cat([logits_seq, decoder_output["logits"]], dim=1)
            values_seq = torch.cat([values_seq, decoder_output["value"]], dim=1)

            kv_cache = decoder_output["past_key_values"]
            obs, attention_mask, rewards_tensor, actions_tensor =  self.env.step_batch(step_batch_obj, logits) 
            actions_seq = torch.cat([actions_seq,  actions_seq.unsqueeze(1)], dim=1)
            step_batch_obj["mask"] =  attention_mask
            step_batch_obj["states_tensors"] = torch.cat([step_batch_obj["states_tensors"], obs], dim=1) 

            rewards_seq = torch.cat([rewards_seq, rewards_tensor.unsqueeze(1)], dim=1)
        return (step_batch_obj["states_tensors"], actions_seq, logits_seq, rewards_seq, hidden_states_seq, values_seq)
    
    def simplified_maple_loop(self, str_batch):
        dynamic_batch  = self.env.get_dynamic_batch(str_batch)
        """
        dynamic_batch = {
            "states_0":
        """

    def maple_loop(self, str_batch, num_supervision_steps=2):
        for i in range(num_supervision_steps):
            #if this is the first supervision step we dont have past trayectory yet we dont make with consolidator
            #else go ahead
            continue
        #TODO:  @MatiasRC
        pass