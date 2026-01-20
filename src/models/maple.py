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
        #consolidator_config = config["consolidator_config"]
        #self.consolidator = Consolidator(consolidator_config) #Currently not used
        self.thinker_optimizer = torch.optim.Adam(self.thinker.parameters(), lr=self.config["thinker_learning_rate"])

        # Memory as nn.Parameter
        #memory_config = config["memory_config"]
        #self.memory = nn.Parameter(torch.randn(memory_config["memory_size"], memory_config["hidden_size"]))

        # Supervision
        #self.num_supervision_steps = config["num_supervision_steps"] 
        # --- Take into account that this is not used yet ---


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
    
    
    def generate(self, dynamic_batch, max_solution_length=120, memory=None):
        """
        The idea is to generate autoregresively the solution to a given batch of levels.
        Symbolic batch: List of Hash Maps that relate  keys like ["walls", "boxes", "goals", "player"] to sets or a tuple in the case of player
        """
        symbolic_batch = dynamic_batch["states_0"] #(B, H, W, C)
        obs, mask = dynamic_batch["states_tensors"], dynamic_batch["attention_mask"] #(B, H, W, C), 

        kv_cache = None

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
        logits = decoder_output["logits"].squeeze(1) # not sure if (B, 1, Vocab) is its true shape 
        values_seq =  decoder_output["value"]
        hidden_states_seq = decoder_output["last_hidden_state"]
        kv_cache = decoder_output["past_key_values"]
        #rewards tensor in the form (B,) wed like it to be (B, T) later on
        step_batch_obj = {
            "states":symbolic_batch,
            "states_tensors":obs,
            "attention_mask": mask
        }

        obs, attention_mask, rewards_tensor,  actions_tensor =  self.env.step_batch(step_batch_obj, logits) #maybe logits should pass through exp first?
        actions_seq = actions_tensor.unsqueeze(1)
        step_batch_obj["attention_mask"] =  attention_mask
        step_batch_obj["states_tensors"] = torch.cat([step_batch_obj["states_tensors"], obs], dim=1) #B, T, H, W, C--- Thus making the seq
        #batch["states"] = new_states alredy done in step_batch

        rewards_seq = rewards_tensor.unsqueeze(1)

        for i in range(max_solution_length-1):
            if attention_mask.max().item() == 0:
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
            actions_seq = torch.cat([actions_seq,  actions_tensor.unsqueeze(1)], dim=1)
            step_batch_obj["attention_mask"] =  attention_mask
            step_batch_obj["states_tensors"] = torch.cat([step_batch_obj["states_tensors"], obs], dim=1) 

            rewards_seq = torch.cat([rewards_seq, rewards_tensor.unsqueeze(1)], dim=1)
        return (step_batch_obj["states_tensors"], actions_seq, logits_seq, rewards_seq, hidden_states_seq, values_seq)
    
    def simplified_maple_loop(self, str_batch):
        dynamic_batch  = self.env.get_dynamic_batch(str_batch)
        """
        dynamic_batch = {
            "states_0": [...], list of hash maps
            "states_tensors": tensor (B, 1, H, W, C)
            "attention_mask": tensor (B, 1)
        """
        #1. Make the trayectory
        obs_seq, actions_seq, logits_seq, rewards_seq, hidden_states_seq, values_seq = self.generate(dynamic_batch, max_solution_length=self.config["max_solution_length"])

        #2. Advantage with TD(0)
        discounts = self.config["discount"] * torch.ones_like(rewards_seq)
        deltas = rewards_seq + discounts * values_seq[:, 1:, :] - values_seq[:, :-1, :]
        advantages = torch.zeros_like(rewards_seq)
        for t in reversed(range(rewards_seq.size(1)-1)):
            advantages[:, t, :] = deltas[:, t, :] + discounts[:, t, :] * advantages[:, t+1, :]

        #3. Prepare the batch for target forward
        batch = {
            "hidden_states": hidden_states_seq,
            "logits": logits_seq,
            "actions": actions_seq,
            "advantages": advantages,
            "gae": advantages + values_seq[:, :-1, :],
        }
        #4. Target forward
        updated_hidden_states = self.target_forward(batch)

        # With the target hidden states we use hubber loss on original hidden states and  advantage loss  on logits to  train the thinker
        self.thinker_optimizer.zero_grad()
        thinker_input = {
            "step_type": "parallel",
            "states_tensors": hidden_states_seq
        }
        thinker_output = self.thinker(thinker_input)
        logits = thinker_output["decoder_output"]["logits"]
        values = thinker_output["decoder_output"]["value"]
        thinker_loss = (
            F.huber_loss(hidden_states_seq, updated_hidden_states)
            - torch.mean(advantages *
                         torch.log_softmax(logits, dim=-1))
            + F.mse_loss(values, batch["gae"])
        )
        thinker_loss.backward()
        self.thinker_optimizer.step()

        return thinker_loss.item()

    def maple_loop(self, str_batch, num_supervision_steps=2):
        """
        Work in progress do not use yet!!
        """
        for i in range(num_supervision_steps):
            #if this is the first supervision step we dont have past trayectory yet we dont make with consolidator
            #else go ahead
            continue
        #TODO:  @MatiasRC
        pass
