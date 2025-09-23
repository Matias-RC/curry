from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

"""def process_action_space(action_space):
    if type(action_space) == spaces.discrete.Discrete:                        
        num_actions = action_space.n    
        dim_actions = 1
        dim_rep_actions = num_actions
        tuple_action = False        
        discrete_action = True
    elif type(action_space) == spaces.tuple.Tuple:              
        num_actions = action_space[0].n    
        dim_actions = len(action_space)    
        dim_rep_actions = dim_actions
        tuple_action = True
        discrete_action = True
    elif type(action_space) == spaces.Box:  
        num_actions = 1   
        dim_actions = action_space.shape[0] 
        dim_rep_actions = dim_actions
        tuple_action = True
        discrete_action = False
    else:
        raise AssertionError(f"Unsupported action space {action_space}")
    return num_actions, dim_actions, dim_rep_actions, tuple_action, discrete_action"""

ActorOut = namedtuple(
    "ActorOut",
    [     
        "pri", # sampled primiary action
        "pri_param", # parameter for primary action dist, can be logit or gaussian mean + log var        
        "reset", # sampled reset action
        "reset_logits", # parameter for reset dist, i.e. logit
        "action", # tuple of the above two actions 
        "action_prob", # prob of primary action 
        "c_action_log_prob", # log prob of chosen action
        "baseline", # baseline 
        "baseline_enc", # baseline encoding, only for non-scalar enc_type
        "entropy_loss", # entropy loss
        "reg_loss", # regularization loss
        "misc",
    ],
)

def compute_discrete_log_prob(logits, actions):
    assert len(logits.shape) == len(actions.shape) + 1
    has_dim = len(actions.shape) == 3    
    end_dim = 2 if has_dim else 1
    log_prob = -torch.nn.CrossEntropyLoss(reduction="none")(
            input=torch.flatten(logits, 0, end_dim), target=torch.flatten(actions, 0, end_dim)
    )
    log_prob = log_prob.view_as(actions)
    if has_dim:
        log_prob = torch.sum(log_prob, dim=-1)
    return log_prob


def sample(logits, greedy, dim=-1):
    if not greedy:
        gumbel_noise = torch.empty_like(logits).uniform_().clamp(1e-10, 1).log().neg_().clamp(1e-10, 1).log().neg_()
        sampled_action = (logits + gumbel_noise).argmax(dim=dim)
        return sampled_action.detach()
    else:
        return torch.argmax(logits, dim=dim)

def atanh(x, eps=1e-6):
    x = torch.clamp(x, -1.0+eps, 1.0-eps)
    return 0.5 * (x.log1p() - (-x).log1p())

class ActorBaseNet(nn.Module):
    # base class for all actor network
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=False, record_state=False):
        super(ActorBaseNet, self).__init__()
        self.disable_thinker = flags.wrapper_type == 1
        self.record_state = record_state        

        self.obs_space = obs_space        
        if not self.disable_thinker:
            self.pri_action_space = action_space[0][0]            
        else:
            self.pri_action_space = action_space[0]

        self.flags = flags      
        self.tree_rep_meaning = tree_rep_meaning

        self.float16 = flags.float16
        self.num_rewards = 1
        self.num_rewards += int(flags.im_cost > 0.0)
        self.num_rewards += int(flags.cur_cost > 0.0)
        self.enc_type = flags.critic_enc_type  
        self.rv_tran = None
        self.critic_zero_init = flags.critic_zero_init         
        self.legacy = getattr(flags, "legacy", False)  

        # action space processing
        # used to equal this -> process_action_space(self.pri_action_space)
        self.num_actions = 4
        self.dim_actions = 1
        self.dim_rep_actions = 4
        self.tuple_action = False
        self.discrete_action = True
        # Yet again I disclose the modifications are because I don't integrate openai gym
        
        self.ordinal = flags.actor_ordinal
        if self.ordinal:
            indices = torch.arange(self.num_actions).view(-1, 1)
            ordinal_mask = (indices + indices.T) <= (self.num_actions - 1)
            ordinal_mask = ordinal_mask.float()
            self.register_buffer("ordinal_mask", ordinal_mask)

        # state space processing
        self.see_tree_rep = flags.see_tree_rep and not self.disable_thinker
        if self.see_tree_rep:
            self.tree_reps_shape = obs_space["tree_reps"].shape[1:]             
            if self.legacy:
                self.tree_reps_shape = list(self.tree_reps_shape)
                self.tree_reps_shape[0] -= 2

        self.see_h = flags.see_h and not self.disable_thinker
        if self.see_h:
            self.hs_shape = obs_space["hs"].shape[1:]
        self.see_x = flags.see_x
        if self.see_x and not self.disable_thinker:
            self.xs_shape = obs_space["xs"].shape[1:]
        self.see_real_state = flags.see_real_state        
        
        if flags.see_real_state:
            assert obs_space["real_states"].dtype in ['uint8', 'float32'], f"Unupported observation sapce {obs_space['real_states']}"            
            low = torch.tensor(obs_space["real_states"].low[0])
            high = torch.tensor(obs_space["real_states"].high[0])
            self.need_norm = torch.isfinite(low).all() and torch.isfinite(high).all()            
            if self.need_norm:
                self.register_buffer("norm_low", low)
                self.register_buffer("norm_high", high)
            self.real_states_shape = obs_space["real_states"].shape[1:]     

        if getattr(flags, "ppo_k", 1) > 1:
            kl_beta = torch.tensor(1.)
            self.register_buffer("kl_beta", kl_beta)

    def normalize(self, x):
        x.dtype == self.obs_space["real_states"].dtype
        if self.need_norm:
            x = (x.float() - self.norm_low) / \
                (self.norm_high -  self.norm_low)
        return x
    
    def ordinal_encode(self, logits):
        norm_softm = F.sigmoid(logits)
        norm_softm_tiled = torch.tile(norm_softm.unsqueeze(-1), [1,1,1,self.num_actions])
        return torch.sum(torch.log(norm_softm_tiled + 1e-8) * self.ordinal_mask + torch.log(1 - norm_softm_tiled + 1e-8) * (1 - self.ordinal_mask), dim=-1)

    def get_weights(self):
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}    

    def set_weights(self, weights, strict=True):
        device = next(self.parameters()).device
        tensor = isinstance(next(iter(weights.values())), torch.Tensor)
        if not tensor:
            self.load_state_dict(
                {k: torch.tensor(v, device=device) for k, v in weights.items()}, strict=strict
            )
        else:
            self.load_state_dict({k: v.to(device) for k, v in weights.items()}, strict=strict)
class ActorNetSingle:
    pass
class ActorNetSep(ActorBaseNet):
    def __init__(self, obs_space, action_space, flags, tree_rep_meaning=None, record_state=False):
        super(ActorNetSep, self).__init__(obs_space, action_space, flags, tree_rep_meaning, record_state)
        self.actor = ActorNetSingle(obs_space, action_space, flags, tree_rep_meaning, record_state, actor=True, critic=False)
        self.critic = ActorNetSingle(obs_space, action_space, flags, tree_rep_meaning, record_state, actor=False, critic=True)
        self.initial_state(1)
        self.rv_tran = self.critic.rv_tran

    def initial_state(self, batch_size, device=None):
        actor_state = self.actor.initial_state(batch_size, device)
        critic_state = self.critic.initial_state(batch_size, device)
        self.state_idx = len(actor_state)
        return actor_state + critic_state
    
    def forward(self, env_out, core_state=(), clamp_action=None, compute_loss=False, greedy=False):
        actor_state = core_state[:self.state_idx]
        critic_state = core_state[self.state_idx:]
        actor_out, actor_state = self.actor(env_out, actor_state, clamp_action, compute_loss, greedy)
        critic_out, critic_state = self.critic(env_out, critic_state, clamp_action, compute_loss, greedy)
        misc = actor_out.misc
        actor_out = ActorOut(
            pri=actor_out.pri,
            pri_param=actor_out.pri_param,
            reset=actor_out.reset,
            reset_logits=actor_out.reset_logits,
            action=actor_out.action,
            action_prob=actor_out.action_prob,
            c_action_log_prob=actor_out.c_action_log_prob,            
            baseline=critic_out.baseline,
            baseline_enc=critic_out.baseline_enc,
            entropy_loss=actor_out.entropy_loss,
            reg_loss=actor_out.reg_loss,
            misc=misc,
        )
        core_state = actor_state + critic_state
        return actor_out, core_state