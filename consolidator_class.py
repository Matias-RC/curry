from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.type_aliases import Schedule
import torch.nn as nn
import torch as th
import numpy as np

class ChannelStackedSpatialAccumulation(nn.Module):
    def __init__(self, C_in, C_out, kernel):
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out

        self.conv = nn.Conv2d(
            C_in,
            C_out,
            kernel,
            padding=kernel // 2
        )

    def forward(self, x):
        B, CT, H, W = x.shape
        C = self.C_in
        assert CT%C == 0
        
        T = CT//C

        x = x.view(B, T, C, H, W)

        x = x.reshape(B*T, C, H, W)
        z = self.conv(x)

        z = z.view(B, T, self.C_out, H, W)

        z = th.tanh(z.sum(dim=1))

        return z   # (B, C_out, H, W)


class Consolidator(nn.Module):
    def __init__(self, 
                 net_arch:dict,
                 lr_schedule: Schedule,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        shared = []
        for i in range(net_arch["n_shared_layers"]):
            shared.append(net_arch["shared_class"][i](**net_arch["shared_kwargs"][i]))
        self.shared = nn.Sequential(shared)

        policy_layers = []
        for i in range(net_arch["n_policy_layers"]):
            policy_layers.append(net_arch["policy_layers"][i](**net_arch["policy_kwargs"][i]))
        self.policy_layers = nn.Sequential(policy_layers)

        vf_layers = []
        for i in range(net_arch["n_vf_layers"]):
            vf_layers.append(net_arch["vf_layers"][i](**net_arch["vf_kwargs"][i]))
        self.vf_layers = nn.Sequential(vf_layers)

        self.optimizer = self.optimizer_class(
            self.parameters(), 
            lr=lr_schedule(1), 
            **self.optimizer_kwargs
        )
    
    def forward(self, trajectory: th.Tensor):
        B, T, C, H, W = trajectory.shape
        obs = trajectory.reshape(B, T*C, H, W)

        features = self.shared(obs) # (B, C_out, H, W)

        return self.policy_layers(features), self.vf_layers(features)
    
    def policy_forward(self, trajectory: th.Tensor):
        B, T, C, H, W = trajectory.shape
        obs = trajectory.reshape(B, T*C, H, W)
        features = self.shared(obs)
        return self.policy_layers(features)
    def vf_forward(self, trajectory: th.Tensor):
        B, T, C, H, W = trajectory.shape
        obs = trajectory.reshape(B, T*C, H, W)
        features = self.shared(obs)
        return self.vf_layers(features)
    
    def set_training_mode(self, mode: bool) -> None:
        pass
    
    def _init_empty(self):
        pass