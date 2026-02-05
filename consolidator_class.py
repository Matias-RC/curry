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
    def __init__(
        self, 
        net_arch: dict,
        actor_prefix_shape: Tuple[int, int, int], # (C, H, W)
        critic_prefix_shape: Tuple[int, int, int], # (C, H, W)
        lr_schedule: Schedule,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        :param net_arch: Dictionary containing layer factories/configs. 
                         Must include 'shared', 'policy', and 'vf' definitions.
                         The first shared layer should likely be ChannelStackedSpatialAccumulation.
        :param actor_prefix_shape: The output shape of the actor prefix (C, H, W).
        :param critic_prefix_shape: The output shape of the critic prefix (C, H, W).
        """
        super().__init__()
        
        self.actor_prefix_shape = actor_prefix_shape
        self.critic_prefix_shape = critic_prefix_shape
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}

        # --- Build Shared Encoder (Evidence Accumulator) ---
        shared = []
        for i in range(net_arch["n_shared_layers"]):
            shared.append(net_arch["shared_class"][i](**net_arch["shared_kwargs"][i]))
        self.shared = nn.Sequential(*shared)

        # --- Build Actor Prefix Generator ---
        policy_layers = []
        for i in range(net_arch["n_policy_layers"]):
            policy_layers.append(net_arch["policy_layers"][i](**net_arch["policy_kwargs"][i]))
        self.policy_layers = nn.Sequential(*policy_layers)

        # --- Build Critic Prefix Generator ---
        vf_layers = []
        for i in range(net_arch["n_vf_layers"]):
            vf_layers.append(net_arch["vf_layers"][i](**net_arch["vf_kwargs"][i]))
        self.vf_layers = nn.Sequential(*vf_layers)

        # --- Optimizer Setup ---
        # Note: In MAPLE, this optimizer is used to train the Consolidator 
        # to predict the "optimized" prefixes found during the test-time inner loop.
        self.optimizer = self.optimizer_class(
            self.parameters(), 
            lr=lr_schedule(1), 
            **self.optimizer_kwargs
        )
    
    def forward(self, trajectory: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Processes a trajectory to generate actor and critic prefixes.
        :param trajectory: (B, T, C, H, W)
        :return: (Actor_Prefix, Critic_Prefix)
        """
        B, T, C, H, W = trajectory.shape
        
        # Flatten Time into Channels for the Spatial Accumulator
        # The ChannelStackedSpatialAccumulation layer expects (B, C*T, H, W)
        # and internally views it back to handle the temporal summation.
        obs = trajectory.reshape(B, T*C, H, W)

        features = self.shared(obs) # Output: (B, C_shared_out, H, W)

        return self.policy_layers(features), self.vf_layers(features)
    
    def policy_forward(self, trajectory: th.Tensor) -> th.Tensor:
        """Forward pass for just the actor head (useful for debugging or specific loss calc)"""
        B, T, C, H, W = trajectory.shape
        obs = trajectory.reshape(B, T*C, H, W)
        features = self.shared(obs)
        return self.policy_layers(features)

    def vf_forward(self, trajectory: th.Tensor) -> th.Tensor:
        """Forward pass for just the critic head"""
        B, T, C, H, W = trajectory.shape
        obs = trajectory.reshape(B, T*C, H, W)
        features = self.shared(obs)
        return self.vf_layers(features)
    
    def set_training_mode(self, mode: bool) -> None:
        """
        Switch between training and evaluation mode.
        This affects layers like BatchNorm or Dropout.
        """
        self.train(mode)
    
    def _init_empty(self, n_envs: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        Creates the zero-initialized prefixes for the very first attempt (Cold Start).
        Used by Maple.collect_rollouts to populate the DynamicBuffer before any history exists.
        
        :param n_envs: Number of environments
        :return: Tuple(Actor_Zero_Tensor, Critic_Zero_Tensor)
        """
        # Determine device based on model parameters
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'

        # Generate zeros matching the expected prefix shapes
        actor_init = th.zeros((n_envs, *self.actor_prefix_shape), dtype=th.float32, device=device)
        critic_init = th.zeros((n_envs, *self.critic_prefix_shape), dtype=th.float32, device=device)
        
        return actor_init, critic_init
    
if __name__ == "__main__":
    # Example Configuration
    obs_channels = 3
    prefix_channels = 16 # C_out

    net_arch = {
        "n_shared_layers": 1,
        "shared_class": [ChannelStackedSpatialAccumulation],
        "shared_kwargs": [{
            "C_in": obs_channels,    # The layer needs to know the channel count per frame
            "C_out": 64,             # Internal feature dimension
            "kernel": 3
        }],
        
        "n_policy_layers": 1,
        "policy_layers": [nn.Conv2d],
        "policy_kwargs": [{"in_channels": 64, "out_channels": prefix_channels, "kernel_size": 3, "padding": 1}],
        
        "n_vf_layers": 1,
        "vf_layers": [nn.Conv2d],
        "vf_kwargs": [{"in_channels": 64, "out_channels": prefix_channels, "kernel_size": 3, "padding": 1}],
    }

    # Instantiate
    consolidator = Consolidator(
        net_arch=net_arch,
        actor_prefix_shape=(prefix_channels, 64, 64), # Match your env H,W
        critic_prefix_shape=(prefix_channels, 64, 64),
        lr_schedule=lambda x: 1e-3
    )