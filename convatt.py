import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Tuple, List

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

@dataclass
class ConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

    def make(self):
        return nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

class ConvAttLayer(nn.Module):
    def  __init__(self, working_channels, qk_dim, kernel_size):
        super(ConvAttLayer, self).__init__()
        self.query_conv = nn.Conv2d(working_channels, qk_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.key_conv = nn.Conv2d(working_channels, qk_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.value_conv = nn.Conv2d(working_channels, working_channels, kernel_size=kernel_size, padding=kernel_size//2)

        #After adaptative pooling
        self.q_linear = nn.Linear(qk_dim, qk_dim)
        self.k_linear = nn.Linear(qk_dim, qk_dim)

        self.scale = torch.sqrt(torch.FloatTensor([qk_dim]))

    def forward(self, x):
        B, M, C, H, W = x.size()
        # Reshape input for convolution then back to original shape
        x_reshaped = x.view(B * M, C, H, W)
        queries = self.query_conv(x_reshaped).view(B, M, -1, H, W)
        keys = self.key_conv(x_reshaped).view(B, M, -1, H, W)
        values = self.value_conv(x_reshaped).view(B, M, -1, H, W)
        
        queries_pooled = F.adaptive_avg_pool2d(queries, (1, 1)).squeeze(-1).squeeze(-1)
        keys_pooled = F.adaptive_avg_pool2d(keys, (1, 1)).squeeze(-1).squeeze(-1)

        queries_linear = self.q_linear(queries_pooled)
        keys_linear = self.k_linear(keys_pooled)

        # Compute attention scores and causal mask
        scores = torch.matmul(queries_linear, keys_linear.transpose(-2, -1)) / self.scale.to(x.device)
        mask = torch.tril(torch.ones(M, M)).to(x.device)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        # Apply attention to values while preserving spatial dimensions
        out = torch.matmul(attn, values.view(B, M, -1, H * W).permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2).view(B, M, -1, H, W)
        return out # (B, M, C, H, W)

class ConvAttResidualRNNCell(nn.Module):
    # h_t = h_t-1 + ConvAtt([x_t;h_t-1])
    def __init__(self, working_channels, qk_dim, kernel_size):
        super(ConvAttResidualRNNCell, self).__init__()
        self.conv_att = ConvAttLayer(working_channels=working_channels,
                                     qk_dim=qk_dim,
                                     kernel_size=kernel_size)
    def forward(self, x, h):
        # x: (B, C, H, W)
        # h: (B, M-1, C, H, W) (External logic manages length M)
        B, C, H, W = x.size()
        h_combined = torch.cat([x.unsqueeze(1), h], dim=1)  # (B, M, C, H, W)
        return self.conv_att(h_combined)

class StackedConvAtt(nn.Module):
    """
    Manages a stack of ConvAttResidualRNN cells.
    Handles passing output of Layer i -> input of Layer i+1.
    """
    def __init__(self, input_channels: int, hidden_channels: int, qk_dim: int, kernel_size: int, stack_size: int, memory_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.stack_size = stack_size
        self.memory_size = memory_size
        self.hidden_channels = hidden_channels

        for i in range(stack_size):
            # First layer takes input_channels, subsequent layers take hidden_channels
            in_ch = input_channels if i == 0 else hidden_channels
            # Projection to match hidden_channels if needed
            if in_ch != hidden_channels:
                self.input_proj = nn.Conv2d(in_ch, hidden_channels, kernel_size=1)
            else:
                self.input_proj = nn.Identity()

            self.layers.append(
                ConvAttResidualRNNCell(hidden_channels, qk_dim, kernel_size)
            )

    def forward(self, x: torch.Tensor, states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x shape: (Batch, Channel, H, W)
        # states: List of tensors (B, M-1, C, H, W), one per layer

        next_states = []
        current_input = x

        for i, layer in enumerate(self.layers):
            # Project input if first layer
            if i == 0:
                current_input = self.input_proj(current_input)

            h = states[i]  # (B, M-1, C, H, W)

            # Feed current input into layer
            # Returns (B, M, C, H, W) where M includes new timestep
            att_output = layer(current_input, h)

            # Take the last output along the memory dimension
            next_h = att_output[:, -1]  # (B, C, H, W)

            # Update state: keep last (M-1) frames for next step
            if att_output.size(1) > self.memory_size:
                new_state = att_output[:, -(self.memory_size):, :, :, :]
            else:
                new_state = att_output

            next_states.append(new_state)

            # Output of this layer is input to next
            current_input = next_h

        # Return final layer output and all new states
        return current_input, next_states

# ==========================================
# Feature Extractor
# ==========================================

class ConvFeatureExtractor(BaseFeaturesExtractor):
    """
    Standard CNN to downsample image before ConvAtt.
    """
    def __init__(self, observation_space: Box, conv_configs: list):
        super().__init__(observation_space, features_dim=1)  # dim is dummy here, calculated below

        layers = []
        for cfg in conv_configs:
            layers.append(cfg.make())
            layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*layers)

        # Calculate output shape
        with torch.no_grad():
            dummy = torch.zeros((1, *observation_space.shape))
            out = self.cnn(dummy)
            self.out_channels, self.h, self.w = out.shape[1:]
            self.features_dim = self.out_channels * self.h * self.w

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs)

# ==========================================
# The Policy
# ==========================================

class ConvAttPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, config, **kwargs):
        # We need to initialize with dummy values then override
        self.config = config
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # 1. Extract Config
        conv_configs = config['conv_configs']
        self.hidden_channels = config['hidden_channels']
        self.qk_dim = config.get('qk_dim', 64)
        kernel_size = config['kernel_size']
        self.stack_size = config.get("stack_size", 1)
        self.memory_size = config.get("memory_size", 4)
        self.pool_output_size = config.get("pool_output_size", (2, 2))

        # 2. Re-Initialize Feature Extractor
        self.features_extractor = ConvFeatureExtractor(observation_space, conv_configs)

        # Get shapes after CNN
        c = self.features_extractor.out_channels
        h = self.features_extractor.h
        w = self.features_extractor.w
        self.spatial_shape = (h, w)
        self.cnn_output_dim = c * h * w

        # 3. Initialize Stacked ConvAtt
        self.att_conv = StackedConvAtt(c, self.hidden_channels, self.qk_dim, kernel_size, self.stack_size, self.memory_size)

        # 4. Adaptive Pooling Layer
        # Pool the last hidden state features (position M) to reduce dimensionality
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.pool_output_size)

        # Calculate pooled dimension
        pooled_h, pooled_w = self.pool_output_size
        self.pooled_dim = self.hidden_channels * pooled_h * pooled_w

        # 5. Overwrite the MLP Extractors
        # SB3 creates these based on features_dim. We need them to accept the pooled features
        self.mlp_extractor = self._build_mlp_extractor()

        # 6. Force value net to match
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

    def _build_mlp_extractor(self):
        """Helper to rebuild the MLP part (Actor/Critic heads) to match pooled features."""
        from stable_baselines3.common.torch_layers import MlpExtractor
        return MlpExtractor(
            feature_dim=self.pooled_dim,  # Input is pooled features from last hidden state
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

    def _process_sequence(self, features, att_states, episode_starts, lstm=None):
        """
        Custom sequence processing for ConvAtt.
        features: (Batch, Seq_Len, C_in, H, W)  <-- Output of CNN
        att_states: Tuple containing a tensor of shape (Num_Layers, Batch, Memory_Size * C * H * W)

        Returns pooled features from the last position (M) in the attention memory.
        """
        # 1. Unpack Dimensions
        n_layers, n_samples, flat_dim = att_states[0].shape
        T = features.size(1)  # Time dimension
        H, W = self.spatial_shape

        # 2. Reshape States: (Layers, Batch, Flat) -> List of (Batch, M, C, H, W)
        current_states = []
        h_flat = att_states[0]

        for i in range(n_layers):
            # Extract layer i, reshape to 5D
            # flat_dim = Memory_Size * C * H * W
            state_i = h_flat[i].view(n_samples, self.memory_size, self.hidden_channels, H, W)
            current_states.append(state_i)

        # 3. Iterate over Time (T)
        pooled_outputs = []
        for t in range(T):
            # Input for this timestep: (Batch, C_in, H, W)
            input_t = features[:, t]

            # Handle Masking (Reset state if episode starts)
            if episode_starts[:, t].any():
                mask = (~episode_starts[:, t].bool()).view(n_samples, 1, 1, 1, 1)

                # Apply mask to ALL layers in the stack
                masked_states = []
                for state in current_states:
                    masked_states.append(state * mask)
                current_states = masked_states

            # Forward pass through Stacked ConvAtt
            # returns: output_t (Batch, C_out, H, W), new_states (List of tensors (B, M, C, H, W))
            output_t, current_states = self.att_conv(input_t, current_states)

            # Extract the last layer's full hidden state (most informed position M)
            # current_states[-1] has shape (B, M, C, H, W)
            last_layer_state = current_states[-1]  # (B, M, C, H, W)

            # Take the last position (M) which is the most informed due to causal masking
            last_position = last_layer_state[:, -1]  # (B, C, H, W)

            # Apply adaptive pooling to reduce spatial dimensions
            pooled = self.adaptive_pool(last_position)  # (B, C, pooled_h, pooled_w)

            # Flatten: (B, C * pooled_h * pooled_w)
            pooled_flat = pooled.flatten(start_dim=1)
            pooled_outputs.append(pooled_flat)

        # 4. Re-pack States for SB3
        # Convert List of (Batch, M, C, H, W) back to (Layers, Batch, Flat)
        new_h_list = []
        for state in current_states:
            new_h_list.append(state.flatten(start_dim=1))

        # Stack layers: (Layers, Batch, Flat)
        final_h = torch.stack(new_h_list, dim=0)

        # 5. Prepare Output
        # (Batch, Seq_Len, Pooled_Features)
        pooled_outputs = torch.stack(pooled_outputs, dim=1)

        return pooled_outputs, (final_h,)

    def forward(self, obs: torch.Tensor, lstm_states: Tuple[torch.Tensor], episode_starts: torch.Tensor, deterministic: bool = False):
        """
        Forward pass override.
        The MLP heads receive pooled features from the last position (M) in the attention memory,
        which is the most informed due to causal masking.
        """
        # obs might be (Batch, C, H, W) or (Batch, Seq, C, H, W)
        if obs.dim() == 4:
            # Add sequence dim -> (Batch, 1, C, H, W)
            obs = obs.unsqueeze(1)

        B, T, C, H, W = obs.shape

        # 1. CNN Feature Extraction
        # Flatten Batch and Time to pass through CNN: (B*T, C, H, W)
        img_flat = obs.view(B * T, C, H, W)
        features = self.features_extractor(img_flat)

        # Reshape back to sequence: (B, T, C_out, H_out, W_out)
        features = features.view(B, T, self.features_extractor.out_channels,
                                self.features_extractor.h, self.features_extractor.w)

        # 2. ConvAtt Processing
        # Returns: pooled_features (B, T, Pooled_Dim) from last position in memory, new_states
        pooled_features, next_att_states = self._process_sequence(features, lstm_states, episode_starts)

        # 3. Take only the last timestep for Action/Value (Standard SB3 behavior for forward)
        last_pooled_features = pooled_features[:, -1]  # (B, Pooled_Dim)

        # 4. Actor / Critic
        # Distribution
        latent_pi = self.mlp_extractor.forward_actor(last_pooled_features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Value
        latent_vf = self.mlp_extractor.forward_critic(last_pooled_features)
        values = self.value_net(latent_vf)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob, next_att_states

