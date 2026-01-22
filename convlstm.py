import torch
import torch.nn as nn
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Tuple, List

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================================
# 1. Configuration & Helper Classes
# ==========================================

@dataclass
class ConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

    def make(self):
        return nn.Conv2d(
            self.in_channels, self.out_channels, 
            self.kernel_size, self.stride, self.padding
        )

class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.
    """
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2
        
        # Gates: input, forget, output, cell_gate (concatenated)
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=self.padding,
            bias=True,
        )
        
        # Initialize forget gate bias to 1.0 for better gradient flow
        with torch.no_grad():
            self.gates.bias[hidden_channels:2 * hidden_channels].fill_(1.0)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = state
        
        # Concatenate input and hidden state along channel dimension
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        
        # Split into 4 gates: Input, Forget, Output, Gate
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        
        return new_h, (new_h, new_c)

class StackedConvLSTM(nn.Module):
    """
    Manages a stack of ConvLSTM cells.
    Handles passing output of Layer i -> input of Layer i+1.
    """
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int, stack_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.stack_size = stack_size
        
        for i in range(stack_size):
            # First layer takes input_channels, subsequent layers take hidden_channels
            in_ch = input_channels if i == 0 else hidden_channels
            self.layers.append(
                ConvLSTMCell(in_ch, hidden_channels, kernel_size)
            )

    def forward(self, x: torch.Tensor, states: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # x shape: (Batch, Channel, H, W)
        # states: List of (h, c) tuples, one per layer
        
        next_states = []
        current_input = x
        
        for i, layer in enumerate(self.layers):
            h, c = states[i]
            # Feed current input into layer
            next_h, (new_h, new_c) = layer(current_input, (h, c))
            
            # Store new state
            next_states.append((new_h, new_c))
            
            # Output of this layer is input to next
            current_input = next_h
            
        # Return final layer output and all new states
        return current_input, next_states

# ==========================================
# 2. Feature Extractor
# ==========================================

class ConvFeatureExtractor(BaseFeaturesExtractor):
    """
    Standard CNN to downsample image before ConvLSTM.
    """
    def __init__(self, observation_space: Box, conv_configs: list[ConvConfig]):
        super().__init__(observation_space, features_dim=1) # dim is dummy here, calculated below
        
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
# 3. The Policy
# ==========================================

class ConvLSTMPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, config, **kwargs):
        # We need to initialize with dummy values then override
        self.config = config
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # 1. Extract Config
        conv_configs = config['conv_configs']
        self.hidden_channels = config['hidden_channels']
        kernel_size = config['kernel_size']
        self.stack_size = config.get("stack_size", 1)

        # 2. Re-Initialize Feature Extractor
        self.features_extractor = ConvFeatureExtractor(observation_space, conv_configs)
        
        # Get shapes after CNN
        c = self.features_extractor.out_channels
        h = self.features_extractor.h
        w = self.features_extractor.w
        self.spatial_shape = (h, w)
        self.cnn_output_dim = c * h * w
        
        # 3. Initialize Stacked ConvLSTM
        self.lstm_conv = StackedConvLSTM(c, self.hidden_channels, kernel_size, self.stack_size)
        
        # 4. Calculate LSTM Output Dimension (Flattened)
        # The output of the ConvLSTM is (Batch, Hidden_Channels, H, W)
        self.lstm_output_dim = self.hidden_channels * h * w
        
        # 5. Overwrite the MLP Extractors
        # SB3 creates these based on features_dim. We need them to accept the FLATTENED ConvLSTM output.
        self.mlp_extractor = self._build_mlp_extractor() # Rebuilds with correct input dim
        
        # 6. Force value net to match
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

    def _build_mlp_extractor(self):
        """Helper to rebuild the MLP part (Actor/Critic heads) to match ConvLSTM output."""
        from stable_baselines3.common.torch_layers import MlpExtractor
        return MlpExtractor(
            feature_dim=self.lstm_output_dim, # Input is flattened ConvLSTM output
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

    def _process_sequence(self, features, lstm_states, episode_starts, lstm=None):
        """
        Custom sequence processing for ConvLSTM.
        features: (Batch, Seq_Len, C_in, H, W)  <-- Output of CNN
        lstm_states: Tuple(h, c) each of shape (Num_Layers, Batch, Flat_Hidden_Dim)
        """
        # 1. Unpack Dimensions
        n_layers, n_samples, _ = lstm_states[0].shape
        T = features.size(1) # Time dimension
        H, W = self.spatial_shape
        
        # 2. Reshape States: (Layers, Batch, Flat) -> List of (Batch, C, H, W)
        # We need a list of (h, c) tuples for the StackedConvLSTM
        current_states = []
        h_flat, c_flat = lstm_states
        
        for i in range(n_layers):
            # Extract layer i, reshape to 4D
            h = h_flat[i].view(n_samples, self.hidden_channels, H, W)
            c = c_flat[i].view(n_samples, self.hidden_channels, H, W)
            current_states.append((h, c))

        # 3. Iterate over Time (T)
        lstm_outputs = []
        for t in range(T):
            # Input for this timestep: (Batch, C_in, H, W)
            input_t = features[:, t]
            
            # Handle Masking (Reset state if episode starts)
            # episode_starts is (Batch, Seq_Len)
            if episode_starts[:, t].any():
                mask = (~episode_starts[:, t].bool()).view(n_samples, 1, 1, 1)
                
                # Apply mask to ALL layers in the stack
                masked_states = []
                for (h, c) in current_states:
                    masked_states.append((h * mask, c * mask))
                current_states = masked_states

            # Forward pass through Stacked ConvLSTM
            # returns: output_t (Batch, C_out, H, W), new_states (List of tuples)
            output_t, current_states = self.lstm_conv(input_t, current_states)
            
            # Flatten output spatial dims: (Batch, C*H*W)
            # We must flatten here because the Actor/Critic MLPs expect 1D vectors
            lstm_outputs.append(output_t.flatten(start_dim=1))

        # 4. Re-pack States for SB3
        # Convert List of (Batch, C, H, W) back to (Layers, Batch, Flat)
        new_h_list = []
        new_c_list = []
        for (h, c) in current_states:
            new_h_list.append(h.flatten(start_dim=1))
            new_c_list.append(c.flatten(start_dim=1))
            
        # Stack layers: (Layers, Batch, Flat)
        final_h = torch.stack(new_h_list, dim=0)
        final_c = torch.stack(new_c_list, dim=0)
        
        # 5. Prepare Output
        # (Batch, Seq_Len, Features)
        lstm_outputs = torch.stack(lstm_outputs, dim=1)
        
        return lstm_outputs, (final_h, final_c)

    def forward(self, obs: torch.Tensor, lstm_states: Tuple[torch.Tensor, torch.Tensor], episode_starts: torch.Tensor, deterministic: bool = False):
        """
        Forward pass override.
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
        features = features.view(B, T, self.features_extractor.out_channels, self.features_extractor.h, self.features_extractor.w)

        # 2. ConvLSTM Processing
        # Returns: latent_features (B, T, Flat_Dim), new_states
        latent_features, next_lstm_states = self._process_sequence(features, lstm_states, episode_starts)

        # 3. Take only the last timestep for Action/Value (Standard SB3 behavior for forward)
        last_features = latent_features[:, -1] # (B, Flat_Dim)

        # 4. Actor / Critic
        # Distribution
        latent_pi = self.mlp_extractor.forward_actor(last_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Value
        latent_vf = self.mlp_extractor.forward_critic(last_features)
        values = self.value_net(latent_vf)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
            
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob, next_lstm_states