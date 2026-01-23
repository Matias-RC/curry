import torch
import torch.nn as nn
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Tuple, List

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
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

class ConvFeatureExtractor(BaseFeaturesExtractor):
    """
    Standard CNN to downsample image before ConvLSTM.
    """
    def __init__(self, observation_space: Box, conv_configs: List[ConvConfig]):
        super().__init__(observation_space, features_dim=1) # dim is dummy here, calculated below
        
        layers = []
        for cfg in conv_configs:
            layers.append(cfg.make())
            layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*layers)

        # Calculate output shape
        # After VecTransposeImage, observation_space.shape is (C, H, W)
        with torch.no_grad():
            dummy = torch.zeros((1, *observation_space.shape))
            out = self.cnn(dummy)
            self.out_channels, self.h, self.w = out.shape[1:]
            self._features_dim = self.out_channels * self.h * self.w

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs)

class ConvLSTMPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Extract config BEFORE super().__init__
        self.config = kwargs.pop("config")

        # 1. Extract Config
        conv_configs_params = self.config['conv_configs']
        conv_configs = [ConvConfig(**params) for params in conv_configs_params]
        self.hidden_channels = self.config['hidden_channels']
        kernel_size = self.config['kernel_size']
        self.stack_size = self.config.get("stack_size", 1)
        self.pool_output_size = self.config.get("pool_output_size", [2, 2])

        # Convert to tuple if needed
        if isinstance(self.pool_output_size, list):
            self.pool_output_size = tuple(self.pool_output_size)

        # Calculate the pooled dimension
        pooled_h, pooled_w = self.pool_output_size
        self.pooled_dim = self.hidden_channels * pooled_h * pooled_w

        # 2. Create temporary feature extractor to get dimensions
        temp_extractor = ConvFeatureExtractor(observation_space, conv_configs)
        h = temp_extractor.h
        w = temp_extractor.w

        # Calculate lstm_hidden_size for SB3 - uses spatial dimensions for state storage
        lstm_hidden_size = self.hidden_channels * h * w
        kwargs['lstm_hidden_size'] = lstm_hidden_size
        kwargs['n_lstm_layers'] = self.stack_size

        # Call super().__init__ with proper lstm_hidden_size
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # 3. Re-Initialize Feature Extractor (replace the one created by super)
        self.features_extractor = ConvFeatureExtractor(observation_space, conv_configs)

        # Get shapes after CNN
        c = self.features_extractor.out_channels
        h = self.features_extractor.h
        w = self.features_extractor.w
        self.spatial_shape = (h, w)
        self.cnn_output_dim = c * h * w

        # 3. Initialize Stacked ConvLSTM
        self.lstm_conv = StackedConvLSTM(c, self.hidden_channels, kernel_size, self.stack_size)

        # 4. Add adaptive pooling to ensure fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.pool_output_size)

        # Note: super().__init__ already created mlp_extractor and value_net
        # with the correct dimensions based on lstm_hidden_size
        # We need to recreate it with pooled_dim
        self._build_mlp_extractor()

    def _build_mlp_extractor(self) -> None:
        """
        Create a new MLP extractor using pooled_dim as input instead of lstm_hidden_size.
        This is necessary because SB3's default uses lstm_hidden_size for the MLP input.
        """
        from stable_baselines3.common.torch_layers import MlpExtractor

        self.mlp_extractor = MlpExtractor(
            self.pooled_dim,  # Use pooled dimension, not lstm_hidden_size
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _get_features_dim(self) -> int:
        """Override to return the pooled dimension after adaptive pooling."""
        return self.pooled_dim

    # Note: We don't override evaluate_actions - let SB3 handle the sequence processing
    # The base class will call our _process_sequence with the right shapes

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        Overridden to handle spatial feature extraction for ConvLSTM.
        """
        # Same preprocessing as get_distribution
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
            if episode_starts.dim() == 1:
                episode_starts = episode_starts.unsqueeze(1)

        B, T, C, H, W = obs.shape

        # Extract CNN features
        img_flat = obs.view(B * T, C, H, W)
        features = self.features_extractor(img_flat)

        # Reshape to sequence
        features = features.view(
            B,
            T,
            self.features_extractor.out_channels,
            self.features_extractor.h,
            self.features_extractor.w,
        )

        # Process through ConvLSTM
        latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, None)

        # Value head
        last_features = latent_vf[:, -1]
        latent_vf = self.mlp_extractor.forward_critic(last_features)
        return self.value_net(latent_vf)

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ):
        """
        Get the current policy distribution given the observations.
        Overridden to handle spatial feature extraction for ConvLSTM.
        """
        # Observations come as (B, C, H, W) or (B*T, C, H, W)
        if obs.dim() == 3:
            # Add batch dimension
            obs = obs.unsqueeze(0)

        if obs.dim() == 4:
            # Single timestep: (B, C, H, W) -> (B, 1, C, H, W)
            obs = obs.unsqueeze(1)
            # episode_starts should also be 2D: (B, T)
            if episode_starts.dim() == 1:
                episode_starts = episode_starts.unsqueeze(1)

        B, T, C, H, W = obs.shape

        # Extract CNN features for all timesteps
        img_flat = obs.view(B * T, C, H, W)
        features = self.features_extractor(img_flat)

        # Reshape back to sequence: (B, T, C_out, H_out, W_out)
        features = features.view(
            B,
            T,
            self.features_extractor.out_channels,
            self.features_extractor.h,
            self.features_extractor.w,
        )

        # Process through ConvLSTM
        latent_features, next_lstm_states = self._process_sequence(features, lstm_states, episode_starts, None)

        # MLP head
        last_features = latent_features[:, -1]
        latent_pi = self.mlp_extractor.forward_actor(last_features)
        return self._get_action_dist_from_latent(latent_pi), next_lstm_states

    def _process_sequence(self, features, lstm_states, episode_starts, lstm=None):
        """
        Custom sequence processing for ConvLSTM.
        features: Can be either:
                  - (B, T, C, H, W) - already has time dimension
                  - (B, C, H, W) - needs time dimension added
        lstm_states: RNNStates or Tuple(h, c) each of shape (Num_Layers, Batch, Flat_Hidden_Dim)
        """
        # Handle both 4D and 5D features
        if features.dim() == 4:
            # Add time dimension: (B, C, H, W) -> (B, 1, C, H, W)
            features = features.unsqueeze(1)
            if episode_starts.dim() == 1:
                episode_starts = episode_starts.unsqueeze(1)
        # Handle RNNStates format from SB3
        if isinstance(lstm_states, RNNStates):
            # Extract policy states (we use the same for both pi and vf)
            h_flat, c_flat = lstm_states.pi
        else:
            h_flat, c_flat = lstm_states

        # 1. Unpack Dimensions
        n_layers, n_samples_state, _ = h_flat.shape
        B = features.size(0)  # Actual batch size from features
        T = features.size(1) # Time dimension
        H, W = self.spatial_shape

        # 2. Reshape States: (Layers, Batch, Flat) -> List of (Batch, C, H, W)
        # We need a list of (h, c) tuples for the StackedConvLSTM
        current_states = []

        # Handle batch size mismatch between states and features
        batch_size_to_use = B if n_samples_state != B else n_samples_state

        for i in range(n_layers):
            # Extract layer i, reshape to 4D
            h = h_flat[i].view(batch_size_to_use, self.hidden_channels, H, W)
            c = c_flat[i].view(batch_size_to_use, self.hidden_channels, H, W)
            current_states.append((h, c))

        # 3. Iterate over Time (T)
        lstm_outputs = []
        for t in range(T):
            # Input for this timestep: (Batch, C_in, H, W)
            input_t = features[:, t]

            # Handle Masking (Reset state if episode starts)
            # episode_starts is (Batch, Seq_Len)
            if episode_starts[:, t].any():
                # Get actual batch size from input and states
                actual_batch_size = input_t.size(0)
                state_batch_size = current_states[0][0].size(0)

                # Only apply mask if batch sizes match
                # During training, SB3 might pass different batch dimensions
                if actual_batch_size == state_batch_size:
                    mask = (~episode_starts[:, t].bool()).view(actual_batch_size, 1, 1, 1)

                    # Apply mask to ALL layers in the stack
                    masked_states = []
                    for (h, c) in current_states:
                        masked_states.append((h * mask, c * mask))
                    current_states = masked_states

            # Forward pass through Stacked ConvLSTM
            # returns: output_t (Batch, C_out, H, W), new_states (List of tuples)
            output_t, current_states = self.lstm_conv(input_t, current_states)

            # Apply adaptive pooling for fixed-size output regardless of spatial dims
            pooled_t = self.adaptive_pool(output_t)  # (Batch, C, pool_h, pool_w)

            # Flatten pooled output: (Batch, C*pool_h*pool_w)
            lstm_outputs.append(pooled_t.flatten(start_dim=1))

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

        # Return RNNStates with pi and vf attributes
        # We use the same state for both policy and value networks
        new_states = RNNStates(pi=(final_h, final_c), vf=(final_h, final_c))
        return lstm_outputs, new_states

    def forward(self, obs: torch.Tensor, lstm_states: Tuple[torch.Tensor, torch.Tensor], episode_starts: torch.Tensor, deterministic: bool = False):
        """
        Forward pass override.
        """
        # obs might be (Batch, C, H, W) or (Batch, Seq, C, H, W)
        if obs.dim() == 4:
            # Add sequence dim -> (Batch, 1, C, H, W)
            obs = obs.unsqueeze(1)
            # episode_starts should also be 2D: (B, T)
            if episode_starts.dim() == 1:
                episode_starts = episode_starts.unsqueeze(1)

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