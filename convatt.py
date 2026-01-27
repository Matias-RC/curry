import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Box
from dataclasses import dataclass
from typing import Tuple, List

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib import RecurrentPPO
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
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

# Recives: (B, M, C, H, W) output: (B, M, C, H, W)
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
        # queries_linear, keys_linear: (B, M, qk_dim)
        scores = torch.matmul(queries_linear, keys_linear.transpose(-2, -1)) / self.scale.to(x.device)  # (B, M, M)
        mask = torch.tril(torch.ones(M, M)).to(x.device)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)  # (B, M, M)

        # Apply attention to values while preserving spatial dimensions
        # values: (B, M, C, H, W) -> reshape to (B, M, C*H*W)
        B_v, M_v, C_v, H_v, W_v = values.shape
        values_flat = values.view(B, M, -1)  # (B, M, C*H*W)

        # attn: (B, M, M) @ values_flat: (B, M, C*H*W) -> (B, M, C*H*W)
        out_flat = torch.matmul(attn, values_flat)  # (B, M, C*H*W)

        # Reshape back to spatial
        out = out_flat.view(B, M, C_v, H_v, W_v)  # (B, M, C, H, W)
        return out

# Receives: x: (B, C, H, W), h: (B, M-1, C, H, W)
# Outputs: (B, M, C, H, W)
class ConvAttResidualRNNCell(nn.Module):
    # h_t = h_t-1 + ConvAtt([x_t;h_t-1])
    def __init__(self, working_channels, qk_dim, kernel_size):
        super(ConvAttResidualRNNCell, self).__init__()
        self.conv_att = ConvAttLayer(working_channels=working_channels,
                                     qk_dim=qk_dim,
                                     kernel_size=kernel_size)
        self.norm1 = nn.LazyInstanceNorm2d(working_channels)
        self.norm2 = nn.LazyInstanceNorm2d(working_channels)
        self.out_conv = nn.Conv2d(working_channels, working_channels, kernel_size=3, padding=1)
    def forward(self, x, h):
        # x: (B, C, H, W)
        # h: (B, M, C, H, W) for memory state
        if x.dim() != 4:
            raise ValueError(f"Unexpected x shape: {x.shape}, expected 4D tensor (B, C, H, W). Got {x.dim()}D tensor.")
        if h.dim() != 5:
            raise ValueError(f"Unexpected h shape: {h.shape}, expected 5D tensor (B, M, C, H, W). Got {h.dim()}D tensor.")

        B, C, H, W = x.size()
        h_combined = torch.cat([x.unsqueeze(1), h], dim=1)  # (B, M+1, C, H, W)
        h_combined = self.norm1(h_combined.view(B * (h_combined.size(1)), C, H, W)).view(B, h_combined.size(1), C, H, W)
        out = h_combined + self.conv_att(h_combined)
        out = self.norm2(out.view(B * out.size(1), C, H, W)).view(B, out.size(1), C, H, W)
        out = self.out_conv(out.view(B * out.size(1), C, H, W)).view(B, out.size(1), C, H, W)
        out = F.relu(out)
        return out

# Receives: x: (B, C, H, W), states: List of (B, M-1, C, H, W)
# Outputs: (B, C, H, W), List of (B, M-1, C, H, W)
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

        # Input projection (only for first layer if needed)
        if input_channels != hidden_channels:
            self.input_proj = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        else:
            self.input_proj = nn.Identity()

        for i in range(stack_size):
            self.layers.append(
                ConvAttResidualRNNCell(hidden_channels, qk_dim, kernel_size)
            )

    def forward(self, x: torch.Tensor, states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x shape: (Batch, Channel, H, W)
        # states: List of tensors (B, M, C, H, W), one per layer

        next_states = []
        current_input = x

        for i, layer in enumerate(self.layers):
            # Project input if first layer
            if i == 0:
                current_input = self.input_proj(current_input)

            h = states[i]  # (B, M, C, H, W)

            # Feed current input into layer
            # Returns (B, M+1, C, H, W) where M includes new timestep
            att_output = layer(current_input, h)

            if att_output.dim() != 5:
                raise ValueError(f"Layer {i} att_output has wrong shape: {att_output.shape}, expected 5D (B, M, C, H, W)")

            # Take the last output along the memory dimension
            next_h = att_output[:, -1]  # (B, C, H, W)

            if next_h.dim() != 4:
                raise ValueError(f"Layer {i} next_h has wrong shape: {next_h.shape}, expected 4D (B, C, H, W). att_output shape was {att_output.shape}")

            # Update state: keep last M frames for next step
            if att_output.size(1) > self.memory_size:
                new_state = att_output[:, -(self.memory_size):, :, :, :]
            else:
                new_state = att_output

            next_states.append(new_state)

            # Output of this layer is input to next
            current_input = next_h

        # Return final layer output and all new states
        return current_input, next_states

# Receives raw images, applies CNN to downsample
# Outputs feature maps
class ConvFeatureExtractor(BaseFeaturesExtractor):
    """
    Standard CNN to downsample image before ConvAtt.
    """
    def __init__(self, observation_space: Box, conv_configs: list):
        super().__init__(observation_space, features_dim=1)  # temporary placeholder

        layers = []
        for cfg in conv_configs:
            layers.append(cfg.make())
            layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*layers)

        # Infer output shape
        # After VecTransposeImage, observation_space.shape is (C, H, W)
        with torch.no_grad():
            dummy = torch.zeros((1, *observation_space.shape))
            out = self.cnn(dummy)
            self.out_channels, self.h, self.w = out.shape[1:]

        self._features_dim = self.out_channels * self.h * self.w

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 5:
            # obs: (B, T, C, H, W)
            B, T, C, H, W = obs.shape
            obs = obs.view(B * T, C, H, W)

            out = self.cnn(obs.to(torch.float32))
            # out: (B*T, C', H', W')

            # restore time dimension
            C2, H2, W2 = out.shape[1:]
            out = out.view(B, T, C2, H2, W2)

            return out
        else:
            return self.cnn(obs.to(torch.float32))

class AlternativeConvFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box):
        super().__init__(observation_space, features_dim=1)
        # Hardcoded for temporal simlicity
        conv_configs = [
            ConvConfig(in_channels=observation_space.shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            ConvConfig(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            ConvConfig(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        ]
        pool_tuple = (2, 2)
        
        layers = []
        for cfg in conv_configs:
            layers.append(cfg.make())
            layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*layers)
        C = conv_configs[-1].out_channels
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_tuple)
        self._features_dim = C * pool_tuple[0] * pool_tuple[1]
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.cnn(obs.to(torch.float32))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        return x

# Recieves images, passes through CNN + ConvAtt + Pooling + MLP heads
# Outputs actions and values
class ConvAttPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # ---- extract custom config BEFORE super ----
        self.config = kwargs.pop("config")

        conv_configs_params = self.config["conv_configs"]
        conv_configs = [ConvConfig(**params) for params in conv_configs_params]

        self.hidden_channels = self.config["hidden_channels"]
        self.qk_dim = self.config.get("qk_dim", 64)
        kernel_size = self.config["kernel_size"]
        self.stack_size = self.config.get("stack_size", 1)
        self.memory_size = self.config.get("memory_size", 4)
        self.pool_output_size = self.config.get("pool_output_size", (2, 2))

        pooled_h, pooled_w = self.pool_output_size
        self.pooled_dim = self.hidden_channels * pooled_h * pooled_w

        # Calculate the state dimension that will be used
        # We need to know the spatial dimensions after CNN, so we need to compute them first
        # Temporarily create feature extractor to get dimensions
        temp_extractor = ConvFeatureExtractor(observation_space, conv_configs)
        h = temp_extractor.h
        w = temp_extractor.w

        # State dimension: memory_size * hidden_channels * H * W
        lstm_hidden_size = self.memory_size * self.hidden_channels * h * w

        # ---- let SB3 construct everything ----
        kwargs['lstm_hidden_size'] = lstm_hidden_size
        kwargs['n_lstm_layers'] = self.stack_size
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # ---- override feature extractor (safe after super) ----
        self.features_extractor = ConvFeatureExtractor(
            observation_space, conv_configs
        )

        c = self.features_extractor.out_channels
        h = self.features_extractor.h
        w = self.features_extractor.w
        self.spatial_shape = (h, w)

        # ---- ConvAtt core ----
        self.att_conv = StackedConvAtt(
            c,
            self.hidden_channels,
            self.qk_dim,
            kernel_size,
            self.stack_size,
            self.memory_size,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.pool_output_size)

        # Set LSTM hidden state shape for SB3 compatibility
        # Shape: (n_layers, 1, flattened_state_dim)
        state_dim = self.memory_size * self.hidden_channels * h * w
        self.lstm_hidden_state_shape = (self.stack_size, 1, state_dim)

        # Override the MLP extractor to use pooled_dim instead of lstm_hidden_size
        # SB3 creates MLP with lstm_hidden_size, but we output pooled_dim
        self._build_mlp_extractor()

    # ------------------------------------------------------------------
    # Build MLP extractor with correct input dimensions
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # SB3 hook: this is THE fix
    # ------------------------------------------------------------------
    def _get_features_dim(self) -> int:
        return self.pooled_dim

    # Note: We don't override evaluate_actions - let SB3 handle the sequence processing
    # The base class will call our _process_sequence with the right shapes

    # ------------------------------------------------------------------
    # Override predict_values to properly extract spatial features
    # ------------------------------------------------------------------
    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        Overridden to handle spatial feature extraction for ConvAtt.
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

        # Process through ConvAtt
        latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, None)

        # Value head
        latent_vf = self.mlp_extractor.forward_critic(latent_vf[:, -1])
        return self.value_net(latent_vf)

    # ------------------------------------------------------------------
    # Override get_distribution to properly extract spatial features
    # ------------------------------------------------------------------
    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states,
        episode_starts: torch.Tensor,
    ):
        """
        Get the current policy distribution given the observations.
        Overridden to handle spatial feature extraction for ConvAtt.
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

        # Process through ConvAtt
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, None)

        # MLP head
        latent_pi = self.mlp_extractor.forward_actor(latent_pi[:, -1])
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    # ------------------------------------------------------------------
    # ConvAtt sequence processing
    # ------------------------------------------------------------------
    def _process_sequence(self, features, att_states, episode_starts, lstm=None):
        """
        features: Can be either:
                  - (B, T, C, H, W) - already has time dimension
                  - (B, C, H, W) - needs time dimension added
        att_states: RNNStates or tuple of (h_flat, c_flat) where h_flat is (n_layers, B, memory_size * hidden_channels * H * W)
                   c_flat is unused but kept for SB3 compatibility
        lstm parameter is unused, kept for SB3 compatibility
        """
        # Handle both 4D and 5D features
        if features.dim() == 4:
            # Add time dimension: (B, C, H, W) -> (B, 1, C, H, W)
            features = features.unsqueeze(1)

        # Handle RNNStates format from SB3
        if isinstance(att_states, RNNStates):
            # Extract policy states (we use the same for both pi and vf)
            h_flat = att_states.pi[0]
        elif isinstance(att_states, tuple):
            h_flat = att_states[0]
            # Check if it's nested tuples
            if isinstance(h_flat, tuple):
                h_flat = h_flat[0]
        else:
            h_flat = att_states

        n_layers, n_samples_state, flat_dim = h_flat.size()
        B = features.size(0)  # Actual batch size from features
        T = features.size(1)
        H, W = self.spatial_shape
        if episode_starts.dim() == 1:
            episode_starts = episode_starts.view(T, B).transpose(0, 1)
        # Verify the flat dimension matches our expectation
        expected_flat = self.memory_size * self.hidden_channels * H * W

        if flat_dim != expected_flat:
            raise ValueError(
                f"State dimension mismatch! "
                f"Got flat_dim={flat_dim}, expected {expected_flat}. "
                f"h_flat shape: {h_flat.shape}, "
                f"memory_size={self.memory_size}, hidden_channels={self.hidden_channels}, "
                f"H={H}, W={W}, n_layers={n_layers}, n_samples_state={n_samples_state}"
            )
        if n_samples_state != B:
            h_flat = h_flat.new_zeros(n_layers, B, flat_dim)
            n_samples_state = B
        current_states = []

        # Always use n_samples_state for reshaping - it's the correct batch dimension from states
        for i in range(n_layers):
            state_i = h_flat[i].view(
                n_samples_state,
                self.memory_size,
                self.hidden_channels,
                H,
                W,
            )
            current_states.append(state_i)

        pooled_outputs = []

        for t in range(T):
            input_t = features[:, t]

            if input_t.dim() != 4:
                raise ValueError(f"input_t at timestep {t} has wrong shape: {input_t.shape}, expected 4D (B, C, H, W). features shape: {features.shape}")

            # Handle episode starts - supports both 1D and 2D tensors
            if episode_starts.dim() == 1:
                episode_start_t = episode_starts
            else:
                episode_start_t = episode_starts[:, t]

            if episode_start_t.any():
                # Get actual batch size from features and states
                actual_batch_size = input_t.size(0)

                mask = (~episode_start_t.bool()).view(
                    actual_batch_size, 1, 1, 1, 1
                )
                current_states = [s * mask for s in current_states]

            _, current_states = self.att_conv(input_t, current_states)

            last_layer_state = current_states[-1]
            last_position = last_layer_state[:, -1]

            pooled = self.adaptive_pool(last_position)
            pooled_outputs.append(pooled.flatten(start_dim=1))

        new_h = torch.stack(
            [s.flatten(start_dim=1) for s in current_states], dim=0
        )

        pooled_outputs = torch.stack(pooled_outputs, dim=1)

        # SB3 expects RNNStates with pi and vf attributes
        # We use the same state for both policy and value networks
        new_states = RNNStates(pi=(new_h, new_h), vf=(new_h, new_h))
        return pooled_outputs, new_states

    # ------------------------------------------------------------------
    # Forward (SB3 recurrent API)
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        lstm_states,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ):
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
            # episode_starts should also be 2D: (B, T)
            if episode_starts.dim() == 1:
                episode_starts = episode_starts.unsqueeze(1)

        B, T, C, H, W = obs.shape

        img_flat = obs.view(B * T, C, H, W)
        features = self.features_extractor(img_flat)

        features = features.view(
            B,
            T,
            self.features_extractor.out_channels,
            self.features_extractor.h,
            self.features_extractor.w,
        )

        pooled_features, next_states = self._process_sequence(
            features, lstm_states, episode_starts
        )

        last_features = pooled_features[:, -1]

        latent_pi = self.mlp_extractor.forward_actor(last_features)
        latent_vf = self.mlp_extractor.forward_critic(last_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)

        actions = (
            distribution.mode()
            if deterministic
            else distribution.sample()
        )

        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob, next_states

