import torch
import torch.nn as nn
import torch.nn.functional as F

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from dataclasses import dataclass

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
    
@dataclass
class ConvLSTMConfig:
    hidden_chanels: int
    perceptual_channels: int
    kernel_size: int
    def make(self):
        padding = self.kernel_size // 2
        return nn.Conv2d(
            in_channels=self.perceptual_channels+self.hidden_chanels,
            out_channels=4 * self.hidden_chanels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding
        )

class ConvLSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, conv_configs):
        super(ConvLSTMFeatureExtractor, self).__init__(observation_space, features_dim=1)

        layers = nn.ModuleList()

        for config in conv_configs:
            layers.append(config.make())
            layers.append(nn.ReLU())
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, observations):
        return self.conv(observations)

class ConvLSTMPolicy(RecurrentActorCriticPolicy):
    """
    Currently stacked ConvLSTM layers are not supported.
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(ConvLSTMPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.lstm_actor = ConvLSTMConfig(
            hidden_chanels=32,
            perceptual_channels=64,
            kernel_size=3
        ).make()
        self.lstm_critic = ConvLSTMConfig(
            hidden_chanels=32,
            perceptual_channels=64,
            kernel_size=3
        ).make()
    
    def _process_sequence(self, features, lstm_states, episode_starts, lstm):
        batch_size, seq_len, *feature_shape = features.size()
        h, c = lstm_states
        h = h.view(batch_size, -1, feature_shape[1], feature_shape[2])
        c = c.view(batch_size, -1, feature_shape[1], feature_shape[2])

        outputs = []
        for t in range(seq_len):
            x_t = features[:, t]
            combined = torch.cat([x_t, h], dim=1)
            gates = lstm(combined)
            i, f, o, g = torch.chunk(gates, 4, dim=1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c = f * c + i * g
            h = o * torch.tanh(c)

            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        new_lstm_states = (h.view(batch_size, -1), c.view(batch_size, -1))
        outputs = outputs.view(batch_size, seq_len, -1)
        return outputs, new_lstm_states

