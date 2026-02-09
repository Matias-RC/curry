import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, TypeVar, Union, Tuple
from gymnasium.spaces import Box

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.policies import ActorCriticPolicy

class PrefixCombinator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, out_shape):
        super().__init__()
        past_out = in_channels
        layers = []

        for _ in range(n_layers):
            layers.append(nn.Conv2d(past_out, hidden_channels,
                                    3, 1, 1))
            layers.append(nn.ReLU())
            past_out = hidden_channels
        layers.append(nn.Conv2d(past_out, out_channels, 3, 1, 1))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.pool_layer = nn.AdaptiveAvgPool2d(out_shape)
    def forward(self, x, y):
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        y = y.to(dtype)
        input = th.cat([x,y], dim=1)
        return th.flatten(self.pool_layer(self.layers(input)),start_dim=1)

class ConvFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, config: dict):
        self.conv_configs = config.get("conv_configs", None)

        assert self.conv_configs is not None, "conv_configs must be provided in extractor_config"

        super(ConvFeatureExtractor, self).__init__(observation_space, features_dim=config["features_dim"])  # Dummy features_dim

        in_channels = observation_space.shape[0]
        conv_layers = []
        for conv_conf in self.conv_configs:
            out_channels = conv_conf['out_channels']
            kernel_size = conv_conf.get('kernel_size', 3)
            stride = conv_conf.get('stride', 1)
            padding = conv_conf.get('padding', 1)

            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch_size, channels, height, width)
        x = self.conv(observations)
        return x

class MaplePolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        prefix_combinator_class: type[nn.Module] = PrefixCombinator,
        prefix_combinator_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        share_prefix_combinator: bool = False,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            False,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )
        self.prefix_combinator_class = prefix_combinator_class
        self.prefix_combinator_kwargs = prefix_combinator_kwargs
        self.share_prefix_combinator = share_prefix_combinator
        self._build_combinator()
        super()._build(lr_schedule)
    
    def _build_combinator(self) -> None:
        if self.share_prefix_combinator:
            self.combinator = self.prefix_combinator_class(**self.prefix_combinator_kwargs)
        else:
            self.policy_combinator = self.prefix_combinator_class(**self.prefix_combinator_kwargs["policy"])
            self.vf_combinator = self.prefix_combinator_class(**self.prefix_combinator_kwargs["vf"])
    
    def forward(self, obs: th.Tensor, prefix: Union[th.Tensor, Tuple[th.Tensor, th.Tensor]], 
                deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        assert self.share_features_extractor or not self.share_prefix_combinator, "logic error, joint combination cannot process two part features"
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor and self.share_prefix_combinator:
            features = self.combinator(features, prefix)
            latent_pi, latent_vf = self.mlp_extractor(features)
        elif self.share_features_extractor and not self.share_prefix_combinator:
            pi_features = self.policy_combinator(features, prefix[0])
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            vf_features = self.vf_combinator(features, prefix[1])
            latent_vf = self.mlp_extractor.forward_critic(vf_features)        
        elif not self.share_features_extractor and not self.share_prefix_combinator:
            pi_features, vf_features = features
            pi_features = self.policy_combinator(pi_features, prefix[0])
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            vf_features = self.vf_combinator(vf_features, prefix[1])
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob
    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, prefix: Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations and prefix.
        """
        # Preprocess the observation
        features = self.extract_features(obs)
        
        # Apply Combinator Logic (Mirrors forward)
        if self.share_features_extractor and self.share_prefix_combinator:
            features = self.combinator(features, prefix)
            latent_pi, latent_vf = self.mlp_extractor(features)
            
        elif self.share_features_extractor and not self.share_prefix_combinator:
            pi_features = self.policy_combinator(features, prefix[0])
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            
            vf_features = self.vf_combinator(features, prefix[1])
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            
        elif not self.share_features_extractor and not self.share_prefix_combinator:
            pi_features, vf_features = features
            
            pi_features = self.policy_combinator(pi_features, prefix[0])
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            
            vf_features = self.vf_combinator(vf_features, prefix[1])
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Standard evaluation
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(
        self, obs: th.Tensor, prefix: Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]
    ) -> Distribution:
        """
        Get the current policy distribution given the observations and prefix.
        """
        features = self.extract_features(obs)

        # Apply Combinator Logic (Actor path only)
        if self.share_features_extractor and self.share_prefix_combinator:
            features = self.combinator(features, prefix)
            latent_pi = self.mlp_extractor.forward_actor(features)
            
        elif self.share_features_extractor and not self.share_prefix_combinator:
            pi_features = self.policy_combinator(features, prefix[0])
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            
        elif not self.share_features_extractor and not self.share_prefix_combinator:
            pi_features, _ = features # Ignore VF features
            pi_features = self.policy_combinator(pi_features, prefix[0])
            latent_pi = self.mlp_extractor.forward_actor(pi_features)

        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(
        self, obs: th.Tensor, prefix: Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations and prefix.
        """
        features = self.extract_features(obs)

        # Apply Combinator Logic (Critic path only)
        if self.share_features_extractor and self.share_prefix_combinator:
            features = self.combinator(features, prefix)
            latent_vf = self.mlp_extractor.forward_critic(features)
            
        elif self.share_features_extractor and not self.share_prefix_combinator:
            vf_features = self.vf_combinator(features, prefix[1])
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            
        elif not self.share_features_extractor and not self.share_prefix_combinator:
            _, vf_features = features # Ignore PI features
            vf_features = self.vf_combinator(vf_features, prefix[1])
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        return self.value_net(latent_vf)

    def _predict(
        self, 
        observation: th.Tensor, 
        prefix: Union[th.Tensor, Tuple[th.Tensor, th.Tensor]], 
        deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        """
        return self.get_distribution(observation, prefix).get_actions(deterministic=deterministic)
