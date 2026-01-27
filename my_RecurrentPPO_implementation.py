from sb3_contrib.ppo_recurrent import RecurrentPPO
from copy import deepcopy
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy

SelfCustomRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="CustomRecurrentPPO")

class CustomRecurrentPPO(RecurrentPPO):
    """
    Modifying the core of the RecurrentPPO such that my custom policies can be used to collect
    rollouts.
    Problems that this class solves:
    - LSTM hidden states have structure (h, c) that is  forced into the RecurrentPPO class.
    - Recive information in collect_rollouts 
    """
    def __int__(self:SelfCustomRecurrentPPO, **kwargs):
        super(CustomRecurrentPPO, self).__init__(**kwargs)
    def _steup_model(self:SelfCustomRecurrentPPO) -> None:
        super(CustomRecurrentPPO, self)._setup_model()
    
    def collect_rollouts(self:SelfCustomRecurrentPPO, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        print("Using CustomRecurrentPPO collect_rollouts")
        out = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        print("Done with CustomRecurrentPPO collect_rollouts")
        return out
    
    def train(self:SelfCustomRecurrentPPO) -> None:
        super().train()
    
    def learn(self:SelfCustomRecurrentPPO, 
              total_timesteps, callback = None, 
              log_interval = 1, 
              tb_log_name = "CustomRecurrentPPO",
              reset_num_timesteps = True, 
              progress_bar = False) -> SelfCustomRecurrentPPO:
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
    
    def _excluded_save_params(self:SelfCustomRecurrentPPO) -> List[str]:
        return super()._excluded_save_params()


if __name__ == "__main__":
    from convlstm import ConvLSTMPolicy
    from convatt import ConvAttPolicy
    from sokoban_wrapper import SokobanCompactWrapper
    from gymnasium.envs.registration import register
    import gymnasium as gym

    this_config = {
        "config": {
            "conv_configs": [
                {
                    "in_channels": 4,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                },
                {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                }
            ],
            "hidden_channels": 64,
            "kernel_size": 3,
            "stack_size": 1,
            "pool_output_size": [2, 2]
        }
    }

    try:
        gym.make('Sokoban-v0')
    except:
        # Register the environment
        register(
            id='Sokoban-v0',
            entry_point='gym_sokoban.envs:SokobanEnv',
            kwargs={
                'dim_room': (7, 7),
                'max_steps': 120,
                'num_boxes': 2,
                'render_mode': 'rgb_array'
            }
        )
    env_registered = gym.make('Sokoban-v0', render_mode='rgb_array')
    env_registered = SokobanCompactWrapper(env_registered)

    model = CustomRecurrentPPO(
        ConvLSTMPolicy,
        env_registered,
        verbose=2,
        policy_kwargs=this_config
    )

    while True:
        obs, _ = env_registered.reset()
        done = False
        lstm_states = None
        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            obs, reward, terminated, truncated, info = env_registered.step(int(action))
            done = terminated or truncated
            env_registered.env.env.env.render("human")