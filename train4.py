import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Import your existing environment helpers ---
# Assuming these are in your local directory as per your snippet
from gym_sokoban.envs import SokobanEnv
from sokoban_wrapper import SokobanCompactWrapper, SokobanRetriesWrapper
import os
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

class BaselineCallback(BaseCallback):
    def __init__(self, save_freq : int, save_path : str, verbose : int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.base_save_path = Path(save_path)

        self.run_dir = None
        self.iteration = 0

    def _init_callback(self) -> None:
        # Create the base directory if it doesn't exist
        self.base_save_path.mkdir(parents=True, exist_ok=True)

        # Determine the next Run ID (e.g., models/0, models/1, etc.)
        ids = [int(d.name) for d in self.base_save_path.iterdir() if d.is_dir() and d.name.isdigit()]
        run_id = max(ids) + 1 if ids else 1
        
        # Create the specific directory for this run
        self.run_dir = self.base_save_path / str(run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose > 0:
            print(f"[Baseline Callback] Logging checkpoints to: {self.run_dir}")
        
    def _on_step(self) -> bool:
        return True    
    
    def _on_rollout_end(self) -> None:
        """
        Triggered before updating the policy, after a rollout is collected.
        """
        self.iteration += 1

        if self.iteration % self.save_freq == 0:
            self.save_checkpoint()

    def save_checkpoint(self):
        """
        Saves the Standard PPO model components in a structure identical 
        to the MAPLE Baseline for easy comparison.
        """
        # Create a sub-directory for this specific iteration
        ckpt_dir = self.run_dir / f"iter_{self.iteration}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose > 0:
            print(f"[Baseline Callback] Saving checkpoint to {ckpt_dir}...")

        # --- 1. Save Policy (ActorCritic) ---
        policy_dir = ckpt_dir / "policy"
        policy_dir.mkdir(exist_ok=True)
        
        # Save the neural network weights (Backbone + MLP Heads)
        th.save(self.model.policy.state_dict(), policy_dir / "policy_state_dict.pt")
        
        # Save the optimizer state (Adam)
        # In SB3 PPO, the optimizer is attached to the policy object
        if hasattr(self.model.policy, "optimizer") and self.model.policy.optimizer is not None:
            th.save(self.model.policy.optimizer.state_dict(), policy_dir / "policy_optimizer_state_dict.pt")

        # --- 2. Save Rollout Buffer ---
        # In PPO, the buffer is full at _on_rollout_end (just before update).
        # Saving this allows you to inspect exactly what the agent saw during this iteration.
        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer is not None:
            buffer_dir = ckpt_dir / "rollout_buffer"
            buffer_dir.mkdir(exist_ok=True)
            # Uses pickle internally to save the buffer object
            th.save(self.model.rollout_buffer, buffer_dir / "rollout_buffer_obj.pt")

        # --- 3. Save Metadata ---
        meta = {
            "iteration": self.iteration, 
            "timesteps": self.num_timesteps,
            "learning_rate": self.model.learning_rate
        }
        th.save(meta, ckpt_dir / "meta.pt")


# ==============================================================================
# 1. The Baseline Comparable CNN
# ==============================================================================
class BaselineComparableCNN(BaseFeaturesExtractor):
    """
    A Feature Extractor that replicates the MAPLE visual pipeline:
    1. ConvFeatureExtractor (Backbone)
    2. PrefixCombinator-like layers (Head) -> But without prefix injection
    3. Adaptive Pooling
    """
    def __init__(self, observation_space: spaces.Box, config: dict, combinator_kwargs: dict):
        
        # Calculate final output dim based on combinator pooling settings
        pool_shape = combinator_kwargs["out_shape"] # e.g. (4,4)
        out_channels = combinator_kwargs["out_channels"] # e.g. 64
        features_dim = out_channels * pool_shape[0] * pool_shape[1]
        
        super().__init__(observation_space, features_dim=features_dim)

        # --- PART A: The Backbone (Identical to ConvFeatureExtractor) ---
        self.conv_configs = config.get("conv_configs")
        in_channels = observation_space.shape[0]
        conv_layers = []
        
        for conv_conf in self.conv_configs:
            c_out = conv_conf['out_channels']
            k = conv_conf.get('kernel_size', 3)
            s = conv_conf.get('stride', 1)
            p = conv_conf.get('padding', 1)
            
            conv_layers.append(nn.Conv2d(in_channels, c_out, k, s, p))
            conv_layers.append(nn.ReLU())
            in_channels = c_out
            
        self.backbone = nn.Sequential(*conv_layers)

        # --- PART B: The Head (Identical to PrefixCombinator layers) ---
        # In MAPLE, input to this is (Backbone + Prefix). 
        # Here, input is just Backbone.
        
        c_in = in_channels 
        c_hidden = combinator_kwargs["hidden_channels"]
        c_out_head = combinator_kwargs["out_channels"]
        n_layers = combinator_kwargs["n_layers"]
        
        head_layers = []
        past_out = c_in
        
        for _ in range(n_layers):
            head_layers.append(nn.Conv2d(past_out, c_hidden, 3, 1, 1))
            head_layers.append(nn.ReLU())
            past_out = c_hidden
            
        head_layers.append(nn.Conv2d(past_out, c_out_head, 3, 1, 1))
        head_layers.append(nn.ReLU())
        
        self.head = nn.Sequential(*head_layers)
        
        # --- PART C: Pooling & Flattening ---
        self.pool = nn.AdaptiveAvgPool2d(pool_shape)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 1. Extract Backbone Features
        x = self.backbone(observations)
        
        # 2. Process through deep "Combinator-like" layers (No prefix concat)
        x = self.head(x)
        
        # 3. Pool and Flatten for MLP
        x = self.pool(x)
        x = self.flatten(x)
        return x

# ==============================================================================
# 2. Configuration (Must match your MAPLE config)
# ==============================================================================
HIDDEN_SIZE_CHANNELS = 64
POOL_SHAPE = 4
SEED = 43
NUM_RETRIES = 2

config = {
    # Same visual backbone config as MAPLE
    "features_extractor_kwargs": {
        "config": {
            "conv_configs": [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': HIDDEN_SIZE_CHANNELS, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ]
        },
        # We pass the combinator settings here so the Baseline class can replicate the depth
        "combinator_kwargs": {
            "hidden_channels": HIDDEN_SIZE_CHANNELS,
            "out_channels": HIDDEN_SIZE_CHANNELS,
            "n_layers": 2, 
            "out_shape": (POOL_SHAPE, POOL_SHAPE)
        }
    },
    # Same MLP Head config as MAPLE
    "net_arch": {
        "pi": [512, 256],
        "vf": [512, 256]
    }
}

# ==============================================================================
# 3. Environment Setup
# ==============================================================================
def make_env(dim_room, max_steps, num_boxes, num_retries, seed=42):
    # We use the same wrappers. Even though standard PPO doesn't use the 'retry' 
    # logic actively, keeping the wrapper ensures observation spaces and 
    # physics/resets are identical.
    env = SokobanRetriesWrapper(
        max_retries=num_retries,
        dim_room=dim_room,
        max_steps=max_steps,
        num_boxes=num_boxes,
        render_mode="rgb_array",
    )
    env = SokobanCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

# ==============================================================================
# 4. Main Execution
# ==============================================================================
if __name__ == "__main__":
    
    # Create Env
    train_env = make_env(dim_room=(6,6), max_steps=22, num_boxes=1, 
                         num_retries=NUM_RETRIES, seed=SEED)
    
    # Instantiate Standard PPO with Comparable Architecture
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=2e-4,
        verbose=1,
        tensorboard_log="./tensorboard_baseline/",
        seed=SEED,
        policy_kwargs={
            "features_extractor_class": BaselineComparableCNN,
            "features_extractor_kwargs": config["features_extractor_kwargs"],
            "net_arch": config["net_arch"], # Ensure MLP heads match
            "activation_fn": nn.Tanh,
            "optimizer_class": th.optim.Adam
        }
    )
    checkpoint_callback = BaselineCallback(
        save_freq=10,  # Save every 10 iterations
        save_path="./models_baseline/",
        verbose=1
    )

    model.learn(total_timesteps=120000, callback=checkpoint_callback)
