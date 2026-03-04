import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor

from src.envs.envs import SokobanCompactWrapper, SokobanRetriesWrapper

# ==============================================================================
# 1. Hyperparameters & Config
# ==============================================================================

SEED = 67
DIM_ROOM = (7, 7)
MAX_STEPS = 60
NUM_BOXES = 1

# Beam Search / Retry Params
NUM_BEAM_SEARCH_STEPS = 30
BEAM_SIZE = 5
NUM_RETRIES = NUM_BEAM_SEARCH_STEPS * BEAM_SIZE

# Optimization Params
INNER_EPOCHS = 5
BATCH_SIZE = 20
ADAPTER_LR = 0.001
MAX_GRAD_NORM = 0.5 

# Architecture Params
HIDDEN_SIZE_CHANNELS = 64
POOL_SHAPE = 4
FEATURES_DIM = HIDDEN_SIZE_CHANNELS * POOL_SHAPE * POOL_SHAPE  # 1024

RUN_ID = 2
ITER = 180
BASE_DIR = Path("./models_baseline") / str(RUN_ID) / f"iter_{ITER}"

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

# ==============================================================================
# 2. Baseline Architecture
# ==============================================================================

class BaselineComparableCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, config: dict, combinator_kwargs: dict):
        pool_shape = combinator_kwargs["out_shape"] 
        out_channels = combinator_kwargs["out_channels"] 
        features_dim = out_channels * pool_shape[0] * pool_shape[1]
        
        super().__init__(observation_space, features_dim=features_dim)

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
        self.pool = nn.AdaptiveAvgPool2d(pool_shape)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.backbone(observations)
        x = self.head(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

# ==============================================================================
# 3. Runtime Adapter & Buffer
# ==============================================================================

class RuntimeAdapter(nn.Module):
    def __init__(self, n_envs, feature_dim, device="cpu", lr=1e-3):
        super().__init__()
        self.n_envs = n_envs
        self.feature_dim = feature_dim
        self.device = device
        self.params = nn.ParameterList()
        
        for _ in range(n_envs):
            w_pi = nn.Parameter(th.eye(feature_dim, device=device))
            b_pi = nn.Parameter(th.zeros(feature_dim, device=device))
            w_vf = nn.Parameter(th.eye(feature_dim, device=device))
            b_vf = nn.Parameter(th.zeros(feature_dim, device=device))
            self.params.append(nn.ParameterList([w_pi, b_pi, w_vf, b_vf]))

        self.optimizers = [th.optim.Adam(self.params[i], lr=lr) for i in range(n_envs)]

    def forward_single(self, features, env_idx):
        w_pi, b_pi, w_vf, b_vf = self.params[env_idx]
        feat_pi = features @ w_pi + b_pi
        feat_vf = features @ w_vf + b_vf
        return feat_pi, feat_vf

    def optimize_step(self, idx: int, loss: th.Tensor):
        opt = self.optimizers[idx]
        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.params[idx], MAX_GRAD_NORM)
        opt.step()
        
    def reset_env(self, idx):
        with th.no_grad():
            self.params[idx][0].copy_(th.eye(self.feature_dim, device=self.device))
            self.params[idx][1].zero_()
            self.params[idx][2].copy_(th.eye(self.feature_dim, device=self.device))
            self.params[idx][3].zero_()

class DynamicBufferForBaseline:
    def __init__(self, n_envs, device:str): 
        self.n_envs = n_envs
        self.device = device
        self.history = [[[],] for _ in range(n_envs)]

    def append_step(self, env_idx, obs, actions, rewards, episode_starts, values, log_probs):
        step_data = {
            'obs': obs, 'actions': actions, 'rewards': rewards,
            'episode_starts': episode_starts, 'values': values, 
            'log_probs': log_probs
        }
        self.history[env_idx][-1].append(step_data)

    def start_new_retry(self, env_idx):
        self.history[env_idx].append([])

    def get_last_attempt(self, env_idx):
        return self._format_traj(self.history[env_idx][-1])
    
    def get_all_attempts(self, env_idx):
         hist = []
         for attempt in self.history[env_idx]:
              hist += attempt
         return self._format_traj(hist)

    def reset_env(self, env_idx):
        self.history[env_idx] = [[],]

    def _format_traj(self, list_of_dicts):
        if not list_of_dicts: return {}
        ret = {}
        for k in list_of_dicts[0].keys():
            temp = [step[k] for step in list_of_dicts]
            if isinstance(temp[0], th.Tensor):
                ret[k] = th.stack(temp)
            else:
                ret[k] = np.array(temp)
        return ret

# ==============================================================================
# 4. Helpers
# ==============================================================================

def compute_returns_and_advantages(rewards, values, episode_starts, gamma=0.99, gae_lambda=0.95):
    returns = th.zeros_like(rewards)
    advantages = th.zeros_like(rewards)
    td0_targets = th.zeros_like(rewards) 
    last_gae_lam = 0
    next_value = 0 
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 0.0 
        else:
            next_non_terminal = 1.0 - episode_starts[t + 1]
        td0_targets[t] = rewards[t] + gamma * next_value * next_non_terminal
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
        returns[t] = advantages[t] + values[t]
        next_value = values[t]
    return returns, advantages, td0_targets

def evaluate_adapted_actions(model, adapter, obs, actions, env_idx):
    with th.no_grad():
        features = model.policy.features_extractor(obs)
    feat_pi, feat_vf = adapter.forward_single(features, env_idx)
    latent_pi = model.policy.mlp_extractor.policy_net(feat_pi)
    latent_vf = model.policy.mlp_extractor.value_net(feat_vf)
    distribution = model.policy._get_action_dist_from_latent(latent_pi)
    log_prob = distribution.log_prob(actions)
    values = model.policy.value_net(latent_vf)
    return values, log_prob

def update_adapter(dynamic_buffer, idx, adapter, model):
    traj = dynamic_buffer.get_all_attempts(idx)
    if not traj: return
    def to_tensor(x):
        if isinstance(x, np.ndarray): x = th.from_numpy(x)
        return x.to(device=DEVICE, dtype=th.float32)

    obs = to_tensor(traj['obs'])
    actions = to_tensor(traj['actions'])
    rewards = to_tensor(traj['rewards'])
    old_values = to_tensor(traj['values'])
    episode_starts = to_tensor(traj['episode_starts'])

    with th.no_grad():
        _, advantages, td0_targets = compute_returns_and_advantages(rewards, old_values, episode_starts)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = TensorDataset(obs, actions, advantages, td0_targets)
    loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(obs)), shuffle=True)
    model.policy.eval() 
    
    for _ in range(INNER_EPOCHS):
        for b_obs, b_act, b_adv, b_td0 in loader:
            values, log_probs = evaluate_adapted_actions(model, adapter, b_obs, b_act, idx)
            values = values.flatten()
            v_loss = 0.5 * F.mse_loss(values, b_td0)
            p_loss = -(log_probs * b_adv).mean()
            total_loss = p_loss + v_loss
            adapter.optimize_step(idx, total_loss)
    
    # We do NOT reset buffer here; we do it in the main loop to keep flow clear

# ==============================================================================
# 5. Main Execution
# ==============================================================================

def make_env(seed):
    env = SokobanRetriesWrapper(
        max_retries=NUM_RETRIES,
        dim_room=DIM_ROOM,
        max_steps=MAX_STEPS,
        num_boxes=NUM_BOXES,
        render_mode="rgb_array",
    )
    env = SokobanCompactWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

def load_baseline_model(env):
    config = {
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
            "combinator_kwargs": {
                "hidden_channels": HIDDEN_SIZE_CHANNELS,
                "out_channels": HIDDEN_SIZE_CHANNELS,
                "n_layers": 2, 
                "out_shape": (POOL_SHAPE, POOL_SHAPE)
            }
        },
        "net_arch": {"pi": [512, 256], "vf": [512, 256]}
    }
    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs={
            "features_extractor_class": BaselineComparableCNN,
            "features_extractor_kwargs": config["features_extractor_kwargs"],
            "net_arch": config["net_arch"],
            "activation_fn": nn.Tanh,
        },
        device=DEVICE
    )
    print(f"[Baseline] Loading checkpoint from {BASE_DIR}")
    policy_path = BASE_DIR / "policy" / "policy_state_dict.pt"
    if policy_path.exists():
        model.policy.load_state_dict(th.load(policy_path, map_location=DEVICE))
        print(" -> Policy loaded.")
    model.policy.eval()
    return model

def main():
    env = make_env(SEED)
    model = load_baseline_model(env)
    adapter = RuntimeAdapter(n_envs=1, feature_dim=FEATURES_DIM, device=DEVICE, lr=ADAPTER_LR)
    buffer = DynamicBufferForBaseline(n_envs=1, device=DEVICE)
    
    beam_stats = {"sum_rewards": 0.0, "count": 0}
    obs, info = env.reset()
    terminated, truncated = False, False
    
    print("[Baseline] Starting No-Viz Adaptation Loop...")
    
    total_episodes = 0
    max_episodes_to_run = 5  # Stop after X full problems (levels)

    while total_episodes < max_episodes_to_run:
        # --- A. Predict ---
        obs_tensor = obs_as_tensor(obs, DEVICE).unsqueeze(0)
        with th.no_grad():
            features = model.policy.features_extractor(obs_tensor)
            f_pi, f_vf = adapter.forward_single(features, env_idx=0)
            latent_pi = model.policy.mlp_extractor.policy_net(f_pi)
            dist = model.policy._get_action_dist_from_latent(latent_pi)
            action = dist.mode()
            latent_vf = model.policy.mlp_extractor.value_net(f_vf)
            value = model.policy.value_net(latent_vf)
            log_prob = dist.log_prob(action)

        action_int = action.item()
        
        # --- B. Step ---
        new_obs, reward, terminated, truncated, new_info = env.step(action_int)
        
        # --- C. Store ---
        buffer.append_step(0, obs, action, reward, terminated, value, log_prob)
        
        # --- D. Check Done ---
        if terminated or truncated:
            retry_count = new_info.get("retry_count", 0)
            retries_left = new_info.get("retries_left", 0)
            
            # Stats
            last_attempt = buffer.get_last_attempt(0)
            if "rewards" in last_attempt:
                ep_rew = np.sum(last_attempt["rewards"])
                beam_stats["sum_rewards"] += ep_rew
                beam_stats["count"] += 1
            
            # Logic: Optimization
            if retry_count > 0 and retry_count % BEAM_SIZE == 0:
                avg_r = beam_stats["sum_rewards"] / max(1, beam_stats["count"])
                print(f"[Env 0] Beam Block Finished (Retry {retry_count}). Avg Reward: {avg_r:.2f} | Optimizing...")
                
                update_adapter(buffer, 0, adapter, model)
                
                beam_stats = {"sum_rewards": 0.0, "count": 0}
                buffer.reset_env(0) # Clear past buffer after optimization
            else:
                buffer.start_new_retry(0) # Prepare list for next attempt

            # Logic: Reset Env
            obs, info = env.reset()

            # Logic: New Level / End of Retries
            if retries_left == 0:
                 print(f"[Env 0] Level Finished. Resetting Adapter for next level.")
                 adapter.reset_env(0)
                 buffer.reset_env(0)
                 beam_stats = {"sum_rewards": 0.0, "count": 0}
                 total_episodes += 1
                 print(f"--- Episode {total_episodes} / {max_episodes_to_run} complete ---\n")
        else:
            obs = new_obs
            info = new_info

if __name__ == "__main__":
    main()