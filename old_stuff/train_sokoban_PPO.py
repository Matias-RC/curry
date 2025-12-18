"""
Patched and simplified PPO trainer for Sokoban with ResNet-like concat residuals.
Key fixes & changes:
  - GAE bootstrap (last_value) is computed and passed from collection -> update
  - normalize advantages ONLY (do not normalize returns used as critic target)
  - replace BatchNorm2d with GroupNorm (more stable in RL / small batches)
  - hyperparameters defined at top for easy tuning
  - smaller default network sizes for quicker debugging
  - simplified / cleaned code layout

Usage: replace `from sokoban_gym_env import SokobanEnv` with your environment module
and run as a script. See --help for args.
"""

import os
import random
import math
from typing import List, Tuple
from collections import namedtuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ----------------------------
# Hyperparameters (top-level)
# ----------------------------
H = {
    'device': 'cpu',
    'gamma': 0.99,
    'lam': 0.95,
    'clip_epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.1,   # slightly larger to encourage exploration
    'lr': 2.5e-4,
    'epochs': 3,
    'minibatch_size': 64,
    'min_steps_per_update': 256,
    'max_steps_per_update': 512,
    'save_interval_updates': 50,
    'save_dir': 'checkpoints',
    'log_csv_name': 'training_log.csv',
}

# ----------------------------
# Utils / small modules
# ----------------------------
Transition = namedtuple('Transition', ['obs', 'action', 'logp', 'reward', 'done', 'value'])


def _gn(channels: int, groups: int = 8) -> nn.Module:
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    if g < 1:
        g = 1
    return nn.GroupNorm(g, channels)


class ResidualConcatBlock(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size, padding=padding)
        self.gn1 = _gn(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.gn2 = _gn(out_ch)

    def forward(self, x):
        h = F.relu(self.gn1(self.conv1(x)))
        h = F.relu(self.gn2(self.conv2(h)))
        return torch.cat([x, h], dim=1)


class ResNetConcat(nn.Module):
    def __init__(self, initial_in_channels: int, block_configs: List[Tuple[int, int, int]]):
        super().__init__()
        modules = []
        cur = initial_in_channels
        for (b_in, b_hidden, b_out) in block_configs:
            # b_in is advisory; use cur
            modules.append(ResidualConcatBlock(cur, b_hidden, b_out))
            cur = cur + b_out
        self.blocks = nn.ModuleList(modules)
        self.out_channels = cur

    def forward(self, x):
        h = x
        for b in self.blocks:
            h = b(h)
        return h


class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int, block_configs: List[Tuple[int, int, int]], final_conv_out=128):
        super().__init__()
        self.backbone = ResNetConcat(input_channels, block_configs)
        self.final_conv = nn.Conv2d(self.backbone.out_channels, final_conv_out, kernel_size=1)
        self.final_gn = _gn(final_conv_out)
        self.final_out = final_conv_out

    def forward(self, x):
        h = self.backbone(x)
        h = F.relu(self.final_gn(self.final_conv(h)))
        return h.mean(dim=[2, 3])


class ActorCritic(nn.Module):
    def __init__(self, input_channels: int, block_configs: List[Tuple[int, int, int]], num_actions: int,
                 final_conv_out=128, actor_hidden=256, critic_hidden=256):
        super().__init__()
        self.features = ConvFeatureExtractor(input_channels, block_configs, final_conv_out=final_conv_out)
        feat_dim = self.features.final_out
        self.actor = nn.Sequential(nn.Linear(feat_dim, actor_hidden), nn.ReLU(), nn.Linear(actor_hidden, num_actions))
        self.critic = nn.Sequential(nn.Linear(feat_dim, critic_hidden), nn.ReLU(), nn.Linear(critic_hidden, 1))

    def forward(self, x):
        # x: (B,C,H,W)
        feat = self.features(x)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value


# ----------------------------
# GAE
# ----------------------------

def compute_gae(transitions: List[Transition], last_value: float, gamma: float, lam: float):
    rewards = [t.reward for t in transitions]
    values = [t.value for t in transitions]
    dones = [t.done for t in transitions]

    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        mask = 0.0 if dones[step] else 1.0
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages[step] = gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ----------------------------
# Simple random level generator
# ----------------------------

def generate_simple_random_easy(n: int, m: int, k: int, seed: int = None) -> List[str]:
    if seed is not None:
        random.seed(seed)
    H = n
    W = m
    grid = [['.' for _ in range(W)] for _ in range(H)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    free_cells = H * W
    if k < 1 or k > free_cells - 1:
        raise ValueError("k out of range for given n,m")

    placed = 0
    attempts_limit = max(200, k * 200)
    attempts = 0
    while placed < k and attempts < attempts_limit:
        attempts += 1
        gy, gx = random.randint(0, H - 1), random.randint(0, W - 1)
        if grid[gy][gx] != '.':
            continue
        working_dirs = []
        for dy, dx in dirs:
            ny, nx = gy + dy, gx + dx
            fy, fx = gy + 2 * dy, gx + 2 * dx
            if 0 <= ny < H and 0 <= nx < W and 0 <= fy < H and 0 <= fx < W:
                if grid[ny][nx] == '.' and grid[fy][fx] == '.':
                    working_dirs.append((dy, dx))
        if not working_dirs:
            continue
        grid[gy][gx] = 'G'
        dy, dx = random.choice(working_dirs)
        by, bx = gy + dy, gx + dx
        grid[by][bx] = '$'
        placed += 1
    if placed < k:
        raise RuntimeError(f"Could not place all {k} boxes/goals after {attempts} attempts")
    candidates = [(y, x) for y in range(H) for x in range(W) if grid[y][x] in ('.', 'G')]
    if not candidates:
        raise RuntimeError("No available cell to place player")
    py, px = random.choice(candidates)
    grid[py][px] = '@' if grid[py][px] == '.' else '+'
    return [''.join(row) for row in grid]


# ----------------------------
# PPO Trainer
# ----------------------------

class PPOTrainer:
    def __init__(self, env_factory, policy: ActorCritic, device='cpu', hparams=None):
        self.env_factory = env_factory
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy = policy.to(self.device)
        self.h = hparams or H
        self.gamma = self.h['gamma']
        self.lam = self.h['lam']
        self.clip_epsilon = self.h['clip_epsilon']
        self.value_coef = self.h['value_coef']
        self.entropy_coef = self.h['entropy_coef']
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.h['lr'])
        self.epochs = self.h['epochs']
        self.minibatch_size = self.h['minibatch_size']
        self.min_steps_per_update = self.h['min_steps_per_update']
        self.max_steps_per_update = self.h['max_steps_per_update']
        self.save_interval_updates = self.h['save_interval_updates']
        self.save_dir = self.h['save_dir']
        self.log_csv_name = self.h['log_csv_name']
        os.makedirs(self.save_dir, exist_ok=True)
        self.update_count = 0

        self.csv_path = os.path.join(self.save_dir, self.log_csv_name)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write('update,num_transitions,avg_episode_reward,avg_step_reward,policy_loss,value_loss,entropy,total_loss\n')

    def _obs_to_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0
        else:
            x = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return x.unsqueeze(0).to(self.device)

    def collect_trajectories(self):
        steps_needed = random.randint(self.min_steps_per_update, self.max_steps_per_update)
        transitions = []
        env = self.env_factory()
        obs, _ = env.reset()
        cur_episode_reward = 0.0
        episode_rewards = []
        safety = steps_needed * 5
        steps = 0

        while len(transitions) < steps_needed and steps < safety:
            steps += 1
            obs_t = self._obs_to_tensor(obs)
            with torch.no_grad():
                logits, value = self.policy(obs_t)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor(action).to(self.device)).item())
                value = float(value.item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            done_flag = bool(terminated or truncated)

            transitions.append(Transition(obs=obs, action=action, logp=logp, reward=float(reward), done=done_flag, value=value))
            cur_episode_reward += float(reward)

            if done_flag:
                episode_rewards.append(cur_episode_reward)
                cur_episode_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

            # if near end compute last_value when appropriate
            if len(transitions) >= steps_needed:
                if not done_flag:
                    # bootstrap value from current observation (obs is the post-step obs)
                    with torch.no_grad():
                        obs_t = self._obs_to_tensor(obs)
                        _, last_value_tensor = self.policy(obs_t)
                        last_value = float(last_value_tensor.item())
                else:
                    last_value = 0.0

        if cur_episode_reward != 0.0:
            episode_rewards.append(cur_episode_reward)

        rewards = [t.reward for t in transitions]
        avg_step_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
        avg_episode_reward = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else 0.0

        # action histogram for debug
        action_counts = {}
        for t in transitions:
            action_counts[t.action] = action_counts.get(t.action, 0) + 1

        stats = {
            'steps_collected': len(transitions),
            'avg_step_reward': avg_step_reward,
            'avg_episode_reward': avg_episode_reward,
            'num_episodes': len(episode_rewards),
            'last_value': last_value,
            'action_counts': action_counts,
        }
        return transitions, stats

    def update(self, transitions: List[Transition], collect_stats: dict = None):
        obs_list = [t.obs for t in transitions]
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        old_logps = np.array([t.logp for t in transitions], dtype=np.float32)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.bool_)
        values = np.array([t.value for t in transitions], dtype=np.float32)

        last_value = collect_stats.get('last_value', 0.0) if collect_stats else 0.0
        advantages, returns = compute_gae(transitions, last_value, self.gamma, self.lam)

        # normalize advantages only
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_t = torch.stack([self._obs_to_tensor(o).squeeze(0) for o in obs_list]).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        old_logps_t = torch.from_numpy(old_logps).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        returns_t = torch.from_numpy(returns).to(self.device)  # raw returns as critic targets

        N = len(transitions)
        accum_policy_loss = accum_value_loss = accum_entropy = accum_total_loss = 0.0
        accum_steps = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(N)
            mb_size = min(self.minibatch_size, N)
            for start in range(0, N, mb_size):
                end = start + mb_size
                mb_idx = indices[start:end]
                if len(mb_idx) == 0:
                    continue
                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logps = old_logps_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                logits, values_pred = self.policy(mb_obs)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                mb_new_logps = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(mb_new_logps - mb_old_logps)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_pred, mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                accum_policy_loss += float(policy_loss.detach().cpu().item())
                accum_value_loss += float(value_loss.detach().cpu().item())
                accum_entropy += float(entropy.detach().cpu().item())
                accum_total_loss += float(loss.detach().cpu().item())
                accum_steps += 1

        if accum_steps > 0:
            mean_policy_loss = accum_policy_loss / accum_steps
            mean_value_loss = accum_value_loss / accum_steps
            mean_entropy = accum_entropy / accum_steps
            mean_total_loss = accum_total_loss / accum_steps
        else:
            mean_policy_loss = mean_value_loss = mean_entropy = mean_total_loss = 0.0

        self.update_count += 1
        if self.update_count % self.save_interval_updates == 0:
            fname = os.path.join(self.save_dir, f"ppo_resnet_concat_update_{self.update_count}.pt")
            torch.save({'update': self.update_count, 'model_state_dict': self.policy.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, fname)
            print(f"[INFO] Saved checkpoint at update {self.update_count} -> {fname}")

        num_transitions = N
        avg_step_reward = collect_stats.get('avg_step_reward') if collect_stats else float(rewards.mean()) if len(rewards) > 0 else 0.0
        avg_episode_reward = collect_stats.get('avg_episode_reward') if collect_stats else 0.0

        print(f"[UPDATE {self.update_count}] transitions={num_transitions} episodes={collect_stats.get('num_episodes', 0) if collect_stats else 'N/A'} "
              f"avg_episode_reward={avg_episode_reward:.4f} avg_step_reward={avg_step_reward:.4f} "
              f"policy_loss={mean_policy_loss:.6f} value_loss={mean_value_loss:.6f} entropy={mean_entropy:.6f} total_loss={mean_total_loss:.6f}")

        try:
            with open(self.csv_path, 'a') as f:
                f.write(f"{self.update_count},{num_transitions},{avg_episode_reward:.6f},{avg_step_reward:.6f},{mean_policy_loss:.8f},{mean_value_loss:.8f},{mean_entropy:.8f},{mean_total_loss:.8f}\n")
        except Exception as e:
            print("[WARN] Failed to write training log CSV:", e)

        return {'policy_loss': mean_policy_loss, 'value_loss': mean_value_loss, 'entropy': mean_entropy, 'total_loss': mean_total_loss, 'num_transitions': num_transitions, 'avg_step_reward': avg_step_reward, 'avg_episode_reward': avg_episode_reward}

    def train(self, total_updates: int = 1000):
        for upd in range(total_updates):
            transitions, stats = self.collect_trajectories()
            print(f"[INFO] Update {upd+1}/{total_updates}: collected {len(transitions)} transitions (episodes={stats.get('num_episodes',0)}) action_counts={stats.get('action_counts')}")
            _ = self.update(transitions, collect_stats=stats)


# ----------------------------
# Main / Example
# ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=H['device'])
    parser.add_argument('--total-updates', type=int, default=1000)
    parser.add_argument('--save-dir', default=H['save_dir'])
    parser.add_argument('--save-every', type=int, default=H['save_interval_updates'])
    parser.add_argument('--min-steps-per-update', type=int, default=H['min_steps_per_update'])
    parser.add_argument('--max-steps-per-update', type=int, default=H['max_steps_per_update'])
    parser.add_argument('--minibatch-size', type=int, default=H['minibatch_size'])
    parser.add_argument('--env-grid', type=int, default=8)
    parser.add_argument('--num-boxes', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # default blocks: keep them small for quicker runs
    block_configs = [
        (3, 32, 16),  # advisory 'in' values won't be strictly used
        (None, 64, 32),
    ]

    # env factories - replace import to match your env implementation
    def env_factory_for_action():
        level = generate_simple_random_easy(args.env_grid, args.env_grid, args.num_boxes)
        from sokoban_gym_env import SokobanEnv
        return SokobanEnv(grid_size=args.env_grid, render_mode="rgb_array", init_type="str", level_map=level, max_steps=300)

    tmp_env = env_factory_for_action()
    tmp_obs, _ = tmp_env.reset()
    C = tmp_obs.shape[2]
    num_actions = tmp_env.action_space.n
    print(f"Detected obs channels={C}, num_actions={num_actions}")
    del tmp_env

    policy_net = ActorCritic(input_channels=C, block_configs=block_configs, num_actions=num_actions, final_conv_out=128, actor_hidden=128, critic_hidden=128)
    some_level = generate_simple_random_easy(args.env_grid, args.env_grid, args.num_boxes)
    def env_factory():
        level = generate_simple_random_easy(args.env_grid, args.env_grid, args.num_boxes)
        from sokoban_gym_env import SokobanEnv
        return SokobanEnv(grid_size=args.env_grid, render_mode="rgb_array", init_type="str", level_map=some_level, max_steps=200)

    trainer = PPOTrainer(env_factory=env_factory, policy=policy_net, device=device, hparams={**H, 'device': device, 'save_dir': args.save_dir, 'save_interval_updates': args.save_every, 'min_steps_per_update': args.min_steps_per_update, 'max_steps_per_update': args.max_steps_per_update, 'minibatch_size': args.minibatch_size})

    # smoke test
    try:
        level = generate_simple_random_easy(args.env_grid, args.env_grid, args.num_boxes)
        from sokoban_gym_env import SokobanEnv
        env = SokobanEnv(grid_size=args.env_grid, render_mode="rgb_array", init_type="str", level_map=level, max_steps=200)
        obs, _ = env.reset()
        print("Initial obs shape (smoke test):", obs.shape)
        for i in range(6):
            a = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(a)
            img = env.render()
            Image.fromarray(img).save(os.path.join(os.path.abspath('.'), f"sokoban_step_{i}.png"))
            if terminated or truncated:
                break
        print("Smoke test done.")
    except Exception as e:
        print("Smoke test skipped: could not import or run SokobanEnv. Error:", e)

    trainer.train(total_updates=args.total_updates)
