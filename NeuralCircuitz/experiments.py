#!/usr/bin/env python3
"""
Recurrent PPO experiment (copy-paste ready)

- Environment: value grid with observation:
    [ sigmoid(value_at_pos), available_moves (5 bits), rel_x, rel_y ]  (length 8)
- Actions: 0=up,1=left,2=right,3=down,4=stay,5=end
- Episode ends when agent selects 'end' OR max_steps reached
- Reward at termination: softmax over all tile values with temperature T (lower T -> sharper)
- Temperature decays as training progresses (reward becomes sharper)
- Agent: embedding -> 4-layer LSTM -> actor/critic
- PPO: on-policy, episode-level recurrent training
- Sweep hidden sizes (H) in increments of 16 until mean eval performance across seeds >= MIN_SUCCESS
- CSV logging (progress.csv): one row per evaluation (update) with:
    update, seed, board_size, hidden_size, avg_reward_this_seed, success_frac_this_seed,
    median_reward_across_seeds_for_this_H_update, median_delta_from_previous_eval, temperature
- Median across seeds is computed from evaluations that have occurred for the same H and update index.
"""

import math
import random
import time
import copy
import csv
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------
# Experiment hyperparameters
# ---------------------------
SEED_LIST = [100, 200, 300]        # seeds to run per H (increase to 5 for final)
START_H = 16
MAX_H = 256
H_STEP = 16
LSTM_LAYERS = 4
BOARD_TRAIN = 10
BOARD_TESTS = [15, 20]
MIN_SUCCESS = 0.90                 # target mean avg_reward across seeds
EPISODES_PER_UPDATE = 12
MAX_UPDATES = 400
GAMMA = 0.995
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 6
MINI_BATCH_EPISODES = 4
LR = 3e-4
ENT_COEF = 1e-3
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
EVAL_EPS = 200
EVAL_EVERY_UPDATES = 10           # evaluate every this many updates
MAX_STEPS_PER_EPISODE_FACTOR = 4

# Temperature schedule (decays multiplicatively per update)
INIT_TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.05
TEMP_DECAY_PER_UPDATE = 0.995     # multiply temperature by this each update

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

# CSV file for progress logging
PROGRESS_CSV = "progress.csv"

# ---------------------------
# Utilities
# ---------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# ---------------------------
# Environment
# ---------------------------
class ValueGridEnv:
    """
    Grid environment with:
    - observation: [sigmoid(value_at_pos), moves_mask(5), rel_x, rel_y]  -> length 8
    - actions: 0=up,1=left,2=right,3=down,4=stay,5=end
    - 'end' triggers termination and final reward computation
    - reward at termination: softmax over tile values with temperature
    """
    def __init__(self, size:int, value_min=0, value_max=20, start_random=True, rng=None, temperature=1.0):
        self.size = size
        self.value_min = value_min
        self.value_max = value_max
        self.start_random = start_random
        self.temperature = float(temperature)
        self.rng = rng if rng is not None else random.Random()
        self.reset()

    def seed(self, s:int):
        self.rng = random.Random(s)

    def make_board(self):
        return [[self.rng.randint(self.value_min, self.value_max) for _ in range(self.size)] for _ in range(self.size)]

    def _available_moves_mask(self, x:int, y:int) -> List[int]:
        return [
            1 if y > 0 else 0,                    # up
            1 if x > 0 else 0,                    # left
            1 if x < self.size - 1 else 0,        # right
            1 if y < self.size - 1 else 0,        # down
            1                                     # stay always available
        ]

    def _make_obs(self) -> np.ndarray:
        x, y = self.pos
        v = self.board[y][x]
        sig = sigmoid(v)
        moves = self._available_moves_mask(x, y)
        rel_x = x / float(max(1, self.size - 1))
        rel_y = y / float(max(1, self.size - 1))
        obs = np.array([sig] + moves + [rel_x, rel_y], dtype=np.float32)
        return obs

    def reset(self) -> np.ndarray:
        self.board = self.make_board()
        flat = [v for row in self.board for v in row]
        self.global_max = max(flat)
        self.max_positions = [(x,y) for y in range(self.size) for x in range(self.size) if self.board[y][x] == self.global_max]
        if self.start_random:
            self.pos = (self.rng.randrange(self.size), self.rng.randrange(self.size))
        else:
            self.pos = (0, 0)
        self.steps = 0
        self.max_steps = max(1, self.size * self.size * MAX_STEPS_PER_EPISODE_FACTOR)
        return self._make_obs()

    def step(self, action:int) -> Tuple[np.ndarray, float, bool, dict]:
        x, y = self.pos
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and x > 0:
            x -= 1
        elif action == 2 and x < self.size - 1:
            x += 1
        elif action == 3 and y < self.size - 1:
            y += 1
        elif action == 4:
            pass
        elif action == 5:
            # end action: compute final reward and terminate
            self.pos = (x, y)
            self.steps += 1
            done = True
            reward = self._compute_end_reward(self.pos)
            obs = self._make_obs()
            info = {'ended_on_max': self.board[y][x] == self.global_max}
            return obs, reward, done, info
        else:
            # invalid -> stay
            pass

        self.pos = (x, y)
        self.steps += 1
        done = False
        reward = 0.0
        info = {'ended_on_max': False}
        if self.steps >= self.max_steps:
            done = True
            reward = self._compute_end_reward(self.pos)
            info = {'ended_on_max': self.board[self.pos[1]][self.pos[0]] == self.global_max}
        obs = self._make_obs()
        return obs, reward, done, info

    def _compute_end_reward(self, pos: Tuple[int,int]) -> float:
        # Compute softmax over all tile values with temperature
        flat_vals = np.array([v for row in self.board for v in row], dtype=np.float64)
        scaled = flat_vals / (self.temperature + 1e-12)
        scaled = scaled - np.max(scaled)  # stability
        exps = np.exp(scaled)
        probs = exps / (np.sum(exps) + 1e-12)
        x, y = pos
        idx = y * self.size + x
        return float(probs[idx])

# ---------------------------
# Recurrent policy (embedding -> LSTM -> actor/critic)
# ---------------------------
class RecurrentPolicy(nn.Module):
    def __init__(self, hidden_size_H:int, embedding_dim:int = 32, n_layers:int = 4, n_actions:int = 6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size_H = hidden_size_H
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.embed = nn.Linear(8, embedding_dim)  # input dim 8
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size_H,
                            num_layers=n_layers, batch_first=True)
        self.actor = nn.Linear(hidden_size_H, n_actions)
        self.critic = nn.Linear(hidden_size_H, 1)

    def forward(self, obs_seq: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None):
        # obs_seq: (B, T, 8)
        B, T, D = obs_seq.shape
        x = F.relu(self.embed(obs_seq.view(B*T, D))).view(B, T, -1)
        out, (h_n, c_n) = self.lstm(x, hx)
        logits = self.actor(out)        # (B, T, A)
        values = self.critic(out).squeeze(-1)  # (B, T)
        return logits, values, (h_n, c_n)

    def get_action_and_value(self, obs: np.ndarray, hx = None, deterministic: bool = False):
        self.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).view(1,1,-1)  # (1,1,8)
            logits, values, (h_n, c_n) = self.forward(obs_t, hx)
            logits_t = logits[0,0]  # (A,)
            probs = F.softmax(logits_t, dim=-1)
            if deterministic:
                action = int(torch.argmax(probs).item())
            else:
                action = int(Categorical(probs).sample().item())
            logprob = float(torch.log(probs[action] + 1e-8).item())
            value = float(values[0,0].item())
            return action, logprob, value, (h_n, c_n)

# ---------------------------
# GAE / padding / helpers
# ---------------------------
def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        nextval = last_value if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * nextval * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns

def pad_sequences(seqs: List[np.ndarray], pad_value=0.0):
    lengths = [len(s) for s in seqs]
    T = max(lengths)
    B = len(seqs)
    if seqs[0].ndim == 1:
        arr = np.full((B, T), pad_value, dtype=np.float32)
    else:
        D = seqs[0].shape[1]
        arr = np.full((B, T, D), pad_value, dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = len(s)
        if s.ndim == 1:
            arr[i, :L] = s
        else:
            arr[i, :L, :] = s
        mask[i, :L] = 1.0
    return arr, mask

# ---------------------------
# Episode collection
# ---------------------------
def collect_episodes(env: ValueGridEnv, policy: RecurrentPolicy, n_episodes: int, seed: int = None):
    episodes = []
    if seed is not None:
        env.seed(seed)
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        hx = (torch.zeros(policy.n_layers, 1, policy.hidden_size_H, device=DEVICE),
              torch.zeros(policy.n_layers, 1, policy.hidden_size_H, device=DEVICE))
        obs_seq = []
        actions = []
        logps = []
        rewards = []
        dones = []
        values = []
        ended_on_max_flags = []
        while not done:
            action, logp, value, hx = policy.get_action_and_value(obs, hx)
            next_obs, reward, done, info = env.step(action)
            obs_seq.append(obs.copy())
            actions.append(int(action))
            logps.append(float(logp))
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(float(value))
            ended_on_max_flags.append(bool(info.get('ended_on_max', False)))
            obs = next_obs
            if len(rewards) >= env.max_steps + 5:
                break
        last_value = 0.0
        if not dones[-1]:
            _, _, last_value, _ = policy.get_action_and_value(obs, hx)
        ep_dict = {
            'obs': np.array(obs_seq, dtype=np.float32),    # (T, D=8)
            'actions': np.array(actions, dtype=np.int64),
            'logps': np.array(logps, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=np.bool_),
            'values': np.array(values, dtype=np.float32),
            'ended_on_max': np.array(ended_on_max_flags, dtype=np.bool_),
            'init_hx': (torch.zeros(policy.n_layers, 1, policy.hidden_size_H, device=DEVICE),
                        torch.zeros(policy.n_layers, 1, policy.hidden_size_H, device=DEVICE)),
            'last_value': float(last_value)
        }
        episodes.append(ep_dict)
    return episodes

# ---------------------------
# PPO update (episode-level minibatches)
# ---------------------------
def ppo_update(policy: RecurrentPolicy, optimizer, episodes: List[dict]):
    policy.train()
    # compute adv & returns per episode
    for ep in episodes:
        adv, ret = compute_gae(ep['rewards'].tolist(), ep['values'].tolist(), ep['dones'].tolist(), ep['last_value'])
        ep['adv'] = adv
        ep['ret'] = ret

    n_eps = len(episodes)
    indices = list(range(n_eps))
    losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    for _ in range(PPO_EPOCHS):
        random.shuffle(indices)
        for start in range(0, n_eps, MINI_BATCH_EPISODES):
            batch_idx = indices[start:start+MINI_BATCH_EPISODES]
            batch_eps = [episodes[i] for i in batch_idx]
            # prepare padded tensors for observations, actions, old logps, advantages, returns
            obs_seqs = [ep['obs'] for ep in batch_eps]  # list of (T, D)
            obs_pad, mask = pad_sequences(obs_seqs, pad_value=0.0)  # obs_pad (B, T, D), mask (B, T)
            B, T, D = obs_pad.shape
            obs_tensor = torch.tensor(obs_pad, dtype=torch.float32, device=DEVICE)
            actions_pad, _ = pad_sequences([ep['actions'] for ep in batch_eps], pad_value=0)
            actions_tensor = torch.tensor(actions_pad, dtype=torch.int64, device=DEVICE)
            old_logp_pad, _ = pad_sequences([ep['logps'] for ep in batch_eps], pad_value=0.0)
            old_logp_tensor = torch.tensor(old_logp_pad, dtype=torch.float32, device=DEVICE)
            adv_pad, _ = pad_sequences([ep['adv'] for ep in batch_eps], pad_value=0.0)
            adv_tensor = torch.tensor(adv_pad, dtype=torch.float32, device=DEVICE)
            ret_pad, _ = pad_sequences([ep['ret'] for ep in batch_eps], pad_value=0.0)
            ret_tensor = torch.tensor(ret_pad, dtype=torch.float32, device=DEVICE)
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=DEVICE)

            # forward (initial hx zeros per batch episode)
            h0 = torch.zeros(policy.n_layers, B, policy.hidden_size_H, device=DEVICE)
            c0 = torch.zeros(policy.n_layers, B, policy.hidden_size_H, device=DEVICE)
            logits, values, _ = policy(obs_tensor, (h0, c0))
            logp_all = F.log_softmax(logits, dim=-1)  # (B, T, A)
            actions_expand = actions_tensor.unsqueeze(-1)
            new_logps = torch.gather(logp_all, dim=-1, index=actions_expand).squeeze(-1)  # (B,T)
            # mask
            new_logps = new_logps * mask_tensor
            old_logp_tensor = old_logp_tensor * mask_tensor
            adv_tensor = adv_tensor * mask_tensor
            ret_tensor = ret_tensor * mask_tensor
            values = values * mask_tensor

            ratio = torch.exp(new_logps - old_logp_tensor + 1e-8)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_tensor
            policy_loss = - (torch.sum(torch.min(surr1, surr2)) / (mask_tensor.sum() + 1e-8))

            value_loss = VF_COEF * (torch.sum(((values - ret_tensor) ** 2) * mask_tensor) / (mask_tensor.sum() + 1e-8))

            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum((probs * logp_all) * mask_tensor.unsqueeze(-1)) / (mask_tensor.sum() + 1e-8)

            loss = policy_loss + value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            losses['policy_loss'] += float(policy_loss.item())
            losses['value_loss'] += float(value_loss.item())
            losses['entropy'] += float(entropy.item())

    denom = (PPO_EPOCHS * max(1, n_eps // MINI_BATCH_EPISODES))
    for k in list(losses.keys()):
        losses[k] /= max(1, denom)
    return losses

# ---------------------------
# Evaluation (deterministic / greedy)
# Returns:
#   avg_reward_over_episodes (avg softmax reward),
#   success_fraction_exact (fraction ended on global max),
#   mean_episode_length
# ---------------------------
def evaluate_policy(env: ValueGridEnv, policy: RecurrentPolicy, n_episodes=100, seed=None):
    if seed is not None:
        env.seed(seed)
    success_count_exact = 0
    total_reward = 0.0
    lengths = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        hx = (torch.zeros(policy.n_layers, 1, policy.hidden_size_H, device=DEVICE),
              torch.zeros(policy.n_layers, 1, policy.hidden_size_H, device=DEVICE))
        steps = 0
        final_reward = 0.0
        ended_on_max_flag = False
        while not done:
            action, _, _, hx = policy.get_action_and_value(obs, hx, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                final_reward = reward
                ended_on_max_flag = bool(info.get('ended_on_max', False))
            if steps > env.max_steps + 5:
                break
        total_reward += float(final_reward)
        success_count_exact += 1 if ended_on_max_flag else 0
        lengths.append(steps)
    avg_reward = total_reward / float(n_episodes)
    success_frac_exact = success_count_exact / float(n_episodes)
    return avg_reward, success_frac_exact, float(np.mean(lengths))

# ---------------------------
# Temperature schedule helper
# ---------------------------
def temperature_for_update(update_idx: int) -> float:
    """Exponential decay per update (clamped at MIN_TEMPERATURE)."""
    temp = INIT_TEMPERATURE * (TEMP_DECAY_PER_UPDATE ** max(0, update_idx))
    if temp < MIN_TEMPERATURE:
        temp = MIN_TEMPERATURE
    return float(temp)

# ---------------------------
# Training single run (per seed)
# ---------------------------
def train_single_run(hidden_size_H:int, board_size:int, seed:int, progress_writer, eval_store:Dict):
    """
    Train single seed for a given hidden size and board size.
    progress_writer: csv.writer that accepts rows
    eval_store: dict[(H, update)] -> list of avg_rewards observed from different seeds
    Returns a dict with training metadata including saved model_state for later zero-shot tests.
    """
    print(f"TRAIN RUN: H={hidden_size_H}, board={board_size}, seed={seed}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = ValueGridEnv(size=board_size, start_random=True, rng=random.Random(seed), temperature=INIT_TEMPERATURE)
    policy = RecurrentPolicy(hidden_size_H, embedding_dim=32, n_layers=LSTM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    best_eval = -1e9
    converged = False
    updates = 0
    total_eps_collected = 0
    start_time = time.time()

    # training loop
    while updates < MAX_UPDATES:
        # collect episodes
        episodes = collect_episodes(env, policy, EPISODES_PER_UPDATE, seed=None)
        total_eps_collected += len(episodes)

        # before update: compute temperature for upcoming evaluation vote; update environment temperature
        current_temp = temperature_for_update(updates)
        env.temperature = current_temp

        # PPO update
        losses = ppo_update(policy, optimizer, episodes)
        updates += 1

        # evaluate periodically (every EVAL_EVERY_UPDATES)
        if updates % EVAL_EVERY_UPDATES == 0:
            # set temperature for evaluation (use decayed temperature at this update)
            eval_temp = temperature_for_update(updates)
            eval_env = ValueGridEnv(size=board_size, start_random=True, rng=random.Random(seed + 1234), temperature=eval_temp)
            avg_reward, success_frac, mean_len = evaluate_policy(eval_env, policy, n_episodes=EVAL_EPS, seed=seed + 1234)
            print(f"[H={hidden_size_H} seed={seed}] update {updates} | eval_avg_reward={avg_reward:.4f} success_frac={success_frac:.3f} mean_len={mean_len:.1f}")

            # store eval in eval_store for median calculation across seeds for this H and update
            key = (hidden_size_H, updates)
            eval_store.setdefault(key, []).append(avg_reward)
            median_reward = float(np.median(eval_store[key]))

            # compute median delta from previous eval for this H (previous eval at updates - EVAL_EVERY_UPDATES)
            prev_key = (hidden_size_H, updates - EVAL_EVERY_UPDATES)
            if prev_key in eval_store and len(eval_store[prev_key]) > 0:
                prev_median = float(np.median(eval_store[prev_key]))
                median_delta = median_reward - prev_median
            else:
                median_delta = 0.0

            # Write CSV row: update, seed, board_size, hidden_size, avg_reward_this_seed, success_frac, median_reward_across_seeds_for_this_H_update, median_delta, temperature
            progress_writer.writerow([updates, seed, board_size, hidden_size_H,
                                      f"{avg_reward:.6f}", f"{success_frac:.6f}",
                                      f"{median_reward:.6f}", f"{median_delta:.6f}", f"{eval_temp:.6f}"])
            # flush writer to disk
            try:
                progress_writer.flush()
            except Exception:
                # progress_writer may be csv.writer with file handle; try to flush underlying file
                pass
            # update best eval and convergence condition
            if avg_reward > best_eval:
                best_eval = avg_reward
            if best_eval >= MIN_SUCCESS:
                converged = True
                break

    elapsed = time.time() - start_time
    return {
        'converged': converged,
        'best_eval': best_eval,
        'updates': updates,
        'episodes_collected': total_eps_collected,
        'time_s': elapsed,
        'model_state': copy.deepcopy(policy.state_dict())
    }

# ---------------------------
# Main sweep & orchestration
# ---------------------------
def run_experiment_and_log():
    # Prepare CSV file and write header
    f_csv = open(PROGRESS_CSV, mode='w', newline='')
    csv_writer = csv.writer(f_csv)
    header = ['update', 'seed', 'board_size', 'hidden_size',
              'avg_reward', 'success_frac', 'median_reward_across_seeds', 'median_delta', 'temperature']
    csv_writer.writerow(header)
    # We'll require a small wrapper to flush easily
    class ProgressWriter:
        def __init__(self, csvfile, writer):
            self.csvfile = csvfile
            self.writer = writer
        def writerow(self, row):
            self.writer.writerow(row)
        def flush(self):
            try:
                self.csvfile.flush()
            except Exception:
                pass
    progress = ProgressWriter(f_csv, csv_writer)

    # evaluation store for medians: key (H, update) -> list of avg_rewards
    eval_store: Dict[Tuple[int, int], List[float]] = {}

    summary_all = {}
    # sweep H
    for H in range(START_H, MAX_H + 1, H_STEP):
        print("\n" + "="*80)
        print(f"SWEEP H = {H}")
        seed_results = []
        for s in SEED_LIST:
            res = train_single_run(H, BOARD_TRAIN, s, progress, eval_store)
            seed_results.append(res)
            print(f" Seed {s} -> converged={res['converged']} best_eval={res['best_eval']:.4f} updates={res['updates']} eps={res['episodes_collected']} time={res['time_s']:.1f}s")
        # aggregate means across seeds for the best_eval metric
        mean_best = float(np.mean([r['best_eval'] for r in seed_results]))
        std_best = float(np.std([r['best_eval'] for r in seed_results], ddof=0))
        summary_all[H] = {'per_seed': seed_results, 'mean_best': mean_best, 'std_best': std_best}
        print(f"H={H} summary: mean_best_eval={mean_best:.4f} std={std_best:.4f}")

        # stopping condition: if mean across seeds reached MIN_SUCCESS
        if mean_best >= MIN_SUCCESS:
            print(f"Reached mean success >= {MIN_SUCCESS:.2f} at H={H}. Recording zero-shot tests and finishing sweep.")
            # Save models for zero-shot testing
            saved_models = [r['model_state'] for r in seed_results]
            zero_shot_results = {}
            for test_size in BOARD_TESTS:
                per_seed_avg = []
                per_seed_success = []
                for i, state in enumerate(saved_models):
                    policy = RecurrentPolicy(H, embedding_dim=32, n_layers=LSTM_LAYERS).to(DEVICE)
                    policy.load_state_dict(state)
                    test_env = ValueGridEnv(size=test_size, start_random=True, rng=random.Random(999+i), temperature=temperature_for_update(0))
                    avg_reward, success_frac, mean_len = evaluate_policy(test_env, policy, n_episodes=EVAL_EPS, seed=1000+i)
                    per_seed_avg.append(avg_reward)
                    per_seed_success.append(success_frac)
                    print(f" Zero-shot model {i} on {test_size}x{test_size}: avg_reward={avg_reward:.4f} success_frac_exact={success_frac:.4f}")
                zero_shot_results[test_size] = {'avg_rewards': per_seed_avg, 'success_fracs': per_seed_success,
                                               'mean_avg': float(np.mean(per_seed_avg)),
                                               'mean_success_frac': float(np.mean(per_seed_success))}
            # close csv
            f_csv.close()
            return {'found_H': H, 'summary': summary_all, 'zero_shot': zero_shot_results}
    # exhausted H range
    f_csv.close()
    return {'found_H': None, 'summary': summary_all}

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    t0 = time.time()
    results = run_experiment_and_log()
    t1 = time.time()
    print("\nExperiment finished in %.1f s" % (t1 - t0))
    if results.get('found_H') is not None:
        print("Found H =", results['found_H'])
        for k, v in results['zero_shot'].items():
            print(f"Zero-shot on {k}x{k}: mean avg_reward {v['mean_avg']:.3f} mean success_frac {v['mean_success_frac']:.3f}")
    else:
        print("No H found that met success threshold within H range. Inspect summary.")
