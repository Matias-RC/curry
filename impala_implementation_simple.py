# impala_fixed.py
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import ctypes
import torch.multiprocessing as mp
import time

# -----------------------------
# Network / Environment (kept similar to original)
# -----------------------------
class BackBone(nn.Module):
    def __init__(self, input_size, hidden_dims, num_hidden_layers):
        super().__init__()
        self.in_size = input_size
        self.hidden = hidden_dims

        layers = []
        for i in range(num_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(self.in_size, self.hidden))
            else:
                layers.append(nn.Linear(self.hidden, self.hidden))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for i in self.layers:
            x = F.relu(i(x))
        return x

class Head(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.out_layer = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.out_layer(x)

class Agent(nn.Module):
    def __init__(self, input_size, hidden_dims, num_hidden_layers, output_size):
        super().__init__()
        self.BackboneNN = BackBone(input_size, hidden_dims, num_hidden_layers)
        self.PolicyHeadNN = Head(hidden_dims, output_size)
        self.ValueHeadNN = Head(hidden_dims, 1)

    def forward(self, board_state):
        x = self.BackboneNN(board_state)
        p = self.PolicyHeadNN(x)
        v = self.ValueHeadNN(x)
        return p, v

# -----------------------------
# Simple grid environment (kept mostly the same)
# -----------------------------
class Environment:
    def __init__(self, board_size, max_steps):
        self._size = board_size
        self.curriculum_stage = 0
        self.state = None
        self.agent_pos = None
        self.action_map = [(1,0),(0,-1),(0,1),(-1,0)]
        self.left_steps = max_steps
        self.max_steps = max_steps
        self.finished = False

    def initialize_state(self):
        size = 10
        if self.curriculum_stage == 0:
            self.agent_pos = (random.randint(0,size-1), random.randint(0,size-1))
            apples = []
            # find adjacent apple
            for dy,dx in random.sample(self.action_map, len(self.action_map)):
                if (0 <= self.agent_pos[0] + dy < size) and (0 <= self.agent_pos[1] + dx < size):
                    apples.append((self.agent_pos[0] + dy, self.agent_pos[1] + dx))
                    break
            for idx in range(2):
                for dy,dx in random.sample(self.action_map, len(self.action_map)):
                    if (0 <= apples[idx][0] + dy < size) and (0 <= apples[idx][1] + dx < size):
                        apples.append((apples[idx][0] + dy, apples[idx][1] + dx))
                        break
        elif  self.curriculum_stage == 1:
            self.agent_pos = (random.randint(0,size-1), random.randint(0,size-1))
            possible_positions = set()
            for i in range(-4, 5):
                for j in range(-4, 5):
                    if i == j == 0:
                        pass
                    elif (0 <= self.agent_pos[0] + i < size) and (0 <= self.agent_pos[1] + j < size):
                        possible_positions.add((self.agent_pos[0] + i, self.agent_pos[1] + j))
            apples = random.sample(list(possible_positions), 3)
        else:
            self.agent_pos = (random.randint(0,size-1), random.randint(0,size-1))
            apples = []
            while len(apples) != 3:
                p = (random.randint(0,size-1), random.randint(0,size-1))
                if p != self.agent_pos:
                    apples.append(p)
        self.finished = False
        self.state = [apples, 0] #state is a tuple (list, int)
        self.left_steps = self.max_steps
    
    def manhattan_distance(self, pos_1, pos_2):
        dy = abs(pos_1[0]-pos_2[0])
        dx = abs(pos_1[1]-pos_2[1])
        return dy + dx

    def update_state(self, action):
        dy, dx = self.action_map[action]
        self.left_steps -= 1
        if self.left_steps < 0:
            self.finished = True
            return 0
        size = 10
        if (0 <= self.agent_pos[0] + dy < size) and (0 <= self.agent_pos[1] + dx < size):
            old_pos = self.agent_pos
            self.agent_pos = (self.agent_pos[0]+ dy, self.agent_pos[1]+dx)

            # current target apple
            apple = self.state[0][self.state[1]]
            if self.agent_pos == apple:
                if self.state[1] == 2:
                    reward = 3
                    self.finished = True
                    return reward
                elif self.left_steps == 0:
                    self.state[1] += 1
                    reward = 2
                    self.finished = True
                    return reward
                else:
                    self.state[1] += 1
                    reward = 2
                    self.finished = False
                    return reward
            elif self.left_steps == 0:
                st = self.manhattan_distance(old_pos, apple)
                nd = self.manhattan_distance(self.agent_pos, apple)
                reward = st-nd
                self.finished = True
                return reward
            else:
                st = self.manhattan_distance(old_pos, apple)
                nd = self.manhattan_distance(self.agent_pos, apple)
                reward = st-nd
                self.finished = False
                return reward
        
        else:
            reward = -1
            self.finished = False
            return reward
        
    def render_state(self):
        board = torch.zeros((2, 10, 10))
        apple_pos = self.state[0][self.state[1]]
        board[0, self.agent_pos[0], self.agent_pos[1]] = 1
        board[1, apple_pos[0], apple_pos[1]] = 1
        return board
    
    def load(self, state, agent_pos):
        self.state = [state[0], 0]
        self.agent_pos = agent_pos
        self.finished = False
        self.left_steps = self.max_steps

# -----------------------------
# Hyperparams / constants
# -----------------------------
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.2
CE = 0.05
VALUE_LOSS_COEF = 0.6
LEVELS_PER_EPISODE = 15
EPISODES_CURR_ZERO = 200
EPISODES_CURR_ONE = 500
EPISODES_CURR_TWO = 500
ALPHA = 0.7

NUM_WORKERS = 5
SYNC_STEPS = 20   # Reduced for faster synchronization and less staleness
WORKER_RENDER_BOARD_SIZE = 10
WORKER_MAX_STEPS = 100

RHO_CLIP = 1.0
C_CLIP = 1.0

torch.set_num_threads(1)

# -----------------------------
# Worker process
# -----------------------------
def worker_proc(worker_id, running, shared_agent, traj_queue, stage_obj, sync_steps=SYNC_STEPS):
    """
    Each worker keeps a local copy of the agent for fast inference and periodically
    copies weights from the shared agent (every `sync_steps` steps).
    """
    env = Environment(10, WORKER_MAX_STEPS)
    local_agent = Agent(203, 40, 3, 4)
    local_agent.eval()
    for p in local_agent.parameters():
        p.requires_grad = False

    # initial sync
    local_agent.load_state_dict(shared_agent.state_dict())

    while running.value:
        env.curriculum_stage = stage_obj.value
        env.initialize_state()

        # initial observation
        x = torch.flatten(env.render_state())
        x_ = F.one_hot(torch.tensor(env.state[1], dtype=torch.long), num_classes=3).float()
        x = torch.cat((x, x_), 0)

        step_logs = []
        steps_since_sync = 0
        total_reward = 0.0

        while not env.finished and running.value:
            # local inference
            with torch.no_grad():
                logits, _ = local_agent(x.unsqueeze(0))  # add batch dim, no value needed
            logits = logits.squeeze(0)

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp_old = dist.log_prob(action)

            reward = env.update_state(action.item())
            total_reward += reward

            xp = torch.flatten(env.render_state())
            xp_ = F.one_hot(torch.tensor(env.state[1], dtype=torch.long), num_classes=3).float()
            xp = torch.cat((xp, xp_), 0)

            # store CPU detached copies to avoid CUDA/pickle issues
            entry = [
                x.detach().cpu(),               # state s_t
                int(action.item()),            # action a_t
                float(reward),                 # reward r_{t+1}
                bool(env.finished),            # done
                xp.detach().cpu(),             # next state s_{t+1}
                logp_old.detach().cpu()        # behavior logp mu(a_t | s_t)
            ]
            step_logs.append(entry)

            x = xp
            steps_since_sync += 1

            # periodic local sync
            if steps_since_sync >= sync_steps:
                try:
                    local_agent.load_state_dict(shared_agent.state_dict())
                except Exception:
                    # ignore intermittent race condition
                    pass
                steps_since_sync = 0

        traj_obj = {
            "total_reward": total_reward,
            "steps_list": step_logs,
            "worker_id": worker_id
        }

        # push to queue (blocking if full)
        try:
            traj_queue.put(traj_obj, block=True, timeout=1.0)
        except Exception:
            # if putting times out (possibly trainer dead), break
            break

    # worker exiting
    return


# -----------------------------
# Offline train / learner
# -----------------------------
def offline_train(batch_trajs, agent, optimizer, device=None):
    """
    batch_trajs: list of trajectory step lists, each step: s_t, a_t, r_{t+1}, done, s_{t+1}, log_mu_a
    Implements V-trace for off-policy correction.
    Returns: loss_item, avg_entropy_item
    """
    if device is None:
        device = next(agent.parameters()).device

    loss_policy_total = 0.0
    loss_value_total = 0.0
    loss_entropy_total = 0.0
    total_steps = 0

    num_trajs = len(batch_trajs)

    if num_trajs == 0:
        return 0.0, 0.0

    for traj in batch_trajs:
        T = len(traj)
        if T == 0:
            continue

        # Convert to tensors
        S = torch.stack([step[0] for step in traj]).to(device)  # (T, state_dim)
        A = torch.tensor([step[1] for step in traj], device=device, dtype=torch.long)  # (T,)
        R = torch.tensor([step[2] for step in traj], device=device, dtype=torch.float)  # (T,)
        Done = torch.tensor([step[3] for step in traj], device=device, dtype=torch.float)  # (T,)
        SP = torch.stack([step[4] for step in traj]).to(device)  # (T, state_dim)
        log_mu = torch.tensor([step[5] for step in traj], device=device, dtype=torch.float)  # (T,)

        # Forward pass for policy and value on S (s_0 to s_{T-1})
        logits, V = agent(S)
        V = V.squeeze(-1)  # (T,)

        dist = torch.distributions.Categorical(logits=logits)
        log_pi = torch.clamp(dist.log_prob(A), min=-1e9) # (T,)

        # Forward pass for V on SP (s_1 to s_T)
        _, V_sp = agent(SP)
        V_sp = V_sp.squeeze(-1)  # (T,)

        # Importance ratios
        rho = torch.exp(log_pi - log_mu)  # (T,)
        c = rho.clone()

        rho = torch.min(rho, torch.tensor(RHO_CLIP, device=device))
        c = torch.min(c, torch.tensor(C_CLIP, device=device))

        # Compute V-trace targets backward
        vs = torch.zeros(T, dtype=torch.float, device=device)
        next_v = 0.0  # Bootstrap 0 for terminal

        for t in reversed(range(T)):
            delta = rho[t] * (R[t] + GAMMA * next_v - V[t])
            vs[t] = V[t] + delta + GAMMA * c[t] * (next_v - V_sp[t])
            next_v = vs[t]

        # Compute advantages A_t = r_{t+1} + gamma * v_{t+1} - V(s_t)
        v_next = torch.cat((vs[1:], torch.tensor([0.0], device=device)))
        A = R + GAMMA * (1 - Done) * v_next - V

        Adv_detached = A.detach()

        # Losses (sum over trajectory)
        loss_policy_total += -(rho * log_pi * Adv_detached).sum()
        loss_value_total += VALUE_LOSS_COEF * F.mse_loss(V, vs.detach(), reduction='sum')
        loss_entropy_total += -CE * dist.entropy().sum()

        total_steps += T

    if total_steps == 0:
        return 0.0, 0.0

    # Average losses
    loss = (loss_policy_total + loss_value_total + loss_entropy_total) / total_steps

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_entropy = loss_entropy_total.item() / total_steps / -CE  # Undo the -CE to get avg entropy

    return loss.item(), avg_entropy

# -----------------------------
# Curriculum runner / trainer loop
# -----------------------------
def run_curriculum_stage(shared_agent, optimizer, num_episodes, stats, traj_queue, device=None):
    if device is None:
        device = next(shared_agent.parameters()).device

    for episode_idx in range(num_episodes):
        # accumulate LEVELS_PER_EPISODE trajectories
        batch_trajs = [traj_queue.get() for _ in range(LEVELS_PER_EPISODE)]  # blocking
        avg_reward = sum(t["total_reward"] for t in batch_trajs) / LEVELS_PER_EPISODE

        steps_lists = [t["steps_list"] for t in batch_trajs]

        if len(steps_lists) == 0:
            continue

        avg_loss, avg_entropy = offline_train(steps_lists, shared_agent, optimizer, device=device)

        stats["reward"].append(avg_reward)
        stats["loss"].append(avg_loss)
        stats["entropy"].append(avg_entropy)
        print(
            f"Episode {episode_idx:5d} | "
            f"Avg Reward: {avg_reward:7.3f} | "
            f"Avg Loss: {avg_loss:7.4f} | "
            f"Avg Entropy: {avg_entropy:7.4f}"
        )

# -----------------------------
# Trainer process wrapper
# -----------------------------
def trainer_proc(running, shared_agent, traj_queue, stage_obj, stats_queue, device=None):
    # optimizer lives in trainer (updates shared agent)
    if device is None:
        device = torch.device("cpu")
    shared_agent.to(device)
    optimizer = torch.optim.Adam(shared_agent.parameters(), lr=1e-3)

    stats = defaultdict(list)

    curriculum = [
        (0, EPISODES_CURR_ZERO),
        (1, EPISODES_CURR_ONE),
        (2, EPISODES_CURR_TWO),
    ]

    for stage, episodes in curriculum:
        if not running.value:
            break
        stage_obj.value = stage
        print(f"[Trainer] Starting curriculum stage {stage} for {episodes} episodes")
        run_curriculum_stage(shared_agent, optimizer, episodes, stats, traj_queue, device=device)

    # tell workers to stop
    running.value = False
    stats_queue.put(dict(stats))
    print("[Trainer] Finished curriculum and set running=False")
    return stats

import matplotlib.pyplot as plt

def plot_stats(stats):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(stats["reward"])
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 3, 2)
    plt.plot(stats["loss"])
    plt.title("Average Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(stats["entropy"])
    plt.title("Average Entropy per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main entry: start trainer + workers
# -----------------------------
def main(num_workers=NUM_WORKERS):
    mp.set_start_method('spawn', force=True)


    shared_agent = Agent(203, 40, 3, 4)
    shared_agent.share_memory()  # important for multiprocessing

    running = mp.Value(ctypes.c_bool, True)
    traj_queue = mp.Queue(maxsize=LEVELS_PER_EPISODE * 4)  # some buffer
    stats_queue = mp.Queue()

    stage_obj = mp.Value(ctypes.c_int, 0)

    # start trainer in a process
    trainer_args = (running, shared_agent, traj_queue, stage_obj, stats_queue)
    trainer_p = mp.Process(target=trainer_proc, args=trainer_args)

    trainer_p.start()
    time.sleep(0.2)  # small allow trainer to start

    # start workers
    workers = []
    for wid in range(num_workers):
        args = (wid, running, shared_agent, traj_queue, stage_obj, SYNC_STEPS)
        p = mp.Process(target=worker_proc, args=args)
        p.start()
        workers.append(p)

    try:
        # Wait for trainer to finish
        trainer_p.join()
        stats = stats_queue.get()
        plot_stats(stats)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: terminating processes")
        running.value = False

    # ensure workers terminate
    for p in workers:
        p.join(timeout=1.0)
        if p.is_alive():
            p.terminate()

    if trainer_p.is_alive():
        trainer_p.terminate()

    print("All processes terminated.")

if __name__ == "__main__":
    main()