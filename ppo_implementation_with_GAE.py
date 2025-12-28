import math
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
        return p,v


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
        if self.curriculum_stage == 0:
            self.agent_pos = (random.randint(0,9), random.randint(0,9))
            apples = []
            for dy,dx in random.sample(self.action_map, 4):
                if (-1 < self.agent_pos[0] + dy < 10) and (-1 < self.agent_pos[1] + dx < 10):
                    apples.append((self.agent_pos[0] + dy, self.agent_pos[1] + dx))
                    break
            for idx in range(2):
                for dy,dx in random.sample(self.action_map, 4):
                    if (-1 < apples[idx][0] + dy < 10) and (-1 < apples[idx][1] + dx < 10):
                        apples.append((apples[idx][0] + dy, apples[idx][1] + dx))
                        break
        elif  self.curriculum_stage == 1:
            self.agent_pos = (random.randint(0,9), random.randint(0,9))
            possible_positions = set()
            for i in range(-4, 5):
                for j in range(-4, 5):
                    if i == j == 0:
                        pass
                    elif (-1 < self.agent_pos[0] + i < 10) and (-1 < self.agent_pos[1] + j < 10):
                        possible_positions.add((self.agent_pos[0] + i, self.agent_pos[1] + j))
            apples = random.sample(list(possible_positions), 3)
        else:
            self.agent_pos = (random.randint(0,9), random.randint(0,9))
            apples = []
            while len(apples) != 3:
                p = (random.randint(0,9), random.randint(0,9))
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
        if (-1 < self.agent_pos[0] + dy < 10) and (-1 < self.agent_pos[1] + dx < 10):
            old_pos = self.agent_pos
            self.agent_pos = (self.agent_pos[0]+ dy, self.agent_pos[1]+dx)

            if self.agent_pos == self.state[0][self.state[1]]:
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
                st = self.manhattan_distance(old_pos, self.state[0][self.state[1]])
                nd = self.manhattan_distance(self.agent_pos, self.state[0][self.state[1]])
                reward = st-nd
                self.finished = True
                return reward
            else:
                st = self.manhattan_distance(old_pos, self.state[0][self.state[1]])
                nd = self.manhattan_distance(self.agent_pos, self.state[0][self.state[1]])
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

GAMMA = 0.9
LAMBDA = 0.8

LAMBDA = 0 #TD(0)
EPSILON = 0.2
CE = 0.08
VALUE_LOSS_COEF = 0.6
LEVELS_PER_EPISODE = 5
REPLAYS = 3
EPISODES_CURR_ZERO = 200
EPISODES_CURR_ONE = 500
EPISODES_CURR_TWO = 500
ALPHA = 0.7

from collections import defaultdict
from typing import List

stats = defaultdict(list)

def run_episode(env: Environment, agent):
    step_logs = []

    total_reward = 0.0
    total_entropy = 0.0
    total_steps = 0

    x = torch.flatten(env.render_state())
    x_ = F.one_hot(torch.tensor(env.state[1]), num_classes=3).float()
    x = torch.cat((x, x_), 0)

    while not env.finished:
        with torch.no_grad():
            logits, value = agent(x)
            value = value.squeeze()

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp_old = dist.log_prob(action).detach()
            entropy = dist.entropy().mean()

            reward = env.update_state(action.item())

            total_reward += reward
            total_entropy += entropy.item()

            xp = torch.flatten(env.render_state())
            xp_ = F.one_hot(torch.tensor(env.state[1]), num_classes=3).float()
            xp = torch.cat((xp, xp_), 0)

            step_logs.append([
                x, action.item(), xp,
                reward, env.finished,
                logp_old, value
            ])

            total_steps += 1
            x = xp

    T = len(step_logs)
    gae = 0.0

    for t in reversed(range(T)):
        s, a, sp, r, done, logp, V_t = step_logs[t]

        if t == T - 1:
            V_tp1 = 0.0  
        else:
            V_tp1 = step_logs[t + 1][6]

        delta = r + GAMMA * (1 - done) * V_tp1 - V_t
        gae = delta + GAMMA * LAMBDA * (1 - done) * gae

        step_logs[t].append(gae)

    return total_reward, total_entropy, total_steps, step_logs

def offline_train(batch, agent, optimizer):
    device = next(agent.parameters()).device

    # Unpack batch
    S = torch.stack([s for s, a, sp, r, d, lp, v, adv in batch]).to(device)
    A = torch.tensor([a for s, a, sp, r, d, lp, v, adv in batch], device=device)
    Done = torch.tensor([d for s, a, sp, r, d, lp, v, adv in batch],
                        device=device, dtype=torch.float)

    Logp_old = torch.stack([lp for s, a, sp, r, d, lp, v, adv in batch]).to(device)
    V_old = torch.stack([v for s, a, sp, r, d, lp, v, adv in batch]).squeeze().to(device)

    # RAW GAE (for value learning)
    GAE_raw = torch.tensor([adv for s, a, sp, r, d, lp, v, adv in batch],
                           device=device, dtype=torch.float)

    # Normalize ONLY for policy
    Adv = (GAE_raw - GAE_raw.mean()) / (GAE_raw.std() + 1e-8)
    Adv_detached = Adv.detach()

    # Forward
    logits, V_new = agent(S)
    V_new = V_new.squeeze()

    dist = torch.distributions.Categorical(logits=logits)
    logp_new = dist.log_prob(A)
    entropy = dist.entropy()

    ratios = torch.exp(logp_new - Logp_old)

    # PPO policy loss
    unclipped = ratios * Adv_detached
    clipped = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON) * Adv_detached
    loss_policy = -torch.min(unclipped, clipped).mean()

    # VALUE TARGET (CRITICAL FIX)
    V_target = V_old + GAE_raw
    loss_value = VALUE_LOSS_COEF * F.mse_loss(V_new, V_target.detach())

    # Entropy bonus
    loss_entropy = -CE * entropy.mean()

    loss = loss_policy + loss_value + loss_entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def run_curriculum_stage(
    env, agent, optimizer,
    num_episodes, stats
):
    for episode_idx in range(num_episodes):

        # initialize episode pool
        episode_states = []
        for _ in range(LEVELS_PER_EPISODE):
            env.initialize_state()
            episode_states.append([env.state, env.agent_pos])

        avg_reward = avg_entropy = 0.0

        for _ in range(REPLAYS):
            batch = []
            for lvl in range(LEVELS_PER_EPISODE):
                env.load(*episode_states[lvl])

                r, e, steps, logs = run_episode(env, agent)
                batch += logs
                avg_reward += r / LEVELS_PER_EPISODE
                avg_entropy += e / max(1, steps * LEVELS_PER_EPISODE)
            batch = random.sample(batch, len(batch))
            avg_loss = offline_train(batch, agent, optimizer)


        stats["reward"].append(avg_reward)
        stats["loss"].append(avg_loss)
        stats["entropy"].append(avg_entropy)
        print(
            f"Episode {episode_idx:5d} | "
            f"Avg Reward: {avg_reward:7.3f} | "
            f"Avg Loss: {avg_loss:7.4f} | "
            f"Avg Entropy: {avg_entropy:7.4f}"
        )

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


def train():
    env = Environment(10, 100)
    agent = Agent(203, 40, 3, 4)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    stats = defaultdict(list)

    curriculum = [
        (0, EPISODES_CURR_ZERO),
        (1, EPISODES_CURR_ONE),
        (2, EPISODES_CURR_TWO),
    ]

    for stage, episodes in curriculum:
        env.curriculum_stage = stage
        run_curriculum_stage(
            env, agent, optimizer,
            episodes, stats
        )

    return stats


if __name__ == "__main__":
    plot_stats(train())