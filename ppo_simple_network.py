import math
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class BackBone(nn.Module):
    def __init__(self, mlp_input_size, mlp_hidden_dims, num_hidden_layers, conv_specs):
        super().__init__()
        self.in_size = mlp_input_size
        self.hidden = mlp_hidden_dims
        conv_layers = []

        for i in conv_specs:
            conv_layers.append(nn.Conv2d(i[0], i[1], i[2], i[3], i[4]))

        self.convLayers = nn.ModuleList(conv_layers)
        layers = []
        for i in range(num_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(self.in_size, self.hidden))
            else:
                layers.append(nn.Linear(self.hidden, self.hidden))
        
        self.mlp_layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for i in self.convLayers:
            x = F.relu(i(x))
        x = x.mean(dim=(2, 3))
        for i in self.mlp_layers:
            x = F.relu(i(x))
        return x

class Head(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.out_layer = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.out_layer(x)

class Agent(nn.Module):
    def __init__(self, mlp_input_size, mlp_hidden_dims, mlp_num_hidden_layers, mlp_output_size, conv_specs):
        super().__init__()
        self.BackboneNN = BackBone(mlp_input_size, mlp_hidden_dims, mlp_num_hidden_layers, conv_specs)
        self.PolicyHeadNN = Head(mlp_hidden_dims, mlp_output_size)
        self.ValueHeadNN = Head(mlp_hidden_dims, 1)

    def forward(self, board_state):
        x = self.BackboneNN(board_state)
        p = self.PolicyHeadNN(x)
        v = self.ValueHeadNN(x)
        return p,v
    
from grid_world_keys import Environment

GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.2
CE = 0.1
CE_ANEAL = 0.99 
CE_EPISODES_PER_ANEAL = 5

VALUE_LOSS_COEF = 0.6
STEPS_PER_EPISODE = 2000

EPISODES_CURR_ZERO = 700
EPISODES_CURR_ONE = 700
EPISODES_CURR_TWO = 2000
ALPHA = 0.7

SUPP_REWARD_COEF = 2
SUPP_DEPLEATION_RATE = 2/(EPISODES_CURR_ZERO+EPISODES_CURR_ONE)

from collections import defaultdict
from typing import List

stats = defaultdict(list)


def run_episode(env: Environment, agent):
    step_logs = []

    total_actual_reward = 0.0
    total_reward = 0.0
    total_entropy = 0.0
    total_steps = 0


    x = env.render()
    curr = env.export_state()
    while not env.finished:
        with torch.no_grad():
            logits, value = agent(x.unsqueeze(0))
            value = value.squeeze()

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp_old = dist.log_prob(action).detach()
            entropy = dist.entropy().mean()

            reward = env.update(action.item())
            new = env.export_state()
            sup_reward = env.suplementary_reward(curr,new)
            
            total_reward += reward + sup_reward*SUPP_REWARD_COEF
            total_actual_reward += reward
            total_entropy += entropy.item()

            xp = env.render()

            step_logs.append([
                x, action.item(), xp,
                reward+ sup_reward*SUPP_REWARD_COEF, env.finished,
                logp_old, value
            ])

            total_steps += 1
            curr = new
            x = xp

    T = len(step_logs)
    gae = 0.0

    for t in reversed(range(T)):
        s, a, sp, r, done, logp, V_t = step_logs[t]

        if t == T - 1:
            V_tp1 = 0.0  
        else:
            V_tp1 = step_logs[t + 1][6]

        delta = r + GAMMA*V_tp1 - V_t
        gae = delta + GAMMA*LAMBDA* gae

        step_logs[t].append(gae)

    return total_reward, total_actual_reward, total_entropy, total_steps, step_logs

def offline_train(batch, agent, optimizer):
    avg_loss = 0
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

    for _ in range(3):
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
        avg_loss += loss.item()

    return avg_loss/3



def run_curriculum_stage(
    env:Environment, agent, optimizer,
    num_episodes, stats
):
    global SUPP_REWARD_COEF, CE
    for episode_idx in range(num_episodes):
        avg_reward = 0.0
        avg_reward_real = 0.0
        avg_entropy = 0.0
        avg_loss = 0.0
        batch = []
        while len(batch) < STEPS_PER_EPISODE:
            env.initialize_state()
            #total_reward, total_actual_reward, total_entropy, total_steps, step_logs
            r, r_real, e, steps, logs = run_episode(env, agent)
            if len(batch)+steps > STEPS_PER_EPISODE:
                batch += logs[:STEPS_PER_EPISODE-len(batch)]
            else:
                batch += logs[:STEPS_PER_EPISODE-len(batch)]
            avg_reward += r
            avg_reward_real += r_real
            avg_entropy += e
        avg_loss = offline_train(batch, agent, optimizer)
        stats["reward"].append(avg_reward/STEPS_PER_EPISODE)
        stats["loss"].append(avg_loss/STEPS_PER_EPISODE)
        stats["entropy"].append(avg_entropy/STEPS_PER_EPISODE)
        stats["true_reward"].append(avg_reward_real/STEPS_PER_EPISODE)
        print(
            f"Episode {episode_idx:5d} | "
            f"Avg Reward: {avg_reward/STEPS_PER_EPISODE:7.3f} | "
            f"Avg Loss: {avg_loss/STEPS_PER_EPISODE:7.4f} | "
            f"Avg Entropy: {avg_entropy/STEPS_PER_EPISODE:7.4f} | "
            f"Avg Reward: {avg_reward_real/STEPS_PER_EPISODE:7.3f} | "
        )
        if episode_idx%CE_EPISODES_PER_ANEAL == 0:
            CE = CE*CE_ANEAL
        if SUPP_REWARD_COEF != 0:
            SUPP_REWARD_COEF -= SUPP_DEPLEATION_RATE
            if SUPP_REWARD_COEF < 0:
                SUPP_REWARD_COEF = 0
def plot_stats(stats):
    plt.figure(figsize=(12, 4))

    plt.subplot(2, 2, 1)
    plt.plot(stats["reward"])
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)
    plt.plot(stats["loss"])
    plt.title("Average Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 3)
    plt.plot(stats["entropy"])
    plt.title("Average Entropy per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")

    plt.subplot(2,2,4)
    plt.plot(stats)

    plt.tight_layout()
    plt.show()


conv_specs = [
    (3, 16, 3, 1, 1),   # keep (12,20) -> (12,20)
    (16, 32, 3, 2, 1),  # stride=2 -> halves: (6,10)
    (32, 64, 3, 2, 1)   # stride=2 -> (3,5)
]
input_shape = (3, 12, 20)

def train():
    env = Environment(6, (12,12), 500)
    agent = Agent(64,64,3,4, conv_specs)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    stats = defaultdict(list)

    curriculum = [
        ((0,1), EPISODES_CURR_ZERO),
        ((1,1), EPISODES_CURR_ONE),
        ((2,1), EPISODES_CURR_TWO),
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