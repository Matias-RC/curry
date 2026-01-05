import math
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from potential_based_environment import PotentialBased, Node, InvalidNode
from collections import defaultdict
from typing import List
import os
import shutil
import imageio.v2 as imageio

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
        self.ValueBackBones = BackBone(hidden_dims, hidden_dims, 1)
        self.PolicyHeadNN = Head(hidden_dims, output_size)
        self.ValueHeadNN = Head(hidden_dims, 1)

    def forward(self, board_state):
        x = self.BackboneNN(board_state)
        p = self.PolicyHeadNN(x)
        v = self.ValueHeadNN(x)
        return p,v


GAMMA = 0.99
LAMBDA = 0.93
EPSILON = 0.2
VALUE_LOSS_COEF = 0.2
LEVELS_PER_EPISODE = 5
REPLAYS = 1
EPISODES_CURR = 800



stats = defaultdict(list)

def run_episode(env:PotentialBased, agent):
    step_logs = []

    total_reward = 0.0
    total_entropy = 0.0
    total_steps = 0
    x = env.render().unsqueeze(0)
    while not env.finished:
        with torch.no_grad():
            logits, value = agent(x)
            value = value.squeeze()

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp_old = dist.log_prob(action).detach()
            entropy = dist.entropy()

            reward, _ = env.update(action.item())

            total_reward += reward
            total_entropy += entropy.item()
            xp = env.render().unsqueeze(0)

            # We store the observation, action, next observation, reward, done flag, old logp, and V_t
            step_logs.append([x, action.item(), xp, reward, env.finished, logp_old, value])
            total_steps += 1

            x = xp

    # Compute GAE backwards now that we have the raw rewards and values stored.
    T = len(step_logs)
    gae = 0.0

    for t in reversed(range(T)):
        s, a, sp, r, done, logp, V_t = step_logs[t]

        if t == T - 1:
            V_tp1 = 0.0  
        else:
            # index 6 contains the stored V_{t+1} (the value for the next timestep)
            V_tp1 = step_logs[t + 1][6]

        delta = r + (GAMMA * V_tp1) - V_t
        gae = delta + GAMMA * LAMBDA * gae

        # append GAE to each log entry (so later offline_train can pick it up)
        step_logs[t].append(gae)

    return total_reward, total_entropy, total_steps, step_logs

# ----------------- CHANGED: offline_train clarifications and stable normalization -----------------
def offline_train(batch, agent, optimizer, CE):
    """
    KEY CHANGES & NOTES:
      * Make sure Advantage (GAE) is normalized (this was already done downstream, kept but clarified).
      * V_target is computed from V_old + GAE_raw (this keeps targets anchored to values at sampling time).
      * We ensure tensors are on the correct device and shapes are correct.
      * This function avoids relying on any 'global' stale V - it uses V values saved inside the batch.
    """
    device = next(agent.parameters()).device

    # Unpack batch entries; each entry looks like:
    # [s, a, sp, r, d, logp_old, V_t, adv]
    S = torch.cat([s for s, a, sp, r, d, lp, v, adv in batch]).to(device)  # stack tensors into a batch
    A = torch.tensor([a for s, a, sp, r, d, lp, v, adv in batch], device=device)
    Done = torch.tensor([d for s, a, sp, r, d, lp, v, adv in batch],
                        device=device, dtype=torch.float)

    # LOGP_OLD: stack log probabilities (they were stored as scalars / tensors)
    Logp_old = torch.stack([lp for s, a, sp, r, d, lp, v, adv in batch]).to(device)
    V_old = torch.stack([v for s, a, sp, r, d, lp, v, adv in batch]).squeeze().to(device)

    # RAW GAE (for value learning)
    GAE_raw = torch.tensor([adv for s, a, sp, r, d, lp, v, adv in batch],
                           device=device, dtype=torch.float)

    # Normalize ONLY for policy (this is important for stable PPO updates)
    Adv = (GAE_raw - GAE_raw.mean()) / (GAE_raw.std() + 1e-8)
    Adv_detached = Adv.detach()

    # Forward pass for the batch
    logits, V_new = agent(S)
    V_new = V_new.squeeze()

    dist = torch.distributions.Categorical(logits=logits)
    logp_new = dist.log_prob(A)
    entropy = dist.entropy()

    ratios = torch.exp(logp_new - Logp_old)

    # PPO clipped surrogate objective
    unclipped = ratios * Adv_detached
    clipped = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON) * Adv_detached
    loss_policy = -torch.min(unclipped, clipped).mean()

    # VALUE TARGET: use the V_old (value estimated at sampling time) plus raw GAE to produce targets.
    # This ensures target = V_sample + advantage_raw (bootstrap-consistent target).
    V_target = V_old + GAE_raw
    loss_value = VALUE_LOSS_COEF * F.mse_loss(V_new, V_target.detach())

    # Entropy bonus (we add it to the loss with a sign that encourages exploration)
    loss_entropy = +CE * entropy.mean()

    loss = loss_policy + loss_value + loss_entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def stream_episode(
    env:PotentialBased,
    agent,
    gif_path="episode.gif",
    total_duration_sec=30,
    frame_dir="images"
):
    os.makedirs(frame_dir, exist_ok=True)

    fps = 200 / total_duration_sec
    frame_duration = 1.0 / fps

    frames = []
    total_steps = 0

    x = env.render()

    while not env.finished:
        with torch.no_grad():
            logits, value = agent(x.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        frame_path = os.path.join(frame_dir, f"step_{total_steps:04d}.png")
        env.render_for_human(filename=frame_path)


        # Load immediately into memory
        frames.append(imageio.imread(frame_path))

        _ = env.update(action.item())
        x = env.render()
        total_steps += 1
    print(f"length: {total_steps}")

    # Write GIF
    imageio.mimsave(
        gif_path,
        frames,
        duration=frame_duration,
        loop=0
    )

    # Cleanup images
    shutil.rmtree(frame_dir)

def run_curriculum_stage(
    env, agent, optimizer,
    num_episodes, stats, CE
):
    for episode_idx in range(num_episodes):
        episode_states = []
        for _ in range(LEVELS_PER_EPISODE+1):
            env.initialize()
            env.generate_potentials(
                max_agent_to_key=50,
                max_key_to_closest_apple=50,
                max_minimum_steps_taken_to_get_here=100
            )
            episode_states.append(env.export())

        avg_reward = 0.0
        avg_entropy = 0.0

        # Collect rollouts. REPLAYS has been reduced to 1 for stability/debugging.
        for _ in range(REPLAYS):
            batch = []
            for idx in range(LEVELS_PER_EPISODE):
                env.load(episode_states[idx])
                r, e, steps, logs = run_episode(env, agent)
                batch += logs

                avg_reward += r / (LEVELS_PER_EPISODE * REPLAYS)
                avg_entropy += e / max(1, steps * LEVELS_PER_EPISODE * REPLAYS)


        # Train on the collected batch
        avg_loss = offline_train(batch, agent, optimizer, CE)

        # Periodic visualization (unchanged)
        if episode_idx % 30 == 0 and episode_idx != 0:
            env.load(episode_states[LEVELS_PER_EPISODE])
            stream_episode(env, agent, f"gifs/episode{episode_idx}.gif")

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


def train(CE):
    env = PotentialBased((6, 6), 100, 5)
    agent = Agent(108, 64, 3, 4)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

    stats = defaultdict(list)

    curriculum = [
        ((1, 7), EPISODES_CURR),
    ]

    for stage, episodes in curriculum:
        run_curriculum_stage(
            env, agent, optimizer,
            episodes, stats, CE
        )
    return stats


if __name__ == "__main__":
    plot_stats(train(0.1))

