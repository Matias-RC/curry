import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import csv
import time
import argparse
import matplotlib.pyplot as plt
from env import GridEnv

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(16 * 10 * 10, 128)
        self.policy = nn.Linear(128, 4)
        self.value = nn.Linear(128, 1)

        # UNIFORM POLICY INIT (critical)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(self.fc(x))
        return self.policy(x), self.value(x)

def worker(rank, args, shared_model, optimizer, counter, lock):
    torch.manual_seed(args.seed + rank)
    env = GridEnv(seed=args.seed + rank)

    model = ActorCritic()
    model.load_state_dict(shared_model.state_dict())

    if args.verbose:
        print(f"[worker {rank}] started")

    obs = env.reset()
    obs = torch.from_numpy(obs).unsqueeze(0)

    while True:
        values, log_probs, rewards = [], [], []
        entropy = 0.0

        for _ in range(args.t_max):
            logits, value = model(obs)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            action = dist.sample()
            entropy += dist.entropy().mean()

            next_obs, reward, done = env.step(action.item())
            next_obs = torch.from_numpy(next_obs).unsqueeze(0)

            values.append(value)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            obs = next_obs
            with lock:
                counter.value += 1
                if counter.value >= args.total_frames:
                    return

            if done:
                obs = torch.from_numpy(env.reset()).unsqueeze(0)
                break

        R = torch.zeros(1, 1)
        if not done:
            _, R = model(obs)

        policy_loss = 0
        value_loss = 0

        for i in reversed(range(len(rewards))):
            R = rewards[i] + args.gamma * R
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)
            policy_loss -= log_probs[i] * advantage.detach()

        loss = policy_loss + value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()

        for lp, gp in zip(model.parameters(), shared_model.parameters()):
            gp._grad = lp.grad

        optimizer.step()
        model.load_state_dict(shared_model.state_dict())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--total-frames", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--t-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    shared_model = ActorCritic()
    shared_model.share_memory()

    optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr)

    counter = mp.Value("i", 0)
    lock = mp.Lock()

    procs = []
    for i in range(args.workers):
        p = mp.Process(
            target=worker,
            args=(i, args, shared_model, optimizer, counter, lock),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
