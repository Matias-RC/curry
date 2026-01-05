"""
Basically here we learn from all of the previous mistakes done.
Moving boxes in a sokoban like environment is a very hard task for agents and it has proven to be hard to fix
with reward shaping, gae, potential based rewards, curriculum steps... What we are doing now is a very simple
proof of concept: Check if for 6x6 girds the task of pushing a box into a goal without walls can be learnt si-
-mply by combining all strategies already tried and excluding BFS graph search.

The agent will begin with just one action per environment and right next to the box while the box is right n-
-ext to the goal in the same direction. until a certain studied threshold of uncertainty is crossed for any 
random state.

Then the agent will be two steps away from the box but in the same axis of pushing intelieved with replays of
the earlier stage. again the agent only gets to have two steps. in this stage we do potentials calculations 
with the manhattan distance between the player and the position of proper pushing. while the actual instance
the agent is in the correct push position it changes to manhattan distance of the box to the goal.

It is important to take into coinsideration that the curriculum progresses this way until the agent is N steps
away from a box right next to the goal in a pushable position. and any push of the box by accident that doesn't
result in correct placement is a strong negative signal. also the environment class should be very versatile
to changes because while training hyper param finetuning is key to checking right learning strategy.
"""

from collections import deque, defaultdict
import random
import math
import copy
import time
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import csv
import matplotlib.pyplot as plt

import shutil
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
# -------------------------- Configurable hyperparameters --------------------------
CONFIG = {
    "GRID_SIZE": 6,
    "GAMMA": 0.99,
    "LAMBDA": 0.95,
    "EPSILON": 0.2,
    "VALUE_LOSS_COEF": 0.3,
    "ENTROPY_COEF": 0.06,
    "LEARNING_RATE": 6e-4,

    # Curriculum parameters (tune these)
    "STAGES": 4,
    # Each stage controls how many distance steps agent may start away from canonical push location
    "STAGE_DISTANCE": [0, 1, 2, 3],
    "STAGE_MAX_STEPS": [2, 4, 8, 12],
    "POTENTIAL_ALPHA": 0.25,  # interpolation between box->goal and agent->pushpos

    # Replay pool
    "REPLAY_POOL_CAPACITY": 500,
    "REPLAY_SAMPLE_PROB": 0.3,  # probability to sample from replay pool when creating a new episode

    # Advancement criteria
    "EVAL_EPISODES": 100,
    "ADVANCE_SUCCESS_RATE": 0.9,
    "ADVANCE_ENTROPY_PROPORTION": 0.3, #(CURRENT ENTROPY/MAX ENTROPY)

    # Base rewards
    "R_SUCCESS": 5.0,
    "R_BAD_PUSH": -2.0,
    "R_STEP": -0.01,

    "ANNEAL_HELPER_CHANNEL_FACTOR":0.99,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- Utility functions --------------------------




def clamp_pos(pos, size):
    y, x = pos
    y = max(0, min(size - 1, y))
    x = max(0, min(size - 1, x))
    return (y, x)

# -------------------------- Environment --------------------------

class PushCurriculumEnv:
    """
    Observation returned: a float tensor of shape (4, GRID_SIZE, GRID_SIZE)
      channel 0: agent one-hot
      channel 1: box one-hot
      channel 2: goal one-hot
      channel 3: push pos one-hot

    State is fully observable by default: (agent_pos, box_pos, goal_pos)
    """

    def __init__(self, grid_size: int = CONFIG["GRID_SIZE"], seed: Optional[int] = None,
                 config: Optional[Dict] = None):
        if config is None:
            config = CONFIG
        self.config = config
        self.grid_size = grid_size
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed if seed is not None else int(time.time()))
        self.action_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # State
        self.agent_pos = (0, 0)
        self.box_pos = (0, 0)
        self.goal_pos = (0, 0)
        self.push_pos = (0, 0)
        self.left_steps = 0
        self.max_steps = 10
        self.finished = True

        # Curriculum state
        self.stage = 0
        self.replay_pool = deque(maxlen=self.config["REPLAY_POOL_CAPACITY"])  # stores tuples of initial states
        self.replay_sample_prob = self.config["REPLAY_SAMPLE_PROB"]

        # diagnostics (rolling)
        self.diagnostics = defaultdict(lambda: deque(maxlen=200))

        # internal
        self.last_reward = 0.0
        self.last_phi = 0.0
    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    def in_bounds(self, pos):
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def sample_goal_box(self):
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    
        while True:
            d = self.rng.choice(dirs)
            gy = self.rng.randint(0, 5)
            gx = self.rng.randint(0, 5)
    
            by, bx = gy - d[0], gx - d[1]
            py, px = by - d[0], bx - d[1]
    
            G = (gy, gx)
            B = (by, bx)
            P = (py, px)
    
            if self.in_bounds(B) and self.in_bounds(P):
                return G, B, P, d
            
    def sample_agent_path(self, P, B, N):
        visited = {P}
        path = [P]

        for _ in range(N):
            y, x = path[-1]
            candidates = []

            for dy, dx in self.action_map:
                ny, nx = y + dy, x + dx
                np = (ny, nx)

                if not self.in_bounds(np):
                    continue
                if np in visited:
                    continue
                if np == B:          # forbid touching box
                    continue

                candidates.append(np)

            if not candidates:
                return None  # fail, resample whole env

            nxt = self.rng.choice(candidates)
            visited.add(nxt)
            path.append(nxt)

        return path[::-1]  # start → ... → P
    
    def sample_phase_N(self, N):
        while True:
            G, B, P, d = self.sample_goal_box()
            path = self.sample_agent_path(P, B, N)
            if path is None:
                continue
            
            A = path[0]
            
            return A, B, G, P, d, path

    # ---------------- helpers for canonical push position ----------------
    def _canonical_push_position(self, agent: Tuple[int, int], box: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, int]:
        by, bx = box
        gy, gx = goal
        dy = int(math.copysign(1, gy - by))
        dx = int(math.copysign(1, gx - bx))
        #these are the directions in which to push the box.
        candidates = []
        if dy != 0:
            if self.in_bounds((by-dy,bx)):
                candidates.append((by-dy,bx))
        if dx != 0:
            if self.in_bounds((by,bx-dx)):
                candidates.append((by,bx-dx))
        if not candidates:
            return None
        i, _ = min(enumerate(candidates), key=lambda x:self.manhattan(x[1], agent))

        return candidates[i]



    # ---------------- potential function (state-only) ----------------
    def _compute_phi(self, agent: Tuple[int, int], box: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Potential is deterministic and depends only on the current state (agent, box, goal).
        It interpolates between box->goal distance and agent->pushpos distance using alpha.
        """
        alpha = self.config["POTENTIAL_ALPHA"]
        push_pos = self._canonical_push_position(agent, box, goal)
        if push_pos == None:
            return None
        d_box_goal = self.manhattan(box, goal)
        d_agent_push = self.manhattan(agent, push_pos)
        # Negative distances as potentials (lower is better), so phi higher when closer -> -distance
        phi = -( (1 - alpha) * d_box_goal + alpha * d_agent_push )
        return float(phi)



    # ------------------ public API: reset, step, render ------------------
    def reset(self, use_replay: bool = True):
        """
        Reset environment. If use_replay and replay_pool not empty, sample with probability replay_sample_prob
        from earlier successful initial states to keep replaying easier tasks.
        Returns observation.
        """
        # possibly sample from replay pool
        if use_replay and len(self.replay_pool) > 0 and self.rng.random() < self.replay_sample_prob:
            a, b, g, p = self.rng.choice(list(self.replay_pool))
        else:
            a, b, g, p, push_dir, path = self.sample_phase_N(self.stage)

        self.agent_pos = a
        self.box_pos = b
        self.goal_pos = g
        self.push_pos = p
        self.max_steps = self.config["STAGE_MAX_STEPS"][self.stage]
        self.left_steps = self.max_steps
        self.finished = False

        # For diagnostics
        self.last_reward = 0.0
        self.last_phi = self._compute_phi(self.agent_pos, self.box_pos, self.goal_pos)
        if self.last_phi == None:
            self.last_phi = 0

        return self.render_state()
    
    def render_state(self):
        """
        Returns a (C, H, W) tensor.
        Channels:\\
          0 = agent\\
          1 = box\\
          2 = goal\\
          3 = push position (optional curriculum signal)
        """
        H = W = self.grid_size
        obs = torch.zeros((4, H, W), dtype=torch.float32)

        ay, ax = self.agent_pos
        by, bx = self.box_pos
        gy, gx = self.goal_pos
        py, px = self.push_pos # TODO

        obs[0, ay, ax] = 1.0
        obs[1, by, bx] = 1.0
        obs[2, gy, gx] = 1.0
        obs[3, py, px] = 1.0

        return obs

    def _is_push(self, agent_from, agent_to, box_pos):
        # If agent moves into the box cell -> push
        return agent_to == box_pos

    def _apply_action(self, action: int) -> Tuple[Tuple[int,int], Tuple[int,int], float, bool]:
        """
        Apply deterministic push physics. Returns new (agent, box), reward, done_flag.

        base rewards (unshaped):
          - success (box on goal) => +R_SUCCESS
          - invalid push (blocked) => R_BAD_PUSH
          - step otherwise => R_STEP
        """
        dy, dx = self.action_map[action]
        a_old = self.agent_pos
        b_old = self.box_pos

        a_new = (a_old[0] + dy, a_old[1] + dx)
        # check bounds for agent
        if not (0 <= a_new[0] < self.grid_size and 0 <= a_new[1] < self.grid_size):
            # invalid move off-grid; treat as small penalty and do not move
            return a_old, b_old, self.config["R_BAD_PUSH"], False

        # push if moving into box
        if self._is_push(a_old, a_new, b_old):
            # compute box target
            b_target = (b_old[0] + dy, b_old[1] + dx)
            if not (0 <= b_target[0] < self.grid_size and 0 <= b_target[1] < self.grid_size):
                # blocked by wall: invalid push
                return a_old, b_old, self.config["R_BAD_PUSH"], False
            # valid push -> move box
            a_new_final = b_old  # agent occupies box's previous cell
            b_new = b_target
            # determine if success
            if b_new == self.goal_pos:
                return a_new_final, b_new, self.config["R_SUCCESS"], True
            else:
                # bad/accidental push (box moved to non-goal)
                return a_new_final, b_new, self.config["R_STEP"], False
        else:
            # normal move
            # if step causes box to be on goal already, it's success only if box on goal (it isn't because we didn't push)
            return a_new, b_old, self.config["R_STEP"], False

    def step(self, action: int):
        if self.finished:
            raise RuntimeError("Step called on finished env; call reset() first")

        self.left_steps -= 1
        agent_prev = self.agent_pos
        box_prev = self.box_pos

        # compute phi(s)
        phi_prev = self._compute_phi(agent_prev, box_prev, self.goal_pos)
        if phi_prev == None:
            phi_prev = 0

        a_new, b_new, reward, base_done = self._apply_action(action)

        # assign deterministic next state
        self.agent_pos = a_new
        self.box_pos = b_new

        self.push_pos = self._canonical_push_position(self.agent_pos, self.box_pos, self.goal_pos)
        if self.push_pos == None:
            self.push_pos = self.box_pos
            self.finished = True

        # compute phi(s')
        phi_post = self._compute_phi(self.agent_pos, self.box_pos, self.goal_pos)
        if phi_post == None:
            phi_post = 0
            self.finished = True

        # shaped reward
        shaped_reward = reward + self.config["GAMMA"] * phi_post - phi_prev

        done = base_done or (self.left_steps <= 0)
        if done:
            self.finished = True

        # diagnostics
        self.last_reward = reward
        self.last_phi = phi_post
        #next_obs, r, done, info
        return self.render_state(), shaped_reward, self.finished, None 

    # ---------------- replay and curriculum management ----------------
    def add_to_replay(self, initial_state: Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int], Tuple[int,int]]):
        self.replay_pool.append(initial_state)
    def reset_buffer(self):
        self.replay_pool = []

    def maybe_advance_stage(self, success_rate: float, avg_entropy: float) -> bool:
        """
        Called by trainer after evaluation. If criteria met, advance stage.
        Returns True if advanced.
        """
        if self.stage + 1 >= len(self.config["STAGE_DISTANCE"]):
            return False
        if success_rate >= self.config["ADVANCE_SUCCESS_RATE"] and avg_entropy <= self.config["ADVANCE_ENTROPY_PROPORTION"] * math.log(4):
            self.reset_buffer()
            for _ in range(self.config["REPLAY_POOL_CAPACITY"]):
                self.reset(use_replay=False)
                self.add_to_replay((self.agent_pos, self.box_pos, self.goal_pos, self.push_pos))

            self.stage += 1
            return True
        return False

    def seed(self, s: int):
        self.rng.seed(s)
        self.np_rng.seed(s)

    # small human render (ASCII) for debugging
    def render_ascii(self):
        grid = [["."] * self.grid_size for _ in range(self.grid_size)]
        ay, ax = self.agent_pos
        by, bx = self.box_pos
        gy, gx = self.goal_pos
        grid[gy][gx] = "G"
        grid[by][bx] = "B"
        grid[ay][ax] = "A"
        lines = ["".join(row) for row in grid]
        print("\n".join(lines))

    def render_for_human(self, filename="env_render.png", cell_size=36, show_grid=True, grid_line_width=1):
        # unchanged from your original
        width = self.grid_size * cell_size
        height = self.grid_size * cell_size
        img = Image.new("RGB", (width, height), (255,255,255))
        draw = ImageDraw.Draw(img)
        ay, ax = self.goal_pos
        y0 = ay*cell_size
        x0 = ax*cell_size
        inset = cell_size // 6
        draw.ellipse([x0 + inset, y0 + inset, x0 + cell_size - inset - 1, y0 + cell_size - inset - 1], fill=(0,255,0))
        py, px = self.agent_pos
        x0 = px * cell_size
        y0 = py * cell_size
        inset = cell_size // 8
        draw.polygon([(x0+(cell_size)/2, y0 + inset),(x0+inset, y0 + cell_size - inset - 1), (x0+cell_size-inset,y0 + cell_size - inset - 1) ], fill=(255,0,0))
        ky, kx = self.box_pos
        x0 = kx * cell_size
        y0 = ky * cell_size
        inset = cell_size // 8
        draw.rectangle([x0+inset, y0+inset, x0 + cell_size - inset, y0 + cell_size - inset], fill=(255,255, 0))
        if show_grid:
            for cx in range(self.grid_size + 1):
                x = cx * cell_size
                draw.line([(x, 0), (x, height)], fill=(150,150,150), width=grid_line_width)
            for cy in range(self.grid_size + 1):
                y = cy * cell_size
                draw.line([(0, y), (width, y)], fill=(150,150,150), width=grid_line_width)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename)
        return filename


# -------------------------- Simple policy / value network --------------------------

class ConvActorCritic(nn.Module):
    def __init__(self, grid_size=6, hidden=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, hidden),
            nn.ReLU(),
        )

        self.policy = nn.Linear(hidden, 4)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self.conv(x)
        h = self.fc(h)
        return self.policy(h), self.value(h)

# -------------------------- Rollout, GAE, Offline PPO training --------------------------

GAMMA = CONFIG["GAMMA"]
LAMBDA = CONFIG["LAMBDA"]
EPSILON = CONFIG["EPSILON"]
VALUE_LOSS_COEF = CONFIG["VALUE_LOSS_COEF"]
ENTROPY_COEF = CONFIG["ENTROPY_COEF"]


def run_episode(env: PushCurriculumEnv, agent: ConvActorCritic, device=DEVICE):
    """
    Run a single episode (on-policy) collecting per-step quantities needed for offline update.
    Returns logs list: entries are (state_tensor, action, next_state_tensor, reward, done, logp_old, value)
    Also returns totals for reward, entropy, steps.
    """
    logs = []
    total_reward = 0.0
    total_entropy = 0.0
    total_steps = 0

    obs = env.reset()
    while not env.finished:
        with torch.no_grad():
            logits, value = agent(obs.to(device))
            logits = logits.squeeze(0)
            value = value.squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp_old = dist.log_prob(action).detach()
            entropy = dist.entropy().mean().item()
        
        next_obs, r, done, info = env.step(int(action.item()))

        logs.append((obs.clone(), int(action.item()), next_obs.clone(), float(r), bool(done), logp_old.clone(), value.clone()))

        obs = next_obs
        total_reward += r
        total_entropy += entropy
        total_steps += 1

    # compute GAE and append advantage to each log entry
    T = len(logs)
    gae = 0.0
    for t in reversed(range(T)):
        s, a, sp, r, done, logp, V = logs[t]
        if t == T - 1:
            V_tp1 = 0.0
        else:
            V_tp1 = logs[t + 1][6]
        delta = r + GAMMA * float(V_tp1) - float(V)
        gae = delta + GAMMA * LAMBDA * gae
        logs[t] = logs[t] + (gae,)

    return total_reward, total_entropy, total_steps, logs


def offline_train(batch, agent: ConvActorCritic, optimizer: torch.optim.Optimizer, device=DEVICE):
    """
    Perform one epoch of PPO-style offline update on the provided batch (list of tuples like run_episode logs).
    Batch is a list of per-step entries (s, a, sp, r, done, logp_old, V_old, adv)
    """
    agent.train()
    device = device

    S = torch.stack([b[0] for b in batch]).to(device)
    A = torch.tensor([b[1] for b in batch], device=device)
    Logp_old = torch.stack([b[5] for b in batch]).to(device)
    V_old = torch.stack([b[6] for b in batch]).squeeze().to(device)
    GAE_raw = torch.tensor([b[7] for b in batch], device=device, dtype=torch.float)

    # Normalize advantages for policy update
    Adv = (GAE_raw - GAE_raw.mean()) / (GAE_raw.std() + 1e-8)
    Adv_detached = Adv.detach()

    logits, V_new = agent(S)
    V_new = V_new.squeeze()
    dist = torch.distributions.Categorical(logits=logits)
    logp_new = dist.log_prob(A)
    entropy = dist.entropy().mean()

    ratios = torch.exp(logp_new - Logp_old)
    unclipped = ratios * Adv_detached
    clipped = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON) * Adv_detached
    loss_policy = -torch.min(unclipped, clipped).mean()

    # Value target using RAW GAE (as in your reference trainer)
    V_target = V_old + GAE_raw
    loss_value = VALUE_LOSS_COEF * F.mse_loss(V_new, V_target.detach())

    loss_entropy = -ENTROPY_COEF * entropy
    loss = loss_policy + loss_value + loss_entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_policy.item(), loss_value.item(), entropy.item()

# -------------------------- Small training loop and evaluation --------------------------

def evaluate_policy(env: PushCurriculumEnv, agent: ConvActorCritic, episodes: int = 50, device=DEVICE):
    agent.eval()
    successes = 0
    total_entropy = 0.0
    for _ in range(episodes):
        obs = env.reset(use_replay=False)
        while not env.finished:
            with torch.no_grad():
                logits, _ = agent(obs.to(device))
                logits = logits.squeeze(0)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                total_entropy += dist.entropy().item()
            _, r, done, _ = env.step(int(action.item()))
        # success if box on goal
        if env.box_pos == env.goal_pos:
            successes += 1
    avg_entropy = total_entropy / max(1, episodes)
    return successes / max(1, episodes), avg_entropy

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_stats_plot(stats, stage: int, out_dir="runs/graphs"):
    ensure_dir(out_dir)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(stats["reward"])
    plt.title("Average Reward")

    plt.subplot(1, 3, 2)
    plt.plot(stats["loss"])
    plt.title("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(stats["entropy"])
    plt.title("Entropy")

    plt.tight_layout()
    path = os.path.join(out_dir, f"stage_{stage:03d}.png")
    plt.savefig(path)
    plt.close()

def save_stats_csv(stats, stage: int, out_dir="runs/logs"):
    ensure_dir(out_dir)

    path = os.path.join(out_dir, f"stage_{stage:03d}.csv")

    keys = list(stats.keys())
    rows = zip(*(stats[k] for k in keys))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)
def stream_episode(
    env:PushCurriculumEnv,
    agent,
    gif_path="episode.gif",
    total_duration_sec=8,
    frame_dir="images"
):
    os.makedirs(frame_dir, exist_ok=True)

    fps = 4 / total_duration_sec
    frame_duration = 1.0 / fps

    frames = []
    total_steps = 0

    env.reset(False)

    x = env.render_state()

    while not env.finished:
        with torch.no_grad():
            logits, value = agent(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        frame_path = os.path.join(frame_dir, f"step_{total_steps:04d}.png")
        env.render_for_human(filename=frame_path)

        # Load immediately into memory
        frames.append(imageio.imread(frame_path))

        _ = env.step(action.item())
        x = env.render_state()
        total_steps += 1
    env.render_for_human(filename=frame_path)
    # Load immediately into memory
    frames.append(imageio.imread(frame_path))
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

def training_demo(num_epochs: int = 20,
                        episodes_per_epoch: int = 40,
                        levels_per_episode: int = 6,
                        device=DEVICE):
    """
    Tiny training demo using the environment + offline PPO. This is intentionally small for a quick demo.
    """
    env = PushCurriculumEnv()
    agent = ConvActorCritic().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=CONFIG["LEARNING_RATE"])

    stats = defaultdict(list)

    curriculum_stage = 0

    for epoch in range(num_epochs):
        # collect episodes and train
        epoch_batch = []
        avg_reward = 0.0
        avg_entropy = 0.0
        total_steps = 0

        for _ in range(episodes_per_epoch):
            # create multiple levels (different initial states) per episode and aggregate logs
            env.initialize_replay_sample_prob = env.replay_sample_prob
            env.reset()
            r, e, steps, logs = run_episode(env, agent, device=device)
            avg_reward += r / episodes_per_epoch
            avg_entropy += e / (max(1, steps) * episodes_per_epoch)
            total_steps += steps
            epoch_batch += logs

        # shuffle and train on collected transitions
        random.shuffle(epoch_batch)
        # sample subset if too large
        train_batch = epoch_batch[:1024]
        loss_info = offline_train(train_batch, agent, optimizer, device=device)
        stats["reward"].append(avg_reward)
        stats["loss"].append(loss_info[0])
        stats["entropy"].append(avg_entropy)
        # periodic evaluation
        if epoch % 10 == 0:
            succ_rate, avg_eval_entropy = evaluate_policy(env, agent, episodes=CONFIG["EVAL_EPISODES"], device=device)
            advanced = env.maybe_advance_stage(succ_rate, avg_eval_entropy)
            if advanced:
                save_stats_plot(stats, curriculum_stage)
                save_stats_csv(stats, curriculum_stage)

                curriculum_stage += 1
                stats = defaultdict(list)
                if curriculum_stage == 5:
                    break
            print(f"Epoch {epoch:3d} | reward {avg_reward:6.3f} | loss {loss_info[0]:.4f} | succ {succ_rate:.3f} | adv:{advanced}")
            if epoch % 100 == 0:
                ensure_dir("gifs")
                stream_episode(env, agent, f"gifs/epoch{epoch}.gif")
        else:
            print(f"Epoch {epoch:3d} | reward {avg_reward:6.3f} | loss {loss_info[0]:.4f}")

    return env, agent



if __name__ == "__main__":

    env, agent = training_demo(num_epochs=3000, episodes_per_epoch=200)


    
