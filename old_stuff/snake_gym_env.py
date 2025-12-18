"""
Self-contained Gymnasium Snake environment + simple PyTorch DQN agent example.

Save this file and run it, or import SnakeEnv from it.

Features:
- Gymnasium-compatible Env (observation_space, action_space, reset, step, render, seed)
- Observation: (H, W, C) float32 array where channels = [empty, snake, food]
- Action space: Discrete(4) -> 0:UP,1:RIGHT,2:DOWN,3:LEFT
- Simple reward: +1 for food, -1 for death, -0.01 per step to incentivize eating.
- Optional sprite rendering if you provide sprite paths in sprite_paths dict.
- Minimal DQN agent implementation (PyTorch optional).

This file contains a `if __name__ == "__main__":` smoke test that runs a few random steps and
prints the outputs.

"""

from typing import Optional, Tuple, Dict, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from PIL import Image, ImageDraw
import os

# Optional import for training agent. It's guarded so environment works even without torch.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

using_gymnasium = False
try:
    import gymnasium as gym
    EnvBase = gym.Env
    using_gymnasium = True
except Exception:
    import gym
    EnvBase = gym.Env
    using_gymnasium = False

def one_hot_grid(grid_size: int, snake: List[Tuple[int, int]], food: Tuple[int, int]) -> np.ndarray:
    """Return observation channels: empty, snake, food"""
    g = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    # empty is implicit; we set explicit channels for snake and food
    for (y, x) in snake:
        if 0 <= y < grid_size and 0 <= x < grid_size:
            g[y, x, 1] = 1.0
    fy, fx = food
    g[fy, fx, 2] = 1.0
    g[:, :, 0] = 1.0 - (g[:, :, 1] + g[:, :, 2])
    return g


class SnakeEnv(EnvBase):
    """A simple Snake environment for Gymnasium.

    Observation: Box(low=0,high=1,shape=(grid,grid,3),dtype=float32)
    Actions: Discrete(4) -> up(0), right(1), down(2), left(3)

    Notes:
    - The snake starts length 3 in the center moving right.
    - Food spawns uniformly on empty cells.
    - Colliding with wall or self -> terminal state with reward -1.
    - Eating food -> reward +1 and grows by 1.
    - Per-step small negative reward to encourage short solutions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 10,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        sprite_paths: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert grid_size >= 5, "grid_size should be >=5"
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_steps = max_steps if max_steps is not None else grid_size * grid_size * 2
        self.sprite_paths = sprite_paths or {}

        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)
        # Observation: channels for empty, snake, food
        self.observation_space = spaces.Box(0.0, 1.0, shape=(grid_size, grid_size, 3), dtype=np.float32)

        # internal state
        self.snake: List[Tuple[int, int]] = []
        self.direction = 1  # start moving right
        self.food: Tuple[int, int] = (0, 0)
        self.steps = 0
        self.terminated = False
        self.truncated = False

        # rendering cache
        self._sprites = {}
        self._load_sprites()

        self._rng = random.Random() if seed is None else random.Random(seed)
        self.np_random = np.random.RandomState(seed)

    def _load_sprites(self):
        # Attempt to load sprites (wall, snake_body, snake_head, food), they are optional
        for key in ["wall", "snake_body", "snake_head", "food"]:
            p = self.sprite_paths.get(key)
            if p and os.path.isfile(p):
                try:
                    self._sprites[key] = Image.open(p).convert("RGBA")
                except Exception:
                    self._sprites[key] = None
            else:
                self._sprites[key] = None

    def seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.np_random = np.random.RandomState(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        # center start
        c = self.grid_size // 2
        self.snake = [(c, c - 1), (c, c), (c, c + 1)]  # head is last element; initially moving right
        self.direction = 1
        self._place_food()
        self.steps = 0
        self.terminated = False
        self.truncated = False
        obs = one_hot_grid(self.grid_size, self.snake, self.food)
        info = {}
        obs = one_hot_grid(self.grid_size, self.snake, self.food)
        info = {}
        if using_gymnasium:
            return obs, info           # gymnasium API
        else:
            return obs 

    def _place_food(self):
        empties = [(y, x) for y in range(self.grid_size) for x in range(self.grid_size) if (y, x) not in self.snake]
        if not empties:
            # no space left: place food off-grid (game effectively finished)
            self.food = (0, 0)
            return
        self.food = self._rng.choice(empties)

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        info = {}

        # If env already ended, return immediately (user must call reset first)
        if self.terminated or self.truncated:
            obs = one_hot_grid(self.grid_size, self.snake, self.food)
            if using_gymnasium:
                return obs, 0.0, True, False, info
            else:
                # old gym: obs, reward, done, info
                return obs, 0.0, True, info

        # disallow 180-degree turns: map actions to directions
        # 0:UP,1:RIGHT,2:DOWN,3:LEFT
        opposite = {0:2, 1:3, 2:0, 3:1}
        if action == opposite[self.direction]:
            # ignore and keep current direction
            action = self.direction
        else:
            self.direction = action

        head_y, head_x = self.snake[-1]
        if self.direction == 0:
            head_y -= 1
        elif self.direction == 1:
            head_x += 1
        elif self.direction == 2:
            head_y += 1
        elif self.direction == 3:
            head_x -= 1

        self.steps += 1

        # check wall collision
        if not (0 <= head_y < self.grid_size and 0 <= head_x < self.grid_size):
            reward = -1.0
            self.terminated = True
            obs = one_hot_grid(self.grid_size, self.snake, self.food)
            if using_gymnasium:
                return obs, float(reward), True, False, info
            else:
                return obs, float(reward), True, info

        new_head = (head_y, head_x)
        # check self collision (allow the tail cell because it moves unless we grow)
        tail = self.snake[0]
        will_grow = new_head == self.food
        if new_head in self.snake and (not will_grow or new_head != tail):
            reward = -1.0
            self.terminated = True
            obs = one_hot_grid(self.grid_size, self.snake, self.food)
            if using_gymnasium:
                return obs, float(reward), True, False, info
            else:
                return obs, float(reward), True, info

        # move
        self.snake.append(new_head)
        if will_grow:
            reward = 1.0
            self._place_food()
        else:
            reward = -0.01
            self.snake.pop(0)

        # truncated by step limit
        if self.steps >= self.max_steps:
            self.truncated = True
            obs = one_hot_grid(self.grid_size, self.snake, self.food)
            if using_gymnasium:
                return obs, float(reward), False, True, info
            else:
                # old gym expects done True when truncated
                return obs, float(reward), True, info

        obs = one_hot_grid(self.grid_size, self.snake, self.food)
        if using_gymnasium:
            return obs, float(reward), False, False, info
        else:
            return obs, float(reward), False, info


    def _render_np(self, cell_px: int = 32) -> np.ndarray:
        # Build an RGB image using sprites if provided, else draw colored rectangles.
        W = self.grid_size * cell_px
        H = self.grid_size * cell_px
        img = Image.new("RGB", (W, H), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)

        # draw grid background
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_box = (x * cell_px, y * cell_px, (x + 1) * cell_px, (y + 1) * cell_px)
                draw.rectangle(cell_box, outline=(60, 60, 60))

        # draw food
        fy, fx = self.food
        if self._sprites.get("food"):
            sprite = self._sprites["food"].resize((cell_px, cell_px), Image.NEAREST)
            img.paste(sprite, (fx * cell_px, fy * cell_px), sprite)
        else:
            draw.ellipse((fx * cell_px + 4, fy * cell_px + 4, (fx + 1) * cell_px - 4, (fy + 1) * cell_px - 4), fill=(255, 80, 80))

        # draw snake body
        for (y, x) in self.snake[:-1]:
            if self._sprites.get("snake_body"):
                sprite = self._sprites["snake_body"].resize((cell_px, cell_px), Image.NEAREST)
                img.paste(sprite, (x * cell_px, y * cell_px), sprite)
            else:
                draw.rectangle((x * cell_px + 2, y * cell_px + 2, (x + 1) * cell_px - 2, (y + 1) * cell_px - 2), fill=(80, 200, 80))

        # draw head
        hy, hx = self.snake[-1]
        if self._sprites.get("snake_head"):
            sprite = self._sprites["snake_head"].resize((cell_px, cell_px), Image.NEAREST)
            img.paste(sprite, (hx * cell_px, hy * cell_px), sprite)
        else:
            draw.rectangle((hx * cell_px + 2, hy * cell_px + 2, (hx + 1) * cell_px - 2, (hy + 1) * cell_px - 2), fill=(40, 160, 40))

        return np.array(img)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_np()
        elif self.render_mode == "human":
            img = self._render_np()
            # use PIL to display
            Image.fromarray(img).show()
            return None
        else:
            # return observation by default
            return one_hot_grid(self.grid_size, self.snake, self.food)

    def close(self):
        return None


# -----------------------------
# Minimal DQN agent implementation (optional, requires PyTorch)
# -----------------------------

if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden: List[int] = [128, 64]):
            super().__init__()
            layers = []
            last = input_dim
            for h in hidden:
                layers.append(nn.Linear(last, h))
                layers.append(nn.ReLU())
                last = h
            layers.append(nn.Linear(last, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    class ReplayBuffer:
        def __init__(self, capacity: int = 10000):
            self.capacity = capacity
            self.buffer = []
            self.pos = 0

        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.pos = (self.pos + 1) % self.capacity

        def sample(self, batch_size: int):
            batch = random.sample(self.buffer, batch_size)
            s, a, r, ns, d = map(np.stack, zip(*batch))
            return s, a, r, ns, d

        def __len__(self):
            return len(self.buffer)


    class DQNAgent:
        def __init__(self, env: SnakeEnv, lr: float = 1e-3, gamma: float = 0.99, buffer_capacity: int = 5000):
            obs_shape = env.observation_space.shape
            input_dim = int(np.prod(obs_shape))
            output_dim = env.action_space.n
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = MLP(input_dim, output_dim).to(self.device)
            self.target = MLP(input_dim, output_dim).to(self.device)
            self.target.load_state_dict(self.net.state_dict())
            self.opt = optim.Adam(self.net.parameters(), lr=lr)
            self.gamma = gamma
            self.replay = ReplayBuffer(capacity=buffer_capacity)
            self.batch_size = 64
            self.update_target_every = 100
            self.steps = 0

        def select_action(self, obs, epsilon: float):
            if random.random() < epsilon:
                return random.randrange(0, 4)
            x = torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q = self.net(x)
            return int(q.argmax().cpu().numpy()[0])

        def store(self, s, a, r, ns, done):
            self.replay.push(s.flatten(), a, r, ns.flatten(), float(done))

        def update(self):
            if len(self.replay) < self.batch_size:
                return
            s, a, r, ns, d = self.replay.sample(self.batch_size)
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
            a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
            r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
            ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
            d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

            q_values = self.net(s).gather(1, a)
            with torch.no_grad():
                q_next = self.target(ns).max(1)[0].unsqueeze(1)
                q_target = r + (1.0 - d) * self.gamma * q_next

            loss = nn.functional.mse_loss(q_values, q_target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self.steps += 1
            if self.steps % self.update_target_every == 0:
                self.target.load_state_dict(self.net.state_dict())


    def train_dqn(env: SnakeEnv, episodes: int = 300, max_steps_per_ep: int = 500):
        agent = DQNAgent(env)
        eps_start = 1.0
        eps_end = 0.05
        eps_decay = 0.995
        epsilon = eps_start
        returns = []
        for ep in range(episodes):
            obs, _ = env.reset()
            total = 0.0
            for t in range(max_steps_per_ep):
                a = agent.select_action(obs, epsilon)
                ns, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                agent.store(obs, a, r, ns, done)
                agent.update()
                obs = ns
                total += r
                if done:
                    break
            returns.append(total)
            epsilon = max(eps_end, epsilon * eps_decay)
            if (ep + 1) % 10 == 0:
                avg = sum(returns[-10:]) / min(10, len(returns))
                print(f"Episode {ep+1}/{episodes} avg return (last10): {avg:.3f} eps={epsilon:.3f}")
        return agent, returns


# If torch not available, we still keep env usable.

if __name__ == "__main__":
    import os
    from PIL import Image

    # Determine folder where the script lives. If __file__ is not available (interactive),
    # fall back to current working directory.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not script_dir:
            script_dir = os.getcwd()
    except NameError:
        script_dir = os.getcwd()

    print("Saving debug frames to folder:", script_dir)

    env = SnakeEnv(grid_size=10, render_mode="rgb_array")
    # reset may return (obs, info) for gymnasium or obs for old gym
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs = reset_out
        info = {}

    print("Initial obs shape:", getattr(obs, "shape", None))

    for i in range(6):
        a = env.action_space.sample()

        step_out = env.step(a)
        # gymnasium: (obs, reward, terminated, truncated, info)
        # old gym:   (obs, reward, done, info)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, r, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        elif isinstance(step_out, tuple) and len(step_out) == 4:
            obs, r, done, info = step_out
        else:
            # best-effort unpack
            try:
                obs, r = step_out[0], step_out[1]
                done = False
            except Exception:
                raise RuntimeError("Unexpected return from env.step()")

        print(f"Step {i}: action={a}, reward={r}, done={done}")
        img = env.render()

        # save image to the same directory the script is running from
        fname = os.path.join(script_dir, f"snake_step_{i}.png")
        Image.fromarray(img).save(fname)
        print("Saved:", fname)

        if done:
            break

    print("Smoke test completed. Images saved to:", script_dir)

