# How to adapt `snake_gym_env.py` into new grid-based game environments — developer tutorial

Below is a single cohesive Markdown guide (no split blocks) on how to modify and extend your Snake environment to create new Gymnasium-compatible games. It includes all adaptation instructions, best practices, and debugging tips.

---

## Overview

Your `snake_gym_env.py` defines a minimal RL-ready environment implementing the Gymnasium API:

- `__init__` sets up grid size, sprites, RNG, and spaces.
- `reset` initializes the grid and entities.
- `step` updates the state, computes rewards, and handles termination.
- `render` visualizes the grid.

To adapt it, you’ll mainly modify the **entities**, **reward logic**, and **action/observation spaces**.

---

## Steps to Adapt

### 1. Rename the Environment
Rename the class and ID if you’re making a new game:
```python
class PacmanEnv(SnakeEnv):
    ...
```
And at the bottom:
```python
if __name__ == "__main__":
    env = PacmanEnv(grid_size=10, render_mode="rgb_array")
```

### 2. Update Action Space
```python
self.action_space = spaces.Discrete(4)  # or another number of actions
```
For continuous control:
```python
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

### 3. Update Observation Space
Snake uses a one-hot grid. You can instead return flattened vectors or images:
```python
self.observation_space = spaces.Box(0.0, 1.0, shape=(grid_size * grid_size * 3,), dtype=np.float32)
```
or for grayscale:
```python
self.observation_space = spaces.Box(0, 255, shape=(grid_size, grid_size, 1), dtype=np.uint8)
```

### 4. Redefine `reset`
Replace snake and food with entities relevant to your new game (player, items, walls, etc.). Example:
```python
self.player = (5, 5)
self.enemies = [(2, 2), (7, 8)]
self.items = [(3, 3)]
```
Return a new observation that encodes them.

### 5. Redefine `step`
Replace movement, collision, and reward logic:
```python
def step(self, action):
    y, x = self.player
    if action == 0: y -= 1
    elif action == 1: x += 1
    elif action == 2: y += 1
    elif action == 3: x -= 1

    self.player = (y, x)
    reward = 0.0
    if self.player in self.items:
        reward += 1.0
        self.items.remove(self.player)

    done = len(self.items) == 0
    obs = self._get_obs()
    return obs, reward, done, False, {}
```

### 6. Redefine `render`
Update `_render_np()` to draw your entities:
```python
def _render_np(self):
    img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
    for y, x in self.items:
        img[y, x] = (255, 255, 0)
    py, px = self.player
    img[py, px] = (0, 255, 0)
    return img
```

### 7. Save Rendered Frames in Current Folder
Replace the `__main__` test block’s save section:
```python
from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

for i in range(6):
    a = env.action_space.sample()
    obs, r, terminated, truncated, info = env.step(a)
    img = env.render()
    fname = os.path.join(script_dir, f"snake_step_{i}.png")
    Image.fromarray(img).save(fname)
    print("Saved:", fname)
```

---

## Reward Shaping Tips
- Use `+1` for good events, `-1` for failure, and small negative step penalties (e.g. `-0.01`) to encourage efficiency.
- Avoid large sparse rewards at first; test reward propagation with random policies.

---

## DQN Integration
Your file includes a simple DQN. If using it:
```python
from snake_gym_env import SnakeEnv, train_dqn

env = SnakeEnv(grid_size=10, render_mode='rgb_array')
agent, rewards = train_dqn(env, episodes=100)
```
If you switch to flattened observations, ensure the DQN’s `obs_dim` matches:
```python
obs_dim = env.observation_space.shape[0]
```

---

## Common Pitfalls
- **Observation shape mismatch**: always verify with `assert env.observation_space.contains(obs)`.
- **180° turn bug**: if using directional control, restrict opposite moves.
- **Gym vs Gymnasium**: standardize imports, avoid mixing both.
- **No render output**: ensure `render_mode='rgb_array'` is set and `_render_np` returns a `uint8` array.

---

## Optional: Register the Env
For clean usage with `gym.make()`:
```python
from gymnasium.envs.registration import register
register(id='MySnake-v0', entry_point='snake_gym_env:SnakeEnv')
```
Then:
```python
import gymnasium as gym
env = gym.make('MySnake-v0', render_mode='rgb_array')
```

---

## Extending Ideas
- **Obstacle Maze**: add static obstacles and maze generation.
- **Multi-Entity Games**: add moving enemies or allies.
- **Partial Observability**: crop the grid around the agent.
- **Dynamic Goals**: moving food or time-dependent rewards.

---

## Debugging
- Test short random rollouts.
- Visualize intermediate frames.
- Print rewards and termination flags.
- Use `env.seed(0)` for deterministic debugging.

---

This file can now serve as a base for any custom Gymnasium environment — swap the `step`, `reset`, and rendering logic to build new grid-based RL games from scratch.

