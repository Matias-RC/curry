"""
SokobanEnv - a minimal Gymnasium-compatible Sokoban-like environment.

Notes:
- Observation: RGB image as uint8, shape = (grid_size, grid_size, 3). Default grid_size=20.
  This is the input you would feed the agent directly (no extra channels).
- Actions: Discrete(4): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
- Rewards: small step penalty (-0.01). +1.0 when a box is pushed onto a goal for the first time.
- Termination: when all boxes are on goals (solved) OR when max_steps reached (truncated).
- Sokoban logic is intentionally shallow here; see TODO hooks where you should plug
  in your extensive code for full rules, deadlock detection, undo stacks, complex levels, etc.

Author: generated for you (based on your Snake env)
"""

from typing import Optional, Tuple, Set, List, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw
import random
import os

# --- colors used in the 20x20 RGB representation (agent input) ---
COLOR_FLOOR = (40, 40, 40)      # empty cell
COLOR_WALL = (100, 100, 100)    # wall
COLOR_BOX = (160, 100, 40)      # box
COLOR_GOAL = (220, 200, 40)     # goal
COLOR_BOX_ON_GOAL = (90, 200, 100)  # box sitting on goal
COLOR_PLAYER = (50, 150, 220)   # player
COLOR_PLAYER_ON_GOAL = (180, 220, 250)

class SokobanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 20,
        render_mode: Optional[str] = "rgb_array",
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        init_type: Optional[str] = "str",
        level_map: Optional[List[str]] = None,  # simple ASCII-level, optional
    ):
        super().__init__()
        assert grid_size >= 5, "grid_size should be >= 5"
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_steps = max_steps if max_steps is not None else grid_size * grid_size * 10
        self.rng = random.Random(seed)
        self.np_random = np.random.RandomState(seed)

        # Actions: up, right, down, left
        self.action_space = spaces.Discrete(4)
        # Observations: RGB uint8 grid the agent consumes (shape: H,W,3)
        self.observation_space = spaces.Box(0, 255, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # internal logical state
        self.walls: Set[Tuple[int,int]] = set()
        self.goals: Set[Tuple[int,int]] = set()
        self.boxes: Set[Tuple[int,int]] = set()
        self.player: Tuple[int,int] = (0,0)

        self.steps = 0
        self.terminated = False
        self.truncated = False

        # allow providing a small ASCII map to initialize a level
        self.level_map = level_map
        self.init_type = init_type

        # Prepare initial layout
        self.reset()

    def deadlock(self):
        rotatePattern = [[0,1,2,3,4,5,6,7,8],
                        [2,5,8,1,4,7,0,3,6],
                        [0,1,2,3,4,5,6,7,8][::-1],
                        [2,5,8,1,4,7,0,3,6][::-1]]
        flipPattern = [[2,1,0,5,4,3,8,7,6],
                        [0,3,6,1,4,7,2,5,8],
                        [2,1,0,5,4,3,8,7,6][::-1],
                        [0,3,6,1,4,7,2,5,8][::-1]]
        allPattern = rotatePattern + flipPattern

        for box in self.boxes:
            if box not in self.goals:
                board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                        (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1),
                        (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
                for pattern in allPattern:
                    newBoard = [board[i] for i in pattern]
                    if newBoard[1] in self.walls and newBoard[5] in self.walls: return True
                    elif newBoard[1] in self.boxes and newBoard[2] in self.walls and newBoard[5] in self.walls: return True
                    elif newBoard[1] in self.boxes and newBoard[2] in self.walls and newBoard[5] in self.boxes: return True
                    elif newBoard[1] in self.boxes and newBoard[2] in self.boxes and newBoard[5] in self.boxes: return True
                    elif newBoard[1] in self.boxes and newBoard[6] in self.boxes and newBoard[2] in self.walls and newBoard[3] in self.walls and newBoard[8] in self.walls: return True
        return False
    # -------------------------
    # Level initialization
    # -------------------------
    def _initialize_level_state(self, init_type="str", level_map: Optional[List[str]] = None):
        """
        Initialize walls, boxes, goals and player.
        Always uses a 20x20 grid. If the provided level_map is smaller, it is centered.
        If larger than 20x20, raises NotImplementedError.
        """
        self.walls.clear()
        self.goals.clear()
        self.boxes.clear()

        if level_map is not None and init_type == "str":
            height = len(level_map)
            width = max(len(row) for row in level_map) if height > 0 else 0

            # Reject oversized maps
            if height > self.grid_size or width > self.grid_size:
                raise NotImplementedError("Level larger than grid size not supported yet.")

            # Compute offsets to center the map
            y_offset = (self.grid_size - height) // 2
            x_offset = (self.grid_size - width) // 2

            self.player = None

            # Parse and center the map
            for ry, row in enumerate(level_map):
                for rx, ch in enumerate(row):
                    y = y_offset + ry
                    x = x_offset + rx
                    if ch == '#':
                        self.walls.add((y, x))
                    elif ch == 'G':
                        self.goals.add((y, x))
                    elif ch == '$':
                        self.boxes.add((y, x))
                    elif ch == '@':
                        self.player = (y, x)
                    elif ch == "*":
                        self.boxes.add((y, x))
                        self.goals.add((y, x))
                    elif ch == "+":
                        self.player = (y, x)
                        self.goals.add((y, x))
                    # Other chars like '.' or ' ' are floor/empty, ignore them

            # Basic safety: ensure a player exists
            if self.player is None:
                self.player = (self.grid_size // 2, self.grid_size // 2)
                while self.player in self.walls:
                    self.player = (self.player[0] + 1, self.player[1] + 1)
                    if self.player[0] >= self.grid_size or self.player[1] >= self.grid_size:
                        self.player = (self.grid_size // 2, self.grid_size // 2)
                        break

        elif level_map is not None and init_type == "matrix":
            h, w = level_map.shape
            if h > self.grid_size or w > self.grid_size:
                raise NotImplementedError("Matrix level larger than 20x20 not supported yet.")

            y_offset = (self.grid_size - h) // 2
            x_offset = (self.grid_size - w) // 2

            self.player = tuple(np.argwhere((level_map == 2) | (level_map == 6))[0])
            self.player = (self.player[0] + y_offset, self.player[1] + x_offset)

            self.boxes = {tuple((y + y_offset, x + x_offset)) for y, x in np.argwhere((level_map == 3) | (level_map == 5))}
            self.walls = {tuple((y + y_offset, x + x_offset)) for y, x in np.argwhere(level_map == 1)}
            self.goals = {tuple((y + y_offset, x + x_offset)) for y, x in np.argwhere((level_map == 4) | (level_map == 5) | (level_map == 6))}

            if self.boxes == self.goals:
                pass  # TODO: handle special cases
        else:
            raise NotImplementedError("init_type must be 'str' or 'matrix' with a valid level_map.")


    def _apply_action_and_update_state(self, action: int) -> Tuple[float, Dict]:
        """
        Apply action and update self.player, self.boxes.
        Shallow push logic:
          - if target cell is wall -> no-op (player stays)
          - if target cell is box:
              - if cell beyond is free (not wall, not box, in-bounds) -> push box into beyond and move player into target
          - else -> no-op
      - else move player into target
        Returns:
          reward (float), info (dict)
        NOTE: This purposely omits advanced Sokoban rules:
          - deadlock detection, undo stack, pushes of multiple boxes (not allowed),
          - box sliding, sokoban-level move counters, advanced reward shaping for subgoals.
        """
        info = {}
        reward = -0.01  # small step penalty by default

        dy_dx = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
        if action not in dy_dx:
            action = 0
        dy, dx = dy_dx[action]
        py, px = self.player
        ty, tx = py + dy, px + dx

        # out of bounds -> treat as wall
        if not (0 <= ty < self.grid_size and 0 <= tx < self.grid_size):
            return reward, info

        if (ty,tx) in self.walls:
            # bump into wall: no movement
            return reward, info

        if (ty,tx) in self.boxes:
            # attempt to push
            by, bx = ty + dy, tx + dx
            # check push destination
            if not (0 <= by < self.grid_size and 0 <= bx < self.grid_size):
                return reward, info
            if (by,bx) in self.walls or (by,bx) in self.boxes:
                # cannot push
                return reward, info

            # perform push: move box, move player
            old_box_pos = (ty,tx)
            new_box_pos = (by,bx)

            # Reward for placing box onto a goal (only when box newly reaches a goal cell)
            was_on_goal_before = old_box_pos in self.goals
            is_on_goal_now = new_box_pos in self.goals

            # update box set
            self.boxes.remove(old_box_pos)
            self.boxes.add(new_box_pos)
            # move player into the old box's cell
            self.player = (ty,tx)

            if (not was_on_goal_before) and is_on_goal_now:
                reward += 1.0
            

            if self.deadlock():
                reward -= 3.0
            # If a box was pushed off a goal, we could penalize; omitted by default.
            # TODO: plug in your more nuanced reward shaping or deadlock checks here.
            return reward, info
        else:
            # free cell or goal cell: move player
            self.player = (ty, tx)
            return reward, info

    # -------------------------
    # Utility / observation / rendering
    # -------------------------
    def _is_solved(self) -> bool:
        """Return True when all boxes are on goals. If #boxes != #goals, this returns False."""
        # shallow: require that every box is located on some goal
        if len(self.boxes) == 0:
            return False
        return self.boxes == self.goals

    def _grid_rgb(self) -> np.ndarray:
        """
        Build the agent-facing small RGB grid (shape: grid_size x grid_size x 3, dtype uint8).
        Color priority:
          - wall
          - box on goal
          - box
          - player on goal
          - player
          - goal
          - floor
        """
        H = self.grid_size
        W = self.grid_size
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, :] = COLOR_FLOOR

        for (y,x) in self.walls:
            img[y,x] = COLOR_WALL

        # mark goals first so we can render box/player on top if they coincide
        for (y,x) in self.goals:
            img[y,x] = COLOR_GOAL

        for (y,x) in self.boxes:
            if (y,x) in self.goals:
                img[y,x] = COLOR_BOX_ON_GOAL
            else:
                img[y,x] = COLOR_BOX

        py, px = self.player
        if (py,px) in self.goals:
            img[py,px] = COLOR_PLAYER_ON_GOAL
        else:
            img[py,px] = COLOR_PLAYER

        return img

    def _render_np(self, cell_px: int = 12) -> np.ndarray:
        """Return an upscaled RGB array for human viewing."""
        small = self._grid_rgb()
        H, W = small.shape[0], small.shape[1]
        big_H = H * cell_px
        big_W = W * cell_px
        img = Image.new("RGB", (big_W, big_H), COLOR_FLOOR)
        draw = ImageDraw.Draw(img)

        # draw cell rectangles
        for y in range(H):
            for x in range(W):
                color = tuple(int(v) for v in small[y,x])
                box = (x*cell_px, y*cell_px, (x+1)*cell_px, (y+1)*cell_px)
                draw.rectangle(box, fill=color, outline=(60,60,60))

                # optionally draw small glyph for boxes/goals/player to make it clearer
                # but we already use color mapping so this is optional

        return np.array(img)

    # -------------------------
    # Gym API: reset / step / render / close
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_random = np.random.RandomState(seed)
        self.steps = 0
        self.terminated = False
        self.truncated = False

        # initialize layout
        self._initialize_level_state(self.init_type, self.level_map)

        obs = self._grid_rgb()
        info = {}
        # gymnasium API expects (obs, info)
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        if self.terminated or self.truncated:
            obs = self._grid_rgb()
            return obs, 0.0, True, False, {}

        reward, info = self._apply_action_and_update_state(int(action))
        self.steps += 1

        # check solved
        if self._is_solved():
            self.terminated = True
            obs = self._grid_rgb()
            return obs, float(reward), True, False, info

        # truncated by step limit
        if self.steps >= self.max_steps:
            self.truncated = True
            obs = self._grid_rgb()
            return obs, float(reward), False, True, info

        obs = self._grid_rgb()
        return obs, float(reward), False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_np()
        elif self.render_mode == "human":
            img = self._render_np()
            Image.fromarray(img).show()
            return None
        else:
            # fallback to small observation
            return self._grid_rgb()

    def close(self):
        return None

# -------------------------
# Quick smoke test & save frames (when run as script)
# -------------------------
if __name__ == "__main__":
    import os
    from PIL import Image

    # try to find script directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not script_dir:
            script_dir = os.getcwd()
    except NameError:
        script_dir = os.getcwd()

    print("Saving Sokoban debug frames to folder:", script_dir)

    # small example ASCII level you can replace with your own
    level = np.array([
        [1,1,1,1,1,1],
        [1,0,0,0,0,1],
        [1,0,0,1,1,1],
        [1,2,0,0,0,1],
        [1,0,3,4,0,1],
        [1,1,1,1,1,1]
    ])
    # instantiate env (grid_size=20 expected; ASCII map may be smaller - it's centered/shallow parsed)
    env = SokobanEnv(grid_size=20, render_mode="rgb_array", init_type="matrix",level_map=level, max_steps=200)
    obs, _ = env.reset()
    print("Initial obs shape:", obs.shape)

    for i in range(12):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        print(f"Step {i}: action={a}, reward={r:.3f}, done={terminated or truncated}")
        img = env.render()
        fname = os.path.join(script_dir, f"sokoban_step_{i}.png")
        Image.fromarray(img).save(fname)
        print("Saved:", fname)
        if terminated or truncated:
            break

    print("Smoke test done. Frames saved to:", script_dir)
