"""
Batched Sokoban Environment for RL training.

Supports multiple levels running in parallel with vectorized action processing.

Example usage:
    # Create config
    class Config:
        def __init__(self):
            self.grid_size = (10, 10)  # (height, width)
            self.initial_max_steps = 120
            self.batch_size = 8
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config()
    env = SokobanEnvironment(config)

    # Load levels
    level_strings = [...]  # List of level strings
    obs = env.load_levels(level_strings)

    # Step
    actions = torch.randint(0, 4, (config.batch_size,))
    obs, rewards, dones, infos = env.step(actions)
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict


class SokobanEnvironment:
    """
    Batched Sokoban environment that supports multiple levels in parallel.

    State representation (channels):
    - 0: walls
    - 1: boxes
    - 2: goals
    - 3: player

    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """

    def __init__(self, config=None):
        assert config is not None, "Config must be provided"

        self.config = config
        self.size_y, self.size_x = config.grid_size  # (height, width)
        self.action_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

        self.max_steps = config.initial_max_steps
        self.batch_size = getattr(config, 'batch_size', 1)

        # Use device from config if available, otherwise default to CPU
        self.device = getattr(config, 'device', 'cpu')

        # Batched state - each is a list of sets/tuples per batch element
        self.reset()

    def reset(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Reset the environment for all batch elements.

        Args:
            batch_size: Optional batch size. If None, uses config batch_size.

        Returns:
            Initial observations as tensor [B, C, H, W]
        """
        if batch_size is not None:
            self.batch_size = batch_size

        # Initialize batched state
        self.player_pos = [(0, 0) for _ in range(self.batch_size)]
        self.boxes = [set() for _ in range(self.batch_size)]
        self.goals = [set() for _ in range(self.batch_size)]
        self.walls = [set() for _ in range(self.batch_size)]
        self.steps_left = [self.max_steps for _ in range(self.batch_size)]
        self.done = [False for _ in range(self.batch_size)]

        return self._get_obs()

    def load_levels(self, level_strings: List[str]) -> torch.Tensor:
        """
        Load levels from string representations.

        Args:
            level_strings: List of strings where each string represents a level.
                Rows are separated by newline characters.
                Characters:
                - '#' = wall
                - '$' = box
                - '.' or 'G' = goal
                - '@' = player
                - '*' = box on goal
                - '+' = player on goal
                - ' ' = empty space

        Returns:
            Initial observations as tensor [B, C, H, W]
        """
        assert len(level_strings) == self.batch_size, \
            f"Expected {self.batch_size} levels, got {len(level_strings)}"

        for i, level_str in enumerate(level_strings):
            self._parse_level_string(i, level_str)

        return self._get_obs()

    def _parse_level_string(self, batch_idx: int, level_str: str):
        """Parse a level string and update state for given batch element."""
        rows = level_str.strip().split('\n')

        self.walls[batch_idx].clear()
        self.boxes[batch_idx].clear()
        self.goals[batch_idx].clear()

        # Parse level - assuming all levels are same size as grid_size
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == '#':
                    self.walls[batch_idx].add((r, c))
                elif ch == '$':
                    self.boxes[batch_idx].add((r, c))
                elif ch in ['.', 'G']:
                    self.goals[batch_idx].add((r, c))
                elif ch == '@':
                    self.player_pos[batch_idx] = (r, c)
                elif ch == '*':  # box on goal
                    self.boxes[batch_idx].add((r, c))
                    self.goals[batch_idx].add((r, c))
                elif ch == '+':  # player on goal
                    self.player_pos[batch_idx] = (r, c)
                    self.goals[batch_idx].add((r, c))

        self.done[batch_idx] = False
        self.steps_left[batch_idx] = self.max_steps

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Execute actions for all batch elements.

        Args:
            actions: Tensor of shape [B] with action indices (0-3)

        Returns:
            observations: Tensor [B, C, H, W]
            rewards: Tensor [B]
            dones: Tensor [B] (boolean)
            infos: List of info dicts, one per batch element
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        else:
            actions = np.array(actions)

        assert len(actions) == self.batch_size, \
            f"Expected {self.batch_size} actions, got {len(actions)}"

        rewards = np.zeros(self.batch_size, dtype=np.float32)
        infos = [{} for _ in range(self.batch_size)]

        # Process each batch element
        for i in range(self.batch_size):
            if not self.done[i]:
                reward, info = self._apply_action(i, int(actions[i]))
                rewards[i] = reward
                infos[i] = info

                # Update step counter
                self.steps_left[i] -= 1

                # Check termination conditions
                if self._is_solved(i):
                    self.done[i] = True
                    rewards[i] += 10.0  # Bonus for solving
                    infos[i]['solved'] = True
                elif self.steps_left[i] <= 0:
                    self.done[i] = True
                    infos[i]['timeout'] = True
            else:
                # Level is already done, set reward to 0 and mark as done in info
                infos[i]['already_done'] = True

        # Get observations (done levels will be padded with zeros)
        obs = self._get_obs()
        rewards_tensor = torch.from_numpy(rewards).to(self.device)
        dones_tensor = torch.tensor(self.done, dtype=torch.bool, device=self.device)

        return obs, rewards_tensor, dones_tensor, infos

    def _apply_action(self, batch_idx: int, action: int) -> Tuple[float, Dict]:
        """
        Apply action for a single batch element.

        Returns:
            reward: Float reward value
            info: Dictionary with step information
        """
        reward = -0.1  # Step penalty
        info = {}

        if action < 0 or action >= len(self.action_map):
            return 0, info

        dy, dx = self.action_map[action]
        py, px = self.player_pos[batch_idx]
        ty, tx = py + dy, px + dx

        # Check bounds
        if not (0 <= ty < self.size_y and 0 <= tx < self.size_x):
            return reward, info

        # Check wall collision
        if (ty, tx) in self.walls[batch_idx]:
            return reward, info

        # Check if pushing a box
        if (ty, tx) in self.boxes[batch_idx]:
            # Calculate box destination
            by, bx = ty + dy, tx + dx

            # Check if box can be pushed
            if not (0 <= by < self.size_y and 0 <= bx < self.size_x):
                return reward, info
            if (by, bx) in self.walls[batch_idx] or (by, bx) in self.boxes[batch_idx]:
                return reward, info

            # Perform push
            old_box_pos = (ty, tx)
            new_box_pos = (by, bx)

            was_on_goal = old_box_pos in self.goals[batch_idx]
            is_on_goal = new_box_pos in self.goals[batch_idx]

            # Update box position
            self.boxes[batch_idx].remove(old_box_pos)
            self.boxes[batch_idx].add(new_box_pos)

            # Update player position
            self.player_pos[batch_idx] = (ty, tx)

            # Reward shaping
            if not was_on_goal and is_on_goal:
                reward += 1.0
                info['box_on_goal'] = True
            elif was_on_goal and not is_on_goal:
                reward -= 1.0
                info['box_off_goal'] = True
        else:
            # Free cell - just move player
            self.player_pos[batch_idx] = (ty, tx)

        return reward, info

    def _is_solved(self, batch_idx: int) -> bool:
        """Check if all boxes are on goals for given batch element."""
        return self.boxes[batch_idx] == self.goals[batch_idx]

    def _get_obs(self) -> torch.Tensor:
        """
        Generate observations for all batch elements.

        For completed levels (done=True), returns padded observations (all zeros).
        This is compatible with the padding used in data/dataset.py collate_fn.

        Returns:
            Tensor of shape [B, C, H, W] where C=4:
            - Channel 0: walls
            - Channel 1: boxes
            - Channel 2: goals
            - Channel 3: player
        """
        obs = torch.zeros(
            (self.batch_size, 4, self.size_y, self.size_x),
            dtype=torch.float32,
            device=self.device
        )

        for i in range(self.batch_size):
            # If level is done, keep it as padding (all zeros)
            if self.done[i]:
                continue

            # Walls (channel 0)
            for wy, wx in self.walls[i]:
                obs[i, 0, wy, wx] = 1.0

            # Boxes (channel 1)
            for by, bx in self.boxes[i]:
                obs[i, 1, by, bx] = 1.0

            # Goals (channel 2)
            for gy, gx in self.goals[i]:
                obs[i, 2, gy, gx] = 1.0

            # Player (channel 3)
            py, px = self.player_pos[i]
            obs[i, 3, py, px] = 1.0

        return obs

    def render(self, batch_idx: int = 0) -> str:
        """
        Render a single batch element as ASCII art.

        Args:
            batch_idx: Which batch element to render

        Returns:
            String representation of the level
        """
        grid = [[' ' for _ in range(self.size_x)] for _ in range(self.size_y)]

        # Walls
        for wy, wx in self.walls[batch_idx]:
            grid[wy][wx] = '#'

        # Goals
        for gy, gx in self.goals[batch_idx]:
            if grid[gy][gx] == ' ':
                grid[gy][gx] = '.'

        # Boxes
        for by, bx in self.boxes[batch_idx]:
            if (by, bx) in self.goals[batch_idx]:
                grid[by][bx] = '*'  # Box on goal
            else:
                grid[by][bx] = '$'

        # Player
        py, px = self.player_pos[batch_idx]
        if (py, px) in self.goals[batch_idx]:
            grid[py][px] = '+'  # Player on goal
        else:
            grid[py][px] = '@'

        return '\n'.join(''.join(row) for row in grid)

    @staticmethod
    def make_config(config_dic):
        """
        Helper method to create a compatible config object.
        Compatible with the config format used in cmds/train.py.
        """
        class EnvConfig:
            pass

        config = EnvConfig()
        config.grid_size = (config_dic["grid_shape_y"], config_dic["grid_shape_x"])  # (height, width)
        config.initial_max_steps = config_dic["max_steps_per_play"]
        config.batch_size = config_dic["batch_size_train"]
        config.device = config_dic["device"]

        return config
