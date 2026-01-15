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

        self.config = config #Config is dict
        self.size_x = config["grid_shape_x"]
        self.size_y = config["grid_shape_y"]

        self.action_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

        self.max_steps = config["max_steps_per_play"]
        self.batch_size = getattr(config, 'batch_size', 1)

        # Use device from config if available, otherwise default to CPU
        self.device = config["device"]

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

    def load_levels(self, level_strings) -> torch.Tensor:
        """
        Load levels from tensor representation
        """
        assert len(level_strings) == self.batch_size, \
            f"Expected {self.batch_size} levels, got {len(level_strings)}"

        for i, level in enumerate(level_strings):
            self._parse_level_tensor(i, level)

        return self._get_obs()

    def tensor_to_symbolic_state(self, tensor):
        H, W, C = tensor.shape
        channels = ["walls", "boxes", "goals", "player"]
        assert C == len(channels)

        state = {}

        for c, key in enumerate(channels):
            # Find all active cells in this channel
            idx = (tensor[:, :, c] > 0.5).nonzero(as_tuple=False)

            if key == "player":
                # Player must be a single position
                if idx.shape[0] != 1:
                    raise ValueError(f"Invalid player channel: {idx.shape[0]} active cells")
                state[key] = (int(idx[0, 0]), int(idx[0, 1]))
            else:
                state[key] = set(
                    (int(i), int(j)) for i, j in idx
                )

        return state


    def _parse_level_tensor(self, batch_idx: int, level):
        """Parse a level string and update state for given batch element."""
        self.walls[batch_idx].clear()
        self.boxes[batch_idx].clear()
        self.goals[batch_idx].clear()

        # Parse level - assuming all levels are same size as grid_size
        self.walls[batch_idx], self.boxes[batch_idx], self.goals[batch_idx], self.player_pos[batch_idx] = self.tensor_to_symbolic_state(level)

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
    
    def symbolic_state_to_tensor(self, state):
        channels = ["walls", "boxes", "goals", "player"]
        num_channels = len(channels)
        tensor = torch.zeros((self.size_y, self.size_x, num_channels), dtype=torch.float32)

        for c, key in enumerate(channels):
            values = state[key] if key != "player" else {state[key]}
            idx = torch.tensor(list(values), dtype=torch.long)  # shape [N, 2]
            tensor[idx[:, 0], idx[:, 1], c] = 1.0

        return tensor
    
    def _get_obs(self) -> torch.Tensor:
        obs = torch.zeros(
            (self.batch_size, 4, self.size_y, self.size_x),
            dtype=torch.float32,
            device=self.device
        )

        for i in range(self.batch_size):
            # If level is done, keep it as padding (all zeros)
            if self.done[i]:
                continue

            obs[i] = self.symbolic_state_to_tensor({"walls": self.walls[i], "player":self.player_pos[i], "goals":self.goals[i], "boxes":self.boxes[i]})
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
