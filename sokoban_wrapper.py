"""
Wrapper to convert Sokoban RGB observations to compact channel representation.
Reduces observation size from (112, 112, 3) to (7, 7, 4) or similar.
"""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class SokobanCompactWrapper(gym.ObservationWrapper):
    """
    Converts Sokoban RGB observations to compact 4-channel representation:
    - Channel 0: Walls
    - Channel 1: Boxes
    - Channel 2: Goals
    - Channel 3: Player

    Also resizes from (112, 112) to the actual grid size (e.g., 7x7).
    """
    def __init__(self, env):
        super().__init__(env)

        # Sokoban uses 16x16 pixel tiles, so actual grid size is:
        original_h, original_w, _ = env.observation_space.shape
        self.grid_h = original_h // 16
        self.grid_w = original_w // 16

        # New observation space: (4, grid_h, grid_w) - channels first for PyTorch
        self.observation_space = Box(
            low=0, high=1,
            shape=(4, self.grid_h, self.grid_w),
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Convert RGB observation to compact representation.

        Sokoban color encoding:
        - (0, 0, 0): Empty/Floor - Black
        - (243, 248, 238): Wall - Light gray
        - (254, 126, 125): Box - Red
        - (142, 121, 56): Box on goal - Brown
        - (254, 253, 135): Goal - Yellow
        - (160, 212, 56): Player - Green
        - (219, 212, 56): Player on goal - Yellow-green
        """
        # Downsample from pixels to grid (channels first: 4, H, W)
        grid_obs = np.zeros((4, self.grid_h, self.grid_w), dtype=np.float32)

        # Sample the center of each tile (avoid borders)
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                # Sample center pixel of each 16x16 tile
                pixel_i = i * 16 + 8
                pixel_j = j * 16 + 8
                rgb = obs[pixel_i, pixel_j]

                # Check colors (with some tolerance)
                r, g, b = rgb

                # Wall (light gray-ish)
                if r > 200 and g > 200 and b > 200:
                    grid_obs[0, i, j] = 1.0

                # Box (red or brown)
                elif (r > 200 and g < 150) or (r > 100 and g > 100 and b < 100):
                    grid_obs[1, i, j] = 1.0

                # Goal (yellow-ish, but not player)
                elif g > 200 and b > 100 and r > 200:
                    grid_obs[2, i, j] = 1.0

                # Player (green-ish)
                elif g > 150 and g > r and g > b:
                    grid_obs[3, i, j] = 1.0

        return grid_obs
