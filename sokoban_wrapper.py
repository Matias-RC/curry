import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class SokobanCompactWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Sokoban uses 16x16 pixel tiles, so actual grid size is:
        original_h, original_w, _ = env.observation_space.shape
        self.grid_h = original_h // 16
        self.grid_w = original_w // 16

        # New observation space: (4, grid_h, grid_w) - channels first for PyTorch
        self.observation_space = Box(
            low=0, high=1,
            shape=(4,self.grid_h,self.grid_w,),
            dtype=np.float32
        )

    def observation(self, obs):
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
                if r == 176 and g == 61 and b == 0:
                    grid_obs[0, i, j] = 1.0

                # Box (red or brown)
                elif r == 215 and g == 103 and b == 0:
                    grid_obs[1, i, j] = 1.0

                # Goal (yellow-ish, but not player)
                elif r == 238:
                    new_rgb = obs[pixel_i - 4, pixel_j - 4]
                    _, new_g, _ = new_rgb
                    if new_g == 114:
                        grid_obs[1, i, j] = 1.0
                    grid_obs[2, i, j] = 1.0

                # Player (green-ish)
                elif r == 41 and g == 202 and b == 26:
                    new_rgb = obs[pixel_i - 7, pixel_j - 7]
                    new_r, _, _ = new_rgb
                    if new_r == 238:
                        grid_obs[2, i, j] = 1.0
                    grid_obs[3, i, j] = 1.0
        return grid_obs
