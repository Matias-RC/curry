import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gym_sokoban.envs import SokobanEnv


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

class SokobanCanonicalCompactWrapper(gym.ObservationWrapper):
    def __init__(self, env, canonical_shape):
        super().__init__(env)

        self.max_h, self.max_w = canonical_shape

        original_h, original_w, _ = env.observation_space.shape
        self.tile_size = 16

        # channels: wall, box, goal, player
        self.channels = 4

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.channels, self.max_h, self.max_w),
            dtype=np.float32,
        )

    def observation(self, obs):
        grid_h = obs.shape[0] // self.tile_size
        grid_w = obs.shape[1] // self.tile_size

        compact = np.zeros((self.channels, grid_h, grid_w), dtype=np.float32)

        # --- same color logic you already wrote ---
        for i in range(grid_h):
            for j in range(grid_w):
                pi = i * 16 + 8
                pj = j * 16 + 8
                r, g, b = obs[pi, pj]

                if (r, g, b) == (176, 61, 0):
                    compact[0, i, j] = 1.0
                elif (r, g, b) == (215, 103, 0):
                    compact[1, i, j] = 1.0
                elif r == 238:
                    compact[2, i, j] = 1.0
                elif (r, g, b) == (41, 202, 26):
                    compact[3, i, j] = 1.0

        # --- center into canonical frame ---
        padded = np.zeros(
            (self.channels, self.max_h, self.max_w),
            dtype=np.float32,
        )

        oh = (self.max_h - grid_h) // 2
        ow = (self.max_w - grid_w) // 2

        padded[:, oh:oh+grid_h, ow:ow+grid_w] = compact
        return padded


class SokobanRetriesWrapper(SokobanEnv):
    def __init__(self, max_retries=3, *args, **kwargs):
        self.max_retries = max_retries
        self.current_retry = 0
        
        # Placeholder for the saved level state
        self.saved_state = None

        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None, second_player=False, render_mode='rgb_array', force_retry=False, force_new_level=False):
        """
        Custom reset that decides whether to:
        1. Fast-Restore the previous level (if retrying)
        2. Generate a new level (if finished or max retries hit)
        """
        
        # --- PATH 1: RETRY ---
        # We only retry if we failed previously AND have retries left AND have a state to restore
        if ((self.current_retry < self.max_retries or force_retry) and self.saved_state is not None) and not force_new_level:
            self.current_retry += 1
            # print(f"DEBUG: Retrying level (Attempt {self.current_retry}/{self.max_retries})")
            
            # Fast restore: Manually reset variables without calling generate_room()
            return self._restore_level(render_mode)

        # --- PATH 2: NEW LEVEL ---
        # Otherwise, we generate a fresh level using the parent logic
        # print("DEBUG: Generating new level")
        obs, info = super().reset(seed=321, options=options, second_player=second_player, render_mode=render_mode)
        
        # Save this fresh configuration immediately
        self._save_level_state()
        
        # Reset retry counters
        self.current_retry = 0
        
        # Add debug info
        info["retry_count"] = 0
        return obs, info

    def step(self, action, observation_mode='rgb_array'):
        obs, reward, terminated, truncated, info = super().step(action, observation_mode)
        
        # Inject retry info into the info dict (useful for logging)
        info["retry_count"] = self.current_retry
        info["retries_left"] = self.max_retries - self.current_retry
        
        return obs, reward, terminated, truncated, info

    def _save_level_state(self):
        """
        Creates a deep copy of the arrays defining the level.
        We must copy because the game modifies room_state in place.
        """
        self.saved_state = {
            # The static walls and targets
            'room_fixed': self.room_fixed.copy(), 
            # The dynamic boxes and player (at start position)
            'room_state': self.room_state.copy(), 
            # The mapping of box IDs (if used)
            'box_mapping': self.box_mapping.copy() if hasattr(self, 'box_mapping') else None,
            # Dimensions
            'dim_room': self.dim_room
        }

    def _restore_level(self, render_mode):
        """
        Manually resets the environment variables to the saved state.
        This mimics the end of the original reset() method.
        """
        state = self.saved_state
        
        # 1. Restore the grids
        self.room_fixed = state['room_fixed'].copy()
        self.room_state = state['room_state'].copy()
        if state['box_mapping'] is not None:
            self.box_mapping = state['box_mapping'].copy()
            
        # 2. Recalculate derived variables (Critical!)
        # Find player again (value 5 is player)
        self.player_position = np.argwhere(self.room_state == 5)[0]
        
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0 
        
        # 3. Generate initial observation
        starting_observation = self.render(mode=render_mode)
        
        info = {
            "retry_count": self.current_retry,
            "retries_left": self.max_retries - self.current_retry,
            "message": "restored_from_save"
        }
        
        return starting_observation, info