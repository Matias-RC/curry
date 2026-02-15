import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gym_sokoban.envs import SokobanEnv
import random

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
        obs, info = super().reset(seed=seed, options=options, second_player=second_player, render_mode=render_mode)
        
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
    

class SokoCanonicalWithAttPadding(gym.ObservationWrapper):
    def __init__(self, env, canonical_shape, window_size):
        super().__init__(env)

        self.max_h, self.max_w = canonical_shape
        self.window_size = window_size
        self.tile_size = 16
        self.channels = 4  # wall, box, goal, player

        if self.max_h % self.window_size != 0 or self.max_w % self.window_size != 0:
            raise ValueError("canonical shape must be divisible by window_size")

        self.Hw = self.max_h // self.window_size
        self.Ww = self.max_w // self.window_size

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.Hw, self.Ww, self.channels,
                   self.window_size, self.window_size),
            dtype=np.float32,
        )

    def observation(self, obs):
        grid_h = obs.shape[0] // self.tile_size
        grid_w = obs.shape[1] // self.tile_size

        centers = obs[8::16, 8::16]          # tile centers
        offset4 = obs[4::16, 4::16]          # (pixel_i - 4, pixel_j - 4)
        offset7 = obs[1::16, 1::16]          # (pixel_i - 7, pixel_j - 7)

        r = centers[..., 0]
        g = centers[..., 1]
        b = centers[..., 2]

        r4 = offset4[..., 0]
        g4 = offset4[..., 1]
        b4 = offset4[..., 2]

        r7 = offset7[..., 0]
        g7 = offset7[..., 1]
        b7 = offset7[..., 2]

        compact = np.zeros((self.channels, grid_h, grid_w), dtype=np.float32)

        # -------------------------
        # Wall
        # -------------------------
        compact[0] = (r == 176) & (g == 61) & (b == 0)

        # -------------------------
        # Box
        # -------------------------
        box_center = (r == 215) & (g == 103) & (b == 0)
        box_on_goal = (r == 238) & (g4 == 114)
        compact[1] = box_center | box_on_goal

        # -------------------------
        # Goal
        # -------------------------
        goal_center = (r == 238)
        player_on_goal = (r == 41) & (g == 202) & (b == 26) & (r7 == 238)
        compact[2] = goal_center | player_on_goal

        # -------------------------
        # Player
        # -------------------------
        compact[3] = (r == 41) & (g == 202) & (b == 26)

        compact = compact.astype(np.float32)

        padded = np.zeros(
            (self.channels, self.max_h, self.max_w),
            dtype=np.float32
        )

        padded[:, :grid_h, :grid_w] = compact

        # mask outside real board
        padded[:, grid_h:, :] = -1.0
        padded[:, :, grid_w:] = -1.0

        # -------------------------------------------------
        # 3) Attention masking per window
        # -------------------------------------------------
        ws = self.window_size

        for i in range(0, self.max_h, ws):
            for j in range(0, self.max_w, ws):
                block = padded[0, i:i+ws, j:j+ws]

                # if no valid empty cell exists → mask window
                if not (block == 0).any():
                    padded[:, i:i+ws, j:j+ws] = -1.0

        # -------------------------------------------------
        # 4) Tile into windows
        # -------------------------------------------------
        # reshape → (C, Hw, ws, Ww, ws)
        windows = padded.reshape(
            self.channels,
            self.Hw, ws,
            self.Ww, ws
        )

        # reorder → (Hw, Ww, C, ws, ws)
        windows = windows.transpose(1, 3, 0, 2, 4)

        return windows.astype(np.float32)

class SokoRetriesCurriculum(SokobanRetriesWrapper):
    def __init__(self,
                 max_retries=3,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 reset=True,
                 render_mode=None,
                 curriculum=False,
                 schedule_dic=None):
        if curriculum == True:
            assert schedule_dic != None, "Probabilistic Schedule was not assigned"
        else:
            schedule_dic = {(("dim_room", dim_room),("max_steps", max_steps), ("num_boxes", num_boxes),
                             ("num_gen_steps", num_gen_steps)):1}

        self.curriculum = curriculum
        self.schedule_dic = schedule_dic    
        super().__init__(max_retries=max_retries, dim_room=dim_room, max_steps=max_steps,
                         num_boxes=num_boxes, num_gen_steps=num_gen_steps, reset=reset,
                         render_mode=render_mode)

    def sample_from_dict(self, prob_dict):

        #TODO: Handle randomness properly (Right now it is apparently not seeded)

        keys = list(prob_dict.keys())
        weights = list(prob_dict.values())
        return random.choices(keys, weights=weights, k=1)[0]

    def reset(self, seed=None, options=None, second_player=False, render_mode='rgb_array', force_retry=False, force_new_level=False):
        if ((self.current_retry < self.max_retries or force_retry) and self.saved_state is not None) and not force_new_level:
            self.current_retry += 1

            return self._restore_level(render_mode)

        if self.curriculum:
            selected_schedule = dict(self.sample_from_dict(self.schedule_dic))
            self.configure(**selected_schedule)

        obs, info = super().reset(seed=seed, options=options, second_player=second_player, render_mode=render_mode)

        self._save_level_state()

        self.current_retry = 0

        info["retry_count"] = 0
        return obs, info