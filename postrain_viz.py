import pygame
import numpy as np
import gymnasium as gym
from gym.spaces import Box
from gym_sokoban.envs import SokobanEnv
import pygame.surfarray as surfarray

# ============================================================
# 1. FIXED WRAPPERS (Must be robust to kwargs)
# ============================================================

class SokobanCompactWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_h, original_w, _ = env.observation_space.shape
        self.grid_h = original_h // 16
        self.grid_w = original_w // 16
        self.observation_space = Box(
            low=0, high=1, shape=(4, self.grid_h, self.grid_w), dtype=np.float32
        )

    def reset(self, **kwargs):
        # PASS KWARGS DOWN (Fixes the TypeError)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        grid_obs = np.zeros((4, self.grid_h, self.grid_w), dtype=np.float32)
        # (Your color logic here - omitted for brevity, assumes standard implementation)
        # For the sake of this test script, we just return zeros or simple logic
        # You should paste your full color logic here if using the trained model
        return grid_obs

class SokobanRetriesWrapper(SokobanEnv):
    def __init__(self, max_retries=3, *args, **kwargs):
        self.max_retries = max_retries
        self.current_retry = 0
        self.saved_state = None
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None, second_player=False, render_mode='rgb_array', force_retry=False, force_new_level=False):
        if force_new_level:
            obs, info = super().reset(seed=seed, options=options, second_player=second_player, render_mode=render_mode)
            self._save_level_state()
            self.current_retry = 0
            return obs, info

        if (self.current_retry < self.max_retries or force_retry) and self.saved_state is not None:
            self.current_retry += 1
            return self._restore_level(render_mode)

        obs, info = super().reset(seed=seed, options=options, second_player=second_player, render_mode=render_mode)
        self._save_level_state()
        self.current_retry = 0
        return obs, info

    def _save_level_state(self):
        self.saved_state = {
            'room_fixed': self.room_fixed.copy(), 
            'room_state': self.room_state.copy(), 
            'box_mapping': self.box_mapping.copy() if hasattr(self, 'box_mapping') else None,
            'dim_room': self.dim_room
        }

    def _restore_level(self, render_mode):
        state = self.saved_state
        self.room_fixed = state['room_fixed'].copy()
        self.room_state = state['room_state'].copy()
        if state['box_mapping'] is not None:
            self.box_mapping = state['box_mapping'].copy()
        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0 
        return self.render(mode=render_mode), {"message": "restored"}

# ============================================================
# 2. ABSTRACTIONS (Agents & Grid Manager)
# ============================================================

class Agent:
    """Base interface so you can mix Random, PPO, Human, etc."""
    def predict(self, obs, state=None, episode_start=None):
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space
    def predict(self, obs, state=None, episode_start=None):
        return self.action_space.sample(), None

class SokobanGrid:
    """
    Manages multiple Sokoban environments in parallel.
    Ensures they all play the SAME level (seed) but can be controlled differently.
    """
    def __init__(self, make_env_fn, agents, rows=1, cols=3, render_scale=4):
        self.envs = []
        self.agents = agents
        self.obs = []
        self.dones = []
        self.lstm_states = [None] * len(agents)
        self.render_scale = render_scale
        self.rows = rows
        self.cols = cols
        
        # Initialize environments
        for _ in range(len(agents)):
            self.envs.append(make_env_fn())
            self.dones.append(False)
            
        # Initialize layout
        dummy_obs, _ = self.envs[0].reset()
        dummy_img = self.envs[0].unwrapped.render(mode='rgb_array')
        self.h, self.w, _ = dummy_img.shape
        self.surface = pygame.Surface((self.w * render_scale * cols, self.h * render_scale * rows))

    def reset_all(self, force_new_level=False, seed=None):
        """Broadcast reset to all envs. If force_new_level, ensures ALL get same seed."""
        common_seed = seed if seed is not None else np.random.randint(0, 100000)
        
        self.obs = []
        self._last_episode_starts = []
        for i, env in enumerate(self.envs):
            # If force_new_level is True, everyone gets the COMMON seed
            # If force_new_level is False (retry), we just call reset(force_retry=True)
            if force_new_level:
                o, _ = env.reset(force_new_level=True, seed=common_seed)
            else:
                o, _ = env.reset(force_retry=True)
            self.obs.append(o)
            self.lstm_states[i] = None # Reset memory if applicable
            self._last_episode_starts.append(True)

    def step_all(self):
        """Step all agents forward one frame."""
        for i, (env, agent) in enumerate(zip(self.envs, self.agents)):
            if agent["type"] == "lstm":
                if self._last_episode_starts[i]:
                    obs, _ = env.reset(force_retry=True)
                    self.obs[i] = obs
                    self.lstm_states[i] = None
                    terminated = False
                    truncated = False
                    self._last_episode_starts[i] = False
                else:
                    # Predict
                    action, self.lstm_states[i] = agent["agent"].predict(self.obs[i], state=self.lstm_states[i])
                
                    # Step
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    
                    self.obs[i] = obs
                
                # Auto-Retry Logic (Per Agent)
                # If this specific agent finishes, it retries immediately while others keep going
                if (terminated or truncated):
                    self.lstm_states[i] = None
                    self._last_episode_starts[i] = True
            elif agent["type"] == "maple":
                pass
            else:
                raise ValueError("This agent type is not recognized")

    def render_to_surface(self):
        """Stitches all environments into the internal surface and returns it."""
        self.surface.fill((0,0,0))
        
        for idx, env in enumerate(self.envs):
            # Get RGB
            rgb = env.unwrapped.render(mode='rgb_array')
            
            # Create Pygame Surface
            # Transpose (H, W, C) -> (W, H, C) for Pygame
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, (self.w * self.render_scale, self.h * self.render_scale))
            
            # Grid Math
            r = idx // self.cols
            c = idx % self.cols
            x = c * (self.w * self.render_scale)
            y = r * (self.h * self.render_scale)
            
            self.surface.blit(surf, (x, y))
            
            # Optional: Draw label (e.g. "Model A", "Random")
            # font.render(...)
            
        return self.surface

# ============================================================
# 3. UI HELPERS (No changes needed, just defined)
# ============================================================

class Button:
    def __init__(self, image:pygame.Surface, x:int, y:int, scale:int):
        width = image.get_width()
        height = image.get_height()
        self.image = pygame.transform.scale(image, (int(width*scale), int(height*scale)))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y) 
        self.clicked = False

    def draw(self, surface:pygame.Surface):
        action = False
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
                action = True
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False
        surface.blit(self.image, (self.rect.x, self.rect.y))
        return action

def create_text_button(text, w=100, h=50, color=(100,100,200)):
    surf = pygame.Surface((w, h))
    surf.fill(color)
    # Basic font rendering
    font = pygame.font.SysFont('Arial', 20, bold=True)
    txt = font.render(text, True, (255,255,255))
    rect = txt.get_rect(center=(w//2, h//2))
    surf.blit(txt, rect)
    pygame.draw.rect(surf, (255,255,255), surf.get_rect(), 3)
    return surf

# ============================================================
# 4. MAIN ENTRY POINT
# ============================================================

def make_env():
    """Factory function for the grid manager"""
    env = SokobanRetriesWrapper(dim_room=(8, 8), max_steps=200, num_boxes=2, render_mode="rgb_array")
    env = SokobanCompactWrapper(env)
    return env

def main():
    pygame.init()
    pygame.font.init()

    # --- SETUP MODELS ---
    # Here you can load your trained model:
    # trained_model = RecurrentPPO.load("path/to/model")
    # agents = [RandomAgent(gym.spaces.Discrete(5)), trained_model, RandomAgent(gym.spaces.Discrete(5))]
    
    # For now, let's use 3 Random Agents to demonstrate side-by-side comparison
    agents = [
        {"agent": RandomAgent(gym.spaces.Discrete(9)), "type":"lstm"}, 
        {"agent": RandomAgent(gym.spaces.Discrete(9)), "type":"lstm"}, 
        {"agent": RandomAgent(gym.spaces.Discrete(9)), "type":"lstm"},
        {"agent": RandomAgent(gym.spaces.Discrete(9)), "type":"lstm"}
    ]

    # --- SETUP GRID MANAGER ---
    # This single line handles the broadcasting logic
    grid_manager = SokobanGrid(make_env, agents, rows=1, cols=3, render_scale=2)
    grid_manager.reset_all(force_new_level=True, seed=42)

    # --- UI SETUP ---
    # Calculate window size based on the grid manager's surface
    grid_w, grid_h = grid_manager.surface.get_size()
    screen = pygame.display.set_mode((grid_w, grid_h + 100)) # +100 for button bar
    pygame.display.set_caption("Sokoban Model Comparison")
    clock = pygame.time.Clock()

    img_play = create_text_button("Play", color=(50, 200, 50))
    img_pause = create_text_button("Pause", color=(200, 150, 50))
    img_step = create_text_button("Step >", color=(50, 50, 200))
    img_skip = create_text_button("New Level", color=(200, 50, 50))

    btn_y = grid_h + 25
    # Center buttons roughly
    offset = (grid_w - 350) // 2
    btn_pp = Button(img_pause, x=offset, y=btn_y, scale=1)
    btn_st = Button(img_step, x=offset + 120, y=btn_y, scale=1)
    btn_sk = Button(img_skip, x=offset + 240, y=btn_y, scale=1)

    running = True
    paused = True

    while running:
        # 1. Input
        step_now = False
        reset_now = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. Draw UI
        screen.fill((30,30,30))
        
        btn_pp.image = img_play if paused else img_pause
        
        if btn_pp.draw(screen):
            paused = not paused
        
        if btn_st.draw(screen):
            if paused: step_now = True
            
        if btn_sk.draw(screen):
            reset_now = True
            paused = True # Pause on new level to let user see it

        # 3. Logic Broadcast
        if reset_now:
            # Generate new seed, broadcast to ALL envs
            seed = np.random.randint(0, 10000)
            print(f"Broadcasting New Level (Seed: {seed})")
            grid_manager.reset_all(force_new_level=True, seed=seed)
        
        elif (not paused) or step_now:
            # Step ALL envs forward
            grid_manager.step_all()

        # 4. Render Broadcast
        # Get the composite image of all envs
        grid_surface = grid_manager.render_to_surface()
        screen.blit(grid_surface, (0, 0))

        pygame.display.flip()
        clock.tick(15 if not paused else 60)

    pygame.quit()
"""
TODO:
-modify the make env function  to allow class assignation for ppo derived models
-implement the logic for the maple agent
-implement the logic for the beam search maple agent
"""


if __name__ == "__main__":
    main()