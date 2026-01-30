import numpy as np
import pygame
import pygame.surfarray as surfarray
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gym_sokoban.envs import SokobanEnv
import random

# Set up matplotlib for displaying images

plt.rcParams['figure.figsize'] = (8, 8)

env = SokobanEnv(
    dim_room=(7, 7),        # Room dimensions (height, width)
    max_steps=120,          # Maximum steps before truncation
    num_boxes=2,            # Number of boxes to push
    render_mode='rgb_array' # Render mode: 'rgb_array', 'human', 'tiny_rgb_array', 'raw'
)
env_registered = gym.make('Sokoban-v0', render_mode='rgb_array')

pygame.init()


SCALE = 4
H, W = 112, 112

screen = pygame.display.set_mode((W*SCALE, H*SCALE))
clock = pygame.time.Clock()

# create surface ONCE
surface = pygame.Surface((W, H))
terminated  = True
truncated = True
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = random.randint(0,8)
    if terminated or truncated:
        rgb, info = env.reset()
        terminated = False
        truncated = False
    else:
        rgb, reward, terminated, truncated, info = env.step(action)


    # write pixels
    surfarray.blit_array(surface, rgb.swapaxes(0, 1))
    surface_scaled = pygame.transform.scale(surface, (W * SCALE, H * SCALE))

    screen.blit(surface_scaled, (0, 0))
    pygame.display.flip()

    clock.tick(10)

pygame.display.quit()
pygame.quit()   