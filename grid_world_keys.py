import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing  import List, Set, Tuple
from PIL import Image, ImageDraw, ImageFont
import os

key_holes = [
    (0,1,0),
    (0,0,1),
    (0,1,1),
    (1,0,1),
    (1,1,0),
    (1,128/255,0),
    (128/255,0,1),
    (0,128/255,128/255),
    (128/255,0,0),
    (0,0,128/255)]

#Floor: (255,255,255)
#Wall: (0,0,0)
#Player: (255,0,0)

templates = [
    {
        (0,0)
    },
    {
        (0,-1),
        (1,-1),
        (1,0)
    },
    {
        (-1,-1)
    },
    {
        (0,-1)
    },
    {
        (0,-1),
        (1,-1),
        (1,0),
        (-1,1)
    },
    {
        (0,-1),
        (0,0),
        (1,-1)
    },
    {
        (0,-1),
        (0,0),
        (1,-1),
        (1,0)
    },
    {
        (-1,-1),
        (1,-1),
        (0,1)
    },
    {
        (-1,-1),
        (-1,1),
        (1,-1),
        (1,1)
    }
]

lock_template = [
    {
        (3, -2), (3, -1), (3, 0), (3, 1), (3, 2),
        (2, -2), (2, 2),
        (1, -2), (1, 1), (1, 2),
        (0, -2), (0, -1), (0, 1), (0, 2)
    },
    {
        (2, 1)
    }
]



class Key:
    def __init__(self, pos, colour, percived_colour):
        self.original_pos = pos
        self.pos = pos
        self.colour = colour
        self.percived_colour = percived_colour
    def reset(self):
        self.pos = self.original_pos

class Lock:
    def __init__(self, pos, colour, my_key, next_key,  last_lock):
        self.pos = pos
        self.colour = colour
        self.my_key = my_key
        self.next_key = next_key
        self.last_lock = last_lock

class Environment:
    def __init__(self, spawn_size, key_maze_size, max_steps):
        self.spawn_x = spawn_size
        self.key_x, self.key_y = key_maze_size

        self.size_x = self.spawn_x + self.key_x + 2 #+2 for padding between maze and spawn
        self.size_y = self.key_y

        self.curriculum_stage = 0
        self.templates = templates
        self.walls = List[Set]
        self.lock_entries = List[Lock]

        self.current_key = None
        self.player_pos = None

        self.action_map = [(1,0),(0,-1),(0,1),(-1,0)]
        self.left_steps = max_steps
        self.max_steps = max_steps

        self.finished = False

    def initialize_state(self):
        if self.curriculum_stage == 0:
            top_bottom = random.random()>self.size_y/(self.spawn_x*2)
            bottom = True
            lock_pos = [0,0]
            if top_bottom:
                lock_pos[1] = random.randint(2,self.spawn_x*2-1) #Avoid corners
                if lock_pos[1] > self.spawn_x:
                    bottom = False
                    lock_pos[1] = lock_pos[1] - self.spawn_x - 1
                lock_pos[1] = lock_pos[1] + self.key_x
                lock_pos[0] = bottom*(self.size_y-1)
            else:
                lock_pos = [random.randint(1, self.size_y-2),self.size_x-1] #Also avoids corners

            #Assign a random colour to the lock and it's assigned key
            colour = random.choice(key_holes)
            key_pos = (random.randint(1, self.size_y-2), random.randint(1, self.key_x-1))#keys should be pushable: padding
            #Opening all the locks gives oficial termination reward
            reward_pos = (random.randint(0, self.size_y-2), random.randint(1, self.key_x-1))
            # ---> reward is not treated like key: It doesnt get put in a walled maze and works with apple mechanics

            level_lock = Lock(tuple(lock_pos), colour, Key(key_pos, colour, (128,128,128)), Key(reward_pos, (128/255,128/255,0),(128/255,128/255,0)), True)

            self.lock_entries = [level_lock]
            self.current_key = 0
            self.player_pos = (random.randint(1,self.size_y-2),  random.randint(self.key_x+2,self.size_x-2))
            self.walls = [set()]
            pos  = (0, self.key_x+2)
            track_x = True
            track_y = True
            direction = (0,1)
            for _ in range(self.spawn_x*2+self.size_y-2):
                self.walls[0].add(pos)
                new_y = pos[0]+direction[0]
                new_x =pos[1]+direction[1]
                if ((new_y == self.size_y) and track_y) or ((new_x == self.key_x+2+self.spawn_x) and track_x):
                    if (new_x == self.key_x+2+self.spawn_x):
                        track_x = False
                    if (new_y == self.size_y-1):
                        track_y = False
                    #rot dir by 90 counter-clockwise
                    direction = (direction[1], -direction[0])
                    new_y = pos[0]+direction[0]
                    new_x =pos[1]+direction[1]

                pos = (new_y,new_x)

                


            for item in self.lock_entries:
                if item.pos in self.walls[0]:
                    self.walls[0].remove(item.pos)
            self.walls.append(self.walls[0].copy())

    def manhattan_distance(self, pos_1, pos_2):
        dy = abs(pos_1[0]-pos_2[0])
        dx = abs(pos_1[1]-pos_2[1])
        return dy + dx
    
    def update(self, action):
        dy, dx = self.action_map[action]
        self.left_steps -= 1
        if self.left_steps < 0:
            self.finished = True
            return 0
        
        new_pos = (self.player_pos[0]+dy, self.player_pos[1]+dx)
        #if new pos in current walls: nope
        #if new pos == key pos:
        #   if new_key is last key: +15 reward finish game
        #   if new_key in current walls: nope
        #   if new_key in lock:
        #       iff  colours match +10 reward, update walls, update key
        #       else -10 reward and re-start level
        #if new pos in lock entries: accept
        #if new pos out of bounds: reject 


    def suplementary_reward(self, last_state):
        pass

    def render(self):
        board = torch.ones((self.size_y,self.size_x,3))
        for i in range(self.size_y):
            for j in range(self.size_x):
                for k in self.lock_entries:
                    if k.pos == (i,j):
                        board[i][j] = torch.tensor(k.colour)
                if (i,j) in self.walls[self.current_key]:
                    board[i][j] = torch.zeros(3)
        this_key = None
        if self.current_key == len(self.lock_entries):
            this_key = self.lock_entries[-1].next_key
        else:
            this_key = self.lock_entries[self.current_key].my_key
        y, x = this_key.pos
        board[y][x] = torch.tensor(this_key.percived_colour)
                
    def _to_rgb(self, col):
        if col is None:
            return (0, 200, 0)
        r, g, b = col
        if max(r, g, b) <= 1.0:
            return (int(r * 255), int(g * 255), int(b * 255))
        return (int(r), int(g), int(b))
    
    def render_for_human(self, filename="env_render.png", cell_size=36, show_grid=True, grid_line_width=1):
        width = self.size_x * cell_size
        height = self.size_y * cell_size
        img = Image.new("RGB", (width, height), (255,255,255))
        draw = ImageDraw.Draw(img)

        # walls
        try:
            wallset = self.walls[self.current_key]
        except Exception:
            wallset = set()
        for (wy, wx) in wallset:
            x0 = wx * cell_size
            y0 = wy * cell_size
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], fill=(0,0,0))

        # locks (holes)
        for k in getattr(self, "lock_entries", []):
            ly, lx = k.pos
            color = self._to_rgb(k.colour)
            x0 = lx * cell_size
            y0 = ly * cell_size
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], fill=color)

        # active key / reward
        if getattr(self, "current_key", 0) == len(getattr(self, "lock_entries", [])):
            this_key = self.lock_entries[-1].next_key
        else:
            this_key = self.lock_entries[self.current_key].my_key if getattr(self, "lock_entries", None) else None

        if this_key is not None:
            ky, kx = this_key.pos
            kcol = self._to_rgb(this_key.colour)
            x0 = kx * cell_size
            y0 = ky * cell_size
            inset = cell_size // 6
            draw.ellipse([x0 + inset, y0 + inset, x0 + cell_size - inset - 1, y0 + cell_size - inset - 1], fill=kcol)

        # player
        if getattr(self, "player_pos", None) is not None:
            py, px = self.player_pos
            x0 = px * cell_size
            y0 = py * cell_size
            inset = cell_size // 8
            draw.polygon([(x0+(cell_size)/2, y0 + inset),(x0+inset, y0 + cell_size - inset - 1), (x0+cell_size-inset,y0 + cell_size - inset - 1) ], fill=(255,0,0))

        # grid lines
        if show_grid:
            for cx in range(self.size_x + 1):
                x = cx * cell_size
                draw.line([(x, 0), (x, height)], fill=(150,150,150), width=grid_line_width)
            for cy in range(self.size_y + 1):
                y = cy * cell_size
                draw.line([(0, y), (width, y)], fill=(150,150,150), width=grid_line_width)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename)
        return filename
    
    def load(self):
        pass


if __name__ == "__main__":
    env = Environment(6, (12,12), 100)
    env.initialize_state()
    for i  in range(3):
        env.render_for_human(filename=f"images/step{i}.png")
        a = int(input())
        env.update(a)