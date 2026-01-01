import copy
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
        self.placed_locks = set()

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
                lock_pos[1] = random.randint(2,self.spawn_x*2-1)
                if lock_pos[1] > self.spawn_x:
                    bottom = False
                    lock_pos[1] = lock_pos[1] - self.spawn_x - 1
                lock_pos[1] = lock_pos[1] + self.key_x
                lock_pos[0] = bottom*(self.size_y-1)
            else:
                lock_pos = [random.randint(1, self.size_y-2),self.size_x-1]

            colour = random.choice(key_holes)
            key_pos = (random.randint(1, self.size_y-2), random.randint(1, self.key_x-1))
            reward_pos = (random.randint(0, self.size_y-2), random.randint(1, self.key_x-1))

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
                    direction = (direction[1], -direction[0])
                    new_y = pos[0]+direction[0]
                    new_x =pos[1]+direction[1]
                pos = (new_y,new_x)

            for item in self.lock_entries:
                if item.pos in self.walls[0]:
                    self.walls[0].remove(item.pos)
            self.walls.append(self.walls[0].copy())

        # snapshot initial level/state for reset
        self._initial_level = copy.deepcopy(self.export_level())
        self._initial_state = copy.deepcopy(self.export_state())

    def manhattan_distance(self, pos_1, pos_2):
        dy = abs(pos_1[0]-pos_2[0])
        dx = abs(pos_1[1]-pos_2[1])
        return dy + dx
    
    def update(self, action: int):
        self.left_steps -= 1
        if self.left_steps < 0:
            self.finished = True
            return 0.0

        dy, dx = self.action_map[action]
        new_pos = (self.player_pos[0] + dy, self.player_pos[1] + dx)

        def in_bounds(pos):
            return 0 <= pos[0] < self.size_y and 0 <= pos[1] < self.size_x

        try:
            walls_current = self.walls[self.current_key]
        except Exception:
            walls_current = set()

        if getattr(self, "lock_entries", None) and self.current_key == len(self.lock_entries):
            final_key = self.lock_entries[-1].next_key
            if new_pos == final_key.pos:
                self.finished = True
                return 15.0

        if (not in_bounds(new_pos)) or (new_pos in walls_current) or (new_pos in self.placed_locks):
            return -0.5

        this_key = None
        if getattr(self, "lock_entries", None) and self.current_key < len(self.lock_entries):
            this_key = self.lock_entries[self.current_key].my_key

        if this_key is not None and new_pos == this_key.pos:
            target_pos = (this_key.pos[0] + dy, this_key.pos[1] + dx)

            if (not in_bounds(target_pos)) or (target_pos in walls_current) or (target_pos in self.placed_locks):
                return -0.5

            lock_hit = None
            for idx, lock in enumerate(self.lock_entries):
                if lock.pos == target_pos:
                    lock_hit = (idx, lock)
                    break

            if lock_hit is not None:
                idx, lock = lock_hit
                if lock.colour == this_key.colour:
                    this_key.pos = lock.pos
                    self.placed_locks.add(lock.pos)
                    if getattr(lock, "next_key", None) is not None:
                        lock.next_key.pos = lock.next_key.original_pos
                    self.current_key += 1
                    self.player_pos = new_pos
                    return 10.0
                else:
                    self.finished = True
                    return -10.0

            this_key.pos = target_pos
            self.player_pos = new_pos
            return -0.5

        self.player_pos = new_pos
        return -0.5
    
    def suplementary_reward(self, state1, state2):
        if len(self.lock_entries) != 1:
            return 0.0

        lock = self.lock_entries[0]

        if state2["current_key"] >= len(self.lock_entries):
            return 0.0

        p1 = tuple(state1["player_pos"])
        p2 = tuple(state2["player_pos"])

        k1 = tuple(state1["locks_pos"][0]["my_key_pos"])
        k2 = tuple(state2["locks_pos"][0]["my_key_pos"])

        lock_pos = lock.pos

        r = 0.0

        d_p_k_1 = self.manhattan_distance(p1, k1)
        d_p_k_2 = self.manhattan_distance(p2, k2)

        if d_p_k_2 < d_p_k_1:
            r += 1.0
        elif d_p_k_2 > d_p_k_1:
            r -= 2.0

        d_k_l_1 = self.manhattan_distance(k1, lock_pos)
        d_k_l_2 = self.manhattan_distance(k2, lock_pos)

        if d_k_l_2 < d_k_l_1:
            r += 1.0
        elif d_k_l_2 > d_k_l_1:
            r -= 2.0

        return r


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
    
    def export_level(self):
        walls_serial = []
        for wset in getattr(self, "walls", []):
            walls_serial.append([ [int(p[0]), int(p[1])] for p in wset ])
        locks_serial = []
        for lock in getattr(self, "lock_entries", []):
            lk = {
                "pos": [int(lock.pos[0]), int(lock.pos[1])],
                "colour": [float(c) for c in lock.colour],
                "last_lock": bool(lock.last_lock),
                "my_key": {
                    "original_pos": [int(lock.my_key.original_pos[0]), int(lock.my_key.original_pos[1])],
                    "colour": [float(c) for c in lock.my_key.colour],
                    "percived_colour": [int(lock.my_key.percived_colour[0]) if isinstance(lock.my_key.percived_colour[0], int) else float(lock.my_key.percived_colour[0]),
                                        int(lock.my_key.percived_colour[1]) if isinstance(lock.my_key.percived_colour[1], int) else float(lock.my_key.percived_colour[1]),
                                        int(lock.my_key.percived_colour[2]) if isinstance(lock.my_key.percived_colour[2], int) else float(lock.my_key.percived_colour[2])]
                },
                "next_key": {
                    "original_pos": [int(lock.next_key.original_pos[0]), int(lock.next_key.original_pos[1])],
                    "colour": [float(c) for c in lock.next_key.colour],
                    "percived_colour": [int(lock.next_key.percived_colour[0]) if isinstance(lock.next_key.percived_colour[0], int) else float(lock.next_key.percived_colour[0]),
                                        int(lock.next_key.percived_colour[1]) if isinstance(lock.next_key.percived_colour[1], int) else float(lock.next_key.percived_colour[1]),
                                        int(lock.next_key.percived_colour[2]) if isinstance(lock.next_key.percived_colour[2], int) else float(lock.next_key.percived_colour[2])]
                }
            }
            locks_serial.append(lk)
        level = {
            "spawn_x": int(self.spawn_x),
            "key_x": int(self.key_x),
            "key_y": int(self.key_y),
            "size_x": int(self.size_x),
            "size_y": int(self.size_y),
            "walls": walls_serial,
            "lock_entries": locks_serial
        }
        return level
    def export_state(self):
        locks_pos = {}
        for idx, lock in enumerate(getattr(self, "lock_entries", [])):
            locks_pos[idx] = {
                "my_key_pos": [int(lock.my_key.pos[0]), int(lock.my_key.pos[1])],
                "next_key_pos": [int(lock.next_key.pos[0]), int(lock.next_key.pos[1])]
            }
        state = {
            "player_pos": [int(self.player_pos[0]), int(self.player_pos[1])],
            "current_key": int(self.current_key),
            "left_steps": int(self.left_steps),
            "placed_locks": [[int(p[0]), int(p[1])] for p in getattr(self, "placed_locks", set())],
            "locks_pos": locks_pos,
            "finished": bool(self.finished)
        }
        return state
    def load_level(self, level_obj):
        self.spawn_x = int(level_obj.get("spawn_x", self.spawn_x))
        self.key_x = int(level_obj.get("key_x", self.key_x))
        self.key_y = int(level_obj.get("key_y", self.key_y))
        self.size_x = int(level_obj.get("size_x", self.size_x))
        self.size_y = int(level_obj.get("size_y", self.size_y))
        walls_raw = level_obj.get("walls", [])
        self.walls = []
        for w in walls_raw:
            self.walls.append(set([ (int(p[0]), int(p[1])) for p in w ]))
        if not self.walls:
            self.walls = [set()]
        locks_raw = level_obj.get("lock_entries", [])
        new_locks = []
        for lr in locks_raw:
            pos = tuple(int(x) for x in lr["pos"])
            colour = tuple(float(x) for x in lr["colour"])
            myk = lr["my_key"]
            nk = lr["next_key"]
            my_key_obj = Key(tuple(int(x) for x in myk["original_pos"]), tuple(float(x) for x in myk["colour"]), tuple(myk["percived_colour"]))
            next_key_obj = Key(tuple(int(x) for x in nk["original_pos"]), tuple(float(x) for x in nk["colour"]), tuple(nk["percived_colour"]))
            lock_obj = Lock(pos, colour, my_key_obj, next_key_obj, bool(lr.get("last_lock", False)))
            new_locks.append(lock_obj)
        self.lock_entries = new_locks
        self.current_key = 0
        self.placed_locks = set()
        self.finished = False

    def load_state(self, state_obj):
        self.player_pos = tuple(int(x) for x in state_obj["player_pos"])
        self.current_key = int(state_obj["current_key"])
        self.left_steps = int(state_obj["left_steps"])
        self.placed_locks = set([ (int(p[0]), int(p[1])) for p in state_obj.get("placed_locks", []) ])
        for idx, posdict in state_obj.get("locks_pos", {}).items():
            idxi = int(idx)
            if idxi < len(self.lock_entries):
                self.lock_entries[idxi].my_key.pos = tuple(int(x) for x in posdict["my_key_pos"])
                self.lock_entries[idxi].next_key.pos = tuple(int(x) for x in posdict["next_key_pos"])
        self.finished = bool(state_obj.get("finished", False))

    def reset(self):
        if hasattr(self, "_initial_level") and hasattr(self, "_initial_state"):
            self.load_level(copy.deepcopy(self._initial_level))
            self.load_state(copy.deepcopy(self._initial_state))
            self.left_steps = int(self.max_steps)
            self.finished = False
        else:
            raise RuntimeError("No initial snapshot available to reset to.")
        

if __name__ == "__main__":
    env = Environment(6, (12,12), 100)
    env.initialize_state()

    saved_level = None
    saved_state = None

    i = 0
    while True:
        env.render_for_human(filename=f"images/step{i}.png")
        print("w=0 a=1 d=2 s=3 | l=export level | k=export state | L=load level | K=load state | r=reset | q=quit")
        cmd = input(">> ").strip()

        if cmd == "q":
            break

        if cmd in ["0","1","2","3"]:
            if not env.finished:
                r = env.update(int(cmd))
                print("reward:", r, "finished:", env.finished)
            else:
                print("episode finished; press r to reset")
            i += 1
            continue

        if cmd == "l":
            saved_level = env.export_level()
            print("level exported")
            continue

        if cmd == "k":
            saved_state = env.export_state()
            print("state exported")
            continue

        if cmd == "L":
            if saved_level is not None:
                env.load_level(saved_level)
                print("level loaded")
            else:
                print("no saved level")
            continue

        if cmd == "K":
            if saved_state is not None:
                env.load_state(saved_state)
                print("state loaded")
            else:
                print("no saved state")
            continue

        if cmd == "r":
            env.reset()
            i = 0
            print("environment reset")
            continue

        print("unknown command")
