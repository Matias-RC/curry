import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Set, Tuple
from PIL import Image, ImageDraw, ImageFont
import os
import time

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
    (0,0,128/255)
]
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

class Key:
    def __init__(self, pos, colour, percived_colour):
        self.original_pos = pos
        self.pos = pos
        self.colour = colour
        self.percived_colour = percived_colour

    def reset(self):
        self.pos = self.original_pos

class Lock:
    def __init__(self, pos, colour, my_key, next_key, last_lock):
        self.pos = pos
        self.colour = colour
        self.my_key = my_key
        self.next_key = next_key
        self.last_lock = last_lock

class Environment:
    def __init__(self, spawn_size, key_maze_size, max_steps):
        self.spawn_x = spawn_size
        self.key_x, self.key_y = key_maze_size
        self.size_x = self.spawn_x + self.key_x + 2  #+2 for padding between maze and spawn
        self.size_y = self.key_y
        self.curriculum_stage = (0,1)
        self.templates = templates
        self.walls: List[Set] = []
        self.lock_entries: List[Lock] = []
        self.placed_locks = set()
        self.current_key = None
        self.player_pos = None
        self.action_map = [(1,0),(0,-1),(0,1),(-1,0)]
        self.left_steps = max_steps
        self.max_steps = max_steps
        self.finished = False

    def initialize_state(self):
        assert isinstance(self.curriculum_stage, tuple)
        num_templates, num_locks = self.curriculum_stage
        num_stages = num_locks + 1
        self.finished = False
        self.left_steps = self.max_steps
        self.placed_locks = set()
        self.current_key = 0

        # -------------------------
        # Template transforms
        # -------------------------
        def rotate(p):
            y, x = p
            return (-x, y)
        def reflect(p):
            y, x = p
            return (y, -x)
        def random_transform(template):
            pts = set(template)
            # random rotation: 0, 90, 180, 270
            k = random.randint(0, 3)
            for _ in range(k):
                pts = set(rotate(p) for p in pts)
            # random reflection
            if random.random() < 0.5:
                pts = set(reflect(p) for p in pts)
            return pts

        # -------------------------
        # Spawn contour walls
        # -------------------------
        contour_walls = set()
        for y in range(self.size_y):
            contour_walls.add((y, self.size_x - 1))
        for x in range(self.key_x + 1, self.size_x):
            contour_walls.add((0, x))
            contour_walls.add((self.size_y - 1, x))

        # -------------------------
        # Expanded contour exclusion
        # -------------------------
        contour_forbidden = set()
        for (y, x) in contour_walls:
            if x >= self.key_x:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        contour_forbidden.add((y + dy, x + dx))

        # -------------------------
        # Per-stage mazes
        # -------------------------
        maze_walls_list = []
        used_key_cells_list = []
        forbidden_list = []
        free_cells_list = []
        for stage in range(num_stages):
            if stage == num_locks:
                maze_walls = set()
                used_key_cells = set()
            else:
                maze_walls = set()
                centers = []
                for y in range(1, self.key_y, 3):
                    for x in range(1, self.key_x, 3):
                        centers.append((y, x))
                random.shuffle(centers)
                centers = centers[:num_templates]
                used_key_cells = set()
                for cy, cx in centers:
                    tpl = random.choice(self.templates)
                    for dy, dx in random_transform(tpl):
                        wy, wx = cy + dy, cx + dx
                        if 0 <= wy < self.key_y and 0 <= wx < self.key_x:
                            maze_walls.add((wy, wx))
                    # Chebyshev-1 exclusion
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            used_key_cells.add((cy + dy, cx + dx))
            forbidden = used_key_cells | contour_forbidden
            free_cells = [
                (y, x)
                for y in range(1, self.key_y - 1)
                for x in range(1, self.key_x - 1)
                if self.has_free_3x3((y, x), maze_walls, forbidden)
            ]
            random.shuffle(free_cells)
            maze_walls_list.append(maze_walls)
            used_key_cells_list.append(used_key_cells)
            forbidden_list.append(forbidden)
            free_cells_list.append(free_cells)

        # Set walls per stage
        self.walls = []
        for stage in range(num_stages):
            self.walls.append(contour_walls.copy() | maze_walls_list[stage])

        # -------------------------
        # Lock placement
        # -------------------------
        contour_positions = []
        for x in range(self.key_x + 2, self.size_x - 1):
            contour_positions.append((0, x))
            contour_positions.append((self.size_y - 1, x))
        for y in range(1, self.size_y - 1):
            contour_positions.append((y, self.size_x - 1))
        random.shuffle(contour_positions)
        occupied = set()
        def valid_lock(pos):
            if pos in occupied:
                return False
            y, x = pos
            for dy, dx in self.action_map:
                if (y + dy, x + dx) in occupied:
                    return False
            return True
        lock_positions = []
        attempts = 0
        for pos in contour_positions:
            if len(lock_positions) >= num_locks:
                break
            if valid_lock(pos):
                lock_positions.append(pos)
                occupied.add(pos)
            attempts += 1
            if attempts > 3 * num_locks:
                break
        if len(lock_positions) < num_locks:
            contour_positions.sort()
            for pos in contour_positions:
                if len(lock_positions) >= num_locks:
                    break
                if valid_lock(pos):
                    lock_positions.append(pos)
                    occupied.add(pos)
        assert len(lock_positions) == num_locks, "Could not place all required locks"

        # Assign unique colors to locks
        lock_colors = random.sample(key_holes, num_locks)
        locks = list(zip(lock_positions, lock_colors))

        # Remove lock positions from all walls
        for stage_walls in self.walls:
            for pos, _ in locks:
                stage_walls.discard(pos)

        # -------------------------
        # Key placement utilities (per stage)
        # -------------------------
        def has_free_3x3(pos, walls, forbidden):
            y, x = pos
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    p = (y + dy, x + dx)
                    if not (0 <= p[0] < self.key_y and 0 <= p[1] < self.key_x):
                        return False
                    if p in walls or p in forbidden:
                        return False
            return True

        def spawn_key(stage):
            nonlocal forbidden_list, free_cells_list, maze_walls_list
            forbidden = forbidden_list[stage]
            free_cells = free_cells_list[stage]
            maze_walls = maze_walls_list[stage]
            if free_cells:
                pos = free_cells.pop()
            else:
                for _ in range(100):
                    y = random.randint(1, self.key_y - 2)
                    x = random.randint(1, self.key_x - 2)
                    pos = (y, x)
                    if has_free_3x3(pos, maze_walls, forbidden):
                        break
                else:
                    raise RuntimeError("No valid key position")
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        maze_walls.discard((pos[0] + dy, pos[1] + dx))
                self.walls[stage] = contour_walls.copy() | maze_walls
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    forbidden.add((pos[0] + dy, pos[1] + dx))
            return pos

        # -------------------------
        # Create keys
        # -------------------------
        key_objects = []
        for stage in range(num_locks):
            pos = spawn_key(stage)
            colour = lock_colors[stage]
            perceived = (128, 128, 128)
            key_objects.append(Key(pos, colour, perceived))

        # Reward key (last stage, no maze)
        reward_pos = spawn_key(num_locks)
        reward_colour = (128/255, 128/255, 0)
        reward_perceived = (128/255, 128/255, 0)
        reward_key = Key(reward_pos, reward_colour, reward_perceived)

        # -------------------------
        # Build lock-key chain
        # -------------------------
        self.lock_entries = []
        for i in range(num_locks):
            pos, colour = locks[i]
            my_key = key_objects[i]
            if i == num_locks - 1:
                next_key = reward_key
                last = True
            else:
                next_key = key_objects[i + 1]
                last = False
            self.lock_entries.append(Lock(pos, colour, my_key, next_key, last))

        # -------------------------
        # Player spawn
        # -------------------------
        self.player_pos = (
            random.randint(1, self.size_y - 2),
            random.randint(self.key_x + 2, self.size_x - 2)
        )

        # -------------------------
        # Snapshot
        # -------------------------
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
                    return 10.0
                else:
                    self.finished = True
                    return -0.5
            this_key.pos = target_pos
            self.player_pos = new_pos
            return -0.5
        self.player_pos = new_pos
        return -0.5
    
    def suplementary_reward(self, state1, state2):
        if len(self.lock_entries) != 1:
            return 0.0
        lock = self.lock_entries[0]
        # 
        # Current key doesn't exist in the exported state dict
        #
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
        H = self.size_y
        W = self.size_x

        # Base board: white floor
        board = torch.ones((H, W, 3), dtype=torch.float32)

        # --------------------------------------------------
        # Walls for current stage
        # --------------------------------------------------
        if hasattr(self, "walls") and self.walls:
            if 0 <= self.current_key < len(self.walls):
                wallset = self.walls[self.current_key]
            else:
                wallset = set()
            for (y, x) in wallset:
                if 0 <= y < H and 0 <= x < W:
                    board[y, x] = torch.tensor((0.0, 0.0, 0.0))

        # --------------------------------------------------
        # Locks (holes)
        # --------------------------------------------------
        placed = getattr(self, "placed_locks", set())

        for lock in getattr(self, "lock_entries", []):
            y, x = lock.pos
            if not (0 <= y < H and 0 <= x < W):
                continue

            if (y, x) in placed:
                # placed locks → perceptually grey
                board[y, x] = torch.tensor((0.5, 0.5, 0.5))
            else:
                # unplaced locks → true colour
                col = lock.colour
                if max(col) > 1.0:
                    col = tuple(c / 255.0 for c in col)
                board[y, x] = torch.tensor(col, dtype=torch.float32)

        # --------------------------------------------------
        # Active key OR final reward
        # --------------------------------------------------
        this_key = None
        if hasattr(self, "lock_entries") and self.lock_entries:
            if self.current_key >= len(self.lock_entries):
                this_key = self.lock_entries[-1].next_key
            else:
                this_key = self.lock_entries[self.current_key].my_key

        if this_key is not None:
            y, x = this_key.pos
            if 0 <= y < H and 0 <= x < W:
                col = this_key.percived_colour
                if max(col) > 1.0:
                    col = tuple(c / 255.0 for c in col)
                board[y, x] = torch.tensor(col, dtype=torch.float32)

        # --------------------------------------------------
        # Player
        # --------------------------------------------------
        if getattr(self, "player_pos", None) is not None:
            py, px = self.player_pos
            if 0 <= py < H and 0 <= px < W:
                board[py, px] = torch.tensor((1.0, 0.0, 0.0))  # red

        # --------------------------------------------------
        # Final formatting
        # --------------------------------------------------
        board = board.clamp(0.0, 1.0)
        board = board.permute(2, 0, 1).contiguous()

        return board


               
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

    # Moved has_free_3x3 to class method for access in initialize_state
    def has_free_3x3(self, pos, walls, forbidden):
        y, x = pos
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                p = (y + dy, x + dx)
                if not (0 <= p[0] < self.key_y and 0 <= p[1] < self.key_x):
                    return False
                if p in walls or p in forbidden:
                    return False
        return True

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
                a = env.export_state()
                r = env.update(int(cmd))
                b = env.export_level()
                r += env.suplementary_reward(a, b)
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
