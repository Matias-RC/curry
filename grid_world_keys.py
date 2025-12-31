from typing  import List, Set, Tuple
import random

key_holes = [
    (0,255,0),
    (0,0,255),
    (0,255,255),
    (255,0,255),
    (255,255,0),
    (255,128,0),
    (128,0,255),
    (128,128,0),
    (0,128,128),
    (128,0,0),
    (0,0,128)]

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
    def __init__(self, pos, colour, terminal):
        self.pos = pos
        self.colour = colour
        self.terminal = terminal

class Lock:
    def __init__(self, pos, colour, my_key, next_key):
        self.pos = pos
        self.colour = colour
        self.my_key = my_key
        self.next_key = next_key

class Environment:
    def __init__(self, spawn_size, key_maze_size, max_steps):
        self.spawn_x = spawn_size
        self.key_x, self.key_y = key_maze_size

        self.size_x = self.spawn_x + self.key_x
        self.size_y = self.key_y

        self.curriculum_stage = 0
        self.templates = templates
        self.keys = List[Key]
        self.walls = List[Set]
        self.lock_entries = List[Lock]

        self.current_key = None
        self.current_walls = None
        self.player_pos = None

        self.action_map = [(1,0),(0,-1),(0,1),(-1,0)]
        self.left_steps = max_steps
        self.max_steps = max_steps

        self.finished = False

    def base_walls_set(self):
        template_dir = (-1, 0)
        for item in self.lock_entries:
            thi_lock_dir = item.direction

    def initialize_stage(self):
        if self.curriculum_stage == 0:
            top

    def initialize_state(self):
        if self.curriculum_stage == 0:
            direction = None
            top_bottom = random.random()>self.spawn_y/(self.spawn_x*2)
            bottom = None

            lock_pos = [0,0]
            if top_bottom:
                direction =  (1,0)
                bottom = True
                lock_pos[0] = random.randint(0,self.spawn_x*2-2)
                if lock_pos[0]  > self.spawn_x-1:
                    bottom = False
                    direction = (-1,0)
                    lock_pos[0] = lock_pos[0] - self.spawn_x+1
                lock_pos[0] = lock_pos[0]+self.key_x
                lock_pos[1] = 7+self.spawn_y*bottom
                lock_pos = tuple(lock_pos)
            else:
                lock_pos = (random.randint(0, self.spawn_y-1)+8,self.size_x-8-1)
                direction = (0,1)
            colour = random.choice(key_holes)
            key_pos = (random.randint(0, self.size_y-1), random.randint(0, self.key_x-1))
            reward_pos = (random.randint(0, self.size_y-1), random.randint(0, self.key_x-1))
            levelLock = Lock(lock_pos, direction, colour, Key(key_pos, colour, False), Key(reward_pos, colour, True))
            #For altternative situations colors are pre organized

            self.lock_entries = [levelLock]
            self.keys = [levelLock.my_key, levelLock.next_key]
            self.current_key = self.keys[0]
            self.player_pos = (random.randint(8, self.size_y-8), random.randint(self.key_x, self.size_x-8))
            self.walls  = [{}]
            self.current_walls = self.walls[0]
        elif self.curriculum_stage < (self.key_x*self.size_y/9)+1:
            pass#Aneal support rewards down to zero
        else:
            pass
        pass

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
        

    def _supp_render_lock(self):
        pass

    def render(self):
        pass
    
    def load(self):
        pass