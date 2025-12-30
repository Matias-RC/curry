from typing  import List, Set, Tuple

"""
first step make the hallgorithm great again

diferentiate between spawn and key_maze

key hole desing with random key color

make key maze a progressive curriculum

make key hole matching a preogressive curriculum
"""

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

class key:
    def __init__(self, pos, colour):
        self.pos = pos
        self.colour = colour

class Lock:
    def __init__(self, pos, direction, colour, key):
        self.pos = pos
        self.direction = direction
        self.colour = colour
        self.key = key

    def render(self):
        pass

class Environment:
    def __init__(self, spawn_size, key_maze_size):
        spawn_x, spawn_y = spawn_size
        key_x = key_maze_size

        self.size_x = spawn_x + key_x + 8
        self.size_y = spawn_y + 16

        self.curriculum_stage = 0
        self.templates = templates
        self.keys = List[key]
        self.walls = List[Set]
        self.lock_entries = List[Lock]

        self.current_key = None
        self.player_pos = None
    
    def initialize_state(self):
        if self.curriculum_stage == 0:
            pass
        elif self.curriculum_stage == 1:
            pass
        else:
            pass
        pass

    def manhattan_distance(self, pos_1, pos_2):
        dy = abs(pos_1[0]-pos_2[0])
        dx = abs(pos_1[1]-pos_2[1])
        return dy + dx
    
    def update(self):
        pass

    def render(self):
        pass
    
    def load(self):
        pass