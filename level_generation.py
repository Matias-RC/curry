import torch
import numpy as np
import random

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


"""
This is an implementation based on Wilson's algorithm for maze generation 
Check: https://en.wikipedia.org/wiki/Maze_generation_algorithm
"""

h = 7
w = 7



class maze_cell:
    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.parent = None
        self.fwrd_links = []
    
    def __repr__(self):
        return f"({self.y},{self.x})"
    def __eq__(self, other):
        if isinstance(other, maze_cell):
            return self.y == other.y and self.x == other.x
        if isinstance(other, tuple):
            return (self.y,self.x) == other
        return NotImplemented
    def __hash__(self):
        return hash((self.y,self.x))
    def __add__(self, other):
        if isinstance(other, maze_cell):
            return maze_cell(self.y + other.y, self.x + other.x)
        if isinstance(other, tuple) and len(other) == 2:
            return (self.y + other[0], self.x + other[1])
        return NotImplemented
    def __radd__(self, other):
        # handles tuple + Position
        return self.__add__(other)
    def to_tuple(self):
        return (self.y,self.x)
    
def generate_maze(height, width):
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    visited = set()
    cells = []
    for i in range(height):
        cells.append([maze_cell(i,j) for j in range(width)])
    current = cells[random.randint(0, height-1)][random.randint(0, width-1)]
    
    visited.add(current.to_tuple())
    while len(visited) < height*width:
        candidates = []
        for dy,dx in dirs:
            new_pos = current + (dy,dx)
            if new_pos not in visited and 0 <= new_pos[0] <= height-1 and 0 <= new_pos[1] <= width-1:
                candidates.append(new_pos)
        if candidates:
            yp,xp = random.choice(candidates)
            new = cells[yp][xp]
            current.fwrd_links.append(new)
            new.parent = current
            visited.add((yp,xp))
            current = new
        else:
            current = current.parent
    return cells

def matrix_repr(cells, height, width):
    matrix = np.ones((2*height+1, 2*width+1))
    for i in cells:
        for j in i:
            relative = j+j
            relative = relative + (1,1)
            matrix[relative] = 0
            for link in j.fwrd_links:
                delta = (link.y - j.y,link.x-j.x)
                matrix[relative[0]+delta[0],relative[1]+delta[1]] = 0
    return matrix.astype(int)

#How to use:
# print(matrix_repr(generate_maze(h,w),h,w))

"""
Prim's Algorithm
"""

def prim_gen(height, width):
    dirs = {(-1,0), (1,0), (0,-1), (0,1)}
    matrix = np.ones((height, width))
    visited = set()
    known_neighbours = set()
    current = (random.randint(1, height-2), random.randint(1, width-2))

    def valid_neighbours(pos):
        neighbours = set()
        for dy, dx in dirs:
            n = (pos[0] + dy, pos[1] + dx)
            if 1 > n[0] or n[0] > height-2 or 1 > n[1] or n[1] > width-2 or n in visited:
                continue
            else:
                valid = True
                neigh_dirs = list(dirs - {(dy, dx), (-dy, -dx)})
                neigh_dirs.append((0, 0))
                for dyp, dxp in neigh_dirs:
                    if (n[0] + dyp, n[1] + dxp) in visited or (n[0] + dyp + dy, n[1] + dxp + dx) in visited:
                        valid = False
                        break
                if valid:
                    neighbours.add(n)
        return neighbours

    visited.add(current)
    known_neighbours |= valid_neighbours(current)

    while known_neighbours:
        matrix[current] = 0
        current = random.choice(list(known_neighbours))
        known_neighbours.discard(current)
        for dy, dx in dirs: #garbage dispossal
            if matrix[(current[0]+dy, current[1]+dx)] == 0:
                neigh_dirs = list(dirs - {(dy, dx), (-dy, -dx)})
                neigh_dirs.append((0, 0))
                for dyp, dxp in neigh_dirs:
                    known_neighbours.discard((current[0] + dyp, current[1] + dxp))
                    known_neighbours.discard((current[0] + dyp-dy, current[1] + dxp -dx))
                break
        visited.add(current)
        known_neighbours |= valid_neighbours(current)
        
    return matrix.astype(int)

# How to use:
# print(prim_gen(20,20))

def prim_longer_halls(height, width, epsilon):
    dirs = {(-1,0), (1,0), (0,-1), (0,1)}
    matrix = np.ones((height, width))
    visited = set()
    known_neighbours = set()
    hall_candidates = set()
    current = (random.randint(1, height-2), random.randint(1, width-2))

    def valid_neighbours(pos):
        neighbours = set()
        halls = set()
        for dy, dx in dirs:
            n = (pos[0] + dy, pos[1] + dx)
            if 1 > n[0] or n[0] > height-2 or 1 > n[1] or n[1] > width-2 or n in visited:
                continue
            else:
                valid = True
                neigh_dirs = list(dirs - {(dy, dx), (-dy, -dx)})
                neigh_dirs.append((0, 0))
                for dyp, dxp in neigh_dirs:
                    if (n[0] + dyp, n[1] + dxp) in visited or (n[0] + dyp + dy, n[1] + dxp + dx) in visited:
                        valid = False
                        break
                if valid:
                    if matrix[(pos[0] - dy, pos[1] - dx)] == 0:
                        halls.add(n)
                    else:
                        neighbours.add(n)
        return neighbours, halls

    visited.add(current)
    neigh, halls = valid_neighbours(current)
    known_neighbours |= neigh
    hall_candidates |= halls

    while known_neighbours or hall_candidates:
        matrix[current] = 0
        if random.random() > epsilon and hall_candidates:
            current = random.choice(list(hall_candidates))
        elif not known_neighbours:
            current = random.choice(list(hall_candidates))
        else:
            current = random.choice(list(known_neighbours))

        known_neighbours.discard(current)
        hall_candidates.discard(current)
        for dy, dx in dirs: #garbage dispossal
            if matrix[(current[0]+dy, current[1]+dx)] == 0:
                neigh_dirs = list(dirs - {(dy, dx), (-dy, -dx)})
                neigh_dirs.append((0, 0))
                for dyp, dxp in neigh_dirs:
                    known_neighbours.discard((current[0] + dyp, current[1] + dxp))
                    known_neighbours.discard((current[0] + dyp-dy, current[1] + dxp -dx))
                    hall_candidates.discard((current[0] + dyp, current[1] + dxp))
                    hall_candidates.discard((current[0] + dyp-dy, current[1] + dxp -dx))
                break
        visited.add(current)
        neigh, halls = valid_neighbours(current)
        known_neighbours |= neigh
        hall_candidates |= halls
        
    return matrix.astype(int)
#example:
#print(prim_longer_halls(10,10, 0.3))

"""
Checking how many halls and how long are they (warnings ahead this code is very badly writen and bearly understandable)
"""

test_case = np.matrix([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])


def hallgorithm(matrix):
    heads = []
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    for i in range(matrix.shape[0])[1:-1]:
        for j in range(matrix.shape[1])[1:-1]:
            if matrix[i,j] == 0:
                free_dirs = []
                for dy,dx in dirs:
                    if matrix[i+dy,j+dx] == 0:
                        free_dirs.append((dy,dx))   
                if len(free_dirs) == 3:
                    works = True
                    odd_index = [i for i,(dy,dx) in enumerate(free_dirs) if sum(dy2==0 for dy2,_ in free_dirs) not in (1,len(free_dirs)-1) or dx==0][0]
                    temp = free_dirs[:odd_index]+free_dirs[odd_index+1:]
                    for dy, dx in temp:
                        if 0< i+dy*2 < matrix.shape[0] and 0<j+dx*2<matrix.shape[1]:
                            if matrix[i+dy*2,j+dx*2] == 0:
                                works = False
                                break
                    if works:
                        heads.append([(i,j),free_dirs[odd_index]])
                elif len(free_dirs) == 2:
                    l_shape = (abs(free_dirs[0][0])+abs(free_dirs[1][0]))==1
                    works = False
                    direction_id = None
                    for dy, dx in free_dirs:
                        if 0< i+dy*2 < matrix.shape[0] and 0<j+dx*2<matrix.shape[1]:
                            if matrix[i+dy*2,j+dx*2] == 0:
                                direction_id = (dy,dx)
                                works = not works
                    if l_shape and works:
                        heads.append([(i,j),direction_id])
                elif len(free_dirs) == 1:
                    dy = free_dirs[0][0]
                    dx = free_dirs[0][1]

                    if 0< i+dy*2 < matrix.shape[0] and 0<j+dx*2<matrix.shape[1]:
                        if matrix[i+dy*2,j+dx*2] == 0:
                            heads.append([(i,j),(free_dirs[0][0],free_dirs[0][1])])
    return heads

def max_hall_length(matrix, head):
    tick_var = 0
    y, x = head[0]
    dy,dx = head[1]
    dirs = {(-1,0), (1,0), (0,-1), (0,1)}-{(dy,dx), (-dy,-dx)}
    while True:
        tick_var += 1
        pos = (y+dy*tick_var,x+dx*tick_var)
        for yp,xp in dirs:
            if matrix[pos[0]+yp,pos[1]+xp] == 0:
                if matrix[pos[0]+yp*2,pos[1]+xp*2] == 0:
                    return tick_var
        if matrix[pos] == 1:
            return tick_var-1

"""
Key note:
    - For buton box dynamic hall length has to be at least 2
Test:
"""

#print(test_case)
#hs = hallgorithm(test_case)
#print(hs)
#print(max_hall_length(test_case, hs[2]))

