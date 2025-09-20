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

"""
Get all combinations for 1X3, 2x3, 3x2, 3x3,3x4, 4x3, and 4x4 for single button env and agent is one move away from win
"""

def get_tutorial():
    tutorial_base = np.matrix([
        [1,1,1,1,1],
        [1,2,3,4,1],
        [1,1,1,1,1]
    ])
    return [np.rot90(tutorial_base, k) for k in range(4)]

#print(get_tutorial())

def get_omanxm(n,m):
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    base = np.pad(np.zeros((n, m)), pad_width=1, mode='constant', constant_values=1)
    listOfCombinations = []
    for i in range(n+2)[1:-1]:
        for j in range(m+2)[1:-1]:
            for dy,dx in dirs:
                matrix = np.copy(base)
                matrix[i,j] = 4
                if 0<i+dy*2<(n+2) and 0<j+dx*2<(m+2):
                    if matrix[i+dy*2,j+dx*2] == 0:
                        matrix[i+dy,j+dx] = 3
                        matrix[i+dy*2,j+dx*2] = 2
                        listOfCombinations.append(matrix)
                        listOfCombinations.append(np.rot90(matrix))
    return listOfCombinations

#print(len(get_omanxm(4,4)))

"""
get the t-step away convos for nxm and 1 box next to goal // I dont recomend asking for over 7x7
"""

def get_nxm(n,m):
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    base = np.pad(np.zeros((n, m)), pad_width=1, mode='constant', constant_values=1)
    listOfCombinations = []
    for i in range(n+2)[1:-1]:
        for j in range(m+2)[1:-1]:
            for dy,dx in dirs:
                matrix = np.copy(base)
                matrix[i,j] = 4
                if 0<i+dy*2<(n+2) and 0<j+dx*2<(m+2):
                    if matrix[i+dy*2,j+dx*2] == 0:
                        matrix[i+dy,j+dx] = 3
                        for k in range(n+2)[1:-1]:
                            for l in range(m+2)[1:-1]:
                                if matrix[k, l] == 0:
                                    temp = np.copy(matrix)
                                    temp[k, l] = 2
                                    listOfCombinations.append(temp)
                                    listOfCombinations.append(np.rot90(temp))
                                if matrix[k, l] == 4:
                                    temp = np.copy(matrix)
                                    temp[k, l] = 6
                                    listOfCombinations.append(temp)
                                    listOfCombinations.append(np.rot90(temp))
    return listOfCombinations

# print(len(get_nxm(7,7))) ---- Dont go over 7x7 it explodes if in cluster maybe 9x9 but id say hard pass

"""
Last step for our introductory part of the curriculum is the random generation of levels for environments of the following characteristics:
    - 2 or 4 goals al next to solvable boxes
    -6x6 or 8x8 environments
"""

def generate_simple_random_easy(n,m,k):
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    base = np.pad(np.zeros((n, m)), pad_width=1, mode='constant', constant_values=1)
    for _ in range(k):
        pos = (random.randint(1,n-1), random.randint(1,m-1))
        woking_directions = []
        while True:
            if base[pos] != 0:
                pos = (random.randint(1,n-1), random.randint(1,m-1))
            else:
                base[pos] = 4
                for dy,dx in dirs:
                    if 0<pos[0]+dy*2<n+2 and 0<pos[1]+dx*2<m+2:
                        if base[pos[0]+dy*2,pos[1]+dx*2] == 0 and (base[pos[0]+dy,pos[1]+dx] == 0 or base[pos[0]+dy,pos[1]+dx]==4):
                            woking_directions.append((dy,dx))
                if woking_directions:
                    break
                else:
                    base[pos] = 0
        box_dir = random.choice(woking_directions)
        base[pos[0]+box_dir[0],pos[1]+box_dir[1]] = 3
    possible_player_positions = np.where((base == 0) | (base == 4))
    p_pos = random.choice(list(zip(*possible_player_positions)))
    base[p_pos] = 2 if base[p_pos] == 0 else 6
    return base.astype(int)

"""
footnotes:
    -if you need easyier levels change this (base[pos[0]+dy,pos[1]+dx] == 0 or base[pos[0]+dy,pos[1]+dx]==4)
"""

#print(generate_simple_random_easy(8,8,4))

"""
Box movement training phase
"""

def n_path(n):
    dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1),(1,1),(-1,1),(1,-1)]
    baned_region = set()
    def get_ocupied_positions(pos):
        oc = {pos}
        for dy,dx in dirs:
            oc.add((pos[0]+dy,pos[1]+dx))
        return oc
    current = (0,0)
    for _ in range(n):
        baned_region.add(current)
        options = []
        for dy, dx in dirs:
            if (current[0]+dy,current[1]+dx) in baned_region:
                pass
            else:
                options.append((current[0]+dy,current[1]+dx))
        for dy, dx in dirs:
            baned_region.add((current[0]+dy,current[1]+dx))
        if not options:
            break
        current = random.choice(options)
    for dy, dx in dirs:
        baned_region.add((current[0]+dy,current[1]+dx))
    ys, xs = zip(*baned_region)
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)
    matrix = np.ones((max_y - min_y + 1, max_x - min_x + 1))
    for y, x in baned_region:
        matrix[y-min_y, x-min_x] = 0
    matrix[(-min_y,-min_x)] = 3
    matrix[(1-min_y,1-min_x)] = 2
    matrix[(current[0]-min_y,current[1]-min_x)] = 4
    return np.pad(matrix, pad_width=1, mode='constant', constant_values=1).astype(int)

#print(n_path(1))

"""
We are almost at level solving the last bit of tutorial is to teach the agent on how to avoid obstacles
"""
templates = [
    np.matrix([
        [0,0,0],
        [3,1,4],
    ]),
    np.matrix([
        [0,1,0],
        [3,1,4],
        [0,0,0]
    ]),
    np.matrix([
        [0,0,0],
        [0,1,0],
        [3,1,4],
        [0,1,0],
    ]),
    np.matrix([
        [0,0,0],
        [3,1,0],
        [0,4,0]
    ]),
    np.matrix([
        [0,0,0,0],
        [3,1,1,0],
        [0,1,4,0],
    ])
]


def nxmfortemps(n,m,temps):
    base = np.pad(np.zeros((n, m)), pad_width=1, mode='constant', constant_values=1)
    temp = random.choice(temps)
    temp = np.rot90(temp, random.randint(0,3))
    possible_positions = []
    for i in range(n+2)[1:-1]:
        for j in range(m+2)[1:-1]:
            if i+temp.shape[0]-1<n+1 and j+temp.shape[1]-1<n+1:
                possible_positions.append((i,j))
    y,x = random.choice(possible_positions)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            base[y+i,x+j] = temp[i,j]
    return base
def nxmfor_k_temps(n, m, temps, k, spacing=1):
    base = np.pad(np.zeros((n, m)), pad_width=1, mode='constant', constant_values=1)
    placed = 0
    attempts = 0
    max_attempts = k * 100
    while placed < k and attempts < max_attempts:
        attempts += 1

        temp = random.choice(temps)
        temp = np.rot90(temp, random.randint(0, 3))

        possible_positions = []
        for i in range(1, n+1):
            for j in range(1, m+1):
                if i + temp.shape[0] - 1 < n+1 and j + temp.shape[1] - 1 < m+1:
                    sub = base[
                        i-spacing : i+temp.shape[0]+spacing,
                        j-spacing : j+temp.shape[1]+spacing
                    ]
                    if sub.shape[0] == temp.shape[0] + 2*spacing and sub.shape[1] == temp.shape[1] + 2*spacing:
                        area = base[i:i+temp.shape[0], j:j+temp.shape[1]]
                        if np.all((area == 0) | (temp == 0)) and np.all(sub == 0):
                            possible_positions.append((i, j))
        if not possible_positions:
            continue
        y, x = random.choice(possible_positions)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if temp[i, j] != 0:
                    base[y+i, x+j] = temp[i, j]

        placed += 1

    return base

#print(nxmfor_k_temps(10, 10, templates, k=5))

def add_char_rand(matrix):
    pos = random.choice(list(zip(*np.argwhere(matrix == 0))))
    matrix[pos] = 2
    return matrix

def lvl_connect(m1,m2, form="any"):
    y,x = m1.shape
    upper_row = m1[1, 1:-1]
    lower_row = m1[y-2, 1:-1]
    left_column = m1[1:-1, 1]
    right_column = m1[1:-1, x-2]
    listed = [upper_row,lower_row,left_column,right_column]
    s = None

    if form == "upper":
        s= 0
    elif form == "lower":
        s=1
    elif form == "left":
        s=2
    elif form == "right":
        s = 3
    else:
        s = random.randint(0,3)

    possible_cons = list(zip(np.argwhere(listed[s] == 0)))
    if not possible_cons:
        raise ValueError("Ensure exterior rows and/or columns (for non padded matrices) have zeroes in them")
    selected_con = random.choice(possible_cons)
    selected_con = tuple(selected_con[0].tolist())
    if s == 0:
        lower_m2 = m2[m2.shape[0]-2, 1:-1]
        m2_con = random.choice(list(zip(np.argwhere(lower_m2 == 0))))[0].tolist()[0]
        m2_length = m2.shape[1]
        m2_substract = m2_con+2 #we add 2 because first we skipped a beat [m2.shape[0]-2, 1:-1] and also indexes are not length
        rm2 = m2_length-m2_substract
        m1_length = x
        m1_substract = selected_con[1]+2
        rm1 = m1_length-m1_substract
        L = m1_substract if m1_substract>m2_substract else m2_substract
        Lp = rm1 if rm1>rm2 else rm2
        tot = L + Lp
        base = np.ones((y+m2.shape[0]-1, tot))
        for i in range(y):
            for j in range(x):
                delta = m2_substract-m1_substract if m2_substract-m1_substract > 0 else 0
                base[i+m2.shape[0]-1,j+delta] = m1[i,j]
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                delta = m1_substract-m2_substract if m1_substract-m2_substract > 0 else 0
                base[i,j+delta] = m2[i,j]
        base[m2.shape[0]-1,L-1] = 0
        return base


#t = nxmfortemps(5,5,templates)
#print(lvl_connect(test_case, t, "upper"))
