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
#print(matrix_repr(generate_maze(h,w),h,w))

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
                            for dyp, dxp in [(dx,dy),(-dx,-dy)]: #Only looking to the sides to check if clear thats rotation
                                if matrix[i+dy+dyp,j+dx+dxp] == 0:
                                    works = False
                                    break
                    if works and matrix[i-free_dirs[odd_index][0],j-free_dirs[odd_index][1]] != 0:
                        heads.append([(i,j),free_dirs[odd_index]])
                elif len(free_dirs) == 2:
                    l_shape = (abs(free_dirs[0][0])+abs(free_dirs[1][0]))==1
                    works = []
                    for dy, dx in free_dirs:
                        t = [0,0,0]
                        if 0< i+dy*2 < matrix.shape[0] and 0<j+dx*2<matrix.shape[1]:
                            if matrix[i+dy*2,j+dx*2] == 0:
                                direction_id = (dy,dx)
                                t[0] = 1
                        item_val = 0
                        for dyp, dxp in [(dx,dy),(-dx,-dy)]: #Only looking to the sides to check if clear thats rotation
                            item_val += 1
                            if 0< i+dy+dyp < matrix.shape[0] and  0<j+dx+dxp<matrix.shape[1]:
                                if matrix[i+dy+dyp,j+dx+dxp] == 0:
                                    t[item_val] = 1
                        works.append(t)
                    truth_value = True
                    for t_val in works[0]:
                        for tp_val in works[1]:
                            if t_val and tp_val:
                                truth_value = False
                    if works[0][0] == 0 and works[1][0] == 0:
                        truth_value = False
                    if l_shape and truth_value:
                        direction_id = None
                        for idx, item in enumerate(works):
                            if item[0] == 1:
                                direction_id = free_dirs[idx]

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
                for ypp, xpp in dirs:
                    if matrix[pos[0]+yp+xpp,pos[1]+xp+ypp] == 0:
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
def nxmfor_k_temps(n, m, temps, k, spacing=2):
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

def lvl_connect(m1,m2, form="any", specific_pos="any"):
    y,x = m1.shape
    upper_row = m1[1, 1:-1]
    lower_row = m1[y-2, 1:-1]
    left_column = m1[1:-1, 1]
    right_column = m1[1:-1, x-2]
    listed = [upper_row,lower_row,left_column,right_column]
    s = None

    if form == "upper" or form == 0:
        s= 0
    elif form == "lower" or form == 1:
        s=1
    elif form == "left" or form == 2:
        s=2
    elif form == "right" or form == 3:
        s = 3
    else:
        s = random.randint(0,3)

    possible_cons = np.argwhere(listed[s] == 0).flatten()
    if len(possible_cons) == 0:
        raise ValueError("Ensure exterior rows and/or columns (for non padded matrices) have zeroes in them")
    selected_con = random.choice(possible_cons).item()
    if specific_pos != "any":
        if s == 0 or s == 1:
            selected_con = specific_pos[1]
        elif s==2 or s==3:
            selected_con = specific_pos[0]

    if s == 0:
        m2_con = random.choice(np.argwhere(m2[m2.shape[0]-2, 1:-1] == 0).flatten()).item()
        m2_length = m2.shape[1]
        m2_substract = m2_con+2
        rm2 = m2_length-m2_substract
        m1_length = x
        m1_substract = selected_con+2
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

    if s == 1:
        m2_con = random.choice(np.argwhere(m2[1, 1:-1] == 0).flatten()).item()
        m2_length = m2.shape[1]
        m2_substract = m2_con+2
        rm2 = m2_length-m2_substract
        m1_length = x
        m1_substract = selected_con+2
        rm1 = m1_length-m1_substract
        L = m1_substract if m1_substract>m2_substract else m2_substract
        Lp = rm1 if rm1>rm2 else rm2
        tot = L + Lp
        base = np.ones((y+m2.shape[0]-1, tot))
        for i in range(y):
            for j in range(x):
                delta = m2_substract-m1_substract if m2_substract-m1_substract > 0 else 0
                base[i,j+delta] = m1[i,j]
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                delta = m1_substract-m2_substract if m1_substract-m2_substract > 0 else 0
                base[i+y-1,j+delta] = m2[i,j]
        base[y-1,L-1] = 0
        return base

    if s == 2:
        m2_con = random.choice(np.argwhere(m2[1:-1, m2.shape[1]-2] == 0).flatten()).item()
        m2_height = m2.shape[0]
        m2_substract = m2_con+2
        rm2 = m2_height-m2_substract
        m1_height = y
        m1_substract = selected_con+2
        rm1 = m1_height-m1_substract
        H = m1_substract if m1_substract>m2_substract else m2_substract
        Hp = rm1 if rm1>rm2 else rm2
        tot = H + Hp
        base = np.ones((tot, x+m2.shape[1]-1))
        for i in range(y):
            for j in range(x):
                delta = m2_substract-m1_substract if m2_substract-m1_substract > 0 else 0
                base[i+delta,j+m2.shape[1]-1] = m1[i,j]
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                delta = m1_substract-m2_substract if m1_substract-m2_substract > 0 else 0
                base[i+delta,j] = m2[i,j]
        base[H-1,m2.shape[1]-1] = 0
        return base

    if s == 3:
        m2_con = random.choice(np.argwhere(m2[1:-1, 1] == 0).flatten()).item()
        m2_height = m2.shape[0]
        m2_substract = m2_con+2
        rm2 = m2_height-m2_substract
        m1_height = y
        m1_substract = selected_con+2
        rm1 = m1_height-m1_substract
        H = m1_substract if m1_substract>m2_substract else m2_substract
        Hp = rm1 if rm1>rm2 else rm2
        tot = H + Hp
        base = np.ones((tot, x+m2.shape[1]-1))
        for i in range(y):
            for j in range(x):
                delta = m2_substract-m1_substract if m2_substract-m1_substract > 0 else 0
                base[i+delta,j] = m1[i,j]
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                delta = m1_substract-m2_substract if m1_substract-m2_substract > 0 else 0
                base[i+delta,j+x-1] = m2[i,j]
        base[H-1,x-1] = 0
        return base


#t = nxmfortemps(5,5,templates)
#print(lvl_connect(test_case,t, "left", [1,1]))


"""
the maze generators are fine but it would be golden if you could add box pushing to the maze complexes while connecting other challenges.

Hypothesis this would test the hability of the encoder to discenr between a generally reconstructable representation and 
                                                                                      a convenient representation.
"""

def hall_boxes(matrix, heads, intended_conection_sides):
    upper_heads = []
    lower_heads = []
    left_heads = []
    right_heads = []
    for i in heads:
        y, x = i[0]
        if y == 1 and 0<x<matrix.shape[1]-2:
            upper_heads.append(i)
        elif y == matrix.shape[0]-2 and 0<x<matrix.shape[1]-1:
            lower_heads.append(i)
        elif 0<y<matrix.shape[0]-2 and x == 1:
            left_heads.append(i)
        elif 0<y<matrix.shape[0]-1 and x == matrix.shape[1]-2:
            right_heads.append(i)
    if intended_conection_sides[0]:
        if upper_heads:
            idx = random.randrange(len(upper_heads))
            upper_heads.pop(idx)
        elif np.argwhere(matrix[1, 1:-1] == 0).size > 0:
            pass
        else:
            matrix[1, random.randint(1,matrix.shape[1]-2)] = 0
    if intended_conection_sides[1]:
        if lower_heads:
            idx = random.randrange(len(lower_heads))
            lower_heads.pop(idx)
        elif np.argwhere(matrix[matrix.shape[0]-2, 1:-1] == 0).size > 0:
            pass
        else:   
            matrix[matrix.shape[0]-2, random.randint(1,matrix.shape[1]-2)] = 0
    if intended_conection_sides[2]:
        if left_heads:
            idx = random.randrange(len(left_heads))
            left_heads.pop(idx)
        elif (matrix[1:-1, 1] == 0).any():
            pass
        else:
            matrix[random.randint(1, matrix.shape[0] - 2), 1] = 0
    if intended_conection_sides[3]:
        if right_heads:
            idx = random.randrange(len(right_heads))
            right_heads.pop(idx)
        elif (matrix[1:-1, matrix.shape[1] - 2] == 0).any():
            pass
        else:
            matrix[random.randint(1, matrix.shape[0] - 2), matrix.shape[1] - 2] = 0
    r = set(tuple(i[0]) for i in (upper_heads + lower_heads + left_heads + right_heads))
    rp = set(tuple(i[0]) for i in heads)
    r_f = r.union(rp)
    heads = [ [h] for h in r_f ]
    return heads, matrix

"""
Finally after lots of procesing you can reliably incorporate boxes into the halls
"""


def incorporate_boxes_and_player(matrix, heads):
    idx = random.randrange(len(heads))
    h = heads.pop(idx)
    y, x = h[0]
    matrix[y, x] = 2

    # Surround h[0] with zeros (excluding contour)
    y_max, x_max = matrix.shape
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if (dy == 0 and dx == 0):
                continue
            if 0 < ny < y_max - 1 and 0 < nx < x_max - 1:
                matrix[ny, nx] = 0

    for i in heads:
        l = max_hall_length(matrix, i)
        if l < 2:
            continue
        s = random.randint(1, l - 1)
        pos = (i[0][0] + i[1][0] * s, i[0][1] + i[1][1] * s)
        matrix[i[0]] = 4
        matrix[pos] = 3

    return matrix


def easy_to_use_halls_without_connect(matrix):
    l = hallgorithm(matrix)
    temp = []
    for i in l:
        if max_hall_length(matrix, i) != 1:
            temp.append(i)
    return incorporate_boxes_and_player(matrix, temp).astype(int)
def easy_to_use_halls_connect(matrix):
    y, x = matrix.shape
    intended_connection = random.randint(0,3)
    icl = [0,0,0,0]
    icl[intended_connection] = 1
    l = hallgorithm(matrix)
    temp = []
    for i in l:
        if max_hall_length(matrix, i) != 1:
            temp.append(i)
    cands = []
    alts = []
    for i in temp:
        if icl[0]:
            if i[0][0] == 1:
                cands.append(i)
            else:
                alts.append(i)
        if icl[1]:
            if i[0][0] == y-2:
                cands.append(i)
            else:
                alts.append(i)
        if icl[2]:
            if i[0][1] == 1:
                cands.append(i)
            else:
                alts.append(i)
        if icl[3]:
            if i[0][1] == x-2:
                cands.append(i)
            else:
                alts.append(i)
    if cands:
        idx = random.randrange(len(cands))
        con_pos_t = cands.pop(idx)[0]
    else:
        con_pos_t = "any"
    temp = alts
    matrix = incorporate_boxes_and_player(matrix, temp).astype(int)
    t = nxmfortemps(5,5,templates)

    return lvl_connect(matrix,t, intended_connection,"any")
"""
Legacy:
#########################################################
#########################################################
#########################################################
#########################################################
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple, Dict, Iterator
import numpy as np
import torch


@dataclass
class Node:
    state: Any
    parent: Optional['Node'] = None
    action: Optional[Any] = None
    
    def trajectory(self) -> List['Node']:
        """
        Reconstructs the trajectory (path) from the root to this node.
        """
        node, path = self, []

        while node:
            path.append(node.action)
            node = node.parent
            
        return list(reversed(path))[1:]
    
    def statesList(self) -> List['Node']:
        node, path = self, []

        while node:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))
    
    def nodesList(self) -> List['Node']:
        node, path = self, []
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    

class PriorityQueue:
    """
    PriorityQueue data structure.
    Code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    def __init__(self):
        self.Heap = []
        self.Count = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        PriorityQueue.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = PriorityQueue.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

    # Heap functions, mimicking heapq operations
    @staticmethod
    def heappush(heap, item):
        heap.append(item)
        PriorityQueue._siftdown(heap, 0, len(heap)-1)

    @staticmethod
    def heappop(heap):
        lastelt = heap.pop()    # raises appropriate IndexError if heap is empty
        if heap:
            returnitem = heap[0]
            heap[0] = lastelt
            PriorityQueue._siftup(heap, 0)
            return returnitem
        return lastelt

    @staticmethod
    def _siftup(heap, pos):
        endpos = len(heap)
        startpos = pos
        newitem = heap[pos]
        childpos = 2 * pos + 1    # leftmost child position
        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and not heap[childpos] < heap[rightpos]:
                childpos = rightpos
            heap[pos] = heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        heap[pos] = newitem
        PriorityQueue._siftdown(heap, startpos, pos)

    @staticmethod
    def _siftdown(heap, startpos, pos):
        newitem = heap[pos]
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if newitem < parent:
                heap[pos] = parent
                pos = parentpos
            else:
                break
        heap[pos] = newitem


class Astar:
    def __init__(self, gameState, max_steps):
        self.gameState = gameState
        self.posWalls = self._pos_of(1)
        # posGoals include cells marked as 4 and 5
        self.posGoals = self._pos_of(4) + self._pos_of(5)+ self._pos_of(6)
        self.max_steps = max_steps

    def _pos_of(self, value):
        return tuple(tuple(x) for x in np.argwhere(self.gameState == value))

    def posOfPlayer(self):
        return tuple(np.argwhere(self.gameState == 2)[0])

    def posOfBoxes(self):
        return tuple(tuple(x) for x in np.argwhere((self.gameState == 3) | (self.gameState == 5)))

    def isEndState(self, posBox):
        return sorted(posBox) == sorted(self.posGoals)

    def isLegalAction(self, action, posPlayer, posBox):
        # Determine next position based on action vector.
        # If action_char is uppercase, it means a push (move two cells)
        direction = action[0:2]
        action_char = action[2]
        if action_char.isupper():
            newPos = (posPlayer[0] + 2 * direction[0], posPlayer[1] + 2 * direction[1])
        else:
            newPos = (posPlayer[0] + direction[0], posPlayer[1] + direction[1])
        # Check if the new position is blocked by a box or a wall.
        return newPos not in posBox and newPos not in self.posWalls

    def legalActions(self, posPlayer, posBox):
        # Each action is defined by its movement vector and two possible characters:
        # lowercase (normal move) and uppercase (push move)
        baseActions = [
            ([-1, 0], 'u', 'U'),
            ([1, 0], 'd', 'D'),
            ([0, -1], 'l', 'L'),
            ([0, 1], 'r', 'R')
        ]
        validActions = []
        for move, low, up in baseActions:
            nextPos = (posPlayer[0] + move[0], posPlayer[1] + move[1])
            # If next position has a box then only a push is allowed; otherwise, only a normal move.
            if nextPos in posBox:
                # Create action tuple: movement vector and push indicator.
                action = move + [up]
            else:
                action = move + [low]
            if self.isLegalAction(action, posPlayer, posBox):
                validActions.append(tuple(action))
        return tuple(validActions)

    def updateState(self, posPlayer, posBox, action):
        newPosPlayer = (posPlayer[0] + action[0], posPlayer[1] + action[1])
        posBox = list(map(list, posBox))
        if action[2].isupper():
            # When pushing, remove the box at the new player position and add it pushed.
            try:
                posBox.remove(list(newPosPlayer))
            except ValueError:
                # Should not happen if action is legal.
                pass
            newBoxPos = [posPlayer[0] + 2 * action[0], posPlayer[1] + 2 * action[1]]
            posBox.append(newBoxPos)
        return newPosPlayer, tuple(tuple(x) for x in posBox)

    def aStarSearch(self):
        start_state = (self.posOfPlayer(), self.posOfBoxes())
        frontier = PriorityQueue()
        actions_queue = PriorityQueue()
        # Push starting state and an empty action sequence
        frontier.push([start_state], self.heuristic(start_state))
        actions_queue.push([""], self.heuristic(start_state))
        exploredSet = set()
        steps_left = self.max_steps
        while not frontier.isEmpty() and steps_left > 0:
            node = frontier.pop()
            node_action = actions_queue.pop()
            current_state = node[-1]
            if self.isEndState(current_state[1]):
                # Return the concatenated action sequence (excluding the initial empty string)
                return ''.join(node_action)[1:]
            if current_state not in exploredSet:
                exploredSet.add(current_state)
                for action in self.legalActions(current_state[0], current_state[1]):
                    newPosPlayer, newPosBox = self.updateState(current_state[0], current_state[1], action)
                    if self.isFailed(newPosBox):
                        continue
                    cost = len(''.join(node_action))  # Using length as cost
                    new_state = (newPosPlayer, newPosBox)
                    h = self.heuristic(new_state)
                    frontier.push(node + [new_state], cost + h)
                    actions_queue.push(node_action + [action[2]], cost + h)
                    steps_left-1
        return 'x'

    def heuristic(self, state):
        """
        Heuristic function: sum of Manhattan distances for boxes not on goal.
        Code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
        """
        posPlayer, posBox = state
        distance = 0
        # Boxes already on goals are not considered.
        boxes_to_move = list(set(posBox) - set(self.posGoals))
        # Remaining goals.
        remaining_goals = list(set(self.posGoals) - set(posBox))
        # For simplicity, pair each box with a goal in order.
        for i in range(min(len(boxes_to_move), len(remaining_goals))):
            distance += abs(boxes_to_move[i][0] - remaining_goals[i][0]) + abs(boxes_to_move[i][1] - remaining_goals[i][1])
        return distance

    def cost(self, actions):
        """
        A simple cost function: one cost per move.
        Code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
        """
        return len(actions)

    def isFailed(self, posBox):
        """
        Check if a state is potentially failed (deadlock), then prune the search.
        """
        # Use self.posGoals and self.posWalls
        rotatePattern = [
            [0,1,2,3,4,5,6,7,8],
            [2,5,8,1,4,7,0,3,6],
            list(reversed([0,1,2,3,4,5,6,7,8])),
            list(reversed([2,5,8,1,4,7,0,3,6]))
        ]
        flipPattern = [
            [2,1,0,5,4,3,8,7,6],
            [0,3,6,1,4,7,2,5,8],
            list(reversed([2,1,0,5,4,3,8,7,6])),
            list(reversed([0,3,6,1,4,7,2,5,8]))
        ]
        allPattern = rotatePattern + flipPattern

        for box in posBox:
            if box not in self.posGoals:
                board = [
                    (box[0]-1, box[1]-1), (box[0]-1, box[1]), (box[0]-1, box[1]+1),
                    (box[0],   box[1]-1), (box[0],   box[1]), (box[0],   box[1]+1),
                    (box[0]+1, box[1]-1), (box[0]+1, box[1]), (box[0]+1, box[1]+1)
                ]
                for pattern in allPattern:
                    newBoard = [board[i] for i in pattern]
                    if newBoard[1] in self.posWalls and newBoard[5] in self.posWalls:
                        return True
                    elif newBoard[1] in posBox and newBoard[2] in self.posWalls and newBoard[5] in self.posWalls:
                        return True
                    elif newBoard[1] in posBox and newBoard[2] in self.posWalls and newBoard[5] in posBox:
                        return True
                    elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox:
                        return True
                    elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in self.posWalls and newBoard[3] in self.posWalls and newBoard[8] in self.posWalls:
                        return True
        return False


def grid_reconstruct(grid_dims, state, posGoals, posWalls):
    posPlayer, posBox = state
    y, x = grid_dims
    grid = np.zeros(grid_dims)
    for i in range(y):
        for j in range(x):
            pos = (i, j)
            if pos == posPlayer:
                if pos in posGoals:
                    grid[pos] = 6
                else:
                    grid[pos] = 2
            elif pos in posWalls:
                grid[pos] = 1
            elif pos in posBox:
                if pos in posGoals:
                    grid[pos] = 5
                else:
                    grid[pos] = 3
            elif pos in posGoals:
                grid[pos] = 4
            else:
                grid[pos] = 0
    return grid

def heuristic(state, posGoals):
    """
    Heuristic function: sum of Manhattan distances for boxes not on goal.
    Code source: https://github.com/dangarfield/sokoban-solver/blob/main/solver.py
    """
    posPlayer, posBox = state
    distance = 0
    # Boxes already on goals are not considered.
    boxes_to_move = list(set(posBox) - set(posGoals))
    # Remaining goals.
    remaining_goals = list(set(posGoals) - set(posBox))
    # For simplicity, pair each box with a goal in order.
    for i in range(min(len(boxes_to_move), len(remaining_goals))):
        distance += abs(boxes_to_move[i][0] - remaining_goals[i][0]) + abs(boxes_to_move[i][1] - remaining_goals[i][1])
    return distance

def parse_level(string_grid):
    """Parses a textual Sokoban-like level into a numeric grid representation."""
    lines = string_grid.strip().split("\n")
    max_width = max(len(line) for line in lines)  # Find the widest line

    height = len(lines)
    width = max_width
    grid = np.ones((height, width), dtype=int) 
    char_map = {
        " ": 0,  # Empty space
        "#": 1,  # Wall
        "@": 2,  # Player
        "$": 3,  # Box
        ".": 4,  # Button/goal
        "*": 5,  # Box on goal
        "+": 6,  # Player on goal
    }

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            grid[y, x] = char_map.get(char, 1)

    return grid


class BaseSokobanManager:
    def __init__(self):
        self.grid_dims = None
        self.grid_base = None
        self.posWalls = None
        self.posGoals = None
    
    def PosOfPlayer(self, grid):
        return tuple(np.argwhere((grid == 2) | (grid == 6))[0])# idplayer = 2

    def PosOfBoxes(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 3) | (grid == 5)))

    def PosOfWalls(self, grid):
        return tuple(tuple(x) for x in np.argwhere(grid == 1))

    def PosOfGoals(self, grid):
        return tuple(tuple(x) for x in np.argwhere((grid == 4) | (grid == 5) | (grid == 6)))

    def isEndState(self, node):
        return sorted(node.state[1]) == sorted(self.posGoals)

    def initializer(self, initial_state):
        self.grid_dims = initial_state.shape
        empty_grid = np.copy(initial_state)
        empty_grid[(empty_grid == 2) | (empty_grid == 3)] = 0
        empty_grid[(empty_grid == 5) | (empty_grid == 6)] = 4
        self.grid_base = empty_grid
        self.posWalls = self.PosOfWalls(initial_state)
        self.posGoals = self.PosOfGoals(initial_state)
        node = Node(state=(self.PosOfPlayer(initial_state), self.PosOfBoxes(initial_state)))
        return node
    
    def isFailed(self, node):
        posPlayer, posBox = node.state
        """This function used to observe if the state is potentially failed, then prune the search. credits to:
            https://github.com/dangarfield/sokoban-solver/blob/main/solver.py for this function and most fast update logic"""
        rotatePattern = [[0,1,2,3,4,5,6,7,8],
                        [2,5,8,1,4,7,0,3,6],
                        [0,1,2,3,4,5,6,7,8][::-1],
                        [2,5,8,1,4,7,0,3,6][::-1]]
        flipPattern = [[2,1,0,5,4,3,8,7,6],
                        [0,3,6,1,4,7,2,5,8],
                        [2,1,0,5,4,3,8,7,6][::-1],
                        [0,3,6,1,4,7,2,5,8][::-1]]
        allPattern = rotatePattern + flipPattern

        for box in posBox:
            if box not in self.posGoals:
                board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                        (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1),
                        (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
                for pattern in allPattern:
                    newBoard = [board[i] for i in pattern]
                    if newBoard[1] in self.posWalls and newBoard[5] in self.posWalls: return True
                    elif newBoard[1] in posBox and newBoard[2] in self.posWalls and newBoard[5] in self.posWalls: return True
                    elif newBoard[1] in posBox and newBoard[2] in self.posWalls and newBoard[5] in posBox: return True
                    elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                    elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in self.posWalls and newBoard[3] in self.posWalls and newBoard[8] in self.posWalls: return True
        return False

    def grid_state(self, player_pos, posBox):
        grid = self.grid_base.copy()
        if player_pos in self.posGoals:
            grid[player_pos] = 6  # Player on Button
        else:
            grid[player_pos] = 2  # Normal Player
        
        for box in list(posBox):
            if box in self.posGoals:
                grid[box] = 5  # Box on Button
            else:
                grid[box] = 3  # Normal Box
        
        return grid
    
    def distance_between_states(state1, state2):
        """
        Euclidean Difference
        """
        difference = state1 -state2
        return np.linalg.norm(difference)
    
    def state_to_tensor(self, grid):
        grid = torch.tensor(grid, dtype=torch.float32)
        grid = grid.flatten()
        return grid
    
    def toTensor(self, state_action_pair, distance_to_finale, library_size):
        """
        state: numpy matrix -> flatten
        action: number -> to one-hot
        """
        grid, action = state_action_pair
        grid = torch.tensor(grid, dtype=torch.float32)
        grid = grid.flatten()
        one_hot = torch.zeros(library_size)
        one_hot[action] = 1
        return (grid, one_hot, distance_to_finale)
    
    def node_to_data(self, node):
        actionsList = node.trajectory() # Trajectory is temporal name (really isn't the trajectory but instead the list of actions taken)
        statesList = node.statesList()[:-1]
        return {"actions":actionsList, "states":statesList}

class SokobanManager(BaseSokobanManager):
    def isLegalAction(self, action, posPlayer, posBoxes):
        dx, dy = action[0]
        factor = 2 if action[1] else 1
        target = (posPlayer[0] + factor * dx, posPlayer[1] + factor * dy)
        return target not in self.posWalls and target not in posBoxes

    def legalUpdate(self, macro, game_data, node):
        player, posBoxes = game_data
        boxes = set(posBoxes)

        for dx, dy in macro[0]:
            nextPos = (player[0] + dx, player[1] + dy)
            push = nextPos in boxes
            action = ((dx, dy), push)
            if not self.isLegalAction(action, player, boxes):
                return False, None
            player = nextPos
            if push:
                boxes.remove(player)
                boxes.add((player[0] + dx, player[1] + dy))
        posBoxes = tuple(boxes)
        new_node = Node(state=(player, posBoxes), parent=node, action=macro[1])
        return not self.isFailed(new_node), new_node
    

class InversedSokobanManager(BaseSokobanManager):
    def isLegalInversion(self, action, posPlayer, posBox): 
        xPlayer, yPlayer = posPlayer
        x1, y1 = xPlayer - action[0], yPlayer - action[1]
        return (x1, y1) not in posBox + self.posWalls and not sorted(posBox) == sorted(self.posGoals)
    
    def legalInvertedUpdate(self, macro, game_data, node):
        player, posBoxes = game_data
        boxes = set(posBoxes)

        for dx, dy in macro[0]:
            new_player = (player[0] - dx, player[1] - dy)

            if new_player in self.posWalls:
                return False, None

            pull_candidate = (player[0] + dx, player[1] + dy)
            pull = pull_candidate in boxes

            if pull:
                boxes.remove(pull_candidate)
                boxes.add(player)

            if not self.isLegalInversion((dx, dy), player, tuple(boxes)):
                return False, None
            player = new_player

        new_state = (player, tuple(boxes))
        new_node = Node(state=new_state, parent=node, action=macro[1])
        return True, new_node

def deadlockagainstwall(posBox, posGoals, posWalls,x,y):
    if (y, x) in posGoals:
        return False

    rotatePattern = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 5, 8, 1, 4, 7, 0, 3, 6],
        list(reversed([0, 1, 2, 3, 4, 5, 6, 7, 8])),
        list(reversed([2, 5, 8, 1, 4, 7, 0, 3, 6]))
    ]
    flipPattern = [
        [2, 1, 0, 5, 4, 3, 8, 7, 6],
        [0, 3, 6, 1, 4, 7, 2, 5, 8],
        list(reversed([2, 1, 0, 5, 4, 3, 8, 7, 6])),
        list(reversed([0, 3, 6, 1, 4, 7, 2, 5, 8]))
    ]
    allPattern = rotatePattern + flipPattern

    board = [
        (y - 1, x - 1), (y - 1, x), (y - 1, x + 1),
        (y,     x - 1), (y,     x), (y,     x + 1),
        (y + 1, x - 1), (y + 1, x), (y + 1, x + 1)
    ]
    
    for pattern in allPattern:
        newBoard = [board[i] for i in pattern]
        if newBoard[1] in posWalls and newBoard[5] in posWalls:
            return True  # simple corner deadlock
        elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls:
            return True
        elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox:
            return True
        elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox:
            return True
        elif (newBoard[1] in posBox and newBoard[6] in posBox and 
              newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls):
            return True
    return False
def is_connected(grid):
    """
    Check if all walkable floor tiles (floor 0 and goal 4) are connected.
    Uses a simple breadth-first search.
    """
    visited = np.zeros_like(grid, dtype=bool)
    floor_tiles = np.argwhere((grid == 0) | (grid == 4))
    if len(floor_tiles) == 0:
        return False

    start = tuple(floor_tiles[0])
    queue = [start]
    while queue:
        y, x = queue.pop(0)
        if visited[y, x]:
            continue
        visited[y, x] = True
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                if not visited[ny, nx] and (grid[ny, nx] == 0 or grid[ny, nx] == 4):
                    queue.append((ny, nx))
    for y, x in floor_tiles:
        if not visited[y, x]:
            return False
    return True


def make_base(width, height, n_boxes, n_walls, seed=None):
    count = 0
    # Loop until a valid, connected grid (and valid player placement, if required) is created.
    while count < 100:
        # Create grid with floor (0) and border walls (1)
        grid = np.zeros((height, width), dtype=int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1

        # Place internal walls randomly on available floor positions.
        floor_positions = list(zip(*np.where(grid == 0)))
        if len(floor_positions) < n_walls:
            raise ValueError("Not enough space to place the requested number of walls.")
        wall_indices = np.random.choice(len(floor_positions), size=n_walls, replace=False)
        for idx in wall_indices:
            grid[floor_positions[idx]] = 1

        # Check connectivity immediately.
        if not is_connected(grid):
            count += 1
            continue

        # Place goals (one per box).
        num_goals = n_boxes
        floor_positions = list(zip(*np.where(grid == 0)))
        if len(floor_positions) < (n_boxes + num_goals + 1):
            raise ValueError("Not enough floor space for boxes, goals, and the player.")
        goal_indices = np.random.choice(len(floor_positions), size=num_goals, replace=False)
        goal_positions = [floor_positions[idx] for idx in goal_indices]
        for pos in goal_positions:
            grid[pos] = 5


        free_positions = list(zip(*np.where(grid == 0)))
        if not free_positions:
            raise ValueError("No free floor space available for the player.")
        player_pos = free_positions[np.random.randint(len(free_positions))]
        
        candidates = []
        free_positions = list(zip(*np.where(grid == 0)))
        random.shuffle(free_positions)
        for pos in free_positions:
            y, x = pos
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                nb_y, nb_x = y + dy, x + dx
                ib_y, ib_x = y - dy, x - dx
                if 0 <= nb_y < grid.shape[0] and 0 <= nb_x < grid.shape[1]:
                    if grid[nb_y, nb_x] == 5 and grid[ib_y, ib_x] == 0:
                        candidates.append(pos)
        if not candidates:
            count += 1
            continue
        player_pos = candidates[np.random.randint(len(candidates))]
        
        grid[player_pos] = 2
        break  # valid level generated; exit loop.

    return grid
def simple_generate(width, height, n_boxes, n_walls, seed=None):
    inver_manager = InversedSokobanManager()
    def invert_states_random(grid, steps):
        node = inver_manager.initializer(grid)
        action_map = [[(-1, 0)], [(1, 0)], [(0, -1)], [(0, 1)]]
        for _ in range(steps):
            nodes = []
            for idx, action in enumerate(action_map):
                condition, new_node = inver_manager.legalInvertedUpdate((action, idx), node.state, node)
                if condition:
                    nodes.append(new_node)
            if len(nodes) == 0:
                break
            node = random.choice(nodes)
        return inver_manager.grid_state(node.state[0], node.state[1])
    grid = make_base(width, height, n_boxes, n_walls,seed)
    return invert_states_random(grid, 8000)