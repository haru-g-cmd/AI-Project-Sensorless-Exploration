import os
import itertools as it
import numpy as np
from collections import namedtuple
from collections import deque
from collections import defaultdict
import random
from time import sleep
from termcolor import colored
from queue import PriorityQueue
import queue
from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
import time
import pickle

class GridAttrib:
    __slots__ = ('open', 'bot_occupied', 'traversed')

    def __init__(self):
        self.open = False
        self.bot_occupied = True #Set to False for Soln 1
        self.traversed = False

class Grid:
    def __init__(self, D=30, debug=True):
        self.D = D
        self.grid = []
        self.debug = debug
        self.gen_grid()

    def valid_index(self, ind):
        return not (ind[0] >= self.D or ind[0] < 0 or ind[1] >= self.D or ind[1] < 0)

    def get_neighbors(self, ind):
        neighbors = []
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        for index in indices:
            if self.valid_index(index):
                neighbors.append(index)
        return neighbors
    
    def get_open_neighbors(self, ind):
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        return [index for index in indices if self.valid_index(index) and self.grid[index[1]][index[0]].open]

    # Gets only the unvisited open neighbors. Used mainly for ops planning.
    def get_untraversed_open_neighbors(self, ind):
        left = (ind[0] - 1, ind[1])
        right = (ind[0] + 1, ind[1])
        up = (ind[0], ind[1] + 1)
        down = (ind[0], ind[1] - 1)
        indices = [left, right, up, down]
        return [index for index in indices if self.valid_index(index) and self.grid[index[1]][index[0]].open and not self.grid[index[1]][index[0]].traversed]
    
    # The steps to be iterated over and over till they cannot be done are implemented here
    def gen_grid_iterate(self):
        cells_to_open = []
        # Get all blocked cells with one open neighbors
        for j in range(self.D):
            for i in range(self.D):
                if self.grid[j][i].open == True:
                    continue
                neighbors_ind = self.get_neighbors((i, j))
                open_neighbors = []
                for neighbor_ind in neighbors_ind:
                    if self.grid[neighbor_ind[1]][neighbor_ind[0]].open is True:
                        open_neighbors.append(neighbor_ind)
                if len(open_neighbors) == 1:
                    cells_to_open.append((i, j))
        # Randomly open one of those cells
        if len(cells_to_open) > 0:
            index = random.choice(cells_to_open)
            self.grid[index[1]][index[0]].open = True
        if self.debug:
            print("After one iteration")
            print(self)
            print(f"Cells to open: {len(cells_to_open)}")
        return len(cells_to_open) != 0

    # Grid generation happens here
    def gen_grid(self):
        for j in range(self.D):
            row = []
            for i in range(self.D):
                row.append(GridAttrib())
            self.grid.append(row)

        # Open Random Cell
        rand_ind = np.random.randint(0, self.D, 2)
        self.grid[rand_ind[1]][rand_ind[0]].open = True
        if self.debug:
            print(self)
        # Iterate on the grid
        while self.gen_grid_iterate():
            pass
        # Randomly open half the dead ends
        cells_to_open = []
        for j in range(self.D):
            for i in range(self.D):
                    all_neighbors = self.get_neighbors((i,j))
                    open_neighbors = [ind for ind in all_neighbors if self.grid[ind[1]][ind[0]].open]
                    closed_neighbors = [ind for ind in all_neighbors if not self.grid[ind[1]][ind[0]].open]
                    # randint is used here to maintain a ~50% chance of any dead end opening
                    if self.grid[j][i].open and random.randint(0, 1) == 1 and len(open_neighbors) == 1:
                        cells_to_open.append(random.choice(closed_neighbors))
        for ind in cells_to_open:
            self.grid[ind[1]][ind[0]].open = True
        if self.debug:
            print("After dead end opening")
            print(self)

    # A bunch of simple helper functions

    def save_grid_state(self, filename):
        file_path = os.ops.join("C:\\Users\\Haru\\Desktop\\", filename)
        with open(file_path, 'wb') as file:
            pickle.dump(self.grid, file)

    @classmethod
    def load_grid_state(cls, filename):
        file_path = os.ops.join("C:\\Users\\Haru\\Desktop\\", filename)
        with open(file_path, 'rb') as file:
            grid = pickle.load(file)
        obj = cls()
        obj.grid = grid
        return obj
    
    def get_zeroes(self):
        zero_coor = [] 
        for j in range(self.D):
            for i in range(self.D):
                if self.grid[j][i].bot_occupied == False and self.grid[j][i].open == True:
                    zero_coor.extend([(j, i)])
        return zero_coor
    
    def place_bot(self, ind):
        self.grid[ind[1]][ind[0]].bot_occupied = True

    def remove_bot(self, ind):
        self.grid[ind[1]][ind[0]].bot_occupied = False

    def set_traversed(self, ind):
        self.grid[ind[1]][ind[0]].traversed = True

    def remove_all_traversal(self):
        for j in range(self.D):
            for i in range(self.D):
                self.grid[j][i].traversed = False

    def get_open_indices(self):
        return [(i, j) for i in range(self.D) for j in range(self.D) if self.grid[j][i].open == True]
    
    def print_grid(self):
        for j in range(self.D):
            for i in range(self.D):
                if self.grid[j][i].open == True:
                    if self.grid[j][i].bot_occupied:
                        print(colored('1', 'green'), end='') # Change to B for Soln 1
                    elif self.grid[j][i].traversed:
                        print(colored('.', 'red'), end='')
                    else:
                        print(colored('0', 'blue'), end='') # Change to - for Soln 1
                else:
                    print('#', end='')
            print()

    def get_unoccupied_open_indices(self):
        return [(i, j) for i in range(self.D) for j in range(self.D) if self.grid[j][i].open == True and self.grid[j][i].bot_occupied == False]

    def reset_grid(self):
        for j in range(self.D):
            for i in range(self.D):
                self.grid[j][i].bot_occupied = False
                self.grid[j][i].traversed = False
    
    def reset_grid2(self, zeroes):
        for j in range(self.D):
            for i in range(self.D):
                if (j, i) in zeroes:
                    self.grid[j][i].bot_occupied = False
                    continue
                if self.grid[j][i].open == True:
                    self.grid[j][i].bot_occupied = True

    def shift_up(self):
        for j in range(self.D - 1): 
            for i in range(self.D):
                if not (0 <= j + 1 < self.D):  
                 continue
                if not self.grid[j + 1][i].bot_occupied or not self.grid[j][i].open:
                  continue
                self.grid[j][i].bot_occupied = self.grid[j + 1][i].bot_occupied
                self.grid[j + 1][i].bot_occupied = False

    def shift_right(self):
        for j in range(self.D):
            for i in range(self.D - 1): 
                if not (0 <= i + 1 < self.D):  
                    continue
                if not self.grid[j][i].bot_occupied or not self.grid[j][i + 1].open:
                    continue
                self.grid[j][i + 1].bot_occupied = self.grid[j][i].bot_occupied
                self.grid[j][i].bot_occupied = False

    def shift_down(self):
        for j in range(self.D):
            for i in range(self.D):
                if not (0 <= j - 1 < self.D):  
                    continue
                if not self.grid[j - 1][i].bot_occupied or not self.grid[j][i].open:
                    continue
                self.grid[j][i].bot_occupied = self.grid[j - 1][i].bot_occupied
                self.grid[j - 1][i].bot_occupied = False

    def shift_left(self):
        for j in range(self.D):
            for i in range(self.D):
                if not (0 <= i - 1 < self.D): 
                    continue
                if not self.grid[j][i].bot_occupied or not self.grid[j][i - 1].open:
                    continue
                # if self.grid[j][i].bot_occupied == False:
                #     continue
                self.grid[j][i - 1].bot_occupied = self.grid[j][i].bot_occupied
                self.grid[j][i].bot_occupied = False

    def __str__(self):
        s = ""
        for j in range(self.D):
            for i in range(self.D):
                if self.grid[j][i].open == True:
                    if self.grid[j][i].bot_occupied:
                        s += colored('1', 'green') # Change to B for Soln 1
                    elif self.grid[j][i].traversed:
                        s += colored('.', 'red')
                    else:
                        s += colored('0', 'blue') # Change to - for Soln 1
                else:
                    s += '#' # Change to # for Soln 1
            s += "\n"
        return s

class PathTreeNode:
    def __init__(self):
        self.children = []
        self.parent = None
        self.data = None

class Bot1:
    def __init__(self, grid, debug=True):
        self.grid = grid
        self.ind = random.choice(self.grid.get_open_indices())
        self.grid.place_bot(self.ind)
        self.ops = None
        self.debug = debug

    def plan_path(self):
        if self.debug:
            print("Planning Path...")
        self.ops = deque([])
        self.grid.remove_all_traversal()
        captain_found = False
        path_tree = PathTreeNode()
        path_tree.data = self.ind
        path_deque = deque([path_tree])
        destination = None
        visited = set()
        compute_counter = 0
        while not captain_found:
            if len(path_deque) == 0:
                self.grid.remove_all_traversal()
                return
            compute_counter += 1
            node = path_deque.popleft()
            ind = node.data
            if ind in visited:
                continue
            visited.add(ind)
            self.grid.set_traversed(ind)
            # if ind == self.captain_ind:
                # destination = node
                # break
            neighbors_ind = self.grid.get_untraversed_open_neighbors(ind)
            for neighbor_ind in neighbors_ind:
                new_node = PathTreeNode()
                new_node.data = neighbor_ind
                new_node.parent = node
                node.children.append(new_node)
            path_deque.extend(node.children)
        self.grid.remove_all_traversal()
        if self.debug:
            print("Planning Done!")
        reverse_path = []
        node = destination
        while node.parent is not None:
            reverse_path.append(node.data)
            node = node.parent
        self.ops.extend(reversed(reverse_path))
        for ind in self.ops:
            self.grid.set_traversed(ind)
        if self.debug:
            print("Planned Path")
            print(self.grid)

    def move(self):
        # if self.ops is None:
            # self.plan_path()
        # if len(self.ops) == 0:
            # if self.debug:
                # print("No ops found!")
            # return
        # open_neighbors = self.grid.get_open_neighbors(self.ind)

        neighbors = self.grid.get_neighbors(self.ind)
        next_move_prob = random.random()
        
        if next_move_prob <= 0.7: 
            next_dest = random.choice([(self.ind[0], self.ind[1] - 1), (self.ind[0] - 1, self.ind[1])])
        elif next_move_prob <= 0.85:  
            next_dest = (self.ind[0] + 1, self.ind[1])
        else:  
            next_dest = (self.ind[0], self.ind[1] + 1)

        if self.grid.valid_index(next_dest) and self.grid.grid[next_dest[1]][next_dest[0]].open:
            self.grid.remove_bot(self.ind)
            self.ind = next_dest
            self.grid.place_bot(self.ind)
            self.grid.set_traversed(self.ind)

#---------------------------------------------------------------------------------------------#
#Solution 1

# grid_loaded = Grid()                    
# #grid_loaded.save_grid_state('grid_state.pkl')
# #grid_loaded = Grid.load_grid_state('grid_state.pkl')
# b = Bot1(grid_loaded)
# print(grid_loaded)
# turns = 0
# t = []
# for i in range(1000):                  #Relative Solution simplified
#     while(1):
#         b.move()
#         turns=turns+1
#         if b.ind == (0,0):
#             t.append(turns)
#             turns = 0
#             b = Bot1(grid_loaded)
#             break
#         print(grid_loaded)
# print(t)


#-----------------------------------------------------------------------------------------------------------#
#Solution 2

D = 15 # To illustrate the idea, a higher D value takes a bit of time to compute
g = Grid(D)

pQ = PriorityQueue() 
zero_coor = g.get_zeroes()
zeroes = len(zero_coor)
pQ.put((zeroes, zero_coor, []))
visited = []
d = ["UP", "DOWN", "LEFT", "RIGHT"]
c = 0
for x in range(D):
    for y in range(D):
        if g.grid[y][x].open == True:
            c = c + 1
open_cells = c
g.print_grid()
print()

# g.shift_up()
# print("After an 'UP' Operation")
# g.print_grid()
# print()
# g.shift_right()
# print("After a 'RIGHT' Operation")
# g.print_grid()
# print()
# g.shift_down()
# print("After an 'DOWN' Operation")
# g.print_grid()
# print()
# g.shift_left()
# print("After a 'LEFT' Operation")
# g.print_grid()
# print()

print("Open cells:", open_cells)
final_coor = []

while not pQ.empty():
    (z, prev_coor, path) = pQ.get()
    g.reset_grid2(prev_coor)

    g.shift_left()
    new_coor = g.get_zeroes()
    new_zeroes = len(new_coor)

    if new_coor != prev_coor and new_coor not in visited:
        visited.append(new_coor)
        pQ.put((-new_zeroes, new_coor, path + [0]))

    if abs(new_zeroes) == open_cells - 1:
        final_coor.append((path, new_coor))
        break
    g.reset_grid2(prev_coor)

    g.shift_right()
    new_coor = g.get_zeroes()
    new_zeroes = len(new_coor)

    if new_coor != prev_coor and new_coor not in visited:
        visited.append(new_coor)
        pQ.put((-new_zeroes, new_coor, path + [1]))

    if abs(new_zeroes) == open_cells - 1:
        final_coor.append((path, new_coor))
        break
    g.reset_grid2(prev_coor)

    g.shift_up()
    new_coor = g.get_zeroes()
    new_zeroes = len(new_coor)

    if new_coor != prev_coor and new_coor not in visited:
        visited.append(new_coor)
        pQ.put((-new_zeroes, new_coor, path + [2]))

    if abs(new_zeroes) == open_cells - 1:
        final_coor.append((path, new_coor))
        break
    g.reset_grid2(prev_coor)

    g.shift_down()
    new_coor = g.get_zeroes()
    new_zeroes = len(new_coor)

    if new_coor != prev_coor and new_coor not in visited:
        visited.append(new_coor)
        pQ.put((-new_zeroes, new_coor, path + [3]))

    if abs(new_zeroes) == open_cells - 1:
        final_coor.append((path, new_coor))
        break
    g.reset_grid2(prev_coor)

ops, coor = [], []
print(final_coor)
(ops, coor) = final_coor[0]
print("Grid after the operations:")
g.reset_grid2(coor)
g.print_grid()
print()
i = 0
while i < len(ops):
    if ops[i] == 0:
        ops[i] = d[0]
    elif ops[i] == 1:
        ops[i] = d[1]
    elif ops[i] == 2:
        ops[i] = d[2]
    elif ops[i] == 3:
        ops[i] = d[3]
    i += 1
print(ops)