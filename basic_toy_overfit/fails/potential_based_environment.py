import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Set, Tuple
from PIL import Image, ImageDraw, ImageFont
import os
import time
from collections import defaultdict, deque
import math
import imageio
import shutil


class InvalidNode:
    def __init__(self):
        self.is_invalid = True
        self.is_goal = False
        self.is_terminated = False
        self.distance = math.inf

class Node:
    def __init__(self, state, is_terminated, is_goal, info):
        self.state = state
        self.is_terminated = is_terminated
        self.is_goal = is_goal
        self.is_invalid = False
        self.children = [InvalidNode() for _ in range(4)]
        self.parents = []
        self.info = info
        self.distance = math.inf

class FrontierManager:
    def __init__(self):
        self.queue = deque()
        self.visited_states = defaultdict(Node)
        self.goal_states = defaultdict(Node)
    
    def get_item(self):
        return self.queue.popleft()
    
    def add_item(self, x):
        self.queue.append(x)
    
    def reset(self):
        self.queue.clear()
        self.visited_states.clear()
        self.goal_states.clear()
    
    def visited_as_dict(self):
        return dict(self.visited_states)


class PotentialBased:
    def __init__(self, size, max_steps, num_apples):
        self.size_y, self.size_x = size
        self.bounds = set()
        self.num_apples = num_apples
        for i in range(self.size_y):
            self.bounds.add((i,0))
            self.bounds.add((i, self.size_x-1))
        for i in range(self.size_x):
            self.bounds.add((0, i))
            self.bounds.add((self.size_y-1, i))
        self.board_as_list = []
        for i in range(self.size_y):
            for j in range(self.size_x):
                self.board_as_list.append((i,j))
        self.apples = set()
        self.key_pos = None
        self.player_pos = None
        self.last_time_key_was_touch = 0
        self.action_map = [(-1,0),(1,0),(0,-1),(0,1)]
        self.expanded_map = self.action_map.copy() + [
            (-1,-1), (1,1), (0,0), (-1,1), (1,-1)
        ]
        self.left_steps = max_steps
        self.max_steps = max_steps
        self.finished = False
        self.current_node = None
        self.manager = FrontierManager()
    
    # ---------- NEW: property for compatibility with external code that checked env.done ----------
    @property
    def done(self):
        # This simply exposes the internal finished flag using the more common 'done' name.
        return self.finished
    # ---------------------------------------------------------------------------------------------

    def initialize(self):
        self.finished = False
        self.left_steps = self.max_steps
        self.manager.reset()
        interior = [
            (i, j)
            for i in range(1, self.size_y - 1)
            for j in range(1, self.size_x - 1)
        ]
        key, player, *apples = random.sample(
            interior,
            2 + self.num_apples
        )
        self.key_pos = key
        self.player_pos = player
        self.apples = set(apples)
        my_info = {
            "swgc_agent_to_key":0,
            "swgc_key_to_closest_apple":0,
            "minimum_steps_taken_to_get_here":0
        }
        this_state = Node(state=(self.player_pos, self.key_pos, tuple(sorted(self.apples)))
                          , is_terminated=False, is_goal=False, info=my_info)
        self.manager.visited_states[this_state.state] = this_state
        self.manager.add_item(this_state)
        self.current_node = this_state
        self.finished = False
    
    def in_grid(self, pos):
        return 0 <= pos[0] < self.size_y and 0 <= pos[1] < self.size_x
    
    def manhattan_distance(self, pos_1, pos_2):
        dy = abs(pos_1[0]-pos_2[0])
        dx = abs(pos_1[1]-pos_2[1])
        return dy + dx
    
    def backtrack(self, start_node: Node):
        queue = deque()
        queue.append((start_node, 0))

        while queue:
            node, dist = queue.popleft()

            if node.distance <= dist:
                continue

            node.distance = dist

            for parent in node.parents:
                queue.append((parent, dist + 1))
           
    def generate_potentials(self, max_agent_to_key, max_key_to_closest_apple, max_minimum_steps_taken_to_get_here):
        while self.manager.queue:
            node:Node = self.manager.get_item()
            swgcatk = node.info["swgc_agent_to_key"]
            swgcktca = node.info["swgc_key_to_closest_apple"]
            msttgh = node.info["minimum_steps_taken_to_get_here"]
            if swgcatk >= max_agent_to_key or swgcktca >= max_key_to_closest_apple or msttgh >= max_minimum_steps_taken_to_get_here:
                node.is_terminated = True
                continue
            ppos, kpos, apples = node.state
            player_to_key = self.manhattan_distance(ppos, kpos)
            if apples:
                key_to_closest_apple = min([self.manhattan_distance(kpos,apple) for apple in apples])
            else:
                key_to_closest_apple = 0  # No apples left, treat as 0
            apples = set(apples)
            for idx, (dy, dx) in enumerate(self.action_map):
                prime_finished = False
                prime_goal = False
                apples_prime = apples.copy()
                ppos_prime = (ppos[0]+dy, ppos[1]+dx)
                if not self.in_grid(ppos_prime):
                    continue
                kpos_prime = kpos
                if ppos_prime == kpos:
                    kpos_prime = (kpos[0] + dy, kpos[1] + dx)
                    if not self.in_grid(kpos_prime):
                        prime_finished = True
                        kpos_prime = kpos
                    if kpos_prime in self.bounds:
                        prime_finished = True
                    if kpos_prime in apples_prime:
                        apples_prime.remove(kpos_prime)
                        if len(apples_prime) == 0:
                            prime_goal = True
                state_prime = (ppos_prime, kpos_prime, tuple(sorted(apples_prime)))
                player_to_key_prime = self.manhattan_distance(ppos_prime, kpos_prime)
                if apples_prime:
                    key_to_closest_apple_prime = min([self.manhattan_distance(kpos_prime,apple) for apple in apples_prime])
                else:
                    key_to_closest_apple_prime = 0
                
                # Initialize primes with current values (carry over if no change)
                swgcatk_prime = swgcatk
                swgcktca_prime = swgcktca
                
                # Logic for agent to key
                if player_to_key != 1:
                    if player_to_key_prime < player_to_key:
                        swgcatk_prime = 0  # Progress: reset
                    else:
                        swgcatk_prime += 1  # No progress or worse: increment
                # Logic for key to apple (only when touching key)
                if player_to_key == 1:
                    if key_to_closest_apple_prime < key_to_closest_apple:
                        swgcktca_prime = 0  # Progress: reset
                    elif key_to_closest_apple_prime > key_to_closest_apple:
                        swgcktca_prime += 1  # Worse: increment
                    else:
                        swgcktca_prime += 1  # No change: increment as stall?
                
                prime_info = {
                    "swgc_agent_to_key": swgcatk_prime,
                    "swgc_key_to_closest_apple": swgcktca_prime,
                    "minimum_steps_taken_to_get_here": msttgh + 1
                }
                
                if state_prime in self.manager.visited_states:
                    pre_existing:Node = self.manager.visited_states[state_prime]
                    # Add parent regardless
                    if node not in [p for p in pre_existing.parents]:
                        pre_existing.parents.append(node)
                    continue
                
                prime = Node(state_prime, prime_finished, prime_goal, prime_info)
                node.children[idx] = prime
                prime.parents.append(node)
                self.manager.visited_states[state_prime] = prime
                if prime_goal:
                    self.manager.goal_states[state_prime] = prime
                self.manager.add_item(prime)
        for found_goal in self.manager.goal_states.values():
            self.backtrack(found_goal)

    # ---------- CHANGED: update() now provides directional shaping and gentler terminal penalties ----------
    def update(self, action: int):
        """
        NOTE: This method was changed in three places:
          1) timeout and terminal penalties were reduced (less dominating negative terminal).
          2) distance-based shaping now gives a *positive* signal when child is closer to goal than parent.
          3) added safe-guards for infinite distances and improved numeric stability.
        """
        if self.finished:
            return 0.0, True
        self.left_steps -= 1
        if self.left_steps <= 0:
            self.finished = True
            # REDUCED terminal penalty on timeout (used to be -1.0) so single end-of-episode reward doesn't dominate learning.
            return -0.2, True  # Time limit termination (less severe)

        reward = 0.0
        node = self.current_node
        # Invalid action index
        if action < 0 or action > 3:
            return -0.5, False
        child = node.children[action]
        # Invalid child → no movement
        if child.is_invalid:
            return -0.5, False
        # Termination check
        if child.is_terminated:
            self.current_node = child
            self.finished = True
            # REDUCED terminal penalty (used to be -1.0). This makes episodic failure less destructive to gradients.
            return -0.2, True

        # ---------- NEW: Distance-based shaping that provides a POSITIVE learning signal ----------
        # We compare the parent's pre-computed `node.distance` with child's `child.distance`.
        # If both distances are finite, give a positive reward proportional to the reduction in distance.
        # This creates a true "progress" signal: taking an action that reduces distance to goal -> +reward.
        # If distances are not available, fall back to a small negative nudging penalty to discourage blind moves.
        valid_children = [
            c for c in node.children
            if not c.is_invalid and not c.is_terminated
        ]
        # Only apply shaping if we have at least one valid child.
        if valid_children:
            # If both node and child have finite (computed) distances, use them.
            if (not math.isinf(node.distance)) and (not math.isinf(child.distance)):
                # Positive when child is closer to goal than the parent node.
                # The scaling 0.1 is small so shaping complements, not overwhelms, the main reward.
                reward += 0.1 * (node.distance - child.distance)
            else:
                # If distances are not available (infinite) we don't want to give a large meaningless signal.
                # Give a tiny negative nudging so the agent still prefers fewer useless moves:
                reward -= 0.05
        # -----------------------------------------------------------------------------------------------

        # Move agent
        self.current_node = child
        # Sync instance variables for rendering
        self.player_pos, self.key_pos, apples_tuple = child.state
        self.apples = set(apples_tuple)
        # Goal reward
        if child.is_goal:
            reward += 1.0
            self.finished = True
            return reward, True
        return reward, False

    # render and render_for_human unchanged (kept quiet per your instruction)
    def render(self):
        H = self.size_y
        W = self.size_x
        board = torch.zeros((H, W, 3), dtype=torch.float32)
        for (y,x) in self.apples:
            board[y, x] = torch.tensor((0.0, 0.0, 1.0))
        py, px = self.player_pos
        if self.player_pos in self.apples:
            board[py, px] = torch.tensor((1.0, 0.0, 1.0))
        else:
            board[py, px] = torch.tensor((1.0, 0.0, 0.0))
        ky, kx = self.key_pos
        board[ky, kx] = torch.tensor((0.0, 1.0, 0.0))
        return torch.flatten(board)
    
    def render_for_human(self, filename="env_render.png", cell_size=36, show_grid=True, grid_line_width=1):
        width = self.size_x * cell_size
        height = self.size_y * cell_size
        img = Image.new("RGB", (width, height), (255,255,255))
        draw = ImageDraw.Draw(img)
        for (ay, ax) in self.apples:
            y0 = ay*cell_size
            x0 = ax*cell_size
            inset = cell_size // 6
            draw.ellipse([x0 + inset, y0 + inset, x0 + cell_size - inset - 1, y0 + cell_size - inset - 1], fill=(0,255,0))
        py, px = self.player_pos
        x0 = px * cell_size
        y0 = py * cell_size
        inset = cell_size // 8
        draw.polygon([(x0+(cell_size)/2, y0 + inset),(x0+inset, y0 + cell_size - inset - 1), (x0+cell_size-inset,y0 + cell_size - inset - 1) ], fill=(255,0,0))
        ky, kx = self.key_pos
        x0 = kx * cell_size
        y0 = ky * cell_size
        inset = cell_size // 8
        draw.rectangle([x0+inset, y0+inset, x0 + cell_size - inset, y0 + cell_size - inset], fill=(255,255, 0))
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
    
    def export(self):
        identifier = {
            "graph":self.manager.visited_as_dict(),
            "origin":self.current_node
        }
        return identifier
    
    def load(self, identifier):
        self.manager.reset()
        self.finished = False
        self.left_steps = self.max_steps

        self.player_pos, self.key_pos, self.apples = identifier["origin"].state
        self.apples = set(self.apples)

        self.manager.visited_states = identifier["graph"].copy()
        state_key = (self.player_pos, self.key_pos, tuple(sorted(self.apples)))
        self.current_node = self.manager.visited_states[state_key]



def stream_best_path(
    env: PotentialBased,
    gif_path="best_path.gif",
    frame_dir="best_path_frames",
    total_duration_sec=20
):
    os.makedirs(frame_dir, exist_ok=True)

    fps = 200 / total_duration_sec
    frame_duration = 1.0 / fps

    frames = []
    total_steps = 0

    while not env.done:
        # Render current state
        frame_path = os.path.join(frame_dir, f"step_{total_steps:04d}.png")
        env.render_for_human(filename=frame_path)
        frames.append(imageio.imread(frame_path))

        node = env.current_node

        # Choose best action by distance
        best_action = None
        best_distance = math.inf

        for action, child in enumerate(node.children):
            if child.is_invalid or child.is_terminated:
                continue
            if child.distance < best_distance:
                best_distance = child.distance
                best_action = action

        if best_action is None:
            print("No valid action found — stuck.")
            break

        _, done = env.update(best_action)
        total_steps += 1

        if done:
            # Render final state
            frame_path = os.path.join(frame_dir, f"step_{total_steps:04d}.png")
            env.render_for_human(filename=frame_path)
            frames.append(imageio.imread(frame_path))
            break

    print(f"Best-path length: {total_steps}")

    imageio.mimsave(
        gif_path,
        frames,
        duration=frame_duration,
        loop=0
    )

    shutil.rmtree(frame_dir)

"""
if __name__ == "__main__":
    import random
    import time

    #random.seed(42)

    size = (20, 20)
    num_apples = 4

    env = PotentialBased(
        size=size,
        max_steps=500,
        num_apples=num_apples
    )

    print("Initializing environment...")
    env.initialize()

    print("Generating potentials...")
    t0 = time.perf_counter()
    env.generate_potentials(
        max_agent_to_key=5,
        max_key_to_closest_apple=5,
        max_minimum_steps_taken_to_get_here=100
    )
    t1 = time.perf_counter()
    print(f"Potential generation took {t1 - t0:.3f}s")

    print("Streaming best path...")
    stream_best_path(
        env,
        gif_path="best_path_15x15_3apples.gif",
        total_duration_sec=25
    )

    print("Done.")"""


if __name__ == "__main__":
    import time
    import random
    import traceback

    tests = [
        ((5, 5), 1),
        ((7, 7), 1),
        ((7, 7), 2),
        ((10, 10), 1),
        ((10, 10), 2),
        ((15, 15), 1),
        ((15, 15), 3),
    ]

    # Parameters that control generation depth/pruning.
    # Increase these to allow more expansion (slower / more memory).
    MAX_AGENT_TO_KEY = 5
    MAX_KEY_TO_CLOSEST_APPLE = 5
    MAX_MINIMUM_STEPS_TAKEN_TO_GET_HERE = 100

    # max_steps for the PotentialBased instance (not directly used by generate_potentials,
    # but kept reasonable)
    INSTANCE_MAX_STEPS = 500

    print("Stress test of PotentialBased.generate_potentials")
    print("Each test: (grid_size, num_apples) -> elapsed_time, visited_nodes, goal_states, max_distance")
    print("-" * 80)

    for idx, (size, n_apples) in enumerate(tests):
        try:
            random.seed(1000 + idx)  # reproducible but different seed per test
            env = PotentialBased(size=size, max_steps=INSTANCE_MAX_STEPS, num_apples=n_apples)
            env.initialize()

            t0 = time.perf_counter()
            env.generate_potentials(
                max_agent_to_key=MAX_AGENT_TO_KEY,
                max_key_to_closest_apple=MAX_KEY_TO_CLOSEST_APPLE,
                max_minimum_steps_taken_to_get_here=MAX_MINIMUM_STEPS_TAKEN_TO_GET_HERE
            )
            t1 = time.perf_counter()

            elapsed = t1 - t0
            visited = len(env.manager.visited_states)
            goals = len(env.manager.goal_states)
            # compute max distance among visited nodes (some may still be inf if unreachable)
            distances = [n.distance for n in env.manager.visited_states.values()]
            finite_distances = [d for d in distances if d != math.inf]
            max_dist = max(finite_distances) if finite_distances else math.inf

            print(f"{size} with {n_apples} apple(s): {elapsed:.3f}s, visited={visited}, goals={goals}, max_distance={max_dist}")

        except Exception as e:
            print(f"{size} with {n_apples} apple(s): ERROR during test")
            traceback.print_exc()
    print("-" * 80)
    print("Done.")
