import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import io
import copy

import torch
import torch.nn as nn
from typing import List, Tuple
import random

from test import visualize_mlp_from_model, TrackedCircuit

class Operator:
    def __init__(self, length: int, shapes: tuple, weights: list, biases: list):
        self.length = length
        self.shapes = shapes
        self.weights = weights
        self.biases = biases

class AndOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        w = torch.zeros(1, total_inputs)
        w[0, input_indices] = 1.0
        b = torch.tensor([len(input_indices) - 1], dtype=torch.float32) * (-1.0)
        super().__init__(length=1, shapes=(total_inputs, 1), weights=[w], biases=[b])

class OrOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        w1 = torch.zeros(1, total_inputs)
        w1[0, input_indices] = -1.0
        b1 = torch.tensor([1.0])
        w2 = torch.tensor([[-1.0]])
        b2 = torch.tensor([1.0])
        super().__init__(length=2, shapes=(total_inputs, 1, 1), weights=[w1, w2], biases=[b1, b2])

class XorOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) == 0:
            raise ValueError("GeneralXorOp requires at least one input index")
        w0 = torch.zeros(2, total_inputs)
        w0[0, input_indices] = 1.0
        w0[1, input_indices] = 1.0
        b0 = torch.zeros(2, dtype=torch.float32)
        w1 = torch.tensor([[-1.0, 0.0],
                           [-1.0, 0.0]], dtype=torch.float32)
        b1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        w2 = torch.tensor([[1.0, 0.0],
                           [0.0, -1.0]], dtype=torch.float32)
        b2 = torch.tensor([0.0, 1.0], dtype=torch.float32)
        w3 = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
        b3 = torch.tensor([1.0], dtype=torch.float32)
        weights = [w0, w1, w2, w3]
        biases = [b0, b1, b2, b3]
        shapes = (total_inputs, 2, 2, 2, 1)
        super().__init__(length=4, shapes=shapes, weights=weights, biases=biases)

class MemoryOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("MemoryOp requires exactly one input index")
        w = torch.zeros(1, total_inputs)
        w[0, input_indices[0]] = 1.0
        b = torch.tensor([0.0])
        super().__init__(length=1, shapes=(total_inputs, 1), weights=[w], biases=[b])
class PlusOneOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("PlusOneOp requires exactly one input index")
        w = torch.zeros(1, total_inputs)
        w[0, input_indices[0]] = 1.0
        b = torch.tensor([1.0], dtype=torch.float32)
        super().__init__(length=1, shapes=(total_inputs, 1), weights=[w], biases=[b])

class NotOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("NotOp requires exactly one input index")
        w = torch.zeros(1, total_inputs)
        w[0, input_indices[0]] = -1.0
        b = torch.tensor([1.0], dtype=torch.float32)
        super().__init__(length=1, shapes=(total_inputs, 1), weights=[w], biases=[b])

class EqualsZeroOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("EqualsZeroOp requires exactly one input index")
        idx = input_indices[0]
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx] = 1.0
        w0[1, idx] = -1.0
        b0 = torch.zeros(2, dtype=torch.float32)
        w1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        w2 = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        b2 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        w3 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b3 = torch.tensor([0.0], dtype=torch.float32)
        w4 = torch.tensor([[-1.0]], dtype=torch.float32)
        b4 = torch.tensor([1.0], dtype=torch.float32)
        weights = [w0, w1, w2, w3, w4]
        biases = [b0, b1, b2, b3, b4]
        shapes = (total_inputs, 2, 1, 2, 1, 1)
        super().__init__(length=5, shapes=shapes, weights=weights, biases=biases)

class NotEqualZeroOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("NotEqualZeroOp requires exactly one input index")
        idx = input_indices[0]
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx] = 1.0
        w0[1, idx] = -1.0
        b0 = torch.zeros(2, dtype=torch.float32)
        w1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        w2 = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        b2 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        w3 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b3 = torch.tensor([0.0], dtype=torch.float32)
        weights = [w0, w1, w2, w3]
        biases = [b0, b1, b2, b3]
        shapes = (total_inputs, 2, 1, 2, 1)
        super().__init__(length=4, shapes=shapes, weights=weights, biases=biases)

class EqualOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 2:
            raise ValueError("EqualOp requires exactly two input indices")
        idx0, idx1 = input_indices
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx0] = 1.0
        w0[0, idx1] = -1.0
        w0[1, idx0] = -1.0
        w0[1, idx1] = 1.0
        b0 = torch.zeros(2, dtype=torch.float32)
        w1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        w2 = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        b2 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        w3 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b3 = torch.tensor([0.0], dtype=torch.float32)
        w4 = torch.tensor([[-1.0]], dtype=torch.float32)
        b4 = torch.tensor([1.0], dtype=torch.float32)
        weights = [w0, w1, w2, w3, w4]
        biases = [b0, b1, b2, b3, b4]
        shapes = (total_inputs, 2, 1, 2, 1, 1)
        super().__init__(length=5, shapes=shapes, weights=weights, biases=biases)

class GreaterThanOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 2:
            raise ValueError("GreaterThanOp requires exactly two input indices")
        idx0, idx1 = input_indices
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx0] = 1.0
        w0[0, idx1] = -1.0
        w0[1, idx0] = 1.0
        w0[1, idx1] = -1.0
        b0 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        w1 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        weights = [w0, w1]
        biases = [b0, b1]
        shapes = (total_inputs, 2, 1)
        super().__init__(length=2, shapes=shapes, weights=weights, biases=biases)

class LessThanOp(Operator):
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 2:
            raise ValueError("LessThanOp requires exactly two input indices")
        idx0, idx1 = input_indices
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx0] = -1.0
        w0[0, idx1] = 1.0
        w0[1, idx0] = -1.0
        w0[1, idx1] = 1.0
        b0 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        w1 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        weights = [w0, w1]
        biases = [b0, b1]
        shapes = (total_inputs, 2, 1)
        super().__init__(length=2, shapes=shapes, weights=weights, biases=biases)
class AbsDiffOp(Operator):
    """
    Computes the absolute difference |input0 - input1| for two integer or real inputs.
    Construction: 
    Layer 0: [d, -d] where d = input0 - input1
    After ReLU: [ReLU(d), ReLU(-d)]
    Layer 1: sum of the two ReLU outputs.
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 2:
            raise ValueError("AbsDiffOp requires exactly two input indices")
        idx0, idx1 = input_indices
        # Layer 0: [input0 - input1, input1 - input0]
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx0] = 1.0
        w0[0, idx1] = -1.0
        w0[1, idx0] = -1.0
        w0[1, idx1] = 1.0
        b0 = torch.zeros(2, dtype=torch.float32)
        # Layer 1: ReLU(d) + ReLU(-d)
        w1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        weights = [w0, w1]
        biases = [b0, b1]
        shapes = (total_inputs, 2, 1)
        super().__init__(length=2, shapes=shapes, weights=weights, biases=biases)
class SumDivTwoOp(Operator):
    """
    Computes (input0 + input1 + input2) / 2 for three inputs.
    Simple linear operation: weights 0.5 for each input, bias 0.
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 3:
            raise ValueError("SumDivTwoOp requires exactly three input indices")
        w = torch.zeros(1, total_inputs)
        for idx in input_indices:
            w[0, idx] = 0.5
        b = torch.tensor([0.0], dtype=torch.float32)
        super().__init__(length=1, shapes=(total_inputs, 1), weights=[w], biases=[b])

def build_neural_circuit(
    structure: List[List[Tuple[Operator, List[int]]]]
) -> nn.Sequential:
    layers = []
    prev_out_size = None
    for col_idx, column in enumerate(structure):
        if not column:
            continue
        col_input_size = column[0][0].shapes[0]
        if prev_out_size is not None and col_input_size != prev_out_size:
            raise ValueError(f"Column {col_idx}: input size mismatch ({col_input_size} vs {prev_out_size})")
        max_depth = max(op.length for op, _ in column)
        operator_ranges = []           
        current_input_size = col_input_size
        for depth in range(max_depth):
            total_out = 0
            for op, _ in column:
                total_out += op.shapes[depth + 1] if depth < op.length else op.shapes[-1]
            W = torch.zeros(total_out, current_input_size)
            b = torch.zeros(total_out)
            pos = 0
            new_ranges = []
            for op_idx, (op, input_indices) in enumerate(column):
                if depth < op.length:
                    local_w = op.weights[depth]
                    local_b = op.biases[depth]
                    out_dim = op.shapes[depth + 1]
                    if depth == 0:
                        W[pos:pos+out_dim] = local_w
                    else:
                        prev_start, _ = operator_ranges[op_idx]
                        in_dim = op.shapes[depth]
                        W[pos:pos+out_dim, prev_start:prev_start+in_dim] = local_w
                    b[pos:pos+out_dim] = local_b
                else:
                    start, _ = operator_ranges[op_idx]
                    out_dim = op.shapes[-1]
                    local_w = torch.eye(out_dim)
                    local_b = torch.zeros(out_dim)
                    W[pos:pos+out_dim, start:start+out_dim] = local_w
                    b[pos:pos+out_dim] = local_b
                new_ranges.append((pos, pos + out_dim))
                pos += out_dim
            linear = nn.Linear(current_input_size, total_out)
            linear.weight.data = W
            linear.bias.data = b
            layers.append(linear)
            is_final_layer = (col_idx == len(structure)-1) and (depth == max_depth-1)
            if not is_final_layer:
                layers.append(nn.ReLU())
            operator_ranges = new_ranges
            current_input_size = total_out
        prev_out_size = len(column)  
    return nn.Sequential(*layers)

LAYERZERO = 19
agent = [
    [
        (MemoryOp([12], LAYERZERO), [12]),
        (LessThanOp([8, 13], LAYERZERO), [8, 13]),
        (EqualOp([8, 13], LAYERZERO), [8,13]),
        (GreaterThanOp([6, 4], LAYERZERO), [6,4]),
        (LessThanOp([6, 4], LAYERZERO), [6,4]),
        (EqualOp([6,4], LAYERZERO), [6,4]),
        (GreaterThanOp([7,5], LAYERZERO), [7,5]),
        (LessThanOp([7,5], LAYERZERO), [7,5]),
        (EqualOp([7,5], LAYERZERO), [7,5]),
        (AbsDiffOp([13, 14], LAYERZERO), [13,14]),
        (MemoryOp([13], LAYERZERO), [13]),
        (MemoryOp([14], LAYERZERO), [14]),
        (MemoryOp([10], LAYERZERO), [10]),
        (NotEqualZeroOp([4], LAYERZERO), [4]),
        (NotEqualZeroOp([5], LAYERZERO), [5]),
        (MemoryOp([16], LAYERZERO), [16]), #Left available 15
        (MemoryOp([15], LAYERZERO), [15]),
        (NotOp([16], LAYERZERO), [16]),
        (EqualsZeroOp([4], LAYERZERO), [4]),
        (EqualsZeroOp([5], LAYERZERO), [5]),
        (MemoryOp([11], LAYERZERO), [11]),
        (AndOp([3, 16], LAYERZERO), [3, 16]),
        (AndOp([3, 17], LAYERZERO), [3, 17]),
        (AndOp([1, 16], LAYERZERO), [1, 16]),
        (AndOp([1, 18], LAYERZERO), [1,18]),
        (AndOp([2,17], LAYERZERO), [2, 17]),
        (AndOp([2, 18], LAYERZERO), [2, 18]),
        (NotOp([18],LAYERZERO), [18]),
        (NotOp([16], LAYERZERO), [16]),
        (NotOp([17], LAYERZERO), [17]),
        (EqualsZeroOp([4], LAYERZERO), [4]),
        (EqualsZeroOp([5], LAYERZERO), [5]),
        (NotOp([3], LAYERZERO), 3),
        (MemoryOp([17], LAYERZERO), [17]), #Right available 33
        (MemoryOp([3], LAYERZERO), [3])
    ],
    [
        (AndOp([0,1], 35), [0,1]),
        (AndOp([0, 2, 3], 35), [0, 2, 3]),
        (AndOp([0,2,4], 35), [0,2,3]),
        (AndOp([0,2,6,5], 35), [0,2,6,5]),
        (AndOp([0,2,7,5], 35), [0,2,7,5]),
        (AndOp([0,2,8,5], 35), [0,2,8,5]),
        (SumDivTwoOp([9, 10, 11], 35), [9,10,11]),
        (MemoryOp([12], 35), [12]),
        (OrOp([13, 14], 35), [13,14]),
        (MemoryOp([15], 35), [15]),
        (AndOp([16, 17], 35), [16, 17]),
        (AndOp([12, 18, 19], 35), [12, 18, 19]),
        (MemoryOp([20], 35), [35]),
        (MemoryOp([21], 35), [21]),
        (AndOp([22, 17], 35), [22, 17]),
        (MemoryOp([23], 35), [23]),
        (AndOp([24, 17], 35), [24, 17]),
        (MemoryOp([25], 35), [25]),
        (AndOp([26, 29], 35), [26,29]),
        (OrOp([17, 29], 35), [16, 29]),
        (AndOp([20, 30, 19], 35), [20,30,19]), #temporary rout to 19 from 31
        (MemoryOp([27], 35), [27]), #21 not (is down in board)
        (MemoryOp([32], 35), [32]), # 22 not (was last movement down)
        (MemoryOp([33], 35), [33]), #23 is right available?
        (MemoryOp([34], 35), [34]) #24 is down in board
    ],
    [
        (MemoryOp([0], 25), [0]),
        (MemoryOp([1], 25), [1]),
        (MemoryOp([2], 25), [2]),
        (MemoryOp([3], 25), [3]),
        (MemoryOp([4], 25), [4]),
        (MemoryOp([5], 25), [5]),
        (AndOp([7,8,9], 25), [7,8,9]),
        (AndOp([7,8,10], 25), [7,8,10]),
        (MemoryOp([11], 25), [11]),
        (AndOp([12, 8, 13], 25), [12,8, 13]),
        (AndOp([14, 12, 8], 25), [14, 12, 8]),
        (AndOp([15, 12, 8], 25), [15, 12, 8]),
        (AndOp([16, 12, 8], 25), [16, 12, 8]),
        (AndOp([17, 12, 8], 25), [17, 12, 8]),
        (AndOp([18, 12, 8], 25), [18, 12, 8]),
        (AndOp([21, 19, 12, 8, 22], 25), [21, 19, 12, 8, 22]), #ERROR 22
        (MemoryOp([20], 25), 20),
        (MemoryOp([6], 25), [6]),
        (AndOp([21, 24, 12, 8, 9], 25), [21, 24, 12, 8, 9]),
        (AndOp([21, 24, 12, 8, 23], 25), [21, 24, 12, 8, 23]),
    ],
    [
        (OrOp([4, 7], 20), [4,7]),
        (OrOp([2,6,9,11, 18], 20), [2,6,9,11,18]),
        (OrOp([1,10,13,16, 19], 20),[1,10,13,16, 19]),
        (OrOp([3, 12, 14], 20), [3, 12, 14]),
        (OrOp([0, 8, 15], 20), [0, 8, 15]),
        (OrOp([5], 20), [5]),
        (OrOp([8, 15], 20), [8, 15]),
        (MemoryOp([17], 20), [17]),
        (OrOp([0], 20), [0])
    ]
]
model_A = build_neural_circuit(agent)  # This is Agent A

# --- Code-based Agent (Agent B) ---
def agent_B(inp):
    # Parse inputs (force ints)
    was_last_movement_up    = int(inp[0])
    was_last_movement_left  = int(inp[1])
    was_last_movement_right = int(inp[2])
    was_last_movement_down  = int(inp[3])

    current_pos_x = int(inp[4])
    current_pos_y = int(inp[5])

    memory_x = int(inp[6])
    memory_y = int(inp[7])
    memory_score_of_position = int(inp[8])

    idx_of_memory = int(inp[9])

    currently_going_to_origin = int(inp[10])
    scouting = int(inp[11])
    going_to_best_score_found = int(inp[12])

    best_score_found = int(inp[13])
    current_pos_score = int(inp[14])

    is_up_in_board    = int(inp[15])
    is_left_in_board  = int(inp[16])
    is_right_in_board = int(inp[17])
    is_down_in_board  = int(inp[18])

    # canonical 11-element output builder
    def OUT(mv_up=0, mv_left=0, mv_right=0, mv_down=0, stay=0, stop=0,
            cur_origin=None, scout=None, to_best=None, best=None, idx=None):
        return [
            int(mv_up), int(mv_left), int(mv_right), int(mv_down), int(stay), int(stop),
            int(cur_origin if cur_origin is not None else currently_going_to_origin),
            int(scout      if scout      is not None else scouting),
            int(to_best    if to_best    is not None else going_to_best_score_found),
            int(best       if best       is not None else best_score_found),
            int(idx        if idx        is not None else idx_of_memory)
        ]

    # --- agent logic (keeps the structure you provided, with minor corrections/assumptions) ---
    if going_to_best_score_found:
        if memory_score_of_position < best_score_found:
            # agent previously wanted to "stay" while incrementing memory index
            return OUT(stay=1, stop=0, cur_origin=0, scout=0, to_best=1, best=best_score_found, idx=idx_of_memory+1)
        if memory_score_of_position == best_score_found:
            # move toward memory pos or stop if same tile
            if memory_x > current_pos_x:
                return OUT(mv_right=1, stop=0, cur_origin=0, scout=0, to_best=1, best=best_score_found, idx=idx_of_memory)
            if memory_x < current_pos_x:
                return OUT(mv_left=1, stop=0, cur_origin=0, scout=0, to_best=1, best=best_score_found, idx=idx_of_memory)
            if (memory_y > current_pos_y) and (memory_x == current_pos_x):
                return OUT(mv_down=1, stop=0, cur_origin=0, scout=0, to_best=1, best=best_score_found, idx=idx_of_memory)
            if (memory_y < current_pos_y) and (memory_x == current_pos_x):
                return OUT(mv_up=1, stop=0, cur_origin=0, scout=0, to_best=1, best=best_score_found, idx=idx_of_memory)
            if (memory_y == current_pos_y) and (memory_x == current_pos_x):
                # same tile -> signal stop
                return OUT(stay=1, stop=1, cur_origin=0, scout=0, to_best=1, best=best_score_found, idx=idx_of_memory)

    # update best if current tile is better
    if best_score_found < current_pos_score:
        best_score_found = current_pos_score

    if currently_going_to_origin and (current_pos_x != 0 or current_pos_y != 0):
        # prefer left if available, otherwise up
        if is_left_in_board:
            return OUT(mv_left=1, stop=0, cur_origin=1, scout=scouting, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)
        if is_up_in_board and not is_left_in_board:
            return OUT(mv_up=1, stop=0, cur_origin=1, scout=scouting, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)

    if currently_going_to_origin and current_pos_x == 0 and current_pos_y == 0:
        # arrived at origin: stay and switch to scouting
        return OUT(stay=1, stop=0, cur_origin=0, scout=1, to_best=0, best=best_score_found, idx=idx_of_memory)

    if scouting and (current_pos_x != 0 or current_pos_y != 0):
        # rules while scouting
        if was_last_movement_down and is_left_in_board:
            return OUT(mv_left=1, stop=0, cur_origin=currently_going_to_origin, scout=1, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)
        if was_last_movement_down and is_right_in_board:
            return OUT(mv_right=1, stop=0, cur_origin=currently_going_to_origin, scout=1, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)
        if was_last_movement_left and is_left_in_board:
            return OUT(mv_left=1, stop=0, cur_origin=currently_going_to_origin, scout=1, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)
        if was_last_movement_left and (not is_left_in_board) and is_down_in_board:
            return OUT(mv_down=1, stop=0, cur_origin=currently_going_to_origin, scout=1, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)
        if was_last_movement_right and is_right_in_board:
            return OUT(mv_right=1, stop=0, cur_origin=currently_going_to_origin, scout=1, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)
        if was_last_movement_right and (not is_right_in_board) and is_down_in_board:
            return OUT(mv_down=1, stop=0, cur_origin=currently_going_to_origin, scout=1, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)
        if (not is_down_in_board) and (current_pos_x == 0 or (not is_right_in_board)):
            return OUT(stay=1, stop=0, cur_origin=0, scout=0, to_best=1, best=best_score_found, idx=idx_of_memory)

    # if scouting and at origin -> move right
    if scouting and current_pos_x == 0 and current_pos_y == 0:
        return OUT(mv_right=1, stop=0, cur_origin=currently_going_to_origin, scout=1, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)

    # default fallback: stay and keep flags
    return OUT(stay=1, stop=0, cur_origin=currently_going_to_origin, scout=scouting, to_best=going_to_best_score_found, best=best_score_found, idx=idx_of_memory)

def run_and_capture(agent_fn, agent_type, board, start_pos=(0,0), init_hidden=None, max_steps=200, fig_size=(700,520)):
    height = len(board)
    width = len(board[0]) if height>0 else 0
    if not (0 <= start_pos[0] < width and 0 <= start_pos[1] < height):
        raise ValueError("start_pos out of board bounds")
    was_last_movement_up = 0
    was_last_movement_left = 0
    was_last_movement_right = 0
    was_last_movement_down = 0
    current_pos_x, current_pos_y = int(start_pos[0]), int(start_pos[1])
    memory = []
    memory_x = 0
    memory_y = 0
    memory_score_of_position = 0
    idx_of_memory = 0
    if init_hidden is None:
        init_hidden = {'currently_going_to_origin':0, 'scouting':1, 'going_to_best_score_found':0}
    currently_going_to_origin = int(init_hidden.get('currently_going_to_origin',0))
    scouting = int(init_hidden.get('scouting',1))
    going_to_best_score_found = int(init_hidden.get('going_to_best_score_found',0))
    current_pos_score = int(board[current_pos_y][current_pos_x])
    best_score_found = int(current_pos_score)
    history = []
    frames = []
    for step in range(max_steps):
        # --- draw frame ---
        fig = plt.figure(figsize=(fig_size[0]/100, fig_size[1]/100), dpi=100)
        gs = fig.add_gridspec(1, 2, width_ratios=[3,1], wspace=0.3)
        ax = fig.add_subplot(gs[0,0])
        ax_text = fig.add_subplot(gs[0,1])
        board_arr = np.array(board)
        ax.imshow(board_arr, interpolation='nearest', aspect='equal')
        ax.set_xticks(np.arange(board_arr.shape[1]))
        ax.set_yticks(np.arange(board_arr.shape[0]))
        ax.set_xticklabels(np.arange(board_arr.shape[1]))
        ax.set_yticklabels(np.arange(board_arr.shape[0]))
        for y in range(board_arr.shape[0]):
            for x in range(board_arr.shape[1]):
                ax.text(x, y, str(board_arr[y,x]), ha='center', va='center')
        ax.scatter([current_pos_x], [current_pos_y], s=200, marker="s")
        ax.set_title(f"Step {step} | pos=({current_pos_x},{current_pos_y}) score={current_pos_score}")
        ax.invert_yaxis()
        ax_text.axis('off')
        hs_lines = [
            "hidden state:",
            f"  going_to_origin: {currently_going_to_origin}",
            f"  scouting        : {scouting}",
            f"  to_best         : {going_to_best_score_found}",
            "",
            f"best current: {best_score_found}",
            "",
            "currently read memory:",
            f"  idx: {idx_of_memory}",
            f"  pos: ({memory_x},{memory_y})",
            f"  score: {memory_score_of_position}",
            "",
            "sparse scratch pad (last few mem entries):"
        ]
        show_mem = memory[-6:]
        if len(show_mem) == 0:
            hs_lines.append("   <empty>")
        else:
            for i,me in enumerate(show_mem):
                global_idx = len(memory) - len(show_mem) + i
                hs_lines.append(f"   [{global_idx}] ({me['x']},{me['y']}) s={me['score']}")
        ax_text.text(0, 1, "\n".join(hs_lines), va='top', ha='left', fontsize=8, family='monospace')
        fig.canvas.draw()
        width_px, height_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((height_px, width_px, 4))
        img = Image.fromarray(buf.copy())
        frames.append(img.convert("RGBA"))
        plt.close(fig)
        # ---- build input and step agent ----
        is_up_in_board = 1 if current_pos_y > 0 else 0
        is_left_in_board = 1 if current_pos_x > 0 else 0
        is_right_in_board = 1 if current_pos_x < width-1 else 0
        is_down_in_board = 1 if current_pos_y < height-1 else 0
        inp = [
            int(was_last_movement_up),
            int(was_last_movement_left),
            int(was_last_movement_right),
            int(was_last_movement_down),
            int(current_pos_x),
            int(current_pos_y),
            int(memory_x),
            int(memory_y),
            int(memory_score_of_position),
            int(idx_of_memory),
            int(currently_going_to_origin),
            int(scouting),
            int(going_to_best_score_found),
            int(best_score_found),
            int(current_pos_score),
            int(is_up_in_board),
            int(is_left_in_board),
            int(is_right_in_board),
            int(is_down_in_board)
        ]
        if agent_type == 'A':
            x = torch.tensor(inp, dtype=torch.float32)
            out = agent_fn(x).squeeze().detach().numpy()
            out = np.round(out).astype(int)
            if len(out) != 9:
                break
            mv_up, mv_left, mv_right, mv_down, stay, stop = out[:6]
            change_hidden = out[6]
            reported_best = out[7]
            inc_idx = out[8]
            next_currently_going_to_origin = currently_going_to_origin
            next_scouting = scouting
            next_going_to_best_score_found = going_to_best_score_found
            if change_hidden:
                if currently_going_to_origin:
                    next_currently_going_to_origin, next_scouting, next_going_to_best_score_found = 0, 1, 0
                elif scouting:
                    next_currently_going_to_origin, next_scouting, next_going_to_best_score_found = 0, 0, 1
                elif going_to_best_score_found:
                    next_currently_going_to_origin, next_scouting, next_going_to_best_score_found = 1, 0, 0  # assuming cycle
            reported_idx = idx_of_memory + inc_idx
        else:  # 'B'
            out = agent_fn(inp)
            if out is None or len(out) < 11:
                break
            mv_up, mv_left, mv_right, mv_down, stay, stop, next_currently_going_to_origin, next_scouting, next_going_to_best_score_found, reported_best, reported_idx = out[:11]
        # --- common movement logic ---
        move_bits = [mv_up, mv_left, mv_right, mv_down, stay]
        chosen_move = 'stay' if move_bits[4] == 1 else None
        if chosen_move is None:
            mapping = ['up', 'left', 'right', 'down']
            for i, b in enumerate(move_bits[:4]):
                if int(b) == 1:
                    chosen_move = mapping[i]
                    break
        if chosen_move is None:
            chosen_move = 'stay'
        prev_x, prev_y = current_pos_x, current_pos_y
        moved = False
        if chosen_move != 'stay':
            if chosen_move == 'up' and is_up_in_board:
                current_pos_y -= 1
                moved = True
            elif chosen_move == 'left' and is_left_in_board:
                current_pos_x -= 1
                moved = True
            elif chosen_move == 'right' and is_right_in_board:
                current_pos_x += 1
                moved = True
            elif chosen_move == 'down' and is_down_in_board:
                current_pos_y += 1
                moved = True
        if not (0 <= current_pos_x < width and 0 <= current_pos_y < height):
            current_pos_x, current_pos_y = prev_x, prev_y
            moved = False
        was_last_movement_up = 1 if chosen_move == 'up' and moved else 0
        was_last_movement_left = 1 if chosen_move == 'left' and moved else 0
        was_last_movement_right = 1 if chosen_move == 'right' and moved else 0
        was_last_movement_down = 1 if chosen_move == 'down' and moved else 0
        if moved:
            score_here = int(board[current_pos_y][current_pos_x])
            memory.append({'x': current_pos_x, 'y': current_pos_y, 'score': score_here})
        current_pos_score = int(board[current_pos_y][current_pos_x])
        currently_going_to_origin = next_currently_going_to_origin
        scouting = next_scouting
        going_to_best_score_found = next_going_to_best_score_found
        best_score_found = max(best_score_found, reported_best, current_pos_score)
        idx_of_memory = reported_idx
        if 0 <= idx_of_memory < len(memory):
            memory_x = memory[idx_of_memory]['x']
            memory_y = memory[idx_of_memory]['y']
            memory_score_of_position = memory[idx_of_memory]['score']
        else:
            memory_x = memory_y = memory_score_of_position = 0
        history.append({
            'step': step,
            'pos': (current_pos_x, current_pos_y),
            'chosen_move': chosen_move,
            'moved': moved,
            'current_pos_score': current_pos_score,
            'best_score_found': best_score_found,
            'idx_of_memory': idx_of_memory,
            'stop': stop
        })
        if stop:
            # final frame
            fig = plt.figure(figsize=(fig_size[0]/100, fig_size[1]/100), dpi=100)
            # ... (same as above, but with "STOP" title)
            # (copy the final frame code from reference)
            # for brevity, assume added similarly
            break
    return {'history': history, 'memory': memory, 'frames': frames, 'final_pos': (current_pos_x, current_pos_y), 'best_score_found': best_score_found}

# --- Main to run and save GIFs ---
if __name__ == "__main__":
    board = [
    [10, 17, 5, 3, 3, 7, 18, 10, 6, 16, 14, 4, 4, 16, 10],
    [6, 16, 13, 17, 6, 13, 2, 5, 7, 0, 10, 10, 14, 18, 8],
    [8, 19, 10, 10, 4, 18, 20, 4, 7, 19, 17, 19, 9, 2, 17],
    [9, 14, 3, 18, 11, 3, 8, 12, 19, 1, 15, 10, 19, 16, 7],
    [3, 3, 1, 20, 9, 15, 1, 13, 2, 15, 9, 4, 4, 20, 11],
    [17, 13, 20, 8, 1, 18, 16, 16, 11, 5, 12, 8, 11, 15, 11],
    [13, 7, 16, 14, 9, 6, 16, 10, 7, 13, 0, 20, 19, 9, 3],
    [8, 1, 9, 7, 13, 7, 18, 6, 17, 20, 5, 12, 14, 8, 17],
    [1, 3, 10, 1, 7, 1, 15, 15, 15, 16, 9, 16, 3, 8, 16],
    [18, 4, 17, 15, 2, 6, 10, 13, 9, 7, 10, 16, 6, 18, 11],
    [12, 14, 11, 12, 15, 12, 11, 20, 6, 2, 19, 18, 18, 9, 3],
    [12, 12, 4, 14, 17, 19, 19, 10, 1, 12, 12, 8, 5, 8, 18],
    [13, 12, 14, 18, 16, 18, 17, 4, 1, 8, 1, 18, 9, 12, 19],
    [20, 12, 5, 0, 20, 2, 17, 8, 15, 18, 8, 14, 11, 15, 7],
    [19, 4, 19, 16, 13, 17, 15, 17, 10, 19, 12, 15, 0, 4, 16],
    ]
    init_hidden = {'currently_going_to_origin':1, 'scouting':0, 'going_to_best_score_found':0}
    res_A = run_and_capture(model_A, 'A', board, start_pos=(4,6), init_hidden=init_hidden, max_steps=2000)
    frames_A = res_A['frames']
    durations = [0.8] * (len(frames_A)-1) + [1.5]
    arrs_A = [np.array(f.convert("RGBA")) for f in frames_A]
    imageio.mimsave("agent_A.gif", arrs_A, duration=durations, loop=0)
    print(f"Saved agent_A.gif ({len(frames_A)} frames). Final pos {res_A['final_pos']}, best {res_A['best_score_found']}")

    res_B = run_and_capture(agent_B, 'B', board, start_pos=(4,6), init_hidden=init_hidden, max_steps=2000)
    frames_B = res_B['frames']
    durations = [0.8] * (len(frames_B)-1) + [1.5]
    arrs_B = [np.array(f.convert("RGBA")) for f in frames_B]
    imageio.mimsave("agent_B.gif", arrs_B, duration=durations, loop=0)
    print(f"Saved agent_B.gif ({len(frames_B)} frames). Final pos {res_B['final_pos']}, best {res_B['best_score_found']}")