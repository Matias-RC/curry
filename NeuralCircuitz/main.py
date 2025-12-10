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
        (MemoryOp([16], LAYERZERO), [16]),
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
        (EqualsZeroOp([5], LAYERZERO), [5])
    ],
    [
        (AndOp([0,1], 32), [0,1]),
        (AndOp([0, 2, 3], 32), [0, 2, 3]),
        (AndOp([0,2,4], 32), [0,2,3]),
        (AndOp([0,2,6,5], 32), [0,2,6,5]),
        (AndOp([0,2,7,5], 32), [0,2,7,5]),
        (AndOp([0,2,8,5], 32), [0,2,8,5]),
        (SumDivTwoOp([9, 10, 11], 32), [9,10,11]),
        (MemoryOp([12], 32), [12]),
        (OrOp([13, 14], 32), [13,14]),
        (MemoryOp([15], 32), [15]),
        (AndOp([16, 17], 32), [16, 17]),
        (AndOp([12, 18, 19], 32), [12, 18, 19]),
        (MemoryOp([20], 32), [32]),
        (MemoryOp([21], 32), [21]),
        (AndOp([22, 17], 32), [22, 17]),
        (MemoryOp([[23], 32]), [23]),
        (AndOp([24, 17], 32), [24, 17]),
        (MemoryOp([[25], 32]), [25]),
        (AndOp([26, 29], 32), [26,29]),
        (OrOp([16, 29], 32), [16, 29]),
        (AndOp([20, 30, 31], 32), [20,30,31]),
        (MemoryOp([27], 32), [27])        
    ],
    [
        (MemoryOp([0], 22), [0]),
        (MemoryOp([1], 22), [1]),
        (MemoryOp([2], 22), [2]),
        (MemoryOp([3], 22), [3]),
        (MemoryOp([4], 22), [4]),
        (MemoryOp([5], 22), [5]),
        (AndOp([7,8,9], 22), [7,8,9]),
        (AndOp([7,8,10], 22), [7,8,10]),
        (MemoryOp([11], 22), [11]),
        (AndOp([12, 8, 13], 22), [12,8, 13]),
        (AndOp([14, 12, 8], 22), [14, 12, 8]),
        (AndOp([15, 12, 8], 22), [15, 12, 8]),
        (AndOp([16, 12, 8], 22), [16, 12, 8]),
        (AndOp([17, 12, 8], 22), [17, 12, 8]),
        (AndOp([18, 12, 8], 22), [18, 12, 8]),
        (AndOp([21, 19, 12, 8], 22), [21, 19, 12, 8]),
        (MemoryOp([20], 22), 20),
        (MemoryOp([6], 22), [6])
    ],
    [
        (OrOp([4, 7], 18), [4,7]),
        (OrOp([2,6,9,11], 18), [2,6,9,11]),
        (OrOp([1,10,13,16], 18),[1,10,13,16]),
        (OrOp([3, 12, 14], 18), [3, 12, 14]),
        (OrOp([0, 8, 15], 18), [0, 8, 15]),
        (OrOp([5], 18), [5]),
        (OrOp([8, 15], 18), [8, 15]),
        (MemoryOp([17], 18), [17]),
        (OrOp([0], 18), [0])
    ]
]





if __name__ == "__main__":
    TOTAL_INPUTS = 14
    structure = [
        [
            (AndOp([0,1], TOTAL_INPUTS), [0,1]),
            (OrOp([2,3], TOTAL_INPUTS), [2,3]),
            (XorOp([4,5], TOTAL_INPUTS), [4,5]),
            (MemoryOp([6], TOTAL_INPUTS), [6]),
            (NotOp([0], TOTAL_INPUTS), [0]),
            (EqualsZeroOp([7], TOTAL_INPUTS), [7]),
            (NotEqualZeroOp([8], TOTAL_INPUTS), [8]),
            (EqualOp([9,10], TOTAL_INPUTS), [9,10]),
            (GreaterThanOp([11,12], TOTAL_INPUTS), [11,12]),
            (LessThanOp([13,12], TOTAL_INPUTS), [13,12]),
        ]
    ]
    model = build_neural_circuit(structure)
    print("Model with all 10 gates built successfully!")
    print(model)
    N_TESTS = 5000
    correct = 0
    with torch.no_grad():
        for test in range(N_TESTS):
            x = torch.zeros(TOTAL_INPUTS)
            x[:6] = torch.randint(0, 2, (6,)).float()
            x[6:] = torch.randint(0, 51, (8,)).float()
            out = model(x)
            bit0 = x[0].item()
            bit1 = x[1].item()
            bit2 = x[2].item()
            bit3 = x[3].item()
            bit4 = x[4].item()
            bit5 = x[5].item()
            mem = x[6].item()
            val7 = x[7].item()
            val8 = x[8].item()
            a9 = x[9].item()
            a10 = x[10].item()
            a11 = x[11].item()
            a12 = x[12].item()
            a13 = x[13].item()
            expected = torch.tensor([
                1.0 if (bit0 == 1 and bit1 == 1) else 0.0,
                1.0 if (bit2 == 1 or bit3 == 1) else 0.0,
                1.0 if (bit4 == 1) != (bit5 == 1) else 0.0,
                mem,
                1.0 if bit0 == 0 else 0.0,
                1.0 if val7 == 0 else 0.0,
                1.0 if val8 != 0 else 0.0,
                1.0 if a9 == a10 else 0.0,
                1.0 if a11 > a12 else 0.0,
                1.0 if a13 < a12 else 0.0,
            ])
            if torch.allclose(out.squeeze(), expected, atol=1e-6):
                correct += 1

    print(f"\n5000 random tests finished!")
    print(f"Success rate: {correct / N_TESTS * 100:.2f}%")





"""    model = TrackedCircuit(model)
    x = torch.zeros(TOTAL_INPUTS)
    x[:6] = torch.randint(0, 2, (6,)).float()
    x[6:] = torch.randint(0, 51, (8,)).float()
    visualize_mlp_from_model(model, x, fname="mlp_neuron_viz.png")"""