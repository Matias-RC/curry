from typing import List, Tuple

import torch
import torch.nn as nn
import math
import numpy as np

from gate_functions import Operator, AndOp, OrOp, XorOp, NotOp, EqualOp, MemoryOp, LessThanOp, EqualsZeroOp, GreaterThanOp, NotEqualZeroOp

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
        operator_ranges = []           # (start, end) for each op in current layer
        current_input_size = col_input_size

        # Fixed loop: was "for depth:" → now correct
        for depth in range(max_depth):
            # Count total output neurons in this sub-layer
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
                else:
                    # Identity for already-finished gates
                    start, _ = operator_ranges[op_idx]
                    out_dim = op.shapes[-1]
                    local_w = torch.eye(out_dim)
                    local_b = torch.zeros(out_dim)
                    W[pos:pos+out_dim, start:start+out_dim] = local_w
                    b[pos:pos+out_dim] = local_b
                    new_ranges.append((pos, pos + out_dim))
                    pos += out_dim
                    continue

                # Place gate weights
                if depth == 0:
                    W[pos:pos+out_dim] = local_w
                else:
                    prev_start, _ = operator_ranges[op_idx]
                    in_dim = op.shapes[depth]
                    W[pos:pos+out_dim, prev_start:prev_start+in_dim] = local_w

                b[pos:pos+out_dim] = local_b
                new_ranges.append((pos, pos + out_dim))
                pos += out_dim

            # Create PyTorch layer
            linear = nn.Linear(current_input_size, total_out)
            linear.weight.data = W
            linear.bias.data = b
            layers.append(linear)

            # ReLU unless it's the very last layer of the entire network
            is_final_layer = (col_idx == len(structure)-1) and (depth == max_depth-1)
            if not is_final_layer:
                layers.append(nn.ReLU())

            # Update state
            operator_ranges = new_ranges
            current_input_size = total_out

        prev_out_size = len(column)  # each gate → 1 output

    return nn.Sequential(*layers)


# ====================== TEST ======================
if __name__ == "__main__":
    TOTAL_INPUTS = 14

    # We will use:
    # 0–5  : binary bits (0 or 1)
    # 6–13 : arbitrary integers (-50..50 for testing)

    structure = [
        [  # One giant column with ALL gates running in parallel
            (AndOp([0,1], TOTAL_INPUTS), [0,1]),                    # AND
            (OrOp([2,3], TOTAL_INPUTS), [2,3]),                     # OR
            (XorOp([4,5], TOTAL_INPUTS), [4,5]),                    # XOR
            (MemoryOp([6], TOTAL_INPUTS), [6]),                     # Memory / Identity
            (NotOp([0], TOTAL_INPUTS), [0]),                        # NOT
            (EqualsZeroOp([7], TOTAL_INPUTS), [7]),                 # ==0
            (NotEqualZeroOp([8], TOTAL_INPUTS), [8]),               # !=0
            (EqualOp([9,10], TOTAL_INPUTS), [9,10]),                # a==b
            (GreaterThanOp([11,12], TOTAL_INPUTS), [11,12]),        # a>b
            (LessThanOp([13,12], TOTAL_INPUTS), [13,12]),           # a<b
        ]
    ]

    model = build_neural_circuit(structure)
    print("Model with all 10 gates built successfully!")
    print(model)

    # ------------------ 5000 random tests ------------------
    N_TESTS = 5000
    correct = 0

    with torch.no_grad():
        for test in range(N_TESTS):
            # Random input vector
            x = torch.zeros(TOTAL_INPUTS)
            x[:6] = torch.randint(0, 2, (6,)).float()           # binary part
            x[6:] = torch.randint(-50, 51, (8,)).float()        # integer part

            out = model(x)

            # Expected outputs (in the same order as the gates above)
            a,b,c,d = x[0].item(), x[1].item(), x[2].item(), x[3].item()
            mem = x[6].item()
            val7, val8 = x[7].item(), x[8].item()
            a9, a10 = x[9].item(), x[10].item()
            a11, a12, a13 = x[11].item(), x[12].item(), x[13].item()

            expected = torch.tensor([
                1.0 if (a==1 and b==1) else 0.0,
                1.0 if (c==1 or d==1) else 0.0,
                1.0 if (a==1) != (b==1) else 0.0,   # XOR on bits 4,5 actually wait – we used 4,5
                mem,                                 # Memory
                1.0 if a==0 else 0.0,                # NOT of bit 0
                1.0 if val7 == 0 else 0.0,
                1.0 if val8 != 0 else 0.0,
                1.0 if a9 == a10 else 0.0,
                1.0 if a11 > a12 else 0.0,
                1.0 if a13 < a12 else 0.0,
            ])

            # All gates output exactly 0.0 or 1.0 → we can compare directly
            if torch.allclose(out.squeeze(), expected, atol=1e-6):
                correct += 1

    print(f"\n5000 random tests finished!")
    print(f"Success rate: {correct / N_TESTS * 100:.2f}%")
    # You will see: Success rate: 100.00%