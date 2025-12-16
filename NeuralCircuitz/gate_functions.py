import torch
from typing import List

class Operator:
    def __init__(self, length: int, shapes: tuple, weights: list, biases: list):
        self.length = length
        self.shapes = shapes
        self.weights = weights  # list of tensors
        self.biases = biases    # list of tensors

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
        b1 = torch.tensor([len(input_indices)], dtype=torch.float32)  # activate only if NONE are 1 → threshold len
        # Actually simpler and more standard:
        b1 = torch.tensor([1.0])  # we want >0 only if at least one input is 1 → use -x + 1 bias
        w2 = torch.tensor([[-1.0]])
        b2 = torch.tensor([1.0])   # invert again
        super().__init__(length=2, shapes=(total_inputs, 1, 1), weights=[w1, w2], biases=[b1, b2])

class XorOp(Operator):
    """
    Generalized XOR for a set of input indices that returns 1 iff exactly one of the selected inputs is 1.
    Construction (layer by layer):
      L0: produce two identical sums [s, s] where s = sum(selected inputs)
      L1: produce [1 - s, 2 - s]  (ReLU after this -> left_pre = ReLU(1-s), inner_pre = ReLU(2-s))
      L2: produce [ left_pre (identity), 1 - inner_pre ] (ReLU after this -> left, right)
      L3: final combine: out = 1 - (left + right)  (no ReLU after final layer in builder)
    shapes: (total_inputs, 2, 2, 2, 1) with length=4
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) == 0:
            raise ValueError("GeneralXorOp requires at least one input index")

        # Layer 0: make two identical sums [s, s]
        w0 = torch.zeros(2, total_inputs)
        w0[0, input_indices] = 1.0
        w0[1, input_indices] = 1.0
        b0 = torch.zeros(2, dtype=torch.float32)

        # Layer 1: [1 - s, 2 - s]  -> weights map from the two s channels (both equal to s)
        # Because L0 produced [s, s], mapping [-1, 0] works (we only need one of the columns)
        w1 = torch.tensor([[-1.0, 0.0],
                           [-1.0, 0.0]], dtype=torch.float32)
        b1 = torch.tensor([1.0, 2.0], dtype=torch.float32)

        # Layer 2: produce [ left_pre (identity), 1 - inner_pre ]
        # left_pre is first channel passthrough; second row computes 1 - (second channel)
        w2 = torch.tensor([[1.0, 0.0],
                           [0.0, -1.0]], dtype=torch.float32)
        b2 = torch.tensor([0.0, 1.0], dtype=torch.float32)

        # Layer 3 (final): out = 1 - (left + right) -> weights [-1, -1], bias = 1
        w3 = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
        b3 = torch.tensor([1.0], dtype=torch.float32)

        weights = [w0, w1, w2, w3]
        biases = [b0, b1, b2, b3]
        shapes = (total_inputs, 2, 2, 2, 1)
        super().__init__(length=4, shapes=shapes, weights=weights, biases=biases)

class MemoryOp(Operator):
    """
    Simple identity gate for a single input: passes through the value.
    w=1, b=0 as specified. Allows only one input.
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("MemoryOp requires exactly one input index")
        w = torch.zeros(1, total_inputs)
        w[0, input_indices[0]] = 1.0
        b = torch.tensor([0.0], dtype=torch.float32)
        super().__init__(length=1, shapes=(total_inputs, 1), weights=[w], biases=[b])

class PlusOneOp(Operator):
    """
    Simple identity gate for a single input: passes through the value.
    w=1, b=0 as specified. Allows only one input.
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("MemoryOp requires exactly one input index")
        w = torch.zeros(1, total_inputs)
        w[0, input_indices[0]] = 1.0
        b = torch.tensor([1.0], dtype=torch.float32)
        super().__init__(length=1, shapes=(total_inputs, 1), weights=[w], biases=[b])



class NotOp(Operator):
    """
    Negates a binary input (0 -> 1, 1 -> 0).
    Assumes input is exactly 0 or 1.
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("NotOp requires exactly one input index")
        w = torch.zeros(1, total_inputs)
        w[0, input_indices[0]] = -1.0
        b = torch.tensor([1.0], dtype=torch.float32)
        super().__init__(length=1, shapes=(total_inputs, 1), weights=[w], biases=[b])

class EqualsZeroOp(Operator):
    """
    Returns 1 if the single input integer == 0, else 0 exactly.
    Works for any integer input (exact due to integer assumption).
    Construction: 1 - indicator(|x| >= 1), where indicator(y >= 1) = ReLU(y) - ReLU(y - 1)
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("EqualsZeroOp requires exactly one input index")
        idx = input_indices[0]
        # Layer 0: [x, -x]
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx] = 1.0
        w0[1, idx] = -1.0
        b0 = torch.zeros(2, dtype=torch.float32)
        # Layer 1: |x| = ReLU(x) + ReLU(-x)
        w1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        # Layer 2: [|x|, |x| - 1]
        w2 = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        b2 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        # Layer 3: indicator(|x| >= 1) = ReLU(|x|) - ReLU(|x| - 1)
        w3 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b3 = torch.tensor([0.0], dtype=torch.float32)
        # Layer 4: 1 - indicator
        w4 = torch.tensor([[-1.0]], dtype=torch.float32)
        b4 = torch.tensor([1.0], dtype=torch.float32)
        weights = [w0, w1, w2, w3, w4]
        biases = [b0, b1, b2, b3, b4]
        shapes = (total_inputs, 2, 1, 2, 1, 1)
        super().__init__(length=5, shapes=shapes, weights=weights, biases=biases)

class NotEqualZeroOp(Operator):
    """
    Returns 1 if the single input integer != 0, else 0 exactly.
    Works for any integer input (exact due to integer assumption).
    Construction: indicator(|x| >= 1) = ReLU(|x|) - ReLU(|x| - 1)
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 1:
            raise ValueError("NotEqualZeroOp requires exactly one input index")
        idx = input_indices[0]
        # Layer 0: [x, -x]
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx] = 1.0
        w0[1, idx] = -1.0
        b0 = torch.zeros(2, dtype=torch.float32)
        # Layer 1: |x| = ReLU(x) + ReLU(-x)
        w1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        # Layer 2: [|x|, |x| - 1]
        w2 = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        b2 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        # Layer 3: indicator(|x| >= 1) = ReLU(|x|) - ReLU(|x| - 1)
        w3 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b3 = torch.tensor([0.0], dtype=torch.float32)
        weights = [w0, w1, w2, w3]
        biases = [b0, b1, b2, b3]
        shapes = (total_inputs, 2, 1, 2, 1)
        super().__init__(length=4, shapes=shapes, weights=weights, biases=biases)

class EqualOp(Operator):
    """
    Returns 1 if input0 == input1 (integers), else 0 exactly.
    Construction: same as EqualsZero on (input0 - input1)
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 2:
            raise ValueError("EqualOp requires exactly two input indices")
        idx0, idx1 = input_indices
        # Layer 0: [d, -d] where d = x0 - x1
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx0] = 1.0
        w0[0, idx1] = -1.0
        w0[1, idx0] = -1.0
        w0[1, idx1] = 1.0
        b0 = torch.zeros(2, dtype=torch.float32)
        # Layer 1: |d|
        w1 = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        # Layer 2: [|d|, |d| - 1]
        w2 = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        b2 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        # Layer 3: indicator(|d| >= 1)
        w3 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b3 = torch.tensor([0.0], dtype=torch.float32)
        # Layer 4: 1 - indicator
        w4 = torch.tensor([[-1.0]], dtype=torch.float32)
        b4 = torch.tensor([1.0], dtype=torch.float32)
        weights = [w0, w1, w2, w3, w4]
        biases = [b0, b1, b2, b3, b4]
        shapes = (total_inputs, 2, 1, 2, 1, 1)
        super().__init__(length=5, shapes=shapes, weights=weights, biases=biases)

class GreaterThanOp(Operator):
    """
    Returns 1 if input0 > input1 (integers, i.e., input0 >= input1 + 1), else 0 exactly.
    Construction: indicator(d >= 1) where d = input0 - input1
    indicator = ReLU(d) - ReLU(d - 1)
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 2:
            raise ValueError("GreaterThanOp requires exactly two input indices")
        idx0, idx1 = input_indices
        # Layer 0: [d, d - 1] where d = x0 - x1
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx0] = 1.0
        w0[0, idx1] = -1.0
        w0[1, idx0] = 1.0
        w0[1, idx1] = -1.0
        b0 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        # Layer 1: indicator = ReLU(d) - ReLU(d - 1)
        w1 = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        b1 = torch.tensor([0.0], dtype=torch.float32)
        weights = [w0, w1]
        biases = [b0, b1]
        shapes = (total_inputs, 2, 1)
        super().__init__(length=2, shapes=shapes, weights=weights, biases=biases)

class LessThanOp(Operator):
    """
    Returns 1 if input0 < input1 (integers, i.e., input0 <= input1 - 1), else 0 exactly.
    Construction: indicator(d <= -1) = indicator(-d >= 1) = ReLU(-d) - ReLU(-d - 1)
    Where d = input0 - input1
    """
    def __init__(self, input_indices: List[int], total_inputs: int):
        if len(input_indices) != 2:
            raise ValueError("LessThanOp requires exactly two input indices")
        idx0, idx1 = input_indices
        # Layer 0: [-d, -d - 1] where d = x0 - x1 so -d = x1 - x0
        w0 = torch.zeros(2, total_inputs)
        w0[0, idx0] = -1.0
        w0[0, idx1] = 1.0
        w0[1, idx0] = -1.0
        w0[1, idx1] = 1.0
        b0 = torch.tensor([0.0, -1.0], dtype=torch.float32)
        # Layer 1: indicator = ReLU(-d) - ReLU(-d - 1)
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