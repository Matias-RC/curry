import torch
import torch.nn as nn
import gc

class StressTestedModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.param = nn.Parameter(torch.empty((0, size)))

    def incorporate(self, vector):
        # 1. Flatten and Validate
        vector = vector.detach().view(1, self.size)
        
        # 2. Concatenate and Clone to break memory links
        if self.param.shape[0] == 0:
            new_data = vector.clone()
        else:
            new_data = torch.cat([self.param.data, vector], dim=0).clone()
        
        # 3. Replace Parameter (Old one is now eligible for GC)
        self.param = nn.Parameter(new_data)

    def erase(self, index):
        if self.param.shape[0] == 0: return
        
        # 1. Create a mask to filter out the index
        indices = torch.arange(self.param.shape[0])
        mask = indices != index
        
        # 2. Slice and Clone to ensure we don't keep a 'view' of the old tensor
        new_data = self.param.data[mask].clone()
        
        # 3. Reassign
        self.param = nn.Parameter(new_data)

    def forward(self):
        return self.param

def run_memory_test(iterations=2000, vector_size=512):
    model = StressTestedModule(vector_size)
    print(f"Starting Stress Test: {iterations} cycles of Add/Backprop/Erase")
    
    for i in range(iterations):
        # Step A: Add a parameter
        model.incorporate(torch.randn(vector_size))
        
        # Step B: Simulate Training (This creates gradients and optimizer state)
        # We use Adam because it has heavy memory buffers (2x param size)
        optimizer = torch.optim.Adam([model.param], lr=1e-3)
        target = torch.randn_like(model.param)
        loss = torch.nn.functional.mse_loss(model.param, target)
        loss.backward()
        optimizer.step()
        
        # Step C: Erase a parameter (The oldest one)
        model.erase(0)
        
        # Step D: Cleanup
        # If we don't delete the optimizer, it keeps a reference to the 
        # specific 'model.param' object we just replaced!
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        
        if i % 500 == 0:
            # Clear Python GC and Torch Cache for accurate reading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem = torch.cuda.memory_reserved() / 1e6
                print(f"Iteration {i} | Reserved Memory: {mem:.2f}MB")
            else:
                print(f"Iteration {i} complete (CPU).")

    print("Test complete. If memory didn't explode, the 'erase' logic is sound.")

run_memory_test()