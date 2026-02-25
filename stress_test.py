import torch
import torch.nn as nn
import gc
import time
import sys

class StatelessDynamicModule(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.param = nn.Parameter(torch.empty((0, size)))

    def incorporate(self, vector):
        # Detach and clone to ensure no graph history is preserved
        vector = vector.detach().view(1, self.size)
        if self.param.numel() == 0:
            new_data = vector.clone()
        else:
            # Re-concatenating into a new memory block
            new_data = torch.cat([self.param.data, vector], dim=0).clone()
        self.param = nn.Parameter(new_data)

    def erase(self, index):
        if self.param.shape[0] == 0: return
        indices = torch.arange(self.param.shape[0], device=self.param.device)
        # Clone ensures we don't keep a 'view' of the larger original tensor
        new_data = self.param.data[indices != index].clone()
        self.param = nn.Parameter(new_data)

def run_stress_test(vector_size=2048, iterations=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StatelessDynamicModule(vector_size).to(device)
    
    print(f"--- HARDWARE CHECK ---", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)
    print(f"Vector Size: {vector_size} | Total Iterations: {iterations}", flush=True)
    
    start_time = time.time()

    for i in range(1, iterations + 1):
        # 1. Expand/Contract Logic (Cycle between 1 and 20 rows)
        if model.param.shape[0] < 20:
            model.incorporate(torch.randn(vector_size, device=device))
        else:
            # Erase the middle element to force memory shifting
            model.erase(10)

        # 2. Optimization
        opt = torch.optim.SGD([model.param], lr=0.01)
        target = torch.ones_like(model.param)
        loss = torch.nn.functional.mse_loss(model.param, target)
        loss.backward()
        opt.step()
        
        # 3. Memory Cleanup
        opt.zero_grad(set_to_none=True)
        del opt
        
        # 4. Periodic Logging (Every 1000 steps)
        if i % 1000 == 0 or i == 1:
            elapsed = time.time() - start_time
            if device.type == "cuda":
                # Check reserved vs allocated memory
                res = torch.cuda.memory_reserved() / 1e6
                alc = torch.cuda.memory_allocated() / 1e6
                print(f"Step: {i:5d} | Rows: {model.param.shape[0]:2d} | Res: {res:7.2f}MB | Alc: {alc:7.2f}MB | Time: {elapsed:.1f}s", flush=True)
                torch.cuda.empty_cache()
            else:
                print(f"Step: {i:5d} | Rows: {model.param.shape[0]:2d} | Time: {elapsed:.1f}s", flush=True)
            
            gc.collect()

    print(f"--- TEST COMPLETE --- Total Time: {time.time() - start_time:.2f}s", flush=True)

if __name__ == "__main__":
    try:
        run_stress_test()
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr, flush=True)