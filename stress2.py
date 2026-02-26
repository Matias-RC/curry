import torch as th
import torch.nn as nn

prefix_memory = nn.Parameter(th.ones((3, 4), requires_grad=True))

with th.no_grad():
    prefix_memory[0] = 1.0
    prefix_memory[1] = 2.0
    prefix_memory[2] = 3.0

batch_indices = th.tensor([0, 0, 1, 2, 0])
target = th.rand((5, 4))
optimizer = th.optim.Adam([prefix_memory], lr=.01)


print(target)
for i in range(1000):
    # RE-ESTABLISH THE GRAPH LINK HERE
    batch_prefixes = prefix_memory[batch_indices] 

    
    loss = nn.functional.mse_loss(batch_prefixes, target)
    if i%100 == 0:
        print(batch_prefixes)
        print(f"Iter {i} | Loss: {loss.item():.4f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # Don't forget to step!
    
print("\nUpdated Source of Truth:")
print(prefix_memory)