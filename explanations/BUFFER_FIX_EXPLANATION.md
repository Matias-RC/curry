# Solution: Fixing RecurrentPPO Hidden State Dimension Mismatch

## Problem Summary

During training with RecurrentPPO and spatial recurrent policies (ConvLSTM/ConvAtt), you encountered a critical dimension mismatch in `evaluate_actions`:

```python
# During training evaluate_actions received:
features: (84, 32, 10, 10)  # 84 samples in minibatch
states:   (1, 2, 9600)       # Only 2 state histories (from n_envs)

# When reshaping states to (n_layers, batch, C, H, W):
x (84, C, H, W) + h (2, C, H, W) → ❌ CANNOT CONCATENATE
```

### Root Cause

The standard `RecurrentRolloutBuffer` from sb3_contrib stores states indexed by `(n_steps, n_layers, n_envs, state_dim)`. During training:

1. **Data Collection Phase**: States are stored for `n_envs` parallel environments (typically 1-2)
2. **Training Phase**: Observations are sampled in minibatches (e.g., 84 samples)
3. **State Sampling Issue**: The buffer only provided states at **sequence start indices**, not for each observation

This meant:
- 84 observations → only 2-4 states (one per sequence start)
- Your policies need states for EVERY observation in the batch
- Result: Dimension mismatch when trying to process features + hidden states

## The Solution: SequenceAwareRolloutBuffer

Created a custom buffer class that samples hidden states **per observation** instead of per sequence start.

### Key Changes

#### File: `sequence_aware_buffer.py`

```python
class SequenceAwareRolloutBuffer(RecurrentRolloutBuffer):
    """
    Custom buffer that properly samples LSTM states per observation.

    Standard approach (parent class):
        lstm_states = states[batch_inds][seq_start_indices]  # Only seq starts

    New approach (this class):
        lstm_states = states[batch_inds]  # All observations
    """

    def _get_samples(self, batch_inds, env):
        # Sample states for EVERY observation in batch_inds
        lstm_states_pi = (
            torch.tensor(
                self.hidden_states_pi[batch_inds].swapaxes(0, 1),
                dtype=torch.float32,
                device=self.device
            ),
            torch.tensor(
                self.cell_states_pi[batch_inds].swapaxes(0, 1),
                dtype=torch.float32,
                device=self.device
            ),
        )
        # Result shape: (n_layers, batch_size, state_dim) ✅
```

### Integration

#### File: `train.py`

```python
from sequence_aware_buffer import SequenceAwareRolloutBuffer

model = RecurrentPPO(
    policy_class,
    env_registered,
    # ... other args ...
    buffer_class=SequenceAwareRolloutBuffer,  # Use custom buffer
)
```

## How It Works

### Before (Standard Buffer)

```
Buffer stores: (n_steps=128, n_layers=1, n_envs=2, state_dim=9600)
                ↓ Flatten to (256, n_layers, state_dim)
                ↓ Sample batch_inds = [5, 12, 23, 45, ...]  (64 indices)
                ↓ Find sequence starts = [5, 23, 45]  (only 3 starts)
                ↓ Sample states at seq starts only

Result: (n_layers=1, n_sequences=3, state_dim=9600)
        ❌ Only 3 states for 64 observations!
```

### After (SequenceAware Buffer)

```
Buffer stores: (n_steps=128, n_layers=1, n_envs=2, state_dim=9600)
                ↓ Flatten to (256, n_layers, state_dim)
                ↓ Sample batch_inds = [5, 12, 23, 45, ...]  (64 indices)
                ↓ Sample states at ALL batch_inds

Result: (n_layers=1, batch_size=64, state_dim=9600)
        ✅ 64 states for 64 observations!
```

## State Flow in Your Policies

### ConvLSTM Policy

```python
# evaluate_actions receives:
obs:    (batch=84, C=3, H=80, W=80)
states: (n_layers=1, batch=84, state_dim=3200)

# In _process_sequence:
# 1. Extract CNN features
features = cnn(obs)  # (84, 32, 10, 10)

# 2. Reshape states to spatial format
h_spatial = states[0].view(84, 32, 10, 10)  # ✅ Works now!

# 3. Process through ConvLSTM
for t in range(T):
    x_t = features[:, t]  # (84, 32, 10, 10)
    h_t, c_t = lstm(x_t, (h_spatial, c_spatial))
```

### ConvAtt Policy

```python
# evaluate_actions receives:
obs:    (batch=84, C=3, H=80, W=80)
states: (n_layers=1, batch=84, state_dim=9600)

# In _process_sequence:
# 1. Extract CNN features
features = cnn(obs)  # (84, 32, 10, 10)

# 2. Reshape states to memory format
# state_dim = memory_size(4) * hidden_channels(32) * H(10) * W(10)
h_memory = states[0].view(84, 4, 32, 10, 10)  # (B, M, C, H, W) ✅

# 3. Process through ConvAtt
for t in range(T):
    x_t = features[:, t]  # (84, 32, 10, 10)
    h_t = conv_att(x_t, h_memory)  # Attention over memory
```

## Verification

Run the test suite to verify the fix:

```bash
source .venv/bin/activate
python test_buffer_dimensions.py
```

Expected output:
```
✅ SUCCESS: State dimensions match expectations!
   Expected: (1, 64, 9600)
   Actual:   torch.Size([1, 64, 9600])

   This means for a minibatch of 64 observations,
   we have 64 corresponding hidden states.
   ✅ No more dimension mismatch!
```

## Training with the Fix

Your existing training script should now work:

```bash
# Train ConvLSTM
python train.py --policy ConvLSTMPolicy --config-file convlstm_config.yaml --total-timesteps 100000

# Train ConvAtt
python train.py --policy ConvAttPolicy --config-file convatt_config.yaml --total-timesteps 100000
```

## Key Insights

### Why the Standard Buffer Doesn't Work

The standard `RecurrentRolloutBuffer` is designed for **sequence-to-sequence** processing where:
- LSTM processes sequences of observations
- Only the initial hidden state (at sequence start) is needed
- The LSTM unrolls from that initial state

### Why Your Policies Need Per-Observation States

Your spatial recurrent policies (ConvLSTM/ConvAtt) operate differently:
- Each observation has spatial dimensions (H, W)
- Hidden states are spatial feature maps, not flat vectors
- During `evaluate_actions`, you need to:
  1. Extract features from each observation
  2. Combine with its corresponding hidden state
  3. Process through the recurrent architecture

This requires a **1-to-1 mapping** between observations and hidden states, which the standard buffer doesn't provide.

## Hidden State Initialization

### Current Approach
Hidden states are initialized as zero tensors at the start of each episode (via `episode_starts` masking in `_process_sequence`).

### Alternative Approaches You Mentioned

1. **Zero Initialization** (current)
   ```python
   h_0 = torch.zeros(n_layers, batch, hidden_channels, H, W)
   ```

2. **Copy First Features** (your proposal)
   ```python
   # In _process_sequence, at episode start:
   if episode_starts[0]:
       h_0 = features[:, 0].unsqueeze(1).repeat(1, memory_size, 1, 1, 1)
       # Uses first frame to initialize all memory slots
   ```

To implement option 2, modify your policy's `_process_sequence` method:

```python
# In ConvAttPolicy._process_sequence
for t in range(T):
    input_t = features[:, t]

    if episode_starts[:, t].any():
        mask = episode_starts[:, t].bool()
        # Initialize memory with first frame
        if t == 0:
            init_memory = input_t[mask].unsqueeze(1).repeat(1, self.memory_size, 1, 1, 1)
            current_states[0][mask] = init_memory
        else:
            # Zero out for mid-sequence resets
            current_states[0][mask] = 0
```

## Files Created/Modified

1. **`sequence_aware_buffer.py`** - Custom buffer implementation
2. **`train.py`** - Updated to use custom buffer
3. **`test_buffer_dimensions.py`** - Test suite for verification
4. **`BUFFER_FIX_EXPLANATION.md`** - This documentation

## Next Steps

1. ✅ Buffer implementation complete and tested
2. ⏭️ Run actual training with ConvLSTM policy
3. ⏭️ Run actual training with ConvAtt policy
4. ⏭️ Monitor TensorBoard for learning curves
5. ⏭️ (Optional) Experiment with hidden state initialization strategies

## Troubleshooting

### If you still see dimension mismatches:

1. **Check batch size**: Ensure your batch_size in config is compatible with buffer_size * n_envs
2. **Verify state dimensions**: Print shapes in `_process_sequence` to debug
3. **Check device placement**: Ensure states and features are on the same device (CPU/GPU)

### Common issues:

- **"Cannot reshape tensor"**: State dimension calculation mismatch
  - Verify: `state_dim = memory_size * hidden_channels * H * W` (for ConvAtt)
  - Verify: `state_dim = hidden_channels * H * W` (for ConvLSTM)

- **"Expected all tensors to be on the same device"**:
  - Check `device` parameter in RecurrentPPO matches policy device
  - Ensure buffer device matches in `SequenceAwareRolloutBuffer`

## References

- Standard buffer: `.venv/lib/python3.12/site-packages/sb3_contrib/common/recurrent/buffers.py`
- RecurrentPPO implementation: sb3_contrib
- Your policies: `convlstm.py`, `convatt.py`
