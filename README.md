# Curry: Spatial Recurrent Policies for Reinforcement Learning

Training recurrent convolutional policies (ConvLSTM and ConvAtt) for partially observable environments using RecurrentPPO.

## Overview

This repository implements two novel spatial recurrent architectures:

1. **ConvLSTMPolicy** - Spatial LSTM with convolutional memory cells that preserve feature map structure
2. **ConvAttPolicy** - Causal self-attention over evolving latent feature maps (see [convatt architecture explanation](explanations/convatt_architecture_explanation.md))

Both policies maintain **spatial hidden states** rather than flat vectors, enabling better spatial reasoning in visual environments.

## Key Innovation: SequenceAwareRolloutBuffer

The standard `RecurrentRolloutBuffer` from sb3_contrib caused dimension mismatches when training spatial recurrent policies:

```python
# Problem: evaluate_actions received
features: (84, 32, 10, 10)  # minibatch of 84 observations
states:   (1, 2, 9600)       # only 2 state histories ❌
```

**Solution**: Custom `SequenceAwareRolloutBuffer` that samples hidden states **per observation** instead of per sequence start, ensuring proper state-feature alignment.

See [BUFFER_FIX_EXPLANATION.md](explanations/BUFFER_FIX_EXPLANATION.md) for technical details.

## Project Structure

```
curry/
├── train.py                           # Main training script
├── sequence_aware_buffer.py           # Custom buffer implementation
├── convlstm.py                        # ConvLSTM policy
├── convatt.py                         # ConvAtt policy
├── sokoban_wrapper.py                 # Environment wrapper
├── convlstm_config.yaml              # ConvLSTM hyperparameters
├── convatt_config.yaml               # ConvAtt hyperparameters
└── explanations/                      # Documentation
    ├── BUFFER_FIX_EXPLANATION.md     # Buffer fix technical details
    └── convatt_architecture_explanation.md  # ConvAtt architecture
```

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install gymnasium
pip install stable-baselines3
pip install sb3-contrib
pip install pyyaml
pip install torch
pip install gym-sokoban  # Or your target environment
```

## Usage

### Train ConvLSTM Policy

```bash
python train.py \
    --policy ConvLSTMPolicy \
    --config-file convlstm_config.yaml \
    --total-timesteps 1000000 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --device cuda
```

### Train ConvAtt Policy

```bash
python train.py \
    --policy ConvAttPolicy \
    --config-file convatt_config.yaml \
    --total-timesteps 1000000 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --device cuda
```

### Arguments

- `--policy`: Policy architecture (`ConvLSTMPolicy` or `ConvAttPolicy`)
- `--config-file`: Path to YAML configuration file
- `--total-timesteps`: Number of training timesteps
- `--batch-size`: Minibatch size for PPO updates
- `--learning-rate`: Optimizer learning rate
- `--device`: Training device (`cpu` or `cuda`)
- `--tensorboard-log`: TensorBoard log directory (default: `./tensorboard/`)
- `--seed`: Random seed for reproducibility (default: 42)

## Policy Architectures

### ConvLSTM Policy

**Architecture:**
```
Observation (H, W, C)
  ↓
CNN Feature Extractor
  ↓
Feature Maps (B, C, H', W')
  ↓
Stacked ConvLSTM Cells
  ↓ (maintains spatial structure)
Hidden States (B, C_hidden, H', W')
  ↓
Adaptive Pooling
  ↓
MLP Actor/Critic Heads
  ↓
Actions & Values
```

**Key Features:**
- Convolutional LSTM cells that process spatial feature maps
- Hidden and cell states preserve spatial dimensions
- Stacked architecture for hierarchical representations
- Adaptive pooling for fixed-size MLP input

**Configuration** ([convlstm_config.yaml](convlstm_config.yaml)):
```yaml
config:
  conv_configs:
    - in_channels: 3
      out_channels: 32
      kernel_size: 8
      stride: 4
      padding: 0
  hidden_channels: 32
  kernel_size: 3
  stack_size: 1
  pool_output_size: [2, 2]
```

### ConvAtt Policy

**Architecture:**
```
Observation (H, W, C)
  ↓
CNN Feature Extractor
  ↓
Feature Maps (B, C, H', W')
  ↓
ConvAtt Residual RNN
  ↓ (causal self-attention)
Memory Stack (B, M, C_hidden, H', W')
  ↓
Adaptive Pooling (last memory slot)
  ↓
MLP Actor/Critic Heads
  ↓
Actions & Values
```

**Key Features:**
- Evolving latent feature map memory
- Causal self-attention over memory stack
- Residual connections preserve stable features
- Memory size controls temporal receptive field

**How It Works:**
1. Current observation features prepended to memory stack: `[x_t, m_t-1, m_t-2, ...]`
2. Causal attention applied (current frame can only attend to itself)
3. Residual update: `new_memory = old_memory + attention(old_memory)`
4. Last memory slot acts as global accumulator (has full context)
5. Drop oldest memory slot, keep last M frames

See [convatt_architecture_explanation.md](explanations/convatt_architecture_explanation.md) for detailed explanation.

**Configuration** ([convatt_config.yaml](convatt_config.yaml)):
```yaml
config:
  conv_configs:
    - in_channels: 3
      out_channels: 32
      kernel_size: 8
      stride: 4
      padding: 0
  hidden_channels: 32
  qk_dim: 64
  kernel_size: 3
  stack_size: 1
  memory_size: 4
  pool_output_size: [2, 2]
```

## Custom Buffer: SequenceAwareRolloutBuffer

The standard `RecurrentRolloutBuffer` samples states only at sequence start indices, which doesn't work for spatial recurrent policies that need per-observation states.

**Standard Buffer:**
```python
# Samples states at sequence starts only
states[seq_start_indices]  # Shape: (n_layers, n_sequences, state_dim)
```

**SequenceAwareRolloutBuffer:**
```python
# Samples states for ALL observations
states[batch_inds]  # Shape: (n_layers, batch_size, state_dim)
```

This ensures proper alignment:
```python
features: (batch=84, C=32, H=10, W=10)
states:   (n_layers=1, batch=84, state_dim=3200)  ✅
```

See [BUFFER_FIX_EXPLANATION.md](explanations/BUFFER_FIX_EXPLANATION.md) for implementation details.

## Training Output

```
================================================================================
Starting training with ConvAttPolicy
  Total timesteps: 1000000
  Batch size: 64
  Learning rate: 0.0003
  Device: cuda
  Buffer: SequenceAwareRolloutBuffer (fixes state dimension matching)
================================================================================

[Training progress with TensorBoard logging]

✅ Training complete! Model saved as: sokoban_convattpolicy_model.zip
```

## Monitoring Training

```bash
tensorboard --logdir=./tensorboard/
```

Open http://localhost:6006 to view:
- Episode rewards
- Policy loss
- Value loss
- Entropy
- Learning rate schedule

## Configuration Files

### ConvLSTM Config

**Standard** ([convlstm_config.yaml](convlstm_config.yaml)): Default spatial dimensions
**Compact** ([convlstm_compact_config.yaml](convlstm_compact_config.yaml)): Smaller pooling output

### ConvAtt Config

**Standard** ([convatt_config.yaml](convatt_config.yaml)): Memory size 4
**Compact** ([convatt_compact_config.yaml](convatt_compact_config.yaml)): Memory size 2

## Hidden State Initialization

Currently, hidden states are initialized as **zero tensors** at episode start:

```python
h_0 = torch.zeros(n_layers, batch, hidden_channels, H, W)
```

### Alternative: Initialize with First Frame

To initialize memory with the first observation's features instead:

```python
# In _process_sequence method
if episode_starts[0]:
    h_0 = features[:, 0].unsqueeze(1).repeat(1, memory_size, 1, 1, 1)
```

Modify your policy's `_process_sequence` method to experiment with different initialization strategies.

## Technical Details

### State Storage Format

**During collection:**
```
(n_steps, n_layers, n_envs, state_dim)
```

**During training (after flattening):**
```
(n_steps * n_envs, n_layers, state_dim)
```

**Sampled batch:**
```
(n_layers, batch_size, state_dim)
```

### Spatial State Reshaping

For ConvLSTM:
```python
state_dim = hidden_channels * H * W
# Reshape: (n_layers, batch, state_dim) → (n_layers, batch, hidden_channels, H, W)
h_spatial = h_flat.view(n_layers, batch, hidden_channels, H, W)
```

For ConvAtt:
```python
state_dim = memory_size * hidden_channels * H * W
# Reshape: (n_layers, batch, state_dim) → (n_layers, batch, memory_size, hidden_channels, H, W)
h_memory = h_flat.view(n_layers, batch, memory_size, hidden_channels, H, W)
```

## Troubleshooting

### Dimension Mismatch Errors

If you see errors like:
```
RuntimeError: Sizes of tensors must match except in dimension 0
```

**Check:**
1. `state_dim` calculation matches your config
2. Batch size is compatible with buffer size
3. States are properly reshaped in `_process_sequence`

### Device Errors

```
RuntimeError: Expected all tensors to be on the same device
```

**Solution:**
```bash
# Ensure consistent device usage
python train.py --device cuda  # or --device cpu
```

### Memory Issues

For large batch sizes or memory sizes:
```bash
# Reduce batch size
python train.py --batch-size 32

# Or use compact config (smaller pooling output)
python train.py --config-file convatt_compact_config.yaml
```

## Citation

If you use this code, please cite:

```bibtex
@misc{curry2025,
  title={Spatial Recurrent Policies with Sequence-Aware Buffers},
  author={Curry Contributors},
  year={2025},
  url={https://github.com/yourusername/curry}
}
```

## License

[Your License Here]

## Acknowledgments

- Built on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Uses [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) RecurrentPPO
- Tested on [Gym-Sokoban](https://github.com/mpSchrader/gym-sokoban)
