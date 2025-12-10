import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Example MLP (editable)
# ---------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # change sizes here to match your model
        self.seq = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
        )

    def forward(self, x):
        # We'll return both final output and a list of activations (per linear layer)
        activs = []
        cur = x
        # Save the input as the "layer 0" activation (so indexing aligns with visual layers)
        activs.append(cur.detach().cpu().numpy().reshape(-1))  # shape (in_features,)
        for m in self.seq:
            cur = m(cur)
            if isinstance(m, nn.Linear):
                # store a 1D array per linear layer (batch dim assumed 1)
                activs.append(cur.detach().cpu().numpy().reshape(-1))
        return cur, activs
class TrackedCircuit(nn.Module):
    def __init__(self, circuit: nn.Sequential):
        super().__init__()
        self.circuit = circuit

    def forward(self, x):
        activs = []
        cur = x

        # store input as activation layer 0
        activs.append(cur.detach().cpu().numpy().reshape(-1))

        for layer in self.circuit:
            cur = layer(cur)

            # Only record activations for modules that behave like layers
            if hasattr(layer, "weight") or hasattr(layer, "weights"):
                activs.append(cur.detach().cpu().numpy().reshape(-1))

        return cur, activs
# ---------------------------
# Visualization function
# ---------------------------
def visualize_mlp_from_model(model, input_tensor, fname="mlp_neuron_viz.png",
                             figsize=(10,6), show=True, draw_activations=True):
    # 1) get list of Linear layers in order
    linear_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear_layers.append(m)
    if not linear_layers:
        # maybe model has a seq attribute
        if hasattr(model, "seq"):
            for m in model.seq:
                if isinstance(m, nn.Linear):
                    linear_layers.append(m)

    # 2) compute layer sizes: input size + each linear out_features
    if len(linear_layers) == 0:
        raise ValueError("No Linear layers found in model.")
    layer_sizes = [linear_layers[0].in_features] + [L.out_features for L in linear_layers]

    # 3) run forward to get activations aligned with layers (input followed by each linear's output)
    with torch.no_grad():
        _, activations = model(input_tensor)

    # safety: ensure activations length matches layer_sizes
    if len(activations) != len(layer_sizes):
        # Try to pad/truncate sensibly
        # If activations missing input, insert input
        if len(activations) == len(layer_sizes) - 1:
            activations = [input_tensor.detach().cpu().numpy().reshape(-1)] + activations
        else:
            # fallback: create zeros for missing
            aligned = []
            for i, size in enumerate(layer_sizes):
                if i < len(activations):
                    a = activations[i].reshape(-1)
                    if a.size != size:
                        # try match by truncation/zero-pad
                        tmp = np.zeros(size, dtype=float)
                        tmp[:min(size, a.size)] = a[:min(size, a.size)]
                        aligned.append(tmp)
                    else:
                        aligned.append(a)
                else:
                    aligned.append(np.zeros(size, dtype=float))
            activations = aligned

    # 4) prepare neuron positions
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    n_layers = len(layer_sizes)
    x_spacing = 1.0 / (n_layers - 1) if n_layers > 1 else 1.0
    neuron_positions = {}

    # radius scales with figure size
    r = 0.03

    for i, size in enumerate(layer_sizes):
        x = i * x_spacing
        # vertical spacing with small margins
        y_spacing = 1.0 / (size + 1)
        for n in range(size):
            y = 1.0 - (n + 1) * y_spacing  # top-to-bottom
            neuron_positions[(i, n)] = (x, y)

            # neuron fill color from activation (optional)
            if draw_activations and activations is not None and i < len(activations):
                act_arr = activations[i]
                if n < act_arr.size:
                    val = float(act_arr[n])
                    # normalize activation to -1..1 by tanh (robust)
                    norm = np.tanh(val)
                    if norm >= 0:
                        color = plt.cm.Reds(0.3 + 0.7 * norm)  # red for positive
                    else:
                        color = plt.cm.Blues(0.3 + 0.7 * (-norm))  # blue for negative
                else:
                    color = 'white'
            else:
                color = 'white'

            circle = plt.Circle((x, y), r, color=color, ec="black", lw=0.5, zorder=4)
            ax.add_patch(circle)

    # 5) draw connections with weight-based alpha and color
    # collect all absolute weights to normalize linewidth/alpha
    all_weights = []
    for L in linear_layers:
        W = L.weight.detach().cpu().numpy()
        all_weights.append(np.abs(W).ravel())
    all_weights = np.concatenate(all_weights) if len(all_weights) > 0 else np.array([1.0])
    w_max = float(all_weights.max()) if all_weights.size > 0 else 1.0

    for layer_i, L in enumerate(linear_layers):
        W = L.weight.detach().cpu().numpy()  # shape (out, in)
        out_n, in_n = W.shape
        for out_idx in range(out_n):
            for in_idx in range(in_n):
                x1, y1 = neuron_positions[(layer_i, in_idx)]
                x2, y2 = neuron_positions[(layer_i+1, out_idx)]
                w = W[out_idx, in_idx]
                alpha = min(1.0, abs(w) / (w_max + 1e-12))  # 0..1
                lw = 0.3 + 2.0 * alpha
                if w >= 0:
                    color = (1.0, 0.0, 0.0, 0.2 + 0.8*alpha)  # red-ish with alpha
                else:
                    color = (0.0, 0.0, 1.0, 0.2 + 0.8*alpha)  # blue-ish
                # draw straight line (for curved, use Bezier or FancyArrowPatch)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1)

    # 6) labels: Input / Hidden / Output
    ax.text(0.01, 0.02, "Input", fontsize=10, ha='left', va='bottom')
    ax.text(0.5, 0.02, "Hidden Layers", fontsize=10, ha='center', va='bottom')
    ax.text(0.99, 0.02, "Output", fontsize=10, ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Saved visualization to {fname}")


# ---------------------------
# Run example
# ---------------------------
if __name__ == "__main__":
    model = MLP()
    sample_input = torch.randn(1, 4)
    visualize_mlp_from_model(model, sample_input, fname="mlp_neuron_viz.png")
