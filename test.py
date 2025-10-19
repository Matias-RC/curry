# plot_training_logs.py
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

# === Hyperparameters ===
SMOOTH_WINDOW = 20     # moving average window size
POINT_ALPHA = 0.3      # transparency for raw points
FIG_SIZE = (14, 10)    # figure size
CLIP_PERCENTILES = (1, 99)  # clip outliers (keep 1st–99th percentile range)


def read_csv_log(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float_list(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def moving_average(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def clip_outliers(y, low=1, high=99):
    """Clip y values to percentile range to remove extreme spikes."""
    if len(y) == 0:
        return y
    lo, hi = np.percentile(y, [low, high])
    return np.clip(y, lo, hi)


def plot_log(csv_path, out_png=None, show=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV log not found: {csv_path}")

    rows = read_csv_log(csv_path)
    if len(rows) == 0:
        raise RuntimeError("CSV log is empty")

    # Use update number as x-axis for uniform spacing
    x = to_float_list(rows, 'update')

    cols = ['avg_episode_reward', 'avg_step_reward',
            'policy_loss', 'value_loss', 'entropy', 'total_loss']

    fig, axes = plt.subplots(3, 2, figsize=FIG_SIZE)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        y = to_float_list(rows, col)
        y = clip_outliers(y, *CLIP_PERCENTILES)

        ax = axes[i]
        ax.scatter(x, y, s=10, alpha=POINT_ALPHA, color='tab:gray', label='Raw (clipped)')

        # Smoothed curve
        if len(y) > SMOOTH_WINDOW:
            y_smooth = moving_average(y, SMOOTH_WINDOW)
            x_smooth = x[:len(y_smooth)]
            ax.plot(x_smooth, y_smooth, color='tab:blue', linewidth=2, label=f'Smoothed ({SMOOTH_WINDOW})')

        ax.set_title(col)
        ax.set_xlabel('update')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()

    if out_png is None:
        out_png = os.path.join(os.path.dirname(csv_path), 'training_log.png')
    plt.savefig(out_png, dpi=200)
    print(f"✅ Saved plot to: {out_png}")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs from CSV")
    parser.add_argument('--csv', default='checkpoints/training_log.csv', help='Path to CSV training log')
    parser.add_argument('--out', default=None, help='Output PNG path (optional)')
    args = parser.parse_args()

    plot_log(args.csv, args.out)
