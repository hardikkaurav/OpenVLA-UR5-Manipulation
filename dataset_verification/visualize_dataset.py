"""Visualization utilities for OpenVLA verification results.

Generates publication-quality plots comparing predicted vs ground-truth actions
from the Berkeley UR5 dataset evaluation. All figures are saved to
``dataset_verification/results/``.

No existing OpenVLA files are modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless servers
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


ACTION_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def _ensure_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required for plots: pip install matplotlib")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── 1. Ground truth vs predicted, per dimension ──────────────────────────────

def plot_gt_vs_predicted(
    gt: np.ndarray,
    pred: np.ndarray,
    output_dir: str | Path = "results",
) -> None:
    """Line plot: ground truth vs predicted for each of the 7 action dimensions.

    Args:
        gt: (T, 7) ground-truth actions.
        pred: (T, 7) predicted actions.
        output_dir: Directory to save the figure.
    """
    _ensure_matplotlib()
    out = _ensure_dir(Path(output_dir))
    T = len(gt)
    timesteps = np.arange(T)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), tight_layout=True)
    axes = axes.flatten()

    for dim in range(7):
        ax = axes[dim]
        ax.plot(timesteps, gt[:, dim], label="Ground Truth", color="#2196F3", linewidth=1.5)
        ax.plot(timesteps, pred[:, dim], label="Predicted", color="#F44336", linewidth=1.5, alpha=0.8)
        ax.set_title(ACTION_LABELS[dim], fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[7].set_visible(False)

    fig.suptitle("Ground Truth vs Predicted Actions", fontsize=16, fontweight="bold", y=1.01)
    path = out / "gt_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 2. Error over time ───────────────────────────────────────────────────────

def plot_error_over_time(
    gt: np.ndarray,
    pred: np.ndarray,
    output_dir: str | Path = "results",
) -> None:
    """Per-timestep absolute error (total and per-dimension).

    Args:
        gt: (T, 7) ground-truth actions.
        pred: (T, 7) predicted actions.
        output_dir: Directory to save the figure.
    """
    _ensure_matplotlib()
    out = _ensure_dir(Path(output_dir))
    T = len(gt)
    timesteps = np.arange(T)
    errors = np.abs(gt - pred)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), tight_layout=True)

    # Total error (L2 norm per timestep)
    total_error = np.linalg.norm(gt - pred, axis=1)
    axes[0].plot(timesteps, total_error, color="#9C27B0", linewidth=1.5)
    axes[0].fill_between(timesteps, 0, total_error, alpha=0.15, color="#9C27B0")
    axes[0].set_title("Total Action Error (L2 Norm) Over Time", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("L2 Error")
    axes[0].grid(True, alpha=0.3)

    # Per-dimension
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]
    for dim in range(7):
        axes[1].plot(timesteps, errors[:, dim], label=ACTION_LABELS[dim],
                     color=colors[dim], linewidth=1.2, alpha=0.85)
    axes[1].set_title("Per-Dimension Absolute Error Over Time", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Absolute Error")
    axes[1].legend(fontsize=9, ncol=4)
    axes[1].grid(True, alpha=0.3)

    path = out / "error_over_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. Histogram of prediction errors ────────────────────────────────────────

def plot_error_histogram(
    gt: np.ndarray,
    pred: np.ndarray,
    output_dir: str | Path = "results",
) -> None:
    """Histogram of per-dimension prediction errors across all timesteps.

    Args:
        gt: (T, 7) ground-truth actions.
        pred: (T, 7) predicted actions.
        output_dir: Directory to save the figure.
    """
    _ensure_matplotlib()
    out = _ensure_dir(Path(output_dir))
    errors = (pred - gt)  # signed errors

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), tight_layout=True)
    axes = axes.flatten()

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]
    for dim in range(7):
        ax = axes[dim]
        ax.hist(errors[:, dim], bins=50, color=colors[dim], alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{ACTION_LABELS[dim]} Error Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Prediction Error (pred − GT)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    axes[7].set_visible(False)

    fig.suptitle("Prediction Error Histograms", fontsize=16, fontweight="bold", y=1.01)
    path = out / "error_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 4. Scatter plot: predicted vs ground truth ────────────────────────────────

def plot_scatter(
    gt: np.ndarray,
    pred: np.ndarray,
    output_dir: str | Path = "results",
) -> None:
    """Scatter plot of predicted vs ground-truth values (per dimension).

    A perfect model would lie on the diagonal y=x line.

    Args:
        gt: (T, 7) ground-truth actions.
        pred: (T, 7) predicted actions.
        output_dir: Directory to save the figure.
    """
    _ensure_matplotlib()
    out = _ensure_dir(Path(output_dir))

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), tight_layout=True)
    axes = axes.flatten()

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]
    for dim in range(7):
        ax = axes[dim]
        ax.scatter(gt[:, dim], pred[:, dim], alpha=0.4, s=12, color=colors[dim], edgecolors="none")
        lims = [
            min(gt[:, dim].min(), pred[:, dim].min()),
            max(gt[:, dim].max(), pred[:, dim].max()),
        ]
        margin = (lims[1] - lims[0]) * 0.05 + 1e-6
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.6, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(ACTION_LABELS[dim], fontsize=12, fontweight="bold")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    axes[7].set_visible(False)

    fig.suptitle("Predicted vs Ground Truth (Scatter)", fontsize=16, fontweight="bold", y=1.01)
    path = out / "scatter_pred_vs_gt.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Generate all plots ────────────────────────────────────────────────────────

def generate_all_plots(
    gt: np.ndarray,
    pred: np.ndarray,
    output_dir: str | Path = "results",
) -> None:
    """Generate all four visualizations and save to *output_dir*.

    Args:
        gt: (T, 7) ground-truth actions.
        pred: (T, 7) predicted actions.
        output_dir: Directory for saved figures.
    """
    print(f"\nGenerating visualizations → {output_dir}/")
    plot_gt_vs_predicted(gt, pred, output_dir)
    plot_error_over_time(gt, pred, output_dir)
    plot_error_histogram(gt, pred, output_dir)
    plot_scatter(gt, pred, output_dir)
    print("All visualizations saved.\n")
