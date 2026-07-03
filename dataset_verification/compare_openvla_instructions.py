"""OpenVLA Instruction Sensitivity and Comparison Tool.

Compares OpenVLA action predictions across multiple language instructions
on the exact same demonstration trajectory:
1. Original dataset instruction.
2. Modified instruction (original + appended irrelevant text).
3. Custom command-line instruction (e.g., "Do not move.").

Generates comprehensive comparison plots and metrics in
`instruction_comparison_results/`. Does NOT modify any project or dataset files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ── Ensure project modules are importable ─────────────────────────────────────
_VERIFICATION_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _VERIFICATION_DIR.parent / "openvla_realtime"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_DIR))

from openvla_policy import OpenVLAPolicy, OpenVLAConfig, OpenVLAError  # noqa: E402
from dataset_loader import load_trajectory, load_all_trajectories, Timestep  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


ACTION_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def _ensure_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── CLI Argument Parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OpenVLA predictions across multiple language instructions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python compare_openvla_instructions.py --source "../bottle (1).npy"
  python compare_openvla_instructions.py --source "../bottle (1).npy" --custom-instruction "Do not move."
  python compare_openvla_instructions.py --source "../bottle (1).npy" --max-timesteps 30
""",
    )

    # Data source
    src = parser.add_argument_group("data source")
    src.add_argument("--source", default=None,
                     help="Path to a .npy trajectory file.")
    src.add_argument("--source-dir", default=None,
                     help="Directory containing .npy trajectory files.")
    src.add_argument("--split", default="train",
                     help="RLDS split: train or test (default: train).")
    src.add_argument("--max-trajectories", type=int, default=1,
                     help="Number of trajectories to process (default: 1).")
    src.add_argument("--max-timesteps", type=int, default=None,
                     help="Cap the number of frames to evaluate per trajectory.")

    # Model configuration
    mdl = parser.add_argument_group("model")
    mdl.add_argument("--model-id", default="openvla/openvla-7b",
                     help="HuggingFace model ID (default: openvla/openvla-7b).")
    mdl.add_argument("--unnorm-key", default="berkeley_autolab_ur5",
                     help="Action unnormalization key (default: berkeley_autolab_ur5).")
    mdl.add_argument("--allow-cpu", action="store_true",
                     help="Allow CPU inference (very slow).")

    # Instruction comparison strings
    ins = parser.add_argument_group("instructions")
    ins.add_argument("--custom-instruction", "--instruction", dest="custom_instruction", default="Do not move.",
                     help="Custom command-line instruction to test (default: 'Do not move.').")
    ins.add_argument("--modified-suffix", default=" Ignore the color of the table. The room is well lit. Today is Monday. This robot is made of metal.",
                     help="Irrelevant text appended to original instruction.")

    # Output
    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", default="instruction_comparison_results",
                     help="Directory to save comparison plots and arrays (default: instruction_comparison_results).")
    out.add_argument("--print-every", type=int, default=1,
                     help="Print progress every N frames (default: 1).")

    return parser.parse_args()


# ── Plotting Generators ───────────────────────────────────────────────────────

def plot_action_comparison_per_dimension(
    gt: np.ndarray,
    pred_orig: np.ndarray,
    pred_mod: np.ndarray,
    pred_cust: np.ndarray,
    orig_str: str,
    mod_str: str,
    cust_str: str,
    output_dir: Path,
) -> None:
    """Plot Ground Truth and 3 Instruction Predictions for every action dimension."""
    _ensure_matplotlib()
    out = _ensure_dir(output_dir)
    T = len(gt)
    timesteps = np.arange(T)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), tight_layout=True)
    axes = axes.flatten()

    orig_label = f"Original: '{orig_str[:30]}...'" if len(orig_str) > 30 else f"Original: '{orig_str}'"
    mod_label = "Modified (+ irrelevant text)"
    cust_label = f"Custom: '{cust_str[:30]}...'" if len(cust_str) > 30 else f"Custom: '{cust_str}'"

    for dim in range(7):
        ax = axes[dim]
        ax.plot(timesteps, gt[:, dim], label="Ground Truth", color="#2196F3", linewidth=2.0)
        ax.plot(timesteps, pred_orig[:, dim], label=orig_label, color="#4CAF50", linewidth=1.5, alpha=0.85)
        ax.plot(timesteps, pred_mod[:, dim], label=mod_label, color="#FF9800", linewidth=1.5, alpha=0.85, linestyle="--")
        ax.plot(timesteps, pred_cust[:, dim], label=cust_label, color="#E91E63", linewidth=1.5, alpha=0.85, linestyle=":")
        ax.set_title(ACTION_LABELS[dim], fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Action Value")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    axes[7].set_visible(False)

    fig.suptitle("Per-Dimension Action Comparison Across Language Instructions", fontsize=16, fontweight="bold", y=1.01)
    path = out / "action_comparison_per_dimension.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_absolute_error_over_time(
    gt: np.ndarray,
    pred_orig: np.ndarray,
    pred_mod: np.ndarray,
    pred_cust: np.ndarray,
    orig_str: str,
    mod_str: str,
    cust_str: str,
    output_dir: Path,
) -> None:
    """Plot per-timestep L2 error and Mean Absolute Error across instructions."""
    _ensure_matplotlib()
    out = _ensure_dir(output_dir)
    T = len(gt)
    timesteps = np.arange(T)

    err_orig_l2 = np.linalg.norm(pred_orig - gt, axis=-1)
    err_mod_l2 = np.linalg.norm(pred_mod - gt, axis=-1)
    err_cust_l2 = np.linalg.norm(pred_cust - gt, axis=-1)

    err_orig_mae = np.mean(np.abs(pred_orig - gt), axis=-1)
    err_mod_mae = np.mean(np.abs(pred_mod - gt), axis=-1)
    err_cust_mae = np.mean(np.abs(pred_cust - gt), axis=-1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), tight_layout=True)

    # Top: L2 Euclidean Trajectory Error
    ax1.plot(timesteps, err_orig_l2, label="Original Instruction Error (L2)", color="#4CAF50", linewidth=1.8)
    ax1.plot(timesteps, err_mod_l2, label="Modified Instruction Error (L2)", color="#FF9800", linewidth=1.8, linestyle="--")
    ax1.plot(timesteps, err_cust_l2, label="Custom Instruction Error (L2)", color="#E91E63", linewidth=1.8, linestyle=":")
    ax1.set_title("Total Trajectory Error (L2 Norm) Over Time", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("L2 Error")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom: Mean Absolute Error across all 7 DoF
    ax2.plot(timesteps, err_orig_mae, label="Original Instruction MAE", color="#4CAF50", linewidth=1.8)
    ax2.plot(timesteps, err_mod_mae, label="Modified Instruction MAE", color="#FF9800", linewidth=1.8, linestyle="--")
    ax2.plot(timesteps, err_cust_mae, label="Custom Instruction MAE", color="#E91E63", linewidth=1.8, linestyle=":")
    ax2.set_title("Mean Absolute Error (MAE across 7 DoF) Over Time", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("MAE")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Absolute Prediction Error Over Time by Language Instruction", fontsize=16, fontweight="bold", y=1.02)
    path = out / "absolute_error_over_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_error_histogram(
    gt: np.ndarray,
    pred_orig: np.ndarray,
    pred_mod: np.ndarray,
    pred_cust: np.ndarray,
    orig_str: str,
    mod_str: str,
    cust_str: str,
    output_dir: Path,
) -> None:
    """Plot overlapping histograms of frame L2 prediction errors for each instruction."""
    _ensure_matplotlib()
    out = _ensure_dir(output_dir)

    err_orig = np.linalg.norm(pred_orig - gt, axis=-1)
    err_mod = np.linalg.norm(pred_mod - gt, axis=-1)
    err_cust = np.linalg.norm(pred_cust - gt, axis=-1)

    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)

    bins = np.histogram_bin_edges(np.concatenate([err_orig, err_mod, err_cust]), bins=25)

    ax.hist(err_orig, bins=bins, alpha=0.55, label=f"Original (Mean L2: {np.mean(err_orig):.4f})", color="#4CAF50", edgecolor="black")
    ax.hist(err_mod, bins=bins, alpha=0.55, label=f"Modified (Mean L2: {np.mean(err_mod):.4f})", color="#FF9800", edgecolor="black")
    ax.hist(err_cust, bins=bins, alpha=0.55, label=f"Custom (Mean L2: {np.mean(err_cust):.4f})", color="#E91E63", edgecolor="black")

    ax.axvline(np.mean(err_orig), color="#388E3C", linestyle="--", linewidth=2)
    ax.axvline(np.mean(err_mod), color="#F57C00", linestyle="--", linewidth=2)
    ax.axvline(np.mean(err_cust), color="#C2185B", linestyle="--", linewidth=2)

    ax.set_title("Histogram of Frame L2 Prediction Errors by Instruction", fontsize=15, fontweight="bold")
    ax.set_xlabel("L2 Euclidean Error")
    ax.set_ylabel("Frame Count")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = out / "error_histogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def plot_instruction_difference(
    pred_orig: np.ndarray,
    pred_other: np.ndarray,
    other_name: str,
    other_color: str,
    file_name: str,
    title_str: str,
    output_dir: Path,
) -> None:
    """Plot per-dimension and total Euclidean difference between original and another instruction."""
    _ensure_matplotlib()
    out = _ensure_dir(output_dir)
    T = len(pred_orig)
    timesteps = np.arange(T)

    diff = pred_other - pred_orig
    total_l2_diff = np.linalg.norm(diff, axis=-1)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), tight_layout=True)
    axes = axes.flatten()

    for dim in range(7):
        ax = axes[dim]
        ax.plot(timesteps, diff[:, dim], label=f"Δ {ACTION_LABELS[dim]} ({other_name} - Original)", color=other_color, linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(f"Difference in {ACTION_LABELS[dim]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Difference")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    # 8th subplot: Total L2 Euclidean Difference over time
    ax8 = axes[7]
    ax8.plot(timesteps, total_l2_diff, label=f"Total L2 Diff (Mean: {np.mean(total_l2_diff):.5f})", color="#673AB7", linewidth=2.0)
    ax8.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax8.set_title("Total L2 Euclidean Difference Over Time", fontsize=13, fontweight="bold")
    ax8.set_xlabel("Timestep")
    ax8.set_ylabel("L2 Diff")
    ax8.legend(fontsize=10, loc="best")
    ax8.grid(True, alpha=0.3)

    fig.suptitle(title_str, fontsize=16, fontweight="bold", y=1.01)
    path = out / file_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ── Core Comparative Evaluation ───────────────────────────────────────────────

def evaluate_instruction_comparison(
    policy: OpenVLAPolicy,
    timesteps: list[Timestep],
    custom_instruction: str,
    modified_suffix: str,
    output_dir: Path,
    print_every: int = 1,
) -> None:
    """Run OpenVLA across 3 language instructions on the same trajectory frames and plot results."""
    T = len(timesteps)
    if T == 0:
        print("No timesteps to evaluate.")
        return

    out = _ensure_dir(output_dir)

    # Define the 3 instruction strings
    orig_str = timesteps[0].instruction if (timesteps and timesteps[0].instruction) else "Put the ranch bottle into the pot."
    mod_str = f"{orig_str.rstrip('.')}.{modified_suffix}"
    cust_str = custom_instruction

    print("\n" + "=" * 90)
    print("  INSTRUCTION SENSITIVITY & RESPONSIVENESS EVALUATION")
    print("=" * 90)
    print(f"  1. Original:   '{orig_str}'")
    print(f"  2. Modified:   '{mod_str}'")
    print(f"  3. Custom:     '{cust_str}'")
    print("=" * 90 + "\n")

    # Allocate arrays
    gt_actions = np.zeros((T, 7), dtype=np.float32)
    pred_orig = np.zeros((T, 7), dtype=np.float32)
    pred_mod = np.zeros((T, 7), dtype=np.float32)
    pred_cust = np.zeros((T, 7), dtype=np.float32)

    print("=" * 95)
    print(f"  {'Frame':>5s}  |  {'Orig L2 Err':>11s}  {'Mod L2 Err':>11s}  {'Cust L2 Err':>11s}  |  {'Δ Mod (L2)':>11s}  {'Δ Cust (L2)':>11s}")
    print("=" * 95)

    for i, ts in enumerate(timesteps):
        gt = ts.action.astype(np.float32)[:7]
        if len(gt) < 7:
            pad = np.zeros(7, dtype=np.float32)
            pad[:len(gt)] = gt
            gt = pad
        gt_actions[i] = gt

        # Inference for each of the 3 instructions
        res_o = policy.predict(ts.image, orig_str)
        res_m = policy.predict(ts.image, mod_str)
        res_c = policy.predict(ts.image, cust_str)

        po = res_o.action.astype(np.float32)[:7]
        pm = res_m.action.astype(np.float32)[:7]
        pc = res_c.action.astype(np.float32)[:7]

        pred_orig[i] = po
        pred_mod[i] = pm
        pred_cust[i] = pc

        err_o = float(np.linalg.norm(po - gt))
        err_m = float(np.linalg.norm(pm - gt))
        err_c = float(np.linalg.norm(pc - gt))

        diff_m = float(np.linalg.norm(pm - po))
        diff_c = float(np.linalg.norm(pc - po))

        if i % print_every == 0 or i == T - 1:
            print(f"  {i:5d}  |  {err_o:11.5f}  {err_m:11.5f}  {err_c:11.5f}  |  {diff_m:11.5f}  {diff_c:11.5f}")

    print("=" * 95 + "\n")

    # Save raw numpy arrays for reproducible analysis
    print("[DEBUG] Saving raw prediction arrays...")
    np.save(str(out / "gt_actions.npy"), gt_actions)
    np.save(str(out / "pred_original.npy"), pred_orig)
    np.save(str(out / "pred_modified.npy"), pred_mod)
    np.save(str(out / "pred_custom.npy"), pred_cust)

    instr_meta = {
        "original": orig_str,
        "modified": mod_str,
        "custom": cust_str,
        "total_frames": T,
    }
    with open(out / "instructions_meta.json", "w", encoding="utf-8") as f:
        json.dump(instr_meta, f, indent=2)
    print(f"  Saved raw arrays and metadata to: {out}\n")

    # Generate all requested comparison plots
    print("[DEBUG] Generating publication-quality comparison figures...")
    plot_action_comparison_per_dimension(gt_actions, pred_orig, pred_mod, pred_cust, orig_str, mod_str, cust_str, out)
    plot_absolute_error_over_time(gt_actions, pred_orig, pred_mod, pred_cust, orig_str, mod_str, cust_str, out)
    plot_error_histogram(gt_actions, pred_orig, pred_mod, pred_cust, orig_str, mod_str, cust_str, out)
    plot_instruction_difference(
        pred_orig, pred_mod,
        other_name="Modified",
        other_color="#FF9800",
        file_name="diff_original_vs_modified.png",
        title_str="Sensitivity to Irrelevant Text (Modified - Original Prediction)",
        output_dir=out,
    )
    plot_instruction_difference(
        pred_orig, pred_cust,
        other_name="Custom",
        other_color="#E91E63",
        file_name="diff_original_vs_custom.png",
        title_str="Responsiveness to Custom Command (Custom - Original Prediction)",
        output_dir=out,
    )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("  INSTRUCTION COMPARISON SUMMARY STATISTICS")
    print("=" * 80)
    print(f"  Total frames evaluated:             {T}")
    print(f"  Original Instruction Mean L2 Error: {np.mean(np.linalg.norm(pred_orig - gt_actions, axis=-1)):.5f}")
    print(f"  Modified Instruction Mean L2 Error: {np.mean(np.linalg.norm(pred_mod - gt_actions, axis=-1)):.5f}")
    print(f"  Custom Instruction Mean L2 Error:   {np.mean(np.linalg.norm(pred_cust - gt_actions, axis=-1)):.5f}")
    print("-" * 80)
    print(f"  Mean L2 Difference (Modified vs Orig): {np.mean(np.linalg.norm(pred_mod - pred_orig, axis=-1)):.5f}")
    print(f"  Max L2 Difference  (Modified vs Orig): {np.max(np.linalg.norm(pred_mod - pred_orig, axis=-1)):.5f}")
    print("-" * 80)
    print(f"  Mean L2 Difference (Custom vs Orig):   {np.mean(np.linalg.norm(pred_cust - pred_orig, axis=-1)):.5f}")
    print(f"  Max L2 Difference  (Custom vs Orig):   {np.max(np.linalg.norm(pred_cust - pred_orig, axis=-1)):.5f}")
    print("=" * 80 + "\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir)
    print()
    print("=" * 70)
    print("  OpenVLA Instruction Sensitivity & Comparison Tool")
    print("  Berkeley Autolab UR5 Demonstration Dataset")
    print("=" * 70)
    print(f"  Model:          {args.model_id}")
    print(f"  Unnorm key:     {args.unnorm_key}")
    print(f"  Source:         {args.source or args.source_dir or 'RLDS (tensorflow_datasets)'}")
    print(f"  Custom Instr:   '{args.custom_instruction}'")
    print(f"  Output Dir:     {out_dir.resolve()}")
    print()

    # 1. Load the pretrained OpenVLA model
    openvla_config = OpenVLAConfig(
        model_id=args.model_id,
        unnorm_key=args.unnorm_key,
        require_gpu=not args.allow_cpu,
    )
    policy = OpenVLAPolicy(openvla_config)

    print("Loading OpenVLA model ...")
    t0 = time.perf_counter()
    try:
        policy.load()
    except OpenVLAError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Model loaded in {time.perf_counter() - t0:.1f} seconds.\n")

    # 2. Load trajectory timesteps
    print("[DEBUG] Loading dataset trajectory...")
    if args.source is not None:
        trajectory_iter = [load_trajectory(args.source)]
    elif args.source_dir is not None:
        trajectory_iter = load_all_trajectories(args.source_dir, max_trajectories=args.max_trajectories)
    else:
        trajectory_iter = load_all_trajectories(split=args.split, max_trajectories=args.max_trajectories)

    trajectory_count = 0
    for timesteps in trajectory_iter:
        trajectory_count += 1
        print(f"\n>>> Processing Trajectory {trajectory_count} ({len(timesteps)} frames)")
        
        traj_out = out_dir if args.max_trajectories <= 1 else (out_dir / f"trajectory_{trajectory_count}")

        evaluate_instruction_comparison(
            policy=policy,
            timesteps=timesteps,
            custom_instruction=args.custom_instruction,
            modified_suffix=args.modified_suffix,
            output_dir=traj_out,
            print_every=args.print_every,
        )

    if trajectory_count == 0:
        print("ERROR: No trajectories were loaded.", file=sys.stderr)
        return 1

    print("✓ Instruction sensitivity comparison complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
