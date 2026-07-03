"""OpenVLA Pretrained Model Verification on the Berkeley UR5 Dataset.

Evaluates the pretrained ``openvla/openvla-7b`` checkpoint against the
ground-truth actions recorded in the Berkeley Autolab UR5 demonstration
dataset.  This is a **verification-only** script — it does not modify
any existing OpenVLA files or perform any training.

The heavy lifting is done by ``trajectory_evaluator.py`` (metrics, CSV,
per-frame comparisons) and ``visualize_dataset.py`` (plots).  This file
is the CLI entry-point that ties everything together.

Usage:
    # Evaluate on RLDS dataset (first trajectory, all frames)
    python verify_openvla.py

    # Quick test — first 50 frames
    python verify_openvla.py --max-timesteps 50

    # Evaluate a local .npy trajectory
    python verify_openvla.py --source /path/to/trajectory.npy

    # Directory of .npy files
    python verify_openvla.py --source-dir /path/to/trajectories/ --max-trajectories 5

    # Visual mode — step through frames with an OpenCV overlay
    python verify_openvla.py --visual --max-timesteps 30

    # Custom tolerance and print frequency
    python verify_openvla.py --tolerance 0.02 --print-every 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Add the existing project to the path so we can reuse OpenVLAPolicy ────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "openvla_realtime"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import OpenVLAConfig                              # noqa: E402
from openvla_policy import OpenVLAPolicy, OpenVLAError        # noqa: E402

from dataset_loader import (                                   # noqa: E402
    load_trajectory,
    load_all_trajectories,
    inspect_trajectory,
)
from trajectory_evaluator import (                             # noqa: E402
    evaluate_trajectory,
    run_visual_evaluation,
    print_trajectory_summary,
    save_results_csv,
    get_arrays,
    TrajectoryResult,
)
from visualize_dataset import generate_all_plots               # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify pretrained OpenVLA on the Berkeley UR5 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python verify_openvla.py --max-timesteps 50
  python verify_openvla.py --source trajectory.npy
  python verify_openvla.py --visual --max-timesteps 30
  python verify_openvla.py --tolerance 0.02 --print-every 5
""",
    )

    # Data source
    src = parser.add_argument_group("data source")
    src.add_argument("--source", default=None,
                     help="Path to a .npy trajectory file.")
    src.add_argument("--source-dir", default=None,
                     help="Directory of .npy trajectory files.")
    src.add_argument("--trajectory-index", type=int, default=0,
                     help="Episode index in RLDS dataset (default: 0).")
    src.add_argument("--split", default="train",
                     help="RLDS split: train or test (default: train).")
    src.add_argument("--max-trajectories", type=int, default=1,
                     help="Number of trajectories to evaluate (default: 1).")
    src.add_argument("--max-timesteps", type=int, default=None,
                     help="Cap frames per trajectory (default: all).")

    # Model
    mdl = parser.add_argument_group("model")
    mdl.add_argument("--model-id", default="openvla/openvla-7b",
                     help="HuggingFace model ID.")
    mdl.add_argument("--unnorm-key", default="berkeley_autolab_ur5",
                     help="Action unnormalization key.")
    mdl.add_argument("--allow-cpu", action="store_true",
                     help="Allow CPU inference (very slow).")

    # Evaluation
    evl = parser.add_argument_group("evaluation")
    evl.add_argument("--tolerance", type=float, default=0.01,
                     help="L2 error threshold for pass/fail (default: 0.01).")
    evl.add_argument("--print-every", type=int, default=10,
                     help="Print detailed comparison every N frames (default: 10).")
    evl.add_argument("--quiet", action="store_true",
                     help="Suppress per-frame output.")
    evl.add_argument("--visual", action="store_true",
                     help="Enable OpenCV visual overlay for each frame.")
    evl.add_argument("--instruction", default=None,
                     help="Override the dataset language instruction with a custom string.")

    # Output
    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", default=None,
                     help="Directory for results (default: dataset_verification/results/).")

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    print()
    print("=" * 70)
    print("  OpenVLA Pretrained Model Verification")
    print("  Berkeley Autolab UR5 Demonstration Dataset")
    print("=" * 70)
    print(f"  Model:          {args.model_id}")
    print(f"  Unnorm key:     {args.unnorm_key}")
    print(f"  Source:         {args.source or args.source_dir or 'RLDS (tensorflow_datasets)'}")
    print(f"  Tolerance:      {args.tolerance}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Visual mode:    {args.visual}")
    if args.instruction:
        print(f"  Instruction:    '{args.instruction}' (command-line override)")
    else:
        print("  Instruction:    (from dataset)")
    print()

    # ── 1. Load the model ─────────────────────────────────────────────────
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
    print("[DEBUG] Model loading complete")

    # ── 2. Load trajectories and evaluate ─────────────────────────────────
    all_results: list[TrajectoryResult] = []
    trajectory_count = 0
    first_image_res = (0, 0)
    first_instruction = ""

    print("[DEBUG] Dataset loading...")
    # Determine the trajectory source
    if args.source is not None:
        trajectory_iter = [load_trajectory(args.source)]
    elif args.source_dir is not None:
        trajectory_iter = load_all_trajectories(
            args.source_dir, max_trajectories=args.max_trajectories,
        )
    else:
        trajectory_iter = load_all_trajectories(
            split=args.split, max_trajectories=args.max_trajectories,
        )
    print("[DEBUG] Dataset loading complete")

    for timesteps in trajectory_iter:
        print("[DEBUG] Beginning trajectory evaluation")
        info = inspect_trajectory(timesteps)
        if trajectory_count == 0:
            first_image_res = info.image_shape
            first_instruction = info.instruction

        print(f"\n>>> Trajectory {trajectory_count + 1}")

        # Choose evaluation mode
        if args.visual:
            result = run_visual_evaluation(
                policy, timesteps,
                max_timesteps=args.max_timesteps,
                tolerance=args.tolerance,
                instruction_override=args.instruction,
            )
        else:
            result = evaluate_trajectory(
                policy, timesteps,
                max_timesteps=args.max_timesteps,
                tolerance=args.tolerance,
                print_every=args.print_every,
                verbose=not args.quiet,
                instruction_override=args.instruction,
            )

        all_results.append(result)
        trajectory_count += 1

        # Per-trajectory summary
        print_trajectory_summary(result, args.tolerance)

        # Per-trajectory CSV
        print("[DEBUG] CSV writing...")
        csv_name = f"evaluation_trajectory_{trajectory_count}.csv"
        save_results_csv(result, output_dir / csv_name)
        print("[DEBUG] CSV writing complete")

    if not all_results:
        print("ERROR: No data was loaded.", file=sys.stderr)
        return 1

    # ── 3. Aggregate across all trajectories ──────────────────────────────
    all_gt_list = []
    all_pred_list = []
    for r in all_results:
        gt, pred = get_arrays(r)
        all_gt_list.append(gt)
        all_pred_list.append(pred)

    gt_combined = np.concatenate(all_gt_list, axis=0)
    pred_combined = np.concatenate(all_pred_list, axis=0)
    total_frames = len(gt_combined)
    total_within = sum(
        1 for r in all_results
        for f in r.frame_results
        if f.within_tolerance
    )

    if trajectory_count > 1:
        # Print combined summary for multi-trajectory runs
        print()
        print("=" * 70)
        print(f"  COMBINED RESULTS ({trajectory_count} trajectories, {total_frames} frames)")
        print("=" * 70)
        print(f"  Overall MAE:                {float(np.mean(np.abs(gt_combined - pred_combined))):.6f}")
        print(f"  Overall MSE:                {float(np.mean((gt_combined - pred_combined)**2)):.8f}")
        per_step_l2 = np.linalg.norm(gt_combined - pred_combined, axis=1)
        print(f"  Max L2 Error:               {float(np.max(per_step_l2)):.6f}")
        print(f"  Min L2 Error:               {float(np.min(per_step_l2)):.6f}")
        print(f"  Frames within tolerance:    {100.0 * total_within / total_frames:.1f}%")
        print("=" * 70)
        print()

    # ── 4. Dataset summary header ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)
    print(f"  Number of trajectories:   {trajectory_count}")
    print(f"  Number of frames:         {total_frames}")
    print(f"  Image resolution:         {first_image_res[0]}x{first_image_res[1]}")
    print(f"  Robot:                    UR5 (Berkeley Autolab)")
    print(f"  Action dimensions:        7")
    print(f"  Instruction:              \"{first_instruction}\"")
    print("=" * 70)
    print()

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("[DEBUG] Visualization...")
    try:
        generate_all_plots(gt_combined, pred_combined, output_dir=output_dir)
    except ImportError:
        print("WARNING: matplotlib not installed — skipping plot generation.")
    except Exception as exc:
        print(f"WARNING: Plot generation failed — {exc}")
    print("[DEBUG] Visualization complete")

    # ── 6. Save raw arrays ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gt_actions.npy", gt_combined)
    np.save(output_dir / "pred_actions.npy", pred_combined)
    print(f"Raw arrays saved: {output_dir}/gt_actions.npy, pred_actions.npy")

    print("\n✓ Verification complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
