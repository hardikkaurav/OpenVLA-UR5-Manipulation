"""Trajectory Evaluator for OpenVLA Verification.

Runs OpenVLA inference on every frame of a loaded trajectory, compares
predictions with ground-truth actions, computes metrics, exports CSV
results, and optionally displays a per-frame visual overlay.

This module is the evaluation engine.  It does **not** load data or load
models — those responsibilities belong to ``dataset_loader.py`` and the
existing ``OpenVLAPolicy`` class respectively.

No existing OpenVLA files are modified.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

# ── Add the existing project to the path so we can reuse OpenVLAPolicy ────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "openvla_realtime"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openvla_policy import OpenVLAPolicy, OpenVLAError       # noqa: E402
from dataset_loader import Timestep                           # noqa: E402


ACTION_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data containers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    """Evaluation result for a single frame."""

    index: int
    instruction: str
    ground_truth: np.ndarray       # (7,)
    predicted: np.ndarray          # (7,)
    absolute_error: np.ndarray     # (7,)
    squared_error: np.ndarray      # (7,)
    l2_error: float
    inference_time_s: float
    within_tolerance: bool


@dataclass
class TrajectoryResult:
    """Aggregate evaluation result for an entire trajectory."""

    frame_results: list[FrameResult] = field(default_factory=list)

    # Populated after evaluation by compute_summary()
    num_frames: int = 0
    overall_mae: float = 0.0
    overall_mse: float = 0.0
    overall_rmse: float = 0.0
    mean_trajectory_error: float = 0.0
    cosine_similarity: float = 0.0
    max_error: float = 0.0
    min_error: float = 0.0
    pct_within_tolerance: float = 0.0
    per_dim_mae: np.ndarray = field(default_factory=lambda: np.zeros(7))
    per_dim_mse: np.ndarray = field(default_factory=lambda: np.zeros(7))
    avg_inference_time_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine_similarity_vector(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ═══════════════════════════════════════════════════════════════════════════════
#  Core evaluator
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_trajectory(
    policy: OpenVLAPolicy,
    timesteps: list[Timestep],
    *,
    max_timesteps: int | None = None,
    tolerance: float = 0.01,
    print_every: int = 10,
    verbose: bool = True,
    instruction_override: str | None = None,
) -> TrajectoryResult:
    """Run OpenVLA on every frame and compare with ground truth.

    Args:
        policy:         A loaded ``OpenVLAPolicy`` instance.
        timesteps:      List of ``Timestep`` objects from ``dataset_loader``.
        max_timesteps:  Cap the number of frames to evaluate (``None`` = all).
        tolerance:      L2 error threshold.  Frames with L2 error below this
                        value are counted as "within tolerance".
        print_every:    Print a detailed comparison every N frames.
        verbose:        If ``True``, print per-frame comparisons.

    Returns:
        A fully populated ``TrajectoryResult``.
    """

    if max_timesteps is not None:
        timesteps = timesteps[:max_timesteps]

    T = len(timesteps)
    result = TrajectoryResult()

    if T == 0:
        return result

    print(f"\n{'─'*70}")
    print(f"  Evaluating {T} frames  |  tolerance = {tolerance}")
    if instruction_override is not None:
        print(f"  [INFO] Using command-line instruction override for all frames: '{instruction_override}'")
    else:
        dataset_inst = timesteps[0].instruction if (timesteps and timesteps[0].instruction) else "pick up the object"
        print(f"  [INFO] Using instruction from dataset: '{dataset_inst}'")
    print(f"{'─'*70}\n")

    for i, ts in enumerate(timesteps):
        print(f"[DEBUG] Beginning timestep {i}")
        gt_raw = ts.action.astype(np.float32)
        # Extract matching 7-DoF control vector (dx, dy, dz, droll, dpitch, dyaw, gripper)
        # and ignore index 7 (the binary terminate_episode flag).
        gt = gt_raw[:7]
        if gt.shape[0] < 7:
            new_gt = np.zeros(7, dtype=np.float32)
            new_gt[:gt.shape[0]] = gt
            gt = new_gt
            
        if instruction_override is not None:
            instruction = instruction_override
        else:
            instruction = ts.instruction if ts.instruction else "pick up the object"

        # ── Image valid check and Preprocessing ──────────────────────────────────
        print("[DEBUG] Image preprocessing...")
        if not isinstance(ts.image, Image.Image):
            print(f"[ERROR] Expected PIL.Image.Image, got {type(ts.image)}")
        elif getattr(ts.image, 'mode', '') != "RGB":
            print(f"[ERROR] Expected image mode 'RGB', got '{getattr(ts.image, 'mode', 'N/A')}'")
            
        print(f"[DEBUG] Before prediction (Frame {i}):")
        print(f"  image type:   {type(ts.image)}")
        print(f"  image mode:   {getattr(ts.image, 'mode', 'N/A')}")
        print(f"  image size:   {getattr(ts.image, 'size', 'N/A')}")
        print(f"  instruction:  '{instruction}'")
        print(f"  robot_state shape: {ts.robot_state.shape if hasattr(ts, 'robot_state') and ts.robot_state is not None else 'N/A'}")
        print(f"  action shape: {ts.action.shape}")

        # ── Inference ─────────────────────────────────────────────────────
        try:
            print("[DEBUG] Calling policy.predict(...)")
            pred_result = policy.predict(ts.image, instruction)
            print("[DEBUG] Returning from policy.predict(...)")
            
            pred_raw = pred_result.action.astype(np.float32)
            inf_time = pred_result.inference_time_s
            
            print("[DEBUG] After prediction:")
            print(f"  predicted action shape:  {pred_raw.shape}")
            print(f"  predicted action values: {pred_raw}")
            print(f"  inference time:          {inf_time:.4f} s")
            
            # The dataset action has 8 dimensions (7 control dims + 1 terminate flag),
            # while OpenVLA predicts only the 7 control dims. Compare matching 7 dimensions.
            if pred_raw.shape[0] != 7:
                print(f"[WARNING] Expected 7-DoF predicted action, got shape {pred_raw.shape}")
                
            pred = pred_raw[:7]
            if pred.shape[0] < 7:
                new_pred = np.zeros(7, dtype=np.float32)
                new_pred[:pred.shape[0]] = pred
                pred = new_pred
                
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise

        # ── Per-frame metrics ─────────────────────────────────────────────
        print("[DEBUG] Error computation...")
        abs_err = np.abs(gt - pred)
        sq_err = (gt - pred) ** 2
        l2 = float(np.linalg.norm(gt - pred))
        within = l2 < tolerance

        fr = FrameResult(
            index=i,
            instruction=instruction,
            ground_truth=gt,
            predicted=pred,
            absolute_error=abs_err,
            squared_error=sq_err,
            l2_error=l2,
            inference_time_s=inf_time,
            within_tolerance=within,
        )
        result.frame_results.append(fr)

        # ── Verbose per-frame print ───────────────────────────────────────
        if verbose and (i % print_every == 0 or i == T - 1):
            _print_frame_comparison(fr, tolerance)

        print(f"[DEBUG] End of timestep {i}")

    # ── Compute aggregate metrics ─────────────────────────────────────────
    _compute_summary(result, tolerance)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_summary(result: TrajectoryResult, tolerance: float) -> None:
    """Populate aggregate fields on a ``TrajectoryResult``."""

    frames = result.frame_results
    T = len(frames)
    result.num_frames = T

    if T == 0:
        return

    gt_all = np.stack([f.ground_truth for f in frames])     # (T, 7)
    pred_all = np.stack([f.predicted for f in frames])       # (T, 7)

    result.overall_mae = float(np.mean(np.abs(gt_all - pred_all)))
    result.overall_mse = float(np.mean((gt_all - pred_all) ** 2))
    result.overall_rmse = float(np.sqrt(result.overall_mse))
    l2_per_step = np.linalg.norm(gt_all - pred_all, axis=1)
    result.mean_trajectory_error = float(np.mean(l2_per_step))
    result.per_dim_mae = np.mean(np.abs(gt_all - pred_all), axis=0)
    result.per_dim_mse = np.mean((gt_all - pred_all) ** 2, axis=0)

    l2_errors = np.array([f.l2_error for f in frames])
    result.max_error = float(np.max(l2_errors))
    result.min_error = float(np.min(l2_errors))

    within_count = sum(1 for f in frames if f.within_tolerance)
    result.pct_within_tolerance = 100.0 * within_count / T

    # Cosine similarity
    sims = [_cosine_similarity_vector(f.ground_truth, f.predicted) for f in frames]
    result.cosine_similarity = float(np.mean(sims))

    inf_times = [f.inference_time_s for f in frames if f.inference_time_s > 0]
    result.avg_inference_time_s = float(np.mean(inf_times)) if inf_times else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Printing
# ═══════════════════════════════════════════════════════════════════════════════

def _print_frame_comparison(fr: FrameResult, tolerance: float) -> None:
    """Print a detailed comparison for one frame."""

    gt_str = "  ".join(f"{v:+.5f}" for v in fr.ground_truth)
    pr_str = "  ".join(f"{v:+.5f}" for v in fr.predicted)
    err_str = "  ".join(f"{v:.5f}" for v in fr.absolute_error)
    status = "✓ PASS" if fr.within_tolerance else "✗ FAIL"

    print(f"Frame {fr.index:>4d}  [{status}]  L2={fr.l2_error:.5f}  "
          f"inference={fr.inference_time_s*1000:.1f} ms")
    print(f"  Instruction:   {fr.instruction}")
    print(f"  Ground Truth:  [{gt_str}]")
    print(f"  Predicted:     [{pr_str}]")
    print(f"  Abs Error:     [{err_str}]")
    print(f"  Within tol ({tolerance}):  {fr.within_tolerance}")
    print()


def print_trajectory_summary(result: TrajectoryResult, tolerance: float) -> None:
    """Print the final evaluation summary to the console."""

    print()
    print("=" * 70)
    print("  TRAJECTORY EVALUATION SUMMARY")
    print("=" * 70)
    print()
    print(f"  Number of frames:              {result.num_frames}")
    print(f"  Tolerance:                     {tolerance}")
    print()
    print(f"  Mean Absolute Error (MAE):     {result.overall_mae:.6f}")
    print(f"  Mean Squared Error (MSE):      {result.overall_mse:.8f}")
    print(f"  Root Mean Squared Error (RMSE):{result.overall_rmse:.6f}")
    print(f"  Mean Trajectory Error (L2):    {result.mean_trajectory_error:.6f}")
    print(f"  Cosine Similarity:             {result.cosine_similarity:.6f}")
    print()
    print(f"  Maximum Error (L2):            {result.max_error:.6f}")
    print(f"  Minimum Error (L2):            {result.min_error:.6f}")
    print(f"  Frames within tolerance:       {result.pct_within_tolerance:.1f}%")
    print()
    print(f"  Average inference time:        {result.avg_inference_time_s*1000:.1f} ms "
          f"({1.0/max(result.avg_inference_time_s, 1e-9):.1f} FPS)")
    print()
    print("  Per-Dimension MAE:")
    for i, label in enumerate(ACTION_LABELS):
        print(f"    {label:>8s}:  {result.per_dim_mae[i]:.6f}")
    print()
    print("  Per-Dimension MSE:")
    for i, label in enumerate(ACTION_LABELS):
        print(f"    {label:>8s}:  {result.per_dim_mse[i]:.8f}")
    print()
    print("=" * 70)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV export
# ═══════════════════════════════════════════════════════════════════════════════

def save_results_csv(
    result: TrajectoryResult,
    output_path: str | Path,
) -> None:
    """Write per-frame results to a CSV file.

    Columns:
        frame_index, instruction,
        gt_dx … gt_gripper,
        pred_dx … pred_gripper,
        abs_err_dx … abs_err_gripper,
        sq_err_dx … sq_err_gripper,
        l2_error, within_tolerance
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gt_cols = [f"gt_{l}" for l in ACTION_LABELS]
    pred_cols = [f"pred_{l}" for l in ACTION_LABELS]
    abs_cols = [f"abs_err_{l}" for l in ACTION_LABELS]
    sq_cols = [f"sq_err_{l}" for l in ACTION_LABELS]
    header = (
        ["frame_index", "instruction"]
        + gt_cols + pred_cols + abs_cols + sq_cols
        + ["l2_error", "within_tolerance"]
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fr in result.frame_results:
            row = [fr.index, fr.instruction]
            row.extend(f"{v:.6f}" for v in fr.ground_truth)
            row.extend(f"{v:.6f}" for v in fr.predicted)
            row.extend(f"{v:.6f}" for v in fr.absolute_error)
            row.extend(f"{v:.8f}" for v in fr.squared_error)
            row.append(f"{fr.l2_error:.6f}")
            row.append(fr.within_tolerance)
            writer.writerow(row)

    print(f"  CSV saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenCV visual overlay (optional)
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_frame(
    timestep: Timestep,
    frame_result: FrameResult,
    tolerance: float = 0.01,
    window_name: str = "OpenVLA Evaluation",
    wait_ms: int = 0,
) -> int | None:
    """Display one frame with a ground-truth / predicted / error overlay.

    Args:
        timestep:       The ``Timestep`` from the dataset.
        frame_result:   The corresponding ``FrameResult``.
        tolerance:      Tolerance used for pass/fail colouring.
        window_name:    OpenCV window title.
        wait_ms:        ``cv2.waitKey`` argument. 0 = wait for key press.

    Returns:
        The key code pressed, or ``None``.
    """

    try:
        import cv2
    except ImportError:
        print("WARNING: opencv-python is required for visualization: pip install opencv-python")
        return None

    # ── Build the image panel ─────────────────────────────────────────────
    rgb = np.asarray(timestep.image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panel = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)

    # Semi-transparent overlay at the top
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (640, 280), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.65, panel, 0.35, 0, dst=panel)

    y = 22
    dy = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.48
    white = (255, 255, 255)
    green = (100, 255, 100)
    red = (100, 100, 255)
    cyan = (255, 220, 130)
    yellow = (100, 255, 255)

    status_color = green if frame_result.within_tolerance else red
    status_text = "PASS" if frame_result.within_tolerance else "FAIL"

    cv2.putText(panel, f"Frame {frame_result.index}  [{status_text}]  "
                f"L2 Error: {frame_result.l2_error:.5f}",
                (10, y), font, 0.55, status_color, 2)
    y += dy + 4

    instr = frame_result.instruction[:80]
    cv2.putText(panel, f"Instruction: {instr}", (10, y), font, fs, white, 1)
    y += dy + 2

    # Header
    cv2.putText(panel, f"{'':>12s}  {'dx':>8s}  {'dy':>8s}  {'dz':>8s}  "
                f"{'droll':>8s}  {'dpitch':>8s}  {'dyaw':>8s}  {'grip':>8s}",
                (10, y), font, 0.38, cyan, 1)
    y += dy

    # Ground truth
    gt_vals = "  ".join(f"{v:+.4f}" for v in frame_result.ground_truth)
    cv2.putText(panel, f"  GT:        {gt_vals}", (10, y), font, 0.42, green, 1)
    y += dy

    # Predicted
    pr_vals = "  ".join(f"{v:+.4f}" for v in frame_result.predicted)
    cv2.putText(panel, f"  Pred:      {pr_vals}", (10, y), font, 0.42, yellow, 1)
    y += dy

    # Absolute error
    err_vals = "  ".join(f"{v:.4f}" for v in frame_result.absolute_error)
    cv2.putText(panel, f"  Abs Err:   {err_vals}", (10, y), font, 0.42, red, 1)
    y += dy

    # Tolerance
    cv2.putText(panel, f"  Tolerance: {tolerance}  |  "
                f"Inference: {frame_result.inference_time_s*1000:.1f} ms",
                (10, y), font, 0.42, white, 1)

    cv2.imshow(window_name, panel)
    key = cv2.waitKey(wait_ms) & 0xFF
    return key if key != 255 else None


def run_visual_evaluation(
    policy: OpenVLAPolicy,
    timesteps: list[Timestep],
    *,
    max_timesteps: int | None = None,
    tolerance: float = 0.01,
    instruction_override: str | None = None,
) -> TrajectoryResult:
    """Evaluate with a live OpenCV window showing each frame.

    Press ``q`` or ``Esc`` to stop early.  Press any other key to advance.
    """

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required for visual mode.", file=sys.stderr)
        raise

    if max_timesteps is not None:
        timesteps = timesteps[:max_timesteps]

    T = len(timesteps)
    result = TrajectoryResult()
    window = "OpenVLA Evaluation — press any key to advance, q/Esc to quit"

    print(f"\nVisual evaluation: {T} frames.  Press any key to step, q/Esc to quit.\n")
    if instruction_override is not None:
        print(f"  [INFO] Using command-line instruction override for all frames: '{instruction_override}'\n")
    else:
        dataset_inst = timesteps[0].instruction if (timesteps and timesteps[0].instruction) else "pick up the object"
        print(f"  [INFO] Using instruction from dataset: '{dataset_inst}'\n")

    for i, ts in enumerate(timesteps):
        print(f"[DEBUG] Beginning timestep {i}")
        gt_raw = ts.action.astype(np.float32)
        # Extract matching 7-DoF control vector (dx, dy, dz, droll, dpitch, dyaw, gripper)
        # and ignore index 7 (the binary terminate_episode flag).
        gt = gt_raw[:7]
        if gt.shape[0] < 7:
            new_gt = np.zeros(7, dtype=np.float32)
            new_gt[:gt.shape[0]] = gt
            gt = new_gt
            
        if instruction_override is not None:
            instruction = instruction_override
        else:
            instruction = ts.instruction if ts.instruction else "pick up the object"

        print("[DEBUG] Image preprocessing...")
        if not isinstance(ts.image, Image.Image):
            print(f"[ERROR] Expected PIL.Image.Image, got {type(ts.image)}")
        elif getattr(ts.image, 'mode', '') != "RGB":
            print(f"[ERROR] Expected image mode 'RGB', got '{getattr(ts.image, 'mode', 'N/A')}'")
            
        print(f"[DEBUG] Before prediction (Frame {i}):")
        print(f"  image type:   {type(ts.image)}")
        print(f"  image mode:   {getattr(ts.image, 'mode', 'N/A')}")
        print(f"  image size:   {getattr(ts.image, 'size', 'N/A')}")
        print(f"  instruction:  '{instruction}'")
        print(f"  robot_state shape: {ts.robot_state.shape if hasattr(ts, 'robot_state') and ts.robot_state is not None else 'N/A'}")
        print(f"  action shape: {ts.action.shape}")

        try:
            print("[DEBUG] Calling policy.predict(...)")
            pred_res = policy.predict(ts.image, instruction)
            print("[DEBUG] Returning from policy.predict(...)")
            
            pred_raw = pred_res.action.astype(np.float32)
            inf_time = pred_res.inference_time_s
            
            print("[DEBUG] After prediction:")
            print(f"  predicted action shape:  {pred_raw.shape}")
            print(f"  predicted action values: {pred_raw}")
            print(f"  inference time:          {inf_time:.4f} s")
            
            # The dataset action has 8 dimensions (7 control dims + 1 terminate flag),
            # while OpenVLA predicts only the 7 control dims. Compare matching 7 dimensions.
            if pred_raw.shape[0] != 7:
                print(f"[WARNING] Expected 7-DoF predicted action, got shape {pred_raw.shape}")
                
            pred = pred_raw[:7]
            if pred.shape[0] < 7:
                new_pred = np.zeros(7, dtype=np.float32)
                new_pred[:pred.shape[0]] = pred
                pred = new_pred
                
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise

        print("[DEBUG] Error computation...")
        abs_err = np.abs(gt - pred)
        sq_err = (gt - pred) ** 2
        l2 = float(np.linalg.norm(gt - pred))

        fr = FrameResult(
            index=i, instruction=instruction,
            ground_truth=gt, predicted=pred,
            absolute_error=abs_err, squared_error=sq_err,
            l2_error=l2, inference_time_s=inf_time,
            within_tolerance=l2 < tolerance,
        )
        result.frame_results.append(fr)

        key = visualize_frame(ts, fr, tolerance=tolerance, window_name=window, wait_ms=0)
        if key in (ord("q"), 27):
            print("Visual evaluation stopped by user.")
            break
            
        print(f"[DEBUG] End of timestep {i}")

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    _compute_summary(result, tolerance)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: extract raw arrays from a result
# ═══════════════════════════════════════════════════════════════════════════════

def get_arrays(result: TrajectoryResult) -> tuple[np.ndarray, np.ndarray]:
    """Return (gt, pred) arrays of shape (T, 7) from a ``TrajectoryResult``."""
    gt = np.stack([f.ground_truth for f in result.frame_results])
    pred = np.stack([f.predicted for f in result.frame_results])
    return gt, pred
