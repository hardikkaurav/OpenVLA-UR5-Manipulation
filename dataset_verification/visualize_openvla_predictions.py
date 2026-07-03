"""OpenVLA Prediction Visualization Script.

Runs live inference with the pretrained OpenVLA model on a trajectory dataset
and visualizes the predicted actions on each image frame. Does NOT compare
against ground-truth actions or compute evaluation metrics.

Features:
- Custom command-line instruction override (--instruction).
- Live OpenCV sequential viewer with keyboard controls (Space, B, P, Q).
- Optional video saving (--save-video).
- Optional PNG frame export (--save-frames).
- Live terminal printing of predicted action vectors per frame.

No existing project files or data collection pipelines are modified.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

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


ACTION_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


# ── CLI Argument Parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OpenVLA action predictions on trajectory frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python visualize_openvla_predictions.py --source "../bottle (1).npy"
  python visualize_openvla_predictions.py --source "../bottle (1).npy" --instruction "Do not move."
  python visualize_openvla_predictions.py --source "../bottle (1).npy" --save-video "results/openvla_pred.mp4"
  python visualize_openvla_predictions.py --source "../bottle (1).npy" --save-frames "results/frames" --max-timesteps 30
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
                     help="Number of trajectories to visualize (default: 1).")
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

    # Override & Visualization options
    vis = parser.add_argument_group("visualization options")
    vis.add_argument("--instruction", default=None,
                     help="Override dataset language instruction with a custom string.")
    vis.add_argument("--save-video", nargs="?", const="results/openvla_predictions.mp4", default=None,
                     help="Save visualization as MP4 video (default path: results/openvla_predictions.mp4).")
    vis.add_argument("--save-frames", nargs="?", const="results/frames", default=None,
                     help="Save each annotated frame as a PNG image (default dir: results/frames).")
    vis.add_argument("--no-display", action="store_true",
                     help="Disable interactive OpenCV window (useful for headless servers).")

    return parser.parse_args()


# ── OpenCV Overlay Construction ───────────────────────────────────────────────

def create_annotated_panel(
    timestep: Timestep,
    frame_index: int,
    total_frames: int,
    instruction: str,
    predicted_action: np.ndarray,
    inference_time_s: float,
) -> np.ndarray:
    """Create an OpenCV BGR image panel with text overlays for prediction inspection."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for visual overlay: pip install opencv-python")

    # Convert PIL Image to OpenCV BGR array
    rgb = np.asarray(timestep.image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panel = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)

    # ── Top semi-transparent banner for metadata & action values ──
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (640, 140), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.70, panel, 0.30, 0, dst=panel)

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    green = (100, 255, 100)
    cyan = (255, 220, 130)
    yellow = (100, 255, 255)
    gray = (180, 180, 180)

    # Line 1: Frame count & inference latency
    cv2.putText(
        panel,
        f"Frame: {frame_index:03d} / {total_frames - 1:03d}   |   Inference: {inference_time_s * 1000:.1f} ms",
        (12, 25),
        font,
        0.52,
        green,
        1,
        cv2.LINE_AA,
    )

    # Line 2: Language instruction
    display_instr = f"Instruction: '{instruction}'"
    if len(display_instr) > 75:
        display_instr = display_instr[:72] + "..."
    cv2.putText(
        panel,
        display_instr,
        (12, 52),
        font,
        0.50,
        yellow,
        1,
        cv2.LINE_AA,
    )

    # Line 3: Table Header for 7-DoF action dimensions
    header_str = f"{'dx':>8s}  {'dy':>8s}  {'dz':>8s}  {'droll':>8s}  {'dpitch':>8s}  {'dyaw':>8s}  {'gripper':>8s}"
    cv2.putText(
        panel,
        header_str,
        (12, 85),
        font,
        0.42,
        cyan,
        1,
        cv2.LINE_AA,
    )

    # Line 4: Predicted action numerical values
    pred_7 = predicted_action[:7]
    if len(pred_7) < 7:
        padded = np.zeros(7, dtype=np.float32)
        padded[:len(pred_7)] = pred_7
        pred_7 = padded
    val_str = "  ".join(f"{v:+.4f}" for v in pred_7)
    cv2.putText(
        panel,
        val_str,
        (12, 115),
        font,
        0.44,
        white,
        1,
        cv2.LINE_AA,
    )

    # ── Bottom semi-transparent banner for keyboard control instructions ──
    footer_overlay = panel.copy()
    cv2.rectangle(footer_overlay, (0, 450), (640, 480), (0, 0, 0), thickness=-1)
    cv2.addWeighted(footer_overlay, 0.70, panel, 0.30, 0, dst=panel)

    cv2.putText(
        panel,
        "Controls: [Space] Next   [B] Prev   [P] Pause/Play   [Q] Quit",
        (15, 470),
        font,
        0.42,
        gray,
        1,
        cv2.LINE_AA,
    )

    return panel


# ── Interactive OpenCV Viewer ─────────────────────────────────────────────────

def run_interactive_viewer(annotated_frames: list[np.ndarray], window_title: str = "OpenVLA Prediction Viewer") -> None:
    """Display annotated frames in an OpenCV window with interactive keyboard navigation."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required for interactive viewing.", file=sys.stderr)
        return

    if not annotated_frames:
        print("No frames to display.")
        return

    print("\n" + "=" * 70)
    print("  INTERACTIVE PREDICTION VIEWER")
    print("=" * 70)
    print("  Controls:")
    print("    [Space] : Advance to next frame (and pause auto-play)")
    print("    [B]     : Go back to previous frame (and pause auto-play)")
    print("    [P]     : Toggle auto-play (pause / resume)")
    print("    [Q]/Esc : Quit interactive viewer")
    print("=" * 70 + "\n")

    idx = 0
    playing = False  # Start paused so the user can inspect frame 0

    while True:
        panel = annotated_frames[idx]
        try:
            cv2.imshow(window_title, panel)
        except cv2.error as err:
            print(f"[WARNING] Unable to open GUI window (headless environment?): {err}")
            break

        # If playing, wait 33ms (~30 FPS); if paused, wait indefinitely (0)
        wait_ms = 33 if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord('q'), ord('Q'), 27):  # Q or Escape
            print("Exiting interactive viewer.")
            break
        elif key in (ord('p'), ord('P')):  # P: Toggle play/pause
            playing = not playing
            state_str = "PLAYING" if playing else "PAUSED"
            print(f"  -> Auto-play {state_str} (Frame {idx})")
        elif key == ord(' '):  # Space: Next frame
            playing = False
            if idx < len(annotated_frames) - 1:
                idx += 1
            else:
                print("  -> Reached end of trajectory.")
        elif key in (ord('b'), ord('B')):  # B: Previous frame
            playing = False
            if idx > 0:
                idx -= 1
            else:
                print("  -> Already at frame 0.")
        elif playing:
            # Automatic advancement when auto-play is enabled
            if idx < len(annotated_frames) - 1:
                idx += 1
            else:
                playing = False
                print("  -> Reached end of trajectory; auto-play paused.")

    try:
        cv2.destroyWindow(window_title)
    except Exception:
        pass


# ── File Export (Video & Frames) ──────────────────────────────────────────────

def save_video_mp4(annotated_frames: list[np.ndarray], output_path: str | Path, fps: int = 15) -> None:
    """Save a sequence of annotated BGR frames as an MP4/AVI video file."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required to save video.", file=sys.stderr)
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not annotated_frames:
        return

    height, width, _ = annotated_frames[0].shape
    fourcc_str = "mp4v" if path.suffix.lower() == ".mp4" else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Failed to open VideoWriter for path: {path}")
        return

    for frame in annotated_frames:
        writer.write(frame)
    writer.release()
    print(f"  ✓ Video saved: {path} ({len(annotated_frames)} frames @ {fps} FPS)")


def save_png_frames(annotated_frames: list[np.ndarray], output_dir: str | Path) -> None:
    """Save each annotated BGR frame as an individual PNG image file."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required to save frames.", file=sys.stderr)
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(annotated_frames):
        file_path = out_path / f"frame_{i:04d}.png"
        cv2.imwrite(str(file_path), frame)

    print(f"  ✓ Saved {len(annotated_frames)} PNG frames to directory: {out_path}")


# ── Core Trajectory Processing ────────────────────────────────────────────────

def visualize_trajectory_predictions(
    policy: OpenVLAPolicy,
    timesteps: list[Timestep],
    instruction_override: str | None = None,
    max_timesteps: int | None = None,
    save_video_path: str | None = None,
    save_frames_dir: str | None = None,
    no_display: bool = False,
) -> None:
    """Run OpenVLA inference on each frame, print predictions, and visualize overlays."""
    if max_timesteps is not None:
        timesteps = timesteps[:max_timesteps]

    T = len(timesteps)
    if T == 0:
        print("No timesteps to evaluate.")
        return

    # Determine active language instruction
    if instruction_override is not None:
        active_instruction = instruction_override
        print(f"\n[INFO] Using command-line instruction override for all {T} frames:")
        print(f"       -> '{active_instruction}'\n")
    else:
        active_instruction = timesteps[0].instruction if (timesteps and timesteps[0].instruction) else "pick up the object"
        print(f"\n[INFO] Using dataset language instruction for all {T} frames:")
        print(f"       -> '{active_instruction}'\n")

    print("=" * 95)
    print(f"  {'Frame':>5s}  |  {'dx':>8s}  {'dy':>8s}  {'dz':>8s}  {'droll':>8s}  {'dpitch':>8s}  {'dyaw':>8s}  {'grip':>8s}  |  {'Latency':>8s}")
    print("=" * 95)

    annotated_frames: list[np.ndarray] = []

    for i, ts in enumerate(timesteps):
        # Use overridden instruction if provided, else timestep instruction
        instr = instruction_override if instruction_override is not None else (ts.instruction if ts.instruction else "pick up the object")

        # Run model inference
        t0 = time.perf_counter()
        pred_res = policy.predict(ts.image, instr)
        inf_time = time.perf_counter() - t0

        pred_action = pred_res.action.astype(np.float32)

        # Print predicted action vector cleanly in terminal
        pred_7 = pred_action[:7]
        if len(pred_7) < 7:
            padded = np.zeros(7, dtype=np.float32)
            padded[:len(pred_7)] = pred_7
            pred_7 = padded

        val_str = "  ".join(f"{v:+8.4f}" for v in pred_7)
        print(f"  {i:5d}  |  {val_str}  |  {inf_time * 1000:6.1f} ms")

        # Build annotated OpenCV BGR panel
        panel = create_annotated_panel(
            timestep=ts,
            frame_index=i,
            total_frames=T,
            instruction=instr,
            predicted_action=pred_action,
            inference_time_s=inf_time,
        )
        annotated_frames.append(panel)

    print("=" * 95 + "\n")

    # Save video if requested
    if save_video_path is not None:
        print("[DEBUG] Saving MP4 video...")
        save_video_mp4(annotated_frames, save_video_path)

    # Save individual PNG frames if requested
    if save_frames_dir is not None:
        print("[DEBUG] Saving PNG frames...")
        save_png_frames(annotated_frames, save_frames_dir)

    # Display interactive OpenCV viewer unless disabled
    if not no_display:
        run_interactive_viewer(annotated_frames)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    print()
    print("=" * 70)
    print("  OpenVLA Prediction Visualization Mode")
    print("  Berkeley Autolab UR5 Demonstration Dataset")
    print("=" * 70)
    print(f"  Model:          {args.model_id}")
    print(f"  Unnorm key:     {args.unnorm_key}")
    print(f"  Source:         {args.source or args.source_dir or 'RLDS (tensorflow_datasets)'}")
    if args.instruction:
        print(f"  Instruction:    '{args.instruction}' (command-line override)")
    else:
        print("  Instruction:    (from dataset)")
    if args.save_video:
        print(f"  Save Video:     {args.save_video}")
    if args.save_frames:
        print(f"  Save Frames:    {args.save_frames}")
    print(f"  Interactive UI: {not args.no_display}")
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
        
        # Adjust save paths if multiple trajectories are evaluated
        vid_path = args.save_video
        if vid_path and trajectory_count > 1:
            p = Path(vid_path)
            vid_path = str(p.with_name(f"{p.stem}_{trajectory_count}{p.suffix}"))

        frm_dir = args.save_frames
        if frm_dir and trajectory_count > 1:
            frm_dir = str(Path(frm_dir) / f"trajectory_{trajectory_count}")

        visualize_trajectory_predictions(
            policy=policy,
            timesteps=timesteps,
            instruction_override=args.instruction,
            max_timesteps=args.max_timesteps,
            save_video_path=vid_path,
            save_frames_dir=frm_dir,
            no_display=args.no_display,
        )

    if trajectory_count == 0:
        print("ERROR: No trajectories were loaded.", file=sys.stderr)
        return 1

    print("✓ Prediction visualization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
