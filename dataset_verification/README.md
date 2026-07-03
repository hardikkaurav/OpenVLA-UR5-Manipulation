# OpenVLA Pretrained Model Verification

Evaluate the pretrained `openvla/openvla-7b` checkpoint against the ground-truth actions from the **Berkeley Autolab UR5 Demonstration Dataset** — without modifying any existing OpenVLA files.

## Directory Structure

```
dataset_verification/
├── verify_openvla.py          # CLI entry-point — ties everything together
├── trajectory_evaluator.py    # Core evaluation engine (metrics, CSV, visual overlay)
├── dataset_loader.py          # Load trajectories from RLDS or local .npy files
├── visualize_dataset.py       # Generate comparison plots (matplotlib)
├── README.md                  # This file
└── results/                   # Output directory (auto-created)
    ├── evaluation_trajectory_1.csv
    ├── gt_vs_predicted.png
    ├── error_over_time.png
    ├── error_histogram.png
    ├── scatter_pred_vs_gt.png
    ├── gt_actions.npy
    └── pred_actions.npy
```

## Module Responsibilities

| Module | Role |
|---|---|
| `dataset_loader.py` | **Data only.** Loads trajectories, prints metadata, displays images. |
| `trajectory_evaluator.py` | **Evaluation engine.** Runs inference, computes metrics, exports CSV, optional OpenCV visual overlay. |
| `visualize_dataset.py` | **Plots.** Generates GT vs predicted, error over time, histograms, scatter plots. |
| `verify_openvla.py` | **CLI orchestrator.** Loads model + data, calls evaluator + visualizer, prints summary. |

## Prerequisites

```bash
# Core (already installed in your OpenVLA venv)
pip install torch transformers pillow numpy

# For RLDS dataset loading
pip install tensorflow tensorflow_datasets

# For plotting
pip install matplotlib

# For visual overlay mode (optional)
pip install opencv-python
```

## Usage

### Quick Test (first trajectory, first 50 timesteps)

```bash
cd dataset_verification
python verify_openvla.py --max-timesteps 50
```

### Full Single-Trajectory Evaluation

```bash
python verify_openvla.py --trajectory-index 0
```

### Evaluate Multiple Trajectories

```bash
python verify_openvla.py --max-trajectories 10
```

### Evaluate a Local `.npy` File

```bash
python verify_openvla.py --source /path/to/trajectory.npy
```

### Evaluate a Directory of `.npy` Files

```bash
python verify_openvla.py --source-dir /path/to/trajectories/ --max-trajectories 5
```

### Visual Mode — Step Through Frames with OpenCV Overlay

```bash
python verify_openvla.py --visual --max-timesteps 30
```

Press any key to advance, `q` or `Esc` to quit.

### Custom Tolerance and Print Frequency

```bash
python verify_openvla.py --tolerance 0.02 --print-every 5
```

### Use a Different Checkpoint

```bash
python verify_openvla.py --model-id openvla/openvla-7b --unnorm-key bridge_orig
```

### CPU Inference (very slow)

```bash
python verify_openvla.py --allow-cpu --max-timesteps 10
```

## Inspect Dataset Only (No Model Needed)

```bash
python dataset_loader.py
python dataset_loader.py --source /path/to/trajectory.npy
```

## Output

### Console — Per-Frame Comparison (every 10th frame by default)

```
Frame   0  [✗ FAIL]  L2=0.04523  inference=97.2 ms
  Instruction:   put the tiger into the grey bowl
  Ground Truth:  [+0.01234  -0.00567  +0.00891  -0.00123  +0.00456  -0.00789  +0.99600]
  Predicted:     [+0.00567  +0.01234  -0.00345  +0.00012  -0.00234  +0.00567  +0.98700]
  Abs Error:     [0.00667  0.01801  0.01236  0.00135  0.00690  0.01356  0.00900]
  Within tol (0.01):  False

Frame  10  [✓ PASS]  L2=0.00812  inference=95.1 ms
  ...
```

### Console — Trajectory Summary

```
======================================================================
  TRAJECTORY EVALUATION SUMMARY
======================================================================

  Number of frames:              120
  Tolerance:                     0.01

  Mean Absolute Error (MAE):     0.023456
  Mean Squared Error (MSE):      0.00123456
  Cosine Similarity:             0.456789

  Maximum Error (L2):            0.123456
  Minimum Error (L2):            0.001234
  Frames within tolerance:       12.5%

  Average inference time:        96.3 ms (10.4 FPS)

  Per-Dimension MAE:
        dx:  0.012345
        dy:  0.023456
        dz:  0.034567
     droll:  0.045678
    dpitch:  0.056789
      dyaw:  0.067890
   gripper:  0.012345

  Per-Dimension MSE:
        dx:  0.00012345
      ...
======================================================================
```

### CSV (per-trajectory)

Saved to `results/evaluation_trajectory_1.csv` with columns:

| Column | Description |
|---|---|
| `frame_index` | Timestep number |
| `instruction` | Language instruction |
| `gt_dx` … `gt_gripper` | Ground-truth action (7 columns) |
| `pred_dx` … `pred_gripper` | Predicted action (7 columns) |
| `abs_err_dx` … `abs_err_gripper` | Absolute error per dimension |
| `sq_err_dx` … `sq_err_gripper` | Squared error per dimension |
| `l2_error` | L2 norm of the error vector |
| `within_tolerance` | True/False |

### Plots

Saved to `results/`:

| Plot | Description |
|---|---|
| `gt_vs_predicted.png` | Line plots per action dimension |
| `error_over_time.png` | L2 error + per-dimension breakdown |
| `error_histogram.png` | Error distribution per dimension |
| `scatter_pred_vs_gt.png` | Perfect model = diagonal line |

## Interpreting Results

| Cosine Similarity | Meaning |
|---|---|
| > 0.8 | Model understands the scene; coordinate transform may fix it |
| 0.4 – 0.8 | Partial understanding; fine-tuning likely needed |
| < 0.4 | Model cannot generalize; fine-tuning required |

| MAE | Meaning |
|---|---|
| < 0.01 | Excellent — near ground-truth accuracy |
| 0.01 – 0.05 | Reasonable — usable with safety clamps |
| > 0.05 | Poor — model is guessing |

## Notes

- **Verification only** — no existing OpenVLA code is modified.
- Reuses `OpenVLAPolicy` from `openvla_realtime/` for model loading and inference.
- The Berkeley UR5 dataset is ~76 GB; first TFDS download may take time.
- Use `--max-timesteps` for quick sanity checks before full runs.
- The `--visual` flag requires `opencv-python` and a display (X11/Wayland).
