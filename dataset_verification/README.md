# OpenVLA Dataset Verification, RLDS Conversion & Policy Fine-Tuning Suite

<div align="center">

[![Dataset Format RLDS](https://img.shields.io/badge/Dataset%20Format-RLDS%20%2F%20TFDS-FF6F00.svg)](https://github.com/google-research/rlds)
[![OpenVLA 7B](https://img.shields.io/badge/Model-OpenVLA--7B-8A2BE2.svg)](https://github.com/openvla/openvla)
[![Robot UR5](https://img.shields.io/badge/Hardware-Universal%20Robots%20UR5-005180.svg)](https://www.universal-robots.com/products/ur5-robot/)
[![Sensor RealSense](https://img.shields.io/badge/Sensor-Intel%20RealSense%20D435i-0071C5.svg)](https://www.intelrealsense.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

An open-source pipeline for **UR5 demonstration data collection, RLDS episode standardization, quantitative verification, and instruction fine-tuning** of 7B Vision-Language-Action (VLA) models.

[Overview](#overview) •
[UR5 & Sensor Setup](#ur5--realsense-d435i-setup) •
[Dataset & Episode Structure](#episode-structure--dataset-statistics) •
[RLDS Conversion](#rlds-conversion) •
[Running Fine-Tuning](#running-fine-tuning) •
[Verification & Evaluation Suite](#verification--evaluation-suite)

</div>

---

## Overview

The `dataset_verification/` directory houses the complete data management, conversion, and evaluation infrastructure for **OpenVLA-UR5-Manipulation**. Before deploying a Vision-Language-Action model on physical hardware, it is critical to verify that:
1. Trajectory episodes are correctly structured, timestamp-aligned, and normalized.
2. Coordinate frames between the vision sensor (`RealSense D435i`) and robot end-effector (`UR5 TCP`) match OpenVLA's expected action space.
3. Pretrained vs. fine-tuned checkpoints can be systematically benchmarked on held-out ground-truth trajectories across all 7 degrees of freedom ($x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper}$).

```
+----------------------------------------------------------------------------------------+
|                            Dataset & Fine-Tuning Workflow                              |
+----------------------------------------------------------------------------------------+
                                            |
                                            v
+-----------------------+      +-------------------------+      +------------------------+
|  1. Data Collection   | ---> |   2. RLDS Conversion    | ---> | 3. Action Normalization|
|  UR5 + RealSense D435i|      |   Standard TFDS Schema  |      |   1st & 99th Quantile  |
+-----------------------+      +-------------------------+      +------------------------+
                                                                             |
                                                                             v
+-----------------------+      +-------------------------+      +------------------------+
|  6. Qualitative Eval  | <--- |  5. Quantitative Eval   | <--- |   4. VLA Fine-Tuning   |
|  OpenCV Overlay Video |      |  MAE / MSE / L2 Metrics |      |   LoRA / Full Weights  |
+-----------------------+      +-------------------------+      +------------------------+
```

---

## UR5 & RealSense D435i Setup

### Robotic Manipulator Configuration

- **Robot Arm**: Universal Robots UR5 (6-DoF Industrial Manipulator)
- **Control Interface**: Real-Time Data Exchange (`UR RTDE`) Ethernet interface running at 100 Hz / 500 Hz internal controller frequency.
- **End-Effector (TCP)**: Standard parallel jaw gripper controlled via digital I/O or Modbus RTU.
- **Cartesian Control Space**: Tool Center Point (TCP) pose expressed in base coordinates $(X, Y, Z, \text{Roll}, \text{Pitch}, \text{Yaw})$. Actions represent relative per-timestep displacement deltas $(\Delta X, \Delta Y, \Delta Z, \Delta \text{Roll}, \Delta \text{Pitch}, \Delta \text{Yaw}, \text{Gripper})$.

### Intel RealSense D435i Sensor Integration

- **Mounting Configuration**: Eye-to-hand fixed tabletop third-person perspective viewing the manipulation workspace.
- **RGB Resolution & Framing**: Streamed at $640 \times 480$ native resolution (RGB888) and center-cropped/resized to $224 \times 224$ pixels to match the input resolution of OpenVLA's Prismatic/SigLIP vision encoder backbone.
- **Hardware Sync**: Timestamp synchronization between camera frame timestamps and UR5 RTDE state telemetry ensures low latency jitter between image acquisition and action execution.

---

## Episode Structure & Dataset Statistics

### Episode Representation

Each recorded demonstration episode consists of a sequence of synchronized multimodal transitions $T = \{ (I_t, s_t, a_t, l) \}_{t=0}^{L-1}$:

```python
{
    "observation": {
        "image": np.ndarray,          # uint8, shape (224, 224, 3) or (480, 640, 3)
        "state": np.ndarray           # float32, shape (7,), absolute TCP pose + gripper state
    },
    "action": np.ndarray,             # float32, shape (7,), target delta action command
    "language_instruction": str,      # String instruction e.g., "pick up the red block"
    "is_first": bool,                 # True at t=0
    "is_last": bool                   # True at t=L-1
}
```

### Action Vector Definition (7-DoF)

| Index | Name | Unit | Range / Interpretation |
| :---: | :--- | :---: | :--- |
| `0` | $\Delta X$ | Meters | End-effector forward/backward translation delta |
| `1` | $\Delta Y$ | Meters | End-effector left/right translation delta |
| `2` | $\Delta Z$ | Meters | End-effector vertical elevation delta |
| `3` | $\Delta \text{Roll}$ | Radians | Rotation delta about X-axis |
| `4` | $\Delta \text{Pitch}$ | Radians | Rotation delta about Y-axis |
| `5` | $\Delta \text{Yaw}$ | Radians | Rotation delta about Z-axis |
| `6` | $\text{Gripper}$ | Normalized | Discrete/Continuous opening ($0.0 = \text{closed}, 1.0 = \text{open}$) |

### Dataset Statistics

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Demonstration Episodes** | ~250+ trajectories | Multi-task table-top pick, place, and assembly demonstrations |
| **Average Episode Length** | ~120 frames | Typical duration at 10 Hz control policy rate |
| **Action Quantile Bounds** | Computed per-dim | 1st and 99th percentile boundaries for robust `bridge_orig` / custom unnormalization |
| **Storage Format** | RLDS / `.npy` | Optimized memory-mapped TFDS / NumPy records |

---

## RLDS Conversion

To enable standardized data loaders across Hugging Face and Open X-Embodiment pipelines, local trajectory arrays are converted into **Robot Learning Dataset Standard (RLDS)** TFDS builders.

### Key Steps in RLDS Pipeline

1. **Ingestion**: `dataset_loader.py` inspects raw `.npy` trajectory arrays or existing TFDS shards.
2. **Metadata Tagging**: Annotates language instructions, step flags (`is_first`, `is_last`, `is_terminal`), and normalization statistics.
3. **Serialization**: Shards data into TFRecord files compliant with the standard `rlds.transformations` API.

```bash
# Verify loading and inspect sample RLDS / local .npy trajectory metadata
python dataset_loader.py --source /path/to/trajectory.npy
```

---

## Directory Layout

```text
dataset_verification/
├── dataset_loader.py                   # Data ingestion engine for RLDS TFDS & NumPy files
├── verify_openvla.py                   # CLI orchestrator evaluating VLA checkpoints against ground truth
├── trajectory_evaluator.py             # Core evaluation metrics (MAE, MSE, Cosine Sim, CSV trace export)
├── visualize_dataset.py                # Publication-grade Matplotlib plotting (histograms, error lines)
├── compare_openvla_instructions.py        # Evaluates sensitivity across prompt phrasing variations
├── visualize_openvla_predictions.py    # Generates qualitative prediction overlay images
├── README.md                           # Documentation (this file)
└── results/                            # Export directory for plots and quantitative logs
    ├── evaluation_trajectory_1.csv     # Step-by-step ground truth vs predicted action trace
    ├── gt_vs_predicted.png             # Multi-DoF action prediction overlay chart
    ├── error_over_time.png             # Temporal L2 error evolution plot
    ├── error_histogram.png             # Error frequency histogram per action dimension
    └── scatter_pred_vs_gt.png          # Correlation scatter plot
```

---

## Module Responsibilities

| Module | Core Functionality |
| :--- | :--- |
| `dataset_loader.py` | **Multimodal Data Loader**: Parses RLDS datasets or local `.npy` files, extracts image frames, ground truth action vectors, and instruction prompts. |
| `trajectory_evaluator.py` | **Evaluation Engine**: Feeds frames into `OpenVLAPolicy`, calculates per-dimension MAE/MSE/Cosine similarity, exports structured CSVs, and supports OpenCV video playback overlays. |
| `visualize_dataset.py` | **Plot Generator**: Produces publication-ready PDF/PNG charts comparing ground truth trajectories against predicted actions. |
| `verify_openvla.py` | **CLI Entrypoint**: Unifies data loading, model inference, metric calculation, and visualization into a single command-line tool. |
| `compare_openvla_instructions.py` | **Instruction Sensitivity Benchmarking**: Tests multiple semantic rephrasings of natural language prompts to quantify prompt robustness. |

---

## Running Fine-Tuning

Fine-tuning OpenVLA 7B adapts the pretrained vision-language representations to UR5 kinematics and RealSense D435i camera perspectives.

### 1. Compute Dataset Normalization Statistics

Before fine-tuning, verify action distribution bounds across the training split:

```bash
python dataset_loader.py --compute-stats
```

### 2. Parameter-Efficient Fine-Tuning (LoRA)

Run instruction fine-tuning using LoRA adapters on the action-token prediction heads:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m openvla.scripts.finetune \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir "/path/to/rlds_dataset" \
    --dataset_name "ur5_manipulation_dataset" \
    --run_root_dir "./work/checkpoints" \
    --lora_rank 32 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --save_steps 2000
```

---

## Verification & Evaluation Suite

Once a checkpoint is fine-tuned (or to benchmark the pretrained baseline), run verification scripts inside `dataset_verification/`.

### Quick Sanity Check (First 50 Timesteps)

```bash
python verify_openvla.py --max-timesteps 50
```

### Full Single-Trajectory Evaluation

```bash
python verify_openvla.py --trajectory-index 0
```

### Batch Evaluation Over Multiple Trajectories

```bash
python verify_openvla.py --max-trajectories 10
```

### Interactive OpenCV Visual Overlay Mode

Step through prediction frames live with visual ground-truth vs. prediction overlay bars:

```bash
python verify_openvla.py --visual --max-timesteps 30
```

### Evaluating Fine-Tuned Checkpoints vs Pretrained

Specify custom model IDs or local checkpoint paths:

```bash
python verify_openvla.py --model-id /path/to/fine_tuned_openvla_ur5 --tolerance 0.01
```

---

## Interpreting Quantitative Output Metrics

### Metric Threshold Guidelines

| Metric | Excellent (Target) | Acceptable (Safe with Clamps) | Poor (Requires Retraining) |
| :--- | :---: | :---: | :---: |
| **Overall MAE** | $< 0.010$ | $0.010 - 0.030$ | $> 0.040$ |
| **Cosine Similarity** | $> 0.85$ | $0.65 - 0.85$ | $< 0.50$ |
| **Binary Gripper Accuracy** | $> 95.0\%$ | $90.0\% - 95.0\%$ | $< 85.0\%$ |

Our fine-tuned UR5 policy achieves an **Overall MAE of 0.006566** (an **86.48% improvement** over the pretrained baseline `0.048560`) and **96.00% binary gripper classification accuracy**.
