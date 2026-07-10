# OpenVLA Real-Time Closed-Loop UR5 & RealSense Inference Pipeline

<div align="center">

[![Real-Time Inference](https://img.shields.io/badge/Inference-Real--Time%20VLA-8A2BE2.svg)](https://github.com/openvla/openvla)
[![Robot UR5](https://img.shields.io/badge/Hardware-Universal%20Robots%20UR5-005180.svg)](https://www.universal-robots.com/products/ur5-robot/)
[![Camera RealSense](https://img.shields.io/badge/Camera-Intel%20RealSense%20D435i-0071C5.svg)](https://www.intelrealsense.com/)
[![CUDA PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.x-ee4c2c.svg)](https://pytorch.org/)
[![Ubuntu 22.04](https://img.shields.io/badge/OS-Ubuntu%2022.04%20LTS-E95420.svg)](https://ubuntu.com/)

An asynchronous, hardware-safe execution engine for running **real-time Vision-Language-Action (VLA) inference** with `openvla/openvla-7b` streaming live RGB frames from an **Intel RealSense D435i** to control a **Universal Robots UR5** manipulator.

[Real-Time Pipeline](#real-time-inference-pipeline) •
[Camera Subsystem](#camera-subsystem) •
[OpenVLA Policy](#openvla-policy-engine) •
[Robot Control & Safety](#robot-control--safety) •
[Running Inference](#running-inference) •
[Configuration](#configuration) •
[Troubleshooting](#troubleshooting)

</div>

---

## Overview

The `openvla_realtime5/` package implements a low-latency, real-time perception-inference-actuation loop for tabletop robotic manipulation. By coupling eye-to-hand RGB streams from an **Intel RealSense D435i** with a quantized or full-precision **OpenVLA 7B** transformer policy, the system continuously decodes natural language instructions into 7-DoF end-effector Cartesian displacement deltas.

```
+-----------------------------------------------------------------------------------------------+
|                             Real-Time Closed-Loop Control Architecture                        |
+-----------------------------------------------------------------------------------------------+

  +-----------------------+              +------------------------+
  | Natural Language Goal |              |  Intel RealSense D435i |
  | "pick up red block"   |              |  Eye-to-Hand RGB Sensor|
  +-----------------------+              +------------------------+
              |                                       |
              v                                       v
  +-------------------------------------------------------------------+
  |               OpenVLA Multimodal Prompt Construction              |
  |   "In: What action should the robot take to [instruction]? Out:"  |
  +-------------------------------------------------------------------+
                                      |
                                      v
  +-------------------------------------------------------------------+
  |                    OpenVLAPolicy Inference Engine                 |
  |          AutoModelForVision2Seq (bfloat16 / 4-bit NF4)            |
  +-------------------------------------------------------------------+
                                      |
                                      v
  +-------------------------------------------------------------------+
  |                  Continuous 7-DoF Action Decoding                 |
  |     [Delta X, Delta Y, Delta Z, Delta Roll, Pitch, Yaw, Gripper]  |
  +-------------------------------------------------------------------+
                                      |
                                      v
  +-------------------------------------------------------------------+
  |                        Safety Controller                          |
  |    * Kinematic Delta Clamping   * Workspace Boundary Enforcement  |
  +-------------------------------------------------------------------+
                    |                                 |
                    v                                 v
  +----------------------------------+   +----------------------------+
  |       DryRunRobot / UR5 RTDE     |   |   Live Telemetry Dashboard |
  |     Simulated or Physical Arm    |   |   OpenCV Action Bar Chart  |
  +----------------------------------+   +----------------------------+
```

---

## Real-Time Inference Pipeline

The execution loop orchestrated by `main.py` operates as an asynchronous control stream:

1. **Frame Capture**: `camera.py` polls the RealSense D435i camera over USB 3.0, returning a freshly captured RGB image and checking stream latency.
2. **Prompt Assembly**: The user's natural language instruction is embedded into the standard OpenVLA prompt template:
   ```text
   In: What action should the robot take to <INSTRUCTION>?
   Out:
   ```
3. **Autoregressive Decoding**: `openvla_policy.py` generates 7 action tokens predicting spatial end-effector displacement deltas and gripper actuation.
4. **Action Unnormalization**: Discrete tokens are mapped back to continuous Cartesian units (meters/radians) using dataset statistics (`bridge_orig` or custom fine-tuned quantiles).
5. **Safety Filtering & Execution**: The decoded action vector passes through a safety governor before being sent to the robot interface (`DryRunRobot` or `UR5RTDERobot`).

---

## Camera Subsystem

The vision module (`camera.py`) manages hardware communication with the **Intel RealSense D435i**:

- **Stream Specifications**: RGB color stream at $640 \times 480$ resolution at 30 FPS.
- **Auto-Exposure & White Balance**: Hardware auto-exposure is stabilized during startup to prevent initial illumination artifacts.
- **Thread Safety**: Frame buffers are managed securely to prevent dropped frames or tearing during multi-threaded VLA inference.
- **Verification Tool**: Run `python test_camera.py` or use Linux `realsense-viewer` to inspect hardware stream integrity before launching full inference.

---

## OpenVLA Policy Engine

The inference wrapper in `openvla_policy.py` handles model instantiation, precision selection, and action decoding for `openvla/openvla-7b` (or local fine-tuned checkpoints):

### Memory & Precision Modes

| Hardware Configuration | Precision Strategy | VRAM Footprint | Typical Inference Rate |
| :--- | :--- | :---: | :---: |
| **NVIDIA RTX 5090 / RTX 4090 / A6000** | Full `torch.bfloat16` weights | ~15.5 GB | **10.0+ FPS** (~95 ms/frame) |
| **NVIDIA RTX 3090 / 4080 (16GB–24GB)** | Automatic 4-Bit `bitsandbytes` NF4 fallback | ~7.2 GB | **4.5–6.0 FPS** (~180 ms/frame) |
| **CPU Fallback (`--allow-cpu`)** | `torch.float32` CPU execution | System RAM | $< 0.3 \text{ FPS}$ (Diagnostic only) |

> [!TIP]
> **Automatic Quantization Fallback**: When loading on GPUs where full `bfloat16` weights would exceed available memory, `openvla_policy.py` automatically falls back to 4-bit `bitsandbytes` quantization without user intervention.

---

## Robot Control & Safety

Safety is paramount when deploying 7B foundation models on physical robotic manipulators. The package isolates control logic within `utils/robot_adapters.py`.

### 1. Default Safety Interface (`DryRunRobot`)

By default, the pipeline runs using `DryRunRobot`. It **never commands physical hardware actuation**:
- Maintains a simulated Tool Center Point (TCP) pose $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$.
- Applies strict conservative delta clipping ($\Delta X, \Delta Y, \Delta Z \in [-0.05 \text{m}, +0.05 \text{m}]$ per step).
- Enforces 3D bounding box workspace limits to ensure simulated poses remain within reachable tabletop boundaries.

### 2. Physical UR5 Hardware Interface (`UR5RTDERobot`)

For closed-loop physical execution using `ur_rtde`, the placeholder `UR5RTDERobot` can be activated once rigorous safety checks are completed.

> [!CAUTION]
> **Physical Hardware Readiness Protocol**: Do **NOT** enable physical UR5 motion until all of the following safety measures are verified:
> - Functional hardware Emergency Stop (E-Stop) button within immediate operator reach.
> - Reduced joint speed / acceleration governors configured in the UR5 controller poly-scope.
> - Verified Cartesian software bounding limits protecting table surfaces and cameras.
> - Audited TCP payload and center-of-gravity parameters.

---

## Running Inference

Ensure your Python virtual environment is activated and hardware drivers are accessible:

```bash
cd openvla_realtime5
source ../.venv/bin/activate
```

### 1. Single-Frame Diagnostic Test Mode (`--test`)

Captures a single camera frame, runs one OpenVLA prediction, displays the telemetry bar chart, logs output to CSV, and exits:

```bash
python main.py --test --instruction "pick up the red block"
```

### 2. Full Real-Time Closed-Loop Execution

Run continuous real-time inference streaming live camera frames until `q`, `Esc`, or `Ctrl+C` is pressed:

```bash
python main.py --instruction "pick up the red block and place it in the bowl"
```

### 3. Execution CLI Flags

| Flag | Argument | Description | Default |
| :--- | :---: | :--- | :---: |
| `--instruction` | `string` | Textual goal prompt sent to the OpenVLA policy | `"pick up the red block"` |
| `--test` | — | Runs a single-step diagnostic execution and exits immediately | `False` |
| `--no-preview` | — | Disables GUI OpenCV window rendering (ideal for headless servers) | `False` |
| `--save-image-every`| `integer`| Saves camera frame + action telemetry snapshot every $N$ frames | `30` (`0` disables) |
| `--max-loop-hz` | `float` | Caps maximum inference loop frequency in Hz | Uncapped |
| `--allow-cpu` | — | Allows CPU inference fallback if no CUDA GPU is detected | `False` |

---

## Configuration

Runtime parameters are centralized in `config.py`:

```python
# Default Language Instruction
DEFAULT_INSTRUCTION = "pick up the red block"

# OpenVLA Model Path / Hugging Face ID
MODEL_ID = "openvla/openvla-7b"
UNNORM_KEY = "bridge_orig"

# Telemetry Logging Parameters
LOG_FILE_PATH = "logs/openvla_inference_log.csv"
SAVE_IMAGE_DIR = "logs/images/"
```

### Log File Schema (`logs/openvla_inference_log.csv`)

Every execution step logs structured telemetry:
`timestamp_utc`, `frame_index`, `instruction`, `dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw`, `gripper`, `inference_ms`, `fps`, `saved_image_path`.

---

## Troubleshooting

### 1. `CUDA GPU was not detected`
- Run `nvidia-smi` to verify driver visibility.
- Check PyTorch CUDA binding: `python -c "import torch; print(torch.cuda.is_available())"`.
- If running on a non-GPU system, use `--allow-cpu` for diagnostic debugging.

### 2. `No Intel RealSense device found`
- Verify camera USB connection (must be connected to a **USB 3.0/3.1** port; USB 2.0 lacks required bandwidth).
- Run `realsense-viewer` to verify sensor recognition.
- Ensure Linux udev rules for librealsense are installed (`/etc/udev/rules.d/99-realsense-libusb.rules`).

### 3. `CUDA Out Of Memory (OOM)`
- Close competing background GPU processes.
- The loader automatically attempts 4-bit `bitsandbytes` quantization if full `bfloat16` exceeds available VRAM. Ensure `bitsandbytes` is installed (`pip install bitsandbytes`).

### 4. `RealSense frame capture timed out`
- Check if another application (e.g., `realsense-viewer`, OpenCV script) is holding an exclusive lock on `/dev/video*`. Close conflicting processes and restart `main.py`.
