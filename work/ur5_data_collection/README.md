# UR5 Pick-and-Place Data Collection for OpenVLA Fine-Tuning

A complete Python data collection system for a UR5 robot performing automated
pick-and-place experiments. Designed to produce demonstrations suitable for
conversion to RLDS format and fine-tuning OpenVLA.

## Project Structure

```
ur5_data_collection/
├── README.md
├── requirements.txt
├── run_experiment.py            # Entry point
├── config/
│   └── poses.json               # Predefined pick/place poses
├── src/
│   ├── __init__.py
│   ├── ur5_controller.py        # UR5Controller class (RTDE interface)
│   ├── camera_logger.py         # CameraLogger class (RGB capture)
│   ├── data_collector.py        # DataCollector class (timestep/episode storage)
│   └── experiment_runner.py     # ExperimentRunner (orchestrates workflow)
└── data/                        # Created at runtime
    └── experiment_YYYYMMDD_HHMMSS/
        ├── metadata.json        # Experiment-level metadata
        ├── episode_0000/
        │   ├── episode_metadata.json
        │   ├── timesteps.csv
        │   └── images/
        │       ├── step_0000.png
        │       ├── step_0001.png
        │       └── ...
        ├── episode_0001/
        │   └── ...
        └── ...
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Poses

Edit `config/poses.json` to set your robot's home pose, pickup grasp pose, and 7 place
positions. All poses use the format `[x, y, z, rx, ry, rz]` in meters and radians.

### 3. Run Data Collection

```bash
# Collect 50 episodes
python run_experiment.py --robot-ip 192.168.1.100 --camera-id 0 --num-episodes 50

# Dry run (simulated robot, no hardware required)
python run_experiment.py --dry-run --num-episodes 5

# Specify object metadata
python run_experiment.py --dry-run \
    --object-type "red_cube" \
    --object-size "0.05,0.05,0.05" \
    --object-weight 0.15 \
    --num-episodes 10

### 4. Automatic Reset Between Episodes

The robot will always pick from a fixed pickup location. After each episode finishes recording (`collector.end_episode()`), the robot automatically executes an unrecorded reset sequence: it moves to the place pose, picks up the object, returns it to the original pickup location, releases it, and returns home. No images, states, or actions are recorded during this sequence.


### 5. Convert to RLDS (Post-Collection)

The saved data is structured for easy RLDS conversion. Each episode contains:
- Sequential RGB images (`step_NNNN.png`)
- A CSV of robot states with columns matching RLDS observation/action fields
- Episode metadata in JSON

See the `convert_to_rlds.py` script (to be provided separately) for TFDS
dataset builder integration.

## Data Format

### Timestep CSV Columns

| Column           | Description                              |
|------------------|------------------------------------------|
| `timestamp`      | UTC ISO-8601 timestamp                   |
| `episode_id`     | Episode index (zero-based)               |
| `step_id`        | Step index within episode (zero-based)   |
| `ee_pos_x`       | End-effector X position (m)              |
| `ee_pos_y`       | End-effector Y position (m)              |
| `ee_pos_z`       | End-effector Z position (m)              |
| `ee_rot_rx`      | End-effector rotation Rx (rad)           |
| `ee_rot_ry`      | End-effector rotation Ry (rad)           |
| `ee_rot_rz`      | End-effector rotation Rz (rad)           |
| `gripper_state`  | Gripper openness normalized to [0, 1]    |
| `image_filename` | Relative path to the RGB image           |

### Episode Metadata JSON

```json
{
  "episode_id": 0,
  "pickup_pose": [x, y, z, rx, ry, rz],
  "place_pose": [x, y, z, rx, ry, rz],
  "place_pose_index": 3,
  "object_type": "red_cube",
  "object_size": [0.05, 0.05, 0.05],
  "object_weight": 0.15,
  "language_instruction": "place the red_cube at position 4",
  "recording_hz": 10.0,
  "num_steps": 42,
  "start_time": "2026-06-22T17:00:00Z",
  "end_time": "2026-06-22T17:01:30Z",
  "success": true
}
```

## RLDS Compatibility Notes

The data layout maps directly to the Bridge-style RLDS format used by OpenVLA:

- **Observation**: `image` (224×224 RGB) + `state` (7-DoF: xyz + rxryrz + gripper)
- **Action**: delta between consecutive states (computed during conversion)
- **Language instruction**: set per-episode from pick/place metadata

During RLDS conversion you will typically:
1. Resize images to 224×224
2. Compute actions as state deltas
3. Generate language instructions like `"pick up the red_cube and place it at position 3"`
4. Package into TFRecord shards

## Hardware Requirements

- Universal Robots UR5 with RTDE interface enabled
- Custom servo gripper controlled by Arduino Uno over USB serial
- USB or IP RGB camera observing the workspace
- Control PC running Ubuntu 20.04+ (or macOS for dry-run testing)
