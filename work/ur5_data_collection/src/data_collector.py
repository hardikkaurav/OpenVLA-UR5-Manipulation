"""
Data Collector
==============
Manages the storage of timestep observations and episode metadata during
pick-and-place data collection. Data is saved in a structured format that
maps cleanly to RLDS (Reinforcement Learning Datasets) for OpenVLA fine-tuning.

Storage layout:
    experiment_dir/
    ├── metadata.json                # Experiment-level metadata
    ├── episode_0000/
    │   ├── episode_metadata.json    # Pick/place poses, object info
    │   ├── timesteps.csv            # Per-step robot state + image refs
    │   └── images/
    │       ├── step_0000.png
    │       └── ...
    └── ...

RLDS Mapping:
    - Each episode → one RLDS episode
    - Each CSV row → one RLDS step
    - observation.image → step_NNNN.png (resize to 224×224)
    - observation.state → [ee_pos_x..z, ee_rot_rx..rz, gripper_state]
    - action → delta between consecutive observation.state rows
    - language_instruction → generated from episode metadata
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# CSV column order — matches RLDS observation fields for easy conversion
TIMESTEP_COLUMNS = [
    "timestamp",
    "episode_id",
    "step_id",
    "ee_pos_x",
    "ee_pos_y",
    "ee_pos_z",
    "ee_rot_rx",
    "ee_rot_ry",
    "ee_rot_rz",
    "gripper_state",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "image_filename",
]


class DataCollector:
    """
    Manages data storage for pick-and-place demonstrations.
    
    Creates a structured directory for each experiment, with per-episode
    subdirectories containing robot state CSVs and camera images. Also
    stores episode-level metadata (pick pose, place pose, object info)
    needed for RLDS language instruction generation.
    
    Attributes:
        base_dir:        Root directory for all experiments.
        experiment_name: Optional name prefix for the experiment folder.
    """

    def __init__(
        self,
        base_dir: str = "data",
        experiment_name: Optional[str] = None,
    ):
        self.base_dir = os.path.abspath(base_dir)

        # Create a timestamped experiment directory
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = experiment_name or "experiment"
        self.experiment_dir = os.path.join(
            self.base_dir, f"{prefix}_{timestamp_str}"
        )

        # Episode tracking
        self._current_episode_id: int = -1
        self._current_step_id: int = 0
        self._episode_dir: Optional[str] = None
        self._images_dir: Optional[str] = None
        self._csv_writer = None
        self._csv_file = None
        self._episode_steps: List[Dict[str, Any]] = []
        self._episode_metadata: Dict[str, Any] = {}

        # Experiment-level counters
        self._total_steps: int = 0
        self._total_episodes: int = 0

    def initialize(self, experiment_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create the experiment directory and save experiment-level metadata.
        
        Args:
            experiment_metadata: Optional dict of experiment-wide information
                                 (robot config, camera settings, etc.).
                                 
        Returns:
            Absolute path to the experiment directory.
        """
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Save experiment metadata
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_dir": self.experiment_dir,
            "timestep_columns": TIMESTEP_COLUMNS,
            "rlds_observation_keys": {
                "image": "images/step_NNNN.png (resize to 224x224)",
                "state": "[ee_pos_x, ee_pos_y, ee_pos_z, ee_rot_rx, ee_rot_ry, ee_rot_rz, gripper_state]",
                "joint_state": "[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]",
            },
            **(experiment_metadata or {}),
        }
        meta_path = os.path.join(self.experiment_dir, "metadata.json")
        self._write_json(meta_path, meta)

        logger.info("Experiment directory created: %s", self.experiment_dir)
        return self.experiment_dir

    # -------------------------------------------------------------------------
    # Episode Lifecycle
    # -------------------------------------------------------------------------

    def start_episode(self, episode_metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Begin a new episode. Creates the episode subdirectory and opens a
        new CSV file for timestep data.
        
        Args:
            episode_metadata: Metadata for this episode (pick_pose, place_pose,
                              object_type, object_size, object_weight, etc.).
                              
        Returns:
            The episode ID (zero-based index).
        """
        # Close any previous episode that wasn't explicitly ended
        if self._csv_file is not None:
            self.end_episode(success=True)

        self._current_episode_id += 1
        self._current_step_id = 0
        self._episode_steps = []

        # Create episode directory
        self._episode_dir = os.path.join(
            self.experiment_dir,
            f"episode_{self._current_episode_id:04d}"
        )
        self._images_dir = os.path.join(self._episode_dir, "images")
        os.makedirs(self._images_dir, exist_ok=True)

        # Store episode metadata (will be finalized in end_episode)
        self._episode_metadata = {
            "episode_id": self._current_episode_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            **(episode_metadata or {}),
        }

        # Open CSV for timestep data
        csv_path = os.path.join(self._episode_dir, "timesteps.csv")
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=TIMESTEP_COLUMNS
        )
        self._csv_writer.writeheader()

        logger.info(
            "Started episode %d → %s",
            self._current_episode_id, self._episode_dir
        )
        return self._current_episode_id

    def end_episode(self, success: bool = True,
                    extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Finalize and close the current episode.
        
        Flushes the CSV, saves episode metadata JSON, and updates counters.
        
        Args:
            success:        Whether the episode completed successfully.
            extra_metadata: Additional metadata to merge (e.g., final object pose).
        """
        if self._csv_file is None:
            logger.warning("end_episode() called but no episode is active.")
            return

        # Close CSV
        self._csv_file.close()
        self._csv_file = None
        self._csv_writer = None

        # Finalize episode metadata
        self._episode_metadata.update({
            "end_time": datetime.now(timezone.utc).isoformat(),
            "num_steps": self._current_step_id,
            "success": success,
            **(extra_metadata or {}),
        })

        # Save episode metadata
        meta_path = os.path.join(self._episode_dir, "episode_metadata.json")
        self._write_json(meta_path, self._episode_metadata)

        self._total_episodes += 1
        self._total_steps += self._current_step_id

        logger.info(
            "Ended episode %d — %d steps, success=%s",
            self._current_episode_id, self._current_step_id, success
        )

    # -------------------------------------------------------------------------
    # Timestep Recording
    # -------------------------------------------------------------------------

    def record_timestep(
        self,
        robot_state: Dict[str, float],
        image: np.ndarray,
        camera_logger,
    ) -> int:
        """
        Record a single timestep: save the image and write robot state to CSV.
        
        This is called at every control step during an episode. The saved data
        includes all fields needed for RLDS conversion.
        
        Args:
            robot_state:   Dict from UR5Controller.get_full_state() with keys:
                           ee_pos_x/y/z, ee_rot_rx/ry/rz, gripper_state,
                           joint_1..joint_6.
            image:         RGB numpy array from CameraLogger.capture_frame().
            camera_logger: CameraLogger instance used to save the frame.
            
        Returns:
            The step ID for this timestep.
        """
        if self._csv_writer is None:
            raise RuntimeError(
                "No active episode. Call start_episode() before recording."
            )

        step_id = self._current_step_id

        # Save image
        image_filename = f"step_{step_id:04d}.png"
        image_path = os.path.join(self._images_dir, image_filename)
        camera_logger.save_frame(image, image_path)

        # Build timestep row
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode_id": self._current_episode_id,
            "step_id": step_id,
            "ee_pos_x": f"{robot_state['ee_pos_x']:.6f}",
            "ee_pos_y": f"{robot_state['ee_pos_y']:.6f}",
            "ee_pos_z": f"{robot_state['ee_pos_z']:.6f}",
            "ee_rot_rx": f"{robot_state['ee_rot_rx']:.6f}",
            "ee_rot_ry": f"{robot_state['ee_rot_ry']:.6f}",
            "ee_rot_rz": f"{robot_state['ee_rot_rz']:.6f}",
            "gripper_state": f"{robot_state['gripper_state']:.4f}",
            "joint_1": f"{robot_state.get('joint_1', 0.0):.6f}",
            "joint_2": f"{robot_state.get('joint_2', 0.0):.6f}",
            "joint_3": f"{robot_state.get('joint_3', 0.0):.6f}",
            "joint_4": f"{robot_state.get('joint_4', 0.0):.6f}",
            "joint_5": f"{robot_state.get('joint_5', 0.0):.6f}",
            "joint_6": f"{robot_state.get('joint_6', 0.0):.6f}",
            "image_filename": f"images/{image_filename}",
        }

        self._csv_writer.writerow(row)
        self._csv_file.flush()  # Ensure data is written even if we crash
        self._episode_steps.append(row)

        self._current_step_id += 1
        logger.debug(
            "Recorded step %d (episode %d)",
            step_id, self._current_episode_id
        )
        return step_id

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _write_json(filepath: str, data: dict) -> None:
        """Write a dictionary to a JSON file with pretty formatting."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def current_episode_id(self) -> int:
        """Current episode index."""
        return self._current_episode_id

    @property
    def current_step_id(self) -> int:
        """Current step index within the active episode."""
        return self._current_step_id

    @property
    def total_episodes(self) -> int:
        """Total number of completed episodes."""
        return self._total_episodes

    @property
    def total_steps(self) -> int:
        """Total number of recorded steps across all episodes."""
        return self._total_steps

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the data collected so far."""
        return {
            "experiment_dir": self.experiment_dir,
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "avg_steps_per_episode": (
                self._total_steps / self._total_episodes
                if self._total_episodes > 0 else 0
            ),
        }
