"""
Experiment Runner
=================
Orchestrates the full pick-and-place data collection workflow.  Coordinates the
UR5Controller, CameraLogger, and DataCollector to execute repeated pick-and-place
cycles while recording demonstrations for OpenVLA fine-tuning.

Design:
    - One fixed pickup location — the object always starts here.
    - Seven predefined place locations loaded from poses.json.
    - After each episode the robot returns to home and waits for the human
      operator to press ENTER, confirming the object has been returned to the
      pickup location.
    - No automatic object-location tracking; the pickup pose is constant.
    - Approach poses are computed automatically from the grasp/place Z plus
      the configured approach_height_offset.
    - Continuous observation recording during every robot motion (5-10 Hz).
    - All language instructions include the target position index so OpenVLA
      can learn position-conditioned placement.

Workflow per episode:
     1. Move to home pose                     (recording continuously)
     2. Move to pickup approach pose           (recording continuously)
     3. Move down to pickup grasp pose         (recording continuously)
     4. Close gripper                          (record observation)
     5. Move back up to pickup approach pose   (recording continuously)
     6. Move to place approach pose            (recording continuously)
     7. Move down to place pose                (recording continuously)
     8. Open gripper                           (record observation)
     9. Move back up to place approach pose    (recording continuously)
    10. Return to home                         (recording continuously)
    11. Prompt operator to reset the object and press ENTER

Expected: ~50-100 observations per episode with continuous recording.
"""

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.ur5_controller import Pose, UR5Controller
from src.camera_logger import CameraLogger
from src.data_collector import DataCollector
from src.pause_manager import PauseManager, QuitRequestedException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language instruction templates — ALL include the target position index.
# This is critical for OpenVLA to learn position-conditioned placement.
# ---------------------------------------------------------------------------
INSTRUCTION_TEMPLATES = [
    "pick up the {obj} and place it at position {idx}",
    "move the {obj} to position {idx}",
    "grasp the {obj} and put it at position {idx}",
    "take the {obj} to target position {idx}",
    "place the {obj} at position {idx}",
]


@dataclass
class ObjectInfo:
    """Metadata about the object being manipulated."""
    object_type: str = "unknown_object"
    object_size: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    object_weight: Optional[float] = None  # kg, optional


class ExperimentRunner:
    """
    Runs automated pick-and-place experiments for data collection.

    The object is always picked from a single fixed pickup location and
    placed at one of the predefined place poses.  After each episode the
    operator manually returns the object to the pickup location before the
    next episode begins.

    Args:
        robot:         Initialized UR5Controller instance.
        camera:        Initialized CameraLogger instance.
        collector:     Initialized DataCollector instance.
        poses_file:    Path to the poses.json configuration file.
        object_info:   Metadata about the target object.
        seed:          Random seed for reproducible place-pose selection.
        recording_hz:  Observation capture rate during motion (Hz). Default: 10.
        iterations_per_pose: Number of times each place pose is used.
    """

    def __init__(
        self,
        robot: UR5Controller,
        camera: CameraLogger,
        collector: DataCollector,
        poses_file: str = "config/poses.json",
        object_info: Optional[ObjectInfo] = None,
        seed: Optional[int] = None,
        recording_hz: float = 10.0,
        iterations_per_pose: int = 125,
    ):
        self.robot = robot
        self.camera = camera
        self.collector = collector
        self.object_info = object_info or ObjectInfo()
        self.recording_hz = recording_hz
        self.iterations_per_pose = iterations_per_pose
        self.pause_manager = PauseManager()

        # Load pose configuration
        self._load_poses(poses_file)

        # Random number generator for place pose selection
        self._rng = random.Random(seed)

        # Track which place poses have been used (for logging/analysis)
        self._place_pose_history: List[int] = []

        # The pre-generated shuffled schedule of place pose indices
        self._schedule: List[int] = []
        self._schedule_idx: int = 0
        


    def _load_poses(self, poses_file: str) -> None:
        """
        Load all predefined poses from the JSON configuration file.

        Expected keys in poses.json:
            - home_pose:              [x, y, z, rx, ry, rz]
            - pickup_grasp_pose:      [x, y, z, rx, ry, rz]
            - approach_height_offset: float (meters above grasp/place for approach)
            - place_poses:            list of [x, y, z, rx, ry, rz] poses
            - move_parameters:        {velocity, acceleration, blend_radius}

        Approach poses are NOT stored in the config — they are computed
        automatically at runtime by adding approach_height_offset to the
        grasp/place Z coordinate.
        """
        poses_path = os.path.abspath(poses_file)
        if not os.path.exists(poses_path):
            raise FileNotFoundError(
                f"Poses configuration not found: {poses_path}\n"
                "Create config/poses.json with home_pose, pickup_grasp_pose, "
                "and place_poses."
            )

        with open(poses_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Parse poses
        self.home_pose = Pose.from_list(config["home_pose"])

        # Fixed pickup location — used for every episode
        if "pickup_grasp_pose" in config:
            self.pickup_grasp_pose = Pose.from_list(config["pickup_grasp_pose"])
        else:
            self.pickup_grasp_pose = Pose.from_list(config["pick_pose"]["grasp"])

        # Approach height used to compute approach poses automatically
        self.approach_height_offset = config.get("approach_height_offset", 0.12)

        # Parse place poses — the code uses len(place_poses) everywhere,
        # so any number of place poses is supported.
        raw_places = config["place_poses"]
        if len(raw_places) < 1:
            raise ValueError("place_poses must contain at least 1 pose.")
        self.place_poses: List[Pose] = [Pose.from_list(p) for p in raw_places]
        self._pose_iteration_counts = {i: 0 for i in range(len(self.place_poses))}

        # Parse move parameters
        move_params = config.get("move_parameters", {})
        if move_params:
            self.robot.velocity = move_params.get("velocity", self.robot.velocity)
            self.robot.acceleration = move_params.get("acceleration", self.robot.acceleration)
            self.robot.blend_radius = move_params.get("blend_radius", self.robot.blend_radius)

        logger.info(
            "Loaded %d place poses from %s", len(self.place_poses), poses_path
        )

    # -------------------------------------------------------------------------
    # Approach-Pose Computation
    # -------------------------------------------------------------------------

    def _approach_pose_for(self, grasp_or_place_pose: Pose) -> Pose:
        """Compute the approach pose by raising Z by approach_height_offset.

        This is used for both the pickup approach and the place approach,
        so approach poses are never stored — only computed on the fly.
        """
        return grasp_or_place_pose.with_z(
            grasp_or_place_pose.z + self.approach_height_offset
        )

    # -------------------------------------------------------------------------
    # Place Pose Selection
    # -------------------------------------------------------------------------

    def _select_place_pose(self) -> int:
        """
        Select the next place pose index from the pre-generated schedule.
        """
        idx = self._schedule[self._schedule_idx]
        self._schedule_idx += 1
        return idx

    def _build_schedule(self) -> None:
        """
        Build a balanced sequence where every place pose appears exactly
        `iterations_per_pose` times, then randomly shuffle it.
        """
        n_poses = len(self.place_poses)
        schedule = list(range(n_poses)) * self.iterations_per_pose
        self._rng.shuffle(schedule)
        self._schedule = schedule
        self._schedule_idx = 0

        logger.info(
            "Built exact schedule: %d total episodes (%d poses × %d iterations each)",
            len(schedule), n_poses, self.iterations_per_pose
        )

    # -------------------------------------------------------------------------
    # Observation Recording Helpers
    # -------------------------------------------------------------------------

    def _record_observation(self) -> None:
        """Capture an image and robot state, and save as a timestep."""
        image = self.camera.capture_frame()
        state = self.robot.get_full_state()
        self.collector.record_timestep(state, image, self.camera)

    def _move_and_record(self, pose: Pose, label: str = "") -> None:
        """
        Move to a target pose while continuously recording observations.

        Starts an async moveL, then captures observations at self.recording_hz
        until the motion completes. Records one final observation at the
        waypoint after the move finishes.

        Args:
            pose:  Target Cartesian pose.
            label: Optional log label (e.g., "PICK: approach").
        """
        if label:
            logger.info("%s → [%.3f, %.3f, %.3f]", label, pose.x, pose.y, pose.z)

        self.pause_manager.check_pause()
        period = 1.0 / self.recording_hz

        # Start the non-blocking move
        self.robot.move_to_async(pose)

        # Record observations while the robot is moving
        while self.robot.is_moving():
            self._record_observation()
            time.sleep(period)

        # Record final observation at the waypoint
        self._record_observation()
        self.pause_manager.check_pause()

    # -------------------------------------------------------------------------
    # Language Instruction Generation
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_instruction(object_type: str, place_idx: int) -> str:
        """
        Generate a natural language instruction for this episode.

        Every instruction explicitly includes the 1-based target position
        index, which is critical for OpenVLA to learn position-conditioned
        placement.

        Args:
            object_type: Name of the object (e.g., "bottle").
            place_idx:   0-based index of the target place position.

        Returns:
            Natural language instruction string.
        """
        # Use 1-based position numbers in human-readable instructions
        display_idx = place_idx + 1
        template = INSTRUCTION_TEMPLATES[place_idx % len(INSTRUCTION_TEMPLATES)]
        return template.format(obj=object_type, idx=display_idx)

    # -------------------------------------------------------------------------
    # Pick-and-Place Primitives
    # -------------------------------------------------------------------------

    def _execute_pick(self) -> None:
        """
        Execute the pick sequence from the fixed pickup location.

        Approach pose is computed automatically from the pickup grasp pose
        and the configured approach_height_offset.

        All moves use continuous recording for dense trajectory data.

        Steps:
            1. Move to pickup approach pose   (recording continuously)
            2. Move down to pickup grasp pose  (recording continuously)
            3. Close gripper                   (record single observation)
            4. Move back up to approach pose   (recording continuously)
        """
        # Compute approach pose automatically
        approach_pose = self._approach_pose_for(self.pickup_grasp_pose)

        self._move_and_record(approach_pose, "PICK: approach above pickup")
        self._move_and_record(self.pickup_grasp_pose, "PICK: move down to grasp")

        logger.info("PICK: Closing gripper")
        self.robot.close_gripper()
        self._record_observation()
        self.pause_manager.check_pause()

        self._move_and_record(approach_pose, "PICK: lifting object")

    def _execute_place(self, place_pose: Pose) -> None:
        """
        Execute the place sequence at the given target pose.

        Approach pose is computed automatically from the place pose and
        the configured approach_height_offset.

        All moves use continuous recording for dense trajectory data.

        Steps:
            1. Move to place approach pose    (recording continuously)
            2. Move down to place pose        (recording continuously)
            3. Open gripper                   (record single observation)
            4. Move back up to approach pose  (recording continuously)
        """
        # Compute approach pose automatically
        approach_pose = self._approach_pose_for(place_pose)

        self._move_and_record(approach_pose, "PLACE: approach above target")
        self._move_and_record(place_pose, "PLACE: move down to place")

        logger.info("PLACE: Opening gripper")
        self.robot.open_gripper()
        self._record_observation()
        self.pause_manager.check_pause()

        self._move_and_record(approach_pose, "PLACE: retracting")

    def _execute_unrecorded_reset(self, place_pose: Pose) -> None:
        """
        Execute an unrecorded reset sequence after the episode finishes.
        
        Moves to the final place pose, picks up the object, returns it to the
        original pickup position, releases it, and returns home.
        No observations (images, robot states, or actions) are recorded.
        """
        logger.info("RESET: Starting unrecorded reset sequence to return object to pickup location ...")
        
        # 1. Approach place pose and pick object
        place_approach = self._approach_pose_for(place_pose)
        logger.info("RESET: Approaching place pose")
        self.robot.move_to(place_approach)
        self.pause_manager.check_pause()
        
        logger.info("RESET: Moving down to grasp object at place pose")
        self.robot.move_to(place_pose)
        self.pause_manager.check_pause()
        
        logger.info("RESET: Closing gripper")
        self.robot.close_gripper()
        self.pause_manager.check_pause()
        
        logger.info("RESET: Lifting object")
        self.robot.move_to(place_approach)
        self.pause_manager.check_pause()
        
        # 2. Return to original pickup pose and place object back
        pickup_approach = self._approach_pose_for(self.pickup_grasp_pose)
        logger.info("RESET: Approaching original pickup pose")
        self.robot.move_to(pickup_approach)
        self.pause_manager.check_pause()
        
        logger.info("RESET: Moving down to original pickup pose")
        self.robot.move_to(self.pickup_grasp_pose)
        self.pause_manager.check_pause()
        
        logger.info("RESET: Opening gripper (releasing object)")
        self.robot.open_gripper()
        self.pause_manager.check_pause()
        
        logger.info("RESET: Retracting from pickup pose")
        self.robot.move_to(pickup_approach)
        self.pause_manager.check_pause()
        
        # 3. Return home
        logger.info("RESET: Returning to home pose")
        self.robot.move_to_home(self.home_pose)
        self.pause_manager.check_pause()
        
        logger.info("RESET: Unrecorded reset sequence complete. Ready for next episode.")

    # -------------------------------------------------------------------------
    # Main Experiment Loop
    # -------------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Execute the full data collection experiment.

        Runs the calculated number of pick-and-place cycles based on iterations_per_pose.
        Each cycle:
            1. Starts from home
            2. Picks from the fixed pickup location
            3. Places at the scheduled pose
            4. Returns to home
            5. Automatically executes an unrecorded reset sequence to return the object to the pickup location

        Returns:
            Summary dictionary with collection statistics.
        """
        num_episodes = len(self.place_poses) * self.iterations_per_pose
        
        logger.info(
            "═══════════════════════════════════════════════════════════"
        )
        logger.info(
            "Starting data collection: %d episodes total", num_episodes
        )
        logger.info(
            "Object: %s | Size: %s | Weight: %s",
            self.object_info.object_type,
            self.object_info.object_size,
            self.object_info.object_weight or "N/A"
        )
        logger.info(
            "Pickup pose: [%.3f, %.3f, %.3f]",
            self.pickup_grasp_pose.x,
            self.pickup_grasp_pose.y,
            self.pickup_grasp_pose.z,
        )
        logger.info(
            "Place poses: %d | Iterations per pose: %d | Recording: %.0f Hz",
            len(self.place_poses),
            self.iterations_per_pose,
            self.recording_hz,
        )
        logger.info(
            "═══════════════════════════════════════════════════════════"
        )

        # Build exact schedule
        self._build_schedule()

        # Initialize the data collector with experiment metadata
        experiment_meta = {
            "num_planned_episodes": num_episodes,
            "object_type": self.object_info.object_type,
            "object_size": self.object_info.object_size,
            "object_weight": self.object_info.object_weight,
            "num_place_poses": len(self.place_poses),
            "home_pose": self.home_pose.to_list(),
            "pickup_grasp_pose": self.pickup_grasp_pose.to_list(),
            "approach_height_offset": self.approach_height_offset,
            "dry_run": self.robot.dry_run,
            "recording_hz": self.recording_hz,
            "iterations_per_pose": self.iterations_per_pose,
            "camera": self.camera.get_metadata(),
        }
        self.collector.initialize(experiment_meta)
        self.pause_manager.start()

        try:
            for ep_idx in range(num_episodes):
                self._run_single_episode(ep_idx, num_episodes)

        except (KeyboardInterrupt, QuitRequestedException) as e:
            logger.warning(
                "Data collection terminated by user after %d completed episodes (%s).",
                self.collector.total_episodes, str(e) or "KeyboardInterrupt"
            )
        except Exception as e:
            logger.error("Data collection failed: %s", e, exc_info=True)
            raise
        finally:
            self.pause_manager.stop()
            # Always try to go home and print summary
            try:
                self.robot.move_to_home(self.home_pose)
            except Exception:
                pass

        summary = self.collector.get_summary()
        logger.info(
            "═══════════════════════════════════════════════════════════"
        )
        logger.info("Data collection complete!")
        logger.info("  Episodes:  %d", summary["total_episodes"])
        logger.info("  Steps:     %d", summary["total_steps"])
        logger.info("  Avg steps: %.1f per episode", summary["avg_steps_per_episode"])
        logger.info("  Saved to:  %s", summary["experiment_dir"])
        
        logger.info("  ")
        logger.info("Final Execution Summary:")
        for idx in range(len(self.place_poses)):
            logger.info("  Position %d was executed %d / %d times.", 
                        idx + 1, self._pose_iteration_counts[idx], self.iterations_per_pose)
        
        logger.info(
            "═══════════════════════════════════════════════════════════"
        )
        return summary

    def _run_single_episode(self, ep_idx: int, total: int) -> None:
        """
        Execute one complete pick-and-place episode.

        The object is always picked from the fixed pickup location.
        After the episode the operator is prompted to reset the object.

        Args:
            ep_idx: Current episode index (0-based).
            total:  Total number of planned episodes.
        """
        # Select a place pose
        place_idx = self._select_place_pose()
        self._place_pose_history.append(place_idx)
        selected_place_pose = self.place_poses[place_idx]
        
        # Increment iteration count
        self._pose_iteration_counts[place_idx] += 1

        # Generate language instruction (always includes position index)
        language_instruction = self.generate_instruction(
            self.object_info.object_type, place_idx
        )

        logger.info(
            "───────────────────────────────────────────────────────────"
        )
        logger.info(
            "Episode %d / %d",
            ep_idx + 1, total
        )
        logger.info(
            "  Destination position: %d", place_idx + 1
        )
        logger.info(
            "  Iteration count:      %d / %d",
            self._pose_iteration_counts[place_idx], self.iterations_per_pose
        )
        logger.info(
            "  Instruction:          \"%s\"", language_instruction
        )

        # Start episode with metadata
        episode_meta = {
            "pickup_pose": self.pickup_grasp_pose.to_list(),
            "place_pose": selected_place_pose.to_list(),
            "place_pose_index": place_idx,
            "object_type": self.object_info.object_type,
            "object_size": self.object_info.object_size,
            "object_weight": self.object_info.object_weight,
            "language_instruction": language_instruction,
            "recording_hz": self.recording_hz,
        }
        self.collector.start_episode(episode_meta)

        try:
            # 0. Verify gripper is open before picking sequence
            if self.robot.get_gripper_state() != self.robot.GRIPPER_OPEN:
                logger.warning("Gripper was not open. Opening it now for safety.")
                self.robot.open_gripper()

            # 1. Move to home pose (continuous recording)
            self._move_and_record(self.home_pose, "HOME: moving to home pose")

            # 2-5. Pick sequence from the fixed pickup location
            self._execute_pick()

            # 6-9. Place sequence at the selected target
            self._execute_place(selected_place_pose)

            # 10. Return to home (continuous recording)
            self._move_and_record(self.home_pose, "HOME: returning to home")

            # End episode successfully
            self.collector.end_episode(success=True)

        except QuitRequestedException as e:
            logger.warning("Episode %d terminated by operator command: %s", ep_idx, e)
            if self.collector._csv_file is not None:
                self.collector.end_episode(
                    success=False,
                    extra_metadata={"error": "Terminated by user via Q command"},
                )
            raise
        except Exception as e:
            logger.error("Episode %d failed: %s", ep_idx, e)
            self.collector.end_episode(
                success=False,
                extra_metadata={"error": str(e)},
            )
            # Re-raise to allow the caller to decide whether to continue
            raise

        # 11. Automatically execute unrecorded reset sequence to return the object
        #     to the original pickup position before the next episode begins.
        self._execute_unrecorded_reset(selected_place_pose)

    @property
    def place_pose_history(self) -> List[int]:
        """Indices of place poses selected in each episode."""
        return self._place_pose_history.copy()
