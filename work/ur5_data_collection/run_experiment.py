#!/usr/bin/env python3
"""
UR5 Pick-and-Place Data Collection — Entry Point
=================================================
Runs automated pick-and-place experiments on a UR5 robot to collect
demonstration data for OpenVLA fine-tuning.

Usage:
    # Dry run (no hardware required)
    python run_experiment.py --dry-run --iterations-per-pose 5

    # Real robot
    python run_experiment.py --robot-ip 192.168.1.100 --camera-id 0 --iterations-per-pose 50

    # With object metadata
    python run_experiment.py --dry-run \
        --object-type "red_cube" \
        --object-size "0.05,0.05,0.05" \
        --object-weight 0.15 \
        --iterations-per-pose 10

    # Custom output directory and seed
    python run_experiment.py --dry-run \
        --output-dir ./my_data \
        --seed 42 \
        --iterations-per-pose 20
"""

import argparse
import logging
import sys
import os
import time

from src.ur5_controller import UR5Controller
from src.camera_logger import CameraLogger
from src.data_collector import DataCollector
from src.experiment_runner import ExperimentRunner, ObjectInfo


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with a clean, informative format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)-20s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("cv2").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="UR5 Pick-and-Place Data Collection for OpenVLA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Hardware
    parser.add_argument(
        "--robot-ip", type=str, default="192.168.1.100",
        help="IP address of the UR5 controller (default: 192.168.1.100)",
    )
    parser.add_argument(
        "--camera-id", type=str, default="realsense",
        help="Camera device to use: 'realsense' (default) or an integer index (e.g. '0') for USB webcams.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate robot and camera without hardware",
    )

    # Arduino / Gripper
    parser.add_argument(
        "--arduino-port", type=str, default="/dev/ttyACM0",
        help="Serial port for the Arduino gripper controller (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--arduino-baudrate", type=int, default=115200,
        help="Baud rate for the Arduino serial connection (default: 115200)",
    )
    # Experiment
    parser.add_argument(
        "--iterations-per-pose", type=int, required=True,
        help="Number of times to execute pick-and-place for EACH predefined target position",
    )
    parser.add_argument(
        "--poses-file", type=str, default="config/poses.json",
        help="Path to the poses configuration JSON (default: config/poses.json)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Root directory for saving collected data (default: data/)",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Optional name prefix for the experiment folder",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible place-pose selection",
    )
    parser.add_argument(
        "--recording-hz", type=float, default=7.0,
        help="Observation capture rate during motion in Hz (default: 10.0)",
    )

    # Object metadata
    parser.add_argument(
        "--object-type", type=str, default="white box",
        help="Type/name of the target object (e.g., 'red_cube')",
    )
    parser.add_argument(
        "--object-size", type=str, default="0.05,0.05,0.05",
        help="Object dimensions as 'w,h,d' in meters (default: 0.05,0.05,0.05)",
    )
    parser.add_argument(
        "--object-weight", type=float, default=None,
        help="Object weight in kg (optional)",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the data collection experiment."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("main")

    # ── Parse object metadata ──────────────────────────────────────────────
    object_size = [float(x.strip()) for x in args.object_size.split(",")]
    if len(object_size) != 3:
        logger.error("--object-size must have 3 comma-separated values (w,h,d).")
        sys.exit(1)

    object_info = ObjectInfo(
        object_type=args.object_type,
        object_size=object_size,
        object_weight=args.object_weight,
    )

    # ── Initialize components ──────────────────────────────────────────────
    logger.info("Initializing UR5 controller ...")
    robot = UR5Controller(
        robot_ip=args.robot_ip,
        dry_run=args.dry_run,
        arduino_port=args.arduino_port,
        arduino_baudrate=args.arduino_baudrate,
    )

    logger.info("Initializing camera logger ...")
    camera = CameraLogger(
        camera_id=args.camera_id,
        dry_run=args.dry_run,
    )

    logger.info("Initializing data collector ...")
    collector = DataCollector(
        base_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    # ── Connect to hardware ────────────────────────────────────────────────
    try:
        robot.connect()
        camera.open()
    except Exception as e:
        logger.error("Hardware initialization failed: %s", e)
        sys.exit(1)

    # ── Build and run the experiment ───────────────────────────────────────
    try:
        runner = ExperimentRunner(
            robot=robot,
            camera=camera,
            collector=collector,
            poses_file=args.poses_file,
            object_info=object_info,
            seed=args.seed,
            recording_hz=args.recording_hz,
            iterations_per_pose=args.iterations_per_pose,
        )

        summary = runner.run()

        logger.info("═══ Collection Summary ═══")
        for key, value in summary.items():
            logger.info("  %-25s %s", key, value)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error("Experiment failed: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        logger.info("Cleaning up ...")
        try:
            camera.close()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass

    logger.info("Done.")


if __name__ == "__main__":
    main()
