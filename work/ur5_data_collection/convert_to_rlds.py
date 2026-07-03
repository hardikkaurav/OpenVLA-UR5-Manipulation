#!/usr/bin/env python3
"""
RLDS Conversion Utility
=======================
Converts collected pick-and-place demonstration data into a format ready for
RLDS (Reinforcement Learning Datasets) dataset building. This script:

    1. Reads episode directories produced by the DataCollector
    2. Resizes images to 224×224 (OpenVLA input resolution)
    3. Computes delta actions between consecutive states
    4. Generates natural language instructions from episode metadata
    5. Outputs a unified JSON manifest per episode for TFDS dataset building

The output can be directly consumed by a TensorFlow Datasets (TFDS) builder
to create the final RLDS dataset for OpenVLA fine-tuning.

Usage:
    python convert_to_rlds.py --experiment-dir data/experiment_20260622_170000 \\
                              --output-dir rlds_output \\
                              --image-size 224
"""

import argparse
import csv
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Core state vector order: [x, y, z, rx, ry, rz, gripper]
STATE_KEYS = [
    "ee_pos_x", "ee_pos_y", "ee_pos_z",
    "ee_rot_rx", "ee_rot_ry", "ee_rot_rz",
    "gripper_state",
]

# Extended state includes joint angles (auto-detected for backward compat)
JOINT_KEYS = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]


def generate_language_instruction(episode_meta: Dict[str, Any]) -> str:
    """
    Generate a natural language instruction from episode metadata.
    
    Every instruction explicitly includes the target position index,
    which is critical for OpenVLA to learn position-conditioned placement.
    
    Args:
        episode_meta: Episode metadata dict from episode_metadata.json.
        
    Returns:
        Natural language instruction string.
    """
    obj_type = episode_meta.get("object_type", "object")
    place_idx = episode_meta.get("place_pose_index",
                                 episode_meta.get("target_position_id", "unknown"))

    # All templates explicitly include the target position index
    templates = [
        f"pick up the {obj_type} and place it at position {place_idx}",
        f"move the {obj_type} to position {place_idx}",
        f"grasp the {obj_type} and put it at position {place_idx}",
        f"take the {obj_type} to target position {place_idx}",
        f"place the {obj_type} at position {place_idx}",
    ]

    # Use place_idx to deterministically select a template for consistency
    if isinstance(place_idx, int):
        return templates[place_idx % len(templates)]
    return templates[0]


def parse_state_vector(row: Dict[str, str], include_joints: bool = False) -> np.ndarray:
    """Extract the state vector from a CSV row.
    
    Args:
        row: CSV row as a dict.
        include_joints: If True and joint columns exist, returns 13-dim state.
                        Otherwise returns 7-dim core state.
    
    Returns:
        State vector as numpy array (7-dim or 13-dim).
    """
    keys = STATE_KEYS[:]
    if include_joints:
        keys.extend(JOINT_KEYS)
    return np.array([float(row[k]) for k in keys if k in row], dtype=np.float64)


def compute_delta_actions(states: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute delta actions between consecutive states.
    
    action[t] = state[t+1] - state[t]
    The last timestep gets a zero action (episode termination).
    
    Returns:
        List of 7-DoF action vectors.
    """
    actions = []
    for i in range(len(states) - 1):
        delta = states[i + 1] - states[i]
        actions.append(delta)
    # Terminal action is zeros
    actions.append(np.zeros_like(states[0]))
    return actions


def resize_image(image_path: str, output_path: str, size: int = 224) -> None:
    """Resize an image to (size × size) and save as PNG."""
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Failed to read image: {image_path}")
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)


def convert_episode(
    episode_dir: str,
    output_dir: str,
    image_size: int = 224,
) -> Optional[Dict[str, Any]]:
    """
    Convert a single episode to RLDS-ready format.
    
    Args:
        episode_dir: Path to the episode directory.
        output_dir:  Path to write converted output.
        image_size:  Target image dimension (square).
        
    Returns:
        Episode manifest dict, or None if conversion failed.
    """
    episode_name = os.path.basename(episode_dir)
    logger.info("Converting %s ...", episode_name)

    # Read episode metadata
    meta_path = os.path.join(episode_dir, "episode_metadata.json")
    if not os.path.exists(meta_path):
        logger.warning("Skipping %s: no episode_metadata.json", episode_name)
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        episode_meta = json.load(f)

    # Skip failed episodes
    if not episode_meta.get("success", True):
        logger.warning("Skipping %s: episode marked as failed", episode_name)
        return None

    # Read timestep CSV
    csv_path = os.path.join(episode_dir, "timesteps.csv")
    if not os.path.exists(csv_path):
        logger.warning("Skipping %s: no timesteps.csv", episode_name)
        return None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 2:
        logger.warning("Skipping %s: too few timesteps (%d)", episode_name, len(rows))
        return None

    # Create output directory for this episode
    ep_output_dir = os.path.join(output_dir, episode_name)
    ep_images_dir = os.path.join(ep_output_dir, "images")
    os.makedirs(ep_images_dir, exist_ok=True)

    # Auto-detect whether this dataset has joint columns
    has_joints = all(k in rows[0] for k in JOINT_KEYS) if rows else False
    if has_joints:
        logger.info("  Joint columns detected — including in state vector.")
    
    # Determine state/action dimensions
    effective_state_keys = STATE_KEYS + (JOINT_KEYS if has_joints else [])
    state_dim = len(effective_state_keys)

    # Process each timestep
    states = []
    steps_manifest = []

    for i, row in enumerate(rows):
        # Parse state
        state = parse_state_vector(row, include_joints=has_joints)
        states.append(state)

        # Resize and copy image
        src_image = os.path.join(episode_dir, row["image_filename"])
        dst_image = os.path.join(ep_images_dir, f"step_{i:04d}.png")

        if os.path.exists(src_image):
            resize_image(src_image, dst_image, size=image_size)
        else:
            logger.warning("Image not found: %s", src_image)
            # Create a placeholder black image
            placeholder = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            cv2.imwrite(dst_image, placeholder)

        step_entry = {
            "step_id": i,
            "image_path": os.path.relpath(dst_image, ep_output_dir),
            "state": state.tolist(),
        }
        if has_joints:
            step_entry["joint_state"] = [float(row[k]) for k in JOINT_KEYS]
        steps_manifest.append(step_entry)

    # Compute delta actions
    actions = compute_delta_actions(states)
    for i, step in enumerate(steps_manifest):
        step["action"] = actions[i].tolist()
        step["is_terminal"] = (i == len(steps_manifest) - 1)
        step["is_first"] = (i == 0)

    # Generate language instruction
    language_instruction = generate_language_instruction(episode_meta)

    # Build episode manifest
    manifest = {
        "episode_id": episode_meta.get("episode_id", 0),
        "language_instruction": language_instruction,
        "num_steps": len(steps_manifest),
        "object_type": episode_meta.get("object_type", "unknown"),
        "pickup_pose": episode_meta.get("pickup_pose"),
        "place_pose": episode_meta.get("place_pose"),
        "target_position_id": episode_meta.get("place_pose_index",
                                                episode_meta.get("target_position_id")),
        "state_dim": state_dim,
        "action_dim": state_dim,
        "has_joint_state": has_joints,
        "steps": steps_manifest,
    }

    # Save episode manifest
    manifest_path = os.path.join(ep_output_dir, "episode_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "  ✓ %s: %d steps, instruction='%s'",
        episode_name, len(steps_manifest), language_instruction
    )
    return manifest


def main() -> None:
    """Main conversion entry point."""
    parser = argparse.ArgumentParser(
        description="Convert collected data to RLDS-ready format for OpenVLA"
    )
    parser.add_argument(
        "--experiment-dir", type=str, required=True,
        help="Path to the experiment directory (e.g., data/experiment_20260622_...)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="rlds_output",
        help="Output directory for RLDS-ready data (default: rlds_output/)"
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Target image size (square, default: 224)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    experiment_dir = os.path.abspath(args.experiment_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(experiment_dir):
        logger.error("Experiment directory not found: %s", experiment_dir)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Find all episode directories
    episode_dirs = sorted([
        os.path.join(experiment_dir, d)
        for d in os.listdir(experiment_dir)
        if d.startswith("episode_") and os.path.isdir(os.path.join(experiment_dir, d))
    ])

    if not episode_dirs:
        logger.error("No episode directories found in %s", experiment_dir)
        sys.exit(1)

    logger.info("Found %d episodes in %s", len(episode_dirs), experiment_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Image size: %d×%d", args.image_size, args.image_size)

    # Convert each episode
    manifests = []
    for ep_dir in episode_dirs:
        manifest = convert_episode(ep_dir, output_dir, args.image_size)
        if manifest is not None:
            manifests.append(manifest)

    # Auto-detect dimensions from first manifest
    detected_state_dim = manifests[0]["state_dim"] if manifests else 7
    detected_has_joints = manifests[0].get("has_joint_state", False) if manifests else False

    # Save dataset-level manifest
    dataset_manifest = {
        "num_episodes": len(manifests),
        "total_steps": sum(m["num_steps"] for m in manifests),
        "image_size": args.image_size,
        "state_dim": detected_state_dim,
        "action_dim": detected_state_dim,
        "state_keys": STATE_KEYS + (JOINT_KEYS if detected_has_joints else []),
        "has_joint_state": detected_has_joints,
        "source_experiment": experiment_dir,
        "episodes": [
            {
                "episode_id": m["episode_id"],
                "num_steps": m["num_steps"],
                "language_instruction": m["language_instruction"],
                "target_position_id": m.get("target_position_id"),
            }
            for m in manifests
        ],
    }

    dataset_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    logger.info("═══════════════════════════════════════════════════════════")
    logger.info("RLDS conversion complete!")
    logger.info("  Episodes converted: %d", len(manifests))
    logger.info("  Total steps:        %d", dataset_manifest["total_steps"])
    logger.info("  Dataset manifest:   %s", dataset_path)
    logger.info("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
