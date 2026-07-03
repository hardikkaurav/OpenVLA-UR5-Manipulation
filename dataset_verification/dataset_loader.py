"""Berkeley UR5 Dataset Loader for OpenVLA Verification.

Loads trajectories from the berkeley_autolab_ur5 RLDS dataset (via TensorFlow
Datasets) and provides them as simple Python objects for evaluation.

Supports two data sources:
  1. RLDS / TensorFlow Datasets (default): ``tfds.load('berkeley_autolab_ur5')``
  2. Local ``.npy`` files: ``np.load(path, allow_pickle=True).item()``

No existing OpenVLA files are modified.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Timestep:
    """A single timestep extracted from a trajectory."""

    image: Image.Image                # workspace camera RGB
    hand_image: Image.Image | None    # hand / wrist camera RGB (may be absent)
    robot_state: np.ndarray | None    # (15,) float32 state vector
    action: np.ndarray                # (7,) ground-truth action
    instruction: str                  # natural-language task description
    index: int = 0


@dataclass
class TrajectoryInfo:
    """Metadata printed by ``inspect_trajectory``."""

    num_timesteps: int = 0
    image_shape: tuple = ()
    hand_image_shape: tuple = ()
    action_dim: int = 0
    robot_state_dim: int = 0
    instruction: str = ""
    keys: list[str] = field(default_factory=list)


# ── RLDS / TFDS loader ───────────────────────────────────────────────────────

def _load_rlds(split: str = "train", max_trajectories: int | None = None):
    """Yield episodes from ``berkeley_autolab_ur5`` via tensorflow_datasets."""

    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf
    except ImportError:
        print(
            "ERROR: tensorflow and tensorflow_datasets are required.\n"
            "  pip install tensorflow tensorflow_datasets",
            file=sys.stderr,
        )
        raise

    builder = tfds.builder("berkeley_autolab_ur5")
    builder.download_and_prepare()
    ds = builder.as_dataset(split=split)

    count = 0
    for episode in ds:
        if max_trajectories is not None and count >= max_trajectories:
            break
        yield episode
        count += 1


def _episode_to_timesteps(episode) -> list[Timestep]:
    """Convert one RLDS episode dict into a list of ``Timestep`` objects."""

    import tensorflow as tf

    steps = episode["steps"]
    timesteps: list[Timestep] = []

    for idx, step in enumerate(steps):
        obs = step["observation"]

        # Image
        img_np = obs["image"].numpy()  # (480, 640, 3) uint8
        pil_image = Image.fromarray(img_np, mode="RGB")

        # Hand image (may or may not exist)
        hand_image = None
        if "hand_image" in obs:
            hand_np = obs["hand_image"].numpy()
            hand_image = Image.fromarray(hand_np, mode="RGB")

        # Robot state
        robot_state = None
        if "robot_state" in obs:
            robot_state = obs["robot_state"].numpy()

        # Action — assemble the 7-DoF vector
        act = step["action"]
        world_vec = act["world_vector"].numpy()        # (3,)
        rot_delta = act["rotation_delta"].numpy()       # (3,)
        gripper = float(act["gripper_closedness_action"].numpy())
        action_7dof = np.concatenate([world_vec, rot_delta, [gripper]]).astype(np.float32)

        # Language instruction
        instruction = ""
        if "natural_language_instruction" in obs:
            raw = obs["natural_language_instruction"]
            instruction = raw.numpy().decode("utf-8") if hasattr(raw, "numpy") else str(raw)
        elif "language_instruction" in step:
            raw = step["language_instruction"]
            instruction = raw.numpy().decode("utf-8") if hasattr(raw, "numpy") else str(raw)

        timesteps.append(Timestep(
            image=pil_image,
            hand_image=hand_image,
            robot_state=robot_state,
            action=action_7dof,
            instruction=instruction,
            index=idx,
        ))

    return timesteps


# ── .npy loader ───────────────────────────────────────────────────────────────

def _load_npy_trajectory(path: str | Path) -> list[Timestep]:
    """Load a single trajectory from a ``.npy`` file.

    Expected structure (after ``np.load(path, allow_pickle=True).item()``):

    * ``image`` or ``images``: (T, H, W, 3) uint8
    * ``actions``: (T, 7) float32
    * ``robot_state`` (optional): (T, D) float32
    * ``language_instruction`` (optional): str

    This is a best-effort loader that inspects available keys and adapts.
    """

    data = np.load(str(path), allow_pickle=True)
    if hasattr(data, "item"):
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict inside the .npy file, got {type(data)}")

    # ── Images ────────────────────────────────────────────────────────────
    images_key = _find_key(data, ["image", "images", "obs", "observations"])
    if images_key is None:
        raise KeyError(f"No image key found. Available keys: {list(data.keys())}")
    images = np.asarray(data[images_key])
    if images.ndim == 3:
        images = images[np.newaxis, ...]  # single image → (1, H, W, 3)

    # ── Third-person / hand image ─────────────────────────────────────────
    hand_key = _find_key(data, ["third_person_image", "hand_image", "wrist_image"])
    hand_images = np.asarray(data[hand_key]) if hand_key else None

    # ── Actions ───────────────────────────────────────────────────────────
    action_key = _find_key(data, ["actions", "action"])
    if action_key is None:
        raise KeyError(f"No action key found. Available keys: {list(data.keys())}")
    actions = np.asarray(data[action_key], dtype=np.float32)

    # ── Robot state ───────────────────────────────────────────────────────
    state_key = _find_key(data, ["robot_state", "state", "states", "robot_states"])
    robot_states = np.asarray(data[state_key], dtype=np.float32) if state_key else None

    # ── Instruction ───────────────────────────────────────────────────────
    instr_key = _find_key(data, [
        "language_instruction", "natural_language_instruction", "instruction", "task",
    ])
    instruction = ""
    if instr_key:
        raw = data[instr_key]
        if isinstance(raw, np.ndarray):
            # Task field may be shape (T, 1) or (T,) — extract first element
            flat = raw.flat[0]
            instruction = flat.decode("utf-8") if isinstance(flat, bytes) else str(flat)
        elif isinstance(raw, bytes):
            instruction = raw.decode("utf-8")
        else:
            instruction = str(raw)

    T = min(len(images), len(actions))
    timesteps: list[Timestep] = []
    for i in range(T):
        pil_img = Image.fromarray(images[i].astype(np.uint8), mode="RGB")
        hand = None
        if hand_images is not None and i < len(hand_images):
            hand = Image.fromarray(hand_images[i].astype(np.uint8), mode="RGB")
        state = robot_states[i] if robot_states is not None and i < len(robot_states) else None
        timesteps.append(Timestep(
            image=pil_img,
            hand_image=hand,
            robot_state=state,
            action=actions[i],
            instruction=instruction,
            index=i,
        ))

    return timesteps


def _find_key(d: dict, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in d:
            return k
    return None


# ── Inspection ────────────────────────────────────────────────────────────────

def inspect_trajectory(timesteps: list[Timestep]) -> TrajectoryInfo:
    """Gather metadata from a loaded trajectory and print a summary."""

    info = TrajectoryInfo()
    if not timesteps:
        return info

    info.num_timesteps = len(timesteps)
    first = timesteps[0]
    info.image_shape = (first.image.width, first.image.height)
    if first.hand_image:
        info.hand_image_shape = (first.hand_image.width, first.hand_image.height)
    info.action_dim = first.action.shape[0] if first.action is not None else 0
    info.robot_state_dim = first.robot_state.shape[0] if first.robot_state is not None else 0
    info.instruction = first.instruction

    print(f"\n{'='*60}")
    print("TRAJECTORY INSPECTION")
    print(f"{'='*60}")
    print(f"  Timesteps:           {info.num_timesteps}")
    print(f"  Image resolution:    {info.image_shape[0]}x{info.image_shape[1]}")
    if info.hand_image_shape:
        print(f"  Hand image res:      {info.hand_image_shape[0]}x{info.hand_image_shape[1]}")
    print(f"  Action dimensions:   {info.action_dim}")
    print(f"  Robot state dims:    {info.robot_state_dim}")
    print(f"  Instruction:         \"{info.instruction}\"")
    print()
    print(f"  First action (GT):   {first.action}")
    if first.robot_state is not None:
        print(f"  First robot state:   {first.robot_state[:6]} ...")
    print(f"{'='*60}\n")

    return info


# ── Public API ────────────────────────────────────────────────────────────────

def load_trajectory(
    source: str | Path | None = None,
    *,
    split: str = "train",
    trajectory_index: int = 0,
) -> list[Timestep]:
    """Load a single trajectory.

    Args:
        source: Path to a ``.npy`` file.  If *None*, the RLDS dataset is used.
        split: RLDS split name (``train`` or ``test``).
        trajectory_index: Which episode to pick from the RLDS dataset.

    Returns:
        A list of ``Timestep`` objects.
    """

    if source is not None:
        p = Path(source)
        if p.suffix == ".npy":
            print(f"Loading trajectory from .npy file: {p}")
            return _load_npy_trajectory(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}. Use .npy or omit to use RLDS.")

    print(f"Loading trajectory #{trajectory_index} from berkeley_autolab_ur5 (split={split}) ...")
    episodes = _load_rlds(split=split, max_trajectories=trajectory_index + 1)
    episode = None
    for i, ep in enumerate(episodes):
        if i == trajectory_index:
            episode = ep
            break
    if episode is None:
        raise IndexError(f"Trajectory index {trajectory_index} not found in split '{split}'.")

    return _episode_to_timesteps(episode)


def load_all_trajectories(
    source_dir: str | Path | None = None,
    *,
    split: str = "train",
    max_trajectories: int | None = None,
) -> Iterator[list[Timestep]]:
    """Yield trajectories one at a time.

    Args:
        source_dir: Directory of ``.npy`` files.  If *None*, the RLDS dataset is used.
        split: RLDS split name.
        max_trajectories: Cap the number of trajectories yielded.
    """

    if source_dir is not None:
        p = Path(source_dir)
        npy_files = sorted(p.glob("*.npy"))
        for i, f in enumerate(npy_files):
            if max_trajectories is not None and i >= max_trajectories:
                break
            yield _load_npy_trajectory(f)
        return

    for ep in _load_rlds(split=split, max_trajectories=max_trajectories):
        yield _episode_to_timesteps(ep)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a Berkeley UR5 trajectory.")
    parser.add_argument("--source", default=None, help="Path to .npy trajectory file. Omit to use RLDS.")
    parser.add_argument("--trajectory-index", type=int, default=0, help="Episode index for RLDS.")
    parser.add_argument("--split", default="train", help="RLDS split (train or test).")
    args = parser.parse_args()

    ts = load_trajectory(args.source, split=args.split, trajectory_index=args.trajectory_index)
    info = inspect_trajectory(ts)

    # Show first image
    if ts:
        print("Displaying first RGB image (close window to continue) ...")
        ts[0].image.show()
        if ts[0].hand_image:
            print("Displaying first hand/third-person image ...")
            ts[0].hand_image.show()
