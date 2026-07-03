"""
Camera Logger
=============
Captures RGB frames from a workspace-mounted camera for data collection.
Supports Intel RealSense cameras (via pyrealsense2) and USB cameras (via OpenCV).
Provides a simulated camera for dry-run testing.

Each captured frame is returned as a numpy array (H, W, 3) in RGB order,
suitable for saving as PNG and later resizing to 224×224 for OpenVLA.
"""

import logging
import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraLogger:
    """
    RGB camera interface for workspace observation.
    
    Captures frames from a RealSense camera or a USB/IP camera using OpenCV.
    The camera should be mounted with a clear view of the pick-and-place
    workspace so that the robot arm, gripper, and objects are all visible.
    
    Attributes:
        camera_id:   'realsense' (default) or OpenCV camera index (int for USB).
        resolution:  Desired capture resolution (width, height).
        dry_run:     If True, generates synthetic frames instead of capturing.
    """

    # Default resolution for OpenCV webcams
    DEFAULT_RESOLUTION = (640, 480)

    def __init__(
        self,
        camera_id: Union[str, int] = "realsense",
        resolution: Optional[Tuple[int, int]] = None,
        dry_run: bool = False,
    ):
        self.camera_id = camera_id
        self.dry_run = dry_run
        
        # Set default resolution based on camera type
        if str(camera_id).lower() == "realsense":
            self.resolution = resolution or (1280, 720)
        else:
            self.resolution = resolution or self.DEFAULT_RESOLUTION

        self._capture = None
        self._pipeline = None
        self._frame_count: int = 0
        self._is_open: bool = False

    def open(self) -> None:
        """
        Open the camera device and configure resolution.
        
        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Simulating camera open (id=%s).", self.camera_id)
            self._is_open = True
            return

        if str(self.camera_id).lower() == "realsense":
            logger.info("Opening Intel RealSense camera ...")
            try:
                import pyrealsense2 as rs
            except ImportError as e:
                raise RuntimeError("pyrealsense2 is not installed. Please install it to use RealSense.") from e

            context = rs.context()
            if len(context.devices) == 0:
                raise RuntimeError("No Intel RealSense camera connected.")
            
            self._pipeline = rs.pipeline()
            
            profiles_to_try = [
                (640, 480, 30),
                (1280, 720, 15),
                (424, 240, 60),
            ]
            
            started = False
            for w, h, fps in profiles_to_try:
                config = rs.config()
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                try:
                    self._pipeline.start(config)
                    self.resolution = (w, h)
                    logger.info("RealSense camera opened. Resolution: %d×%d @ %d FPS (BGR8)", w, h, fps)
                    started = True
                    break
                except Exception:
                    # Profile not supported or failed to start, try next
                    continue
            
            if not started:
                raise RuntimeError("Failed to start RealSense pipeline with any of the preferred fallback profiles.")
            
            self._is_open = True
            return

        # Fallback to OpenCV VideoCapture
        try:
            cam_idx = int(self.camera_id)
        except ValueError:
            raise ValueError(f"Invalid camera_id '{self.camera_id}'. Must be 'realsense' or an integer.")

        logger.info("Opening OpenCV camera %s ...", cam_idx)
        self._capture = cv2.VideoCapture(cam_idx)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"Failed to open camera with id={cam_idx}. "
                "Check that the camera is connected and not in use."
            )

        # Set resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Verify actual resolution
        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera opened. Resolution: %d×%d", actual_w, actual_h)

        self._is_open = True

    def close(self) -> None:
        """Release the camera device."""
        if self.dry_run:
            logger.info("[DRY RUN] Simulating camera close.")
            self._is_open = False
            return

        if str(self.camera_id).lower() == "realsense":
            if self._pipeline is not None:
                self._pipeline.stop()
                logger.info("RealSense camera released.")
                self._pipeline = None
        else:
            if self._capture is not None:
                self._capture.release()
                logger.info("Camera released.")
                self._capture = None

        self._is_open = False

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single RGB frame from the camera.
        
        Returns:
            numpy array of shape (H, W, 3) in RGB color order.
            
        Raises:
            RuntimeError: If the camera is not open or the frame cannot be read.
        """
        if not self._is_open:
            raise RuntimeError("Camera is not open. Call open() first.")

        if self.dry_run:
            frame = self._generate_synthetic_frame()
            self._frame_count += 1
            return frame

        if str(self.camera_id).lower() == "realsense":
            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to capture color frame from RealSense.")
            
            # Format is BGR8, so convert to RGB to maintain API compatibility
            frame_bgr = np.asanyarray(color_frame.get_data())
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._frame_count += 1
            return frame_rgb
        else:
            ret, frame_bgr = self._capture.read()
            if not ret or frame_bgr is None:
                raise RuntimeError(
                    "Failed to capture frame from camera. "
                    "The device may have been disconnected."
                )

            # Convert BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._frame_count += 1
            return frame_rgb

    def save_frame(self, frame: np.ndarray, filepath: str) -> str:
        """
        Save an RGB frame to disk as PNG.
        
        Args:
            frame:    RGB numpy array (H, W, 3).
            filepath: Destination path (should end in .png).
            
        Returns:
            The absolute path where the image was saved.
        """
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert RGB back to BGR for OpenCV's imwrite
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(filepath, frame_bgr)

        if not success:
            raise IOError(f"Failed to save image to {filepath}")

        logger.debug("Saved frame to %s", filepath)
        return os.path.abspath(filepath)

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generate a synthetic RGB frame for dry-run testing.
        
        Creates a visually distinct frame with a gradient background, a
        simulated workspace grid, and a frame counter so you can verify
        that different timesteps produce different images.
        """
        w, h = self.resolution
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Gradient background (dark blue to teal)
        for row in range(h):
            t = row / h
            frame[row, :, 0] = int(20 + 30 * t)    # R
            frame[row, :, 1] = int(40 + 80 * t)    # G
            frame[row, :, 2] = int(80 + 100 * t)   # B

        # Grid lines (simulated workspace)
        grid_spacing = 60
        for x in range(0, w, grid_spacing):
            frame[:, max(0, x):x+1, :] = [60, 90, 120]
        for y in range(0, h, grid_spacing):
            frame[max(0, y):y+1, :, :] = [60, 90, 120]

        # Simulated object (colored rectangle that moves slightly per frame)
        obj_x = 200 + (self._frame_count * 7) % 200
        obj_y = 150 + (self._frame_count * 3) % 150
        obj_w, obj_h = 40, 40
        frame[obj_y:obj_y+obj_h, obj_x:obj_x+obj_w] = [220, 80, 80]  # Red cube

        # Add frame counter text using OpenCV
        cv2.putText(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            f"DRY RUN | Frame {self._frame_count:05d}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        # We drew on BGR, so re-read the array (putText modifies in-place on BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame_bgr,
            f"DRY RUN | Frame {self._frame_count:05d}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        return frame

    def get_metadata(self) -> dict:
        """
        Return camera configuration metadata for experiment-level storage.
        
        This metadata is saved once in the experiment's metadata.json,
        future-proofing the dataset for later processing or replication.
        
        Returns:
            Dictionary with image_width, image_height, camera_id, image_format.
        """
        return {
            "image_width": self.resolution[0],
            "image_height": self.resolution[1],
            "camera_id": str(self.camera_id),
            "image_format": "png",
        }

    @property
    def frame_count(self) -> int:
        """Number of frames captured so far."""
        return self._frame_count

    @property
    def is_open(self) -> bool:
        """Whether the camera device is currently open."""
        return self._is_open
