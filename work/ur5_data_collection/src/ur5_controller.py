"""
UR5 Robot Controller
====================
Provides a clean interface to the UR5 robot via the RTDE (Real-Time Data Exchange)
protocol. Uses moveL (linear Cartesian moves) exclusively for smooth, predictable
trajectories ideal for data collection.

The gripper is a custom servo driven by an Arduino Uno connected over USB serial.
The Arduino expects an integer servo angle followed by a newline (e.g. ``"118\n"``).
Gripper open/close angles are configurable via constructor arguments.

Supports both blocking and non-blocking (async) moveL commands. The async variant
enables continuous observation recording during motion at a configurable Hz rate.

For dry-run / simulation mode, all robot commands are logged but not executed,
and simulated TCP pose + gripper state are returned. During async dry-run moves,
the TCP pose is linearly interpolated between start and target.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """Represents a 6-DoF Cartesian pose [x, y, z, rx, ry, rz].
    
    Positions are in meters; rotations are axis-angle in radians.
    This matches the UR5 moveL pose representation.
    """
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def to_list(self) -> List[float]:
        """Convert to [x, y, z, rx, ry, rz] list for RTDE commands."""
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    @classmethod
    def from_list(cls, values: List[float]) -> "Pose":
        """Create Pose from a 6-element list."""
        if len(values) != 6:
            raise ValueError(f"Pose requires 6 values, got {len(values)}: {values}")
        return cls(*values)

    def with_z(self, z: float) -> "Pose":
        """Return a copy of this pose with a different Z height."""
        return Pose(self.x, self.y, z, self.rx, self.ry, self.rz)

    def position(self) -> np.ndarray:
        """Return position as numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])

    def orientation(self) -> np.ndarray:
        """Return orientation as numpy array [rx, ry, rz]."""
        return np.array([self.rx, self.ry, self.rz])

    def to_array(self) -> np.ndarray:
        """Return full pose as numpy array [x, y, z, rx, ry, rz]."""
        return np.array([self.x, self.y, self.z, self.rx, self.ry, self.rz])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Pose":
        """Create Pose from a numpy array."""
        return cls(*arr.tolist())

    @classmethod
    def lerp(cls, a: "Pose", b: "Pose", t: float) -> "Pose":
        """Linearly interpolate between two poses. t=0 returns a, t=1 returns b."""
        t = max(0.0, min(1.0, t))
        arr = (1.0 - t) * a.to_array() + t * b.to_array()
        return cls.from_array(arr)


class UR5Controller:
    """
    Controller for the UR5 robot arm with an Arduino-driven servo gripper.
    
    Connects to the UR5 via RTDE for real-time state reading and moveL commands.
    The gripper is a custom servo controlled by an Arduino Uno over USB serial.
    The Arduino expects integer servo angles followed by a newline character.
    
    Supports both blocking ``move_to()`` and non-blocking ``move_to_async()``.
    The async variant is used by the ExperimentRunner for continuous observation
    recording during robot motion.
    
    Attributes:
        robot_ip:            IP address of the UR5 controller.
        velocity:            Default linear velocity for moveL (m/s).
        acceleration:        Default linear acceleration for moveL (m/s²).
        blend_radius:        Default blend radius for moveL (m). 0 = stop at waypoint.
        dry_run:             If True, no hardware connection is made; commands are simulated.
        arduino_port:        Serial port for the Arduino (e.g. ``/dev/ttyACM0``).
        arduino_baudrate:    Baud rate for the Arduino serial connection.
        gripper_open_angle:  Servo angle in degrees for the fully-open position.
        gripper_close_angle: Servo angle in degrees for the fully-closed position.
    """

    # Gripper servo calibration constants
    OPEN_ANGLE = 40
    CLOSED_ANGLE = 110

    # Gripper state constants
    GRIPPER_OPEN = 1.0    # Normalized open state
    GRIPPER_CLOSED = 0.0  # Normalized closed state

    def __init__(
        self,
        robot_ip: str = "192.168.1.100",
        velocity: float = 0.25,
        acceleration: float = 0.50,
        blend_radius: float = 0.0,
        dry_run: bool = False,
        arduino_port: str = "/dev/ttyACM0",
        arduino_baudrate: int = 115200,
    ):
        self.robot_ip = robot_ip
        self.velocity = velocity
        self.acceleration = acceleration
        self.blend_radius = blend_radius
        self.dry_run = dry_run

        # Arduino / gripper configuration
        self.arduino_port = arduino_port
        self.arduino_baudrate = arduino_baudrate

        # Internal state tracking
        self._current_pose: Optional[Pose] = None
        self._gripper_state: Optional[float] = None  # Start unknown
        self._rtde_control = None
        self._rtde_receive = None
        self._arduino = None  # pyserial Serial instance
        self._connected = False

        # Simulated joint positions (6 joints for UR5)
        self._simulated_joints: List[float] = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

        # Async motion state (for dry-run interpolation)
        self._async_start_pose: Optional[Pose] = None
        self._async_target_pose: Optional[Pose] = None
        self._async_start_time: float = 0.0
        self._async_duration: float = 0.0
        self._async_moving: bool = False

    def connect(self) -> None:
        """
        Establish connection to the UR5 via RTDE and open the Arduino serial
        port for gripper control.
        
        In dry-run mode, this initializes simulated state instead — no serial
        connection is opened.
        
        Raises:
            ConnectionError: If the robot is unreachable or the Arduino
                serial port cannot be opened.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Simulating UR5 connection to %s", self.robot_ip)
            self._connected = True
            self._gripper_state = self.GRIPPER_OPEN
            return

        try:
            # Import RTDE libraries only when actually connecting to hardware
            import rtde_control  # type: ignore
            import rtde_receive  # type: ignore

            logger.info("Connecting to UR5 at %s ...", self.robot_ip)
            self._rtde_control = rtde_control.RTDEControlInterface(self.robot_ip)
            self._rtde_receive = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            logger.info("Successfully connected to UR5.")

        except Exception as e:
            self._connected = False
            raise ConnectionError(
                f"Failed to connect to UR5 at {self.robot_ip}: {e}"
            ) from e

        # --- Arduino serial connection for the servo gripper ---
        try:
            import serial  # type: ignore  # pyserial

            logger.info(
                "Opening Arduino serial connection on %s at %d baud ...",
                self.arduino_port, self.arduino_baudrate,
            )
            self._arduino = serial.Serial(
                self.arduino_port,
                self.arduino_baudrate,
                timeout=2,
            )
            # The Arduino resets when the serial connection opens.
            # Wait for it to finish its setup() before sending commands.
            time.sleep(2)
            logger.info("Arduino serial connection established.")

        except Exception as e:
            self._connected = False
            # Clean up the RTDE connection that was already opened
            try:
                if self._rtde_control is not None:
                    self._rtde_control.stopScript()
                    self._rtde_control.disconnect()
                if self._arduino is not None and self._arduino.is_open:
                    self._arduino.close()
            except Exception:
                pass
            raise ConnectionError(
                f"Failed to open Arduino serial port {self.arduino_port}: {e}"
            ) from e

        logger.info("Synchronizing gripper state to OPEN (40°) ...")
        self._send_servo_angle(self.OPEN_ANGLE)
        time.sleep(1)
        self._gripper_state = self.GRIPPER_OPEN

        self._connected = True

    def disconnect(self) -> None:
        """Safely disconnect from the UR5 and close the Arduino serial port."""
        if self.dry_run:
            logger.info("[DRY RUN] Simulating UR5 disconnect.")
            self._connected = False
            return

        # Close the Arduino serial connection first
        try:
            if self._arduino is not None and self._arduino.is_open:
                self._arduino.close()
                logger.info("Arduino disconnected.")
        except Exception as e:
            logger.warning("Error closing Arduino serial port: %s", e)
        finally:
            self._arduino = None

        # Disconnect RTDE
        try:
            if self._rtde_control is not None:
                self._rtde_control.stopScript()
                self._rtde_control.disconnect()
            if self._rtde_receive is not None:
                self._rtde_receive.disconnect()
            logger.info("Disconnected from UR5.")
        except Exception as e:
            logger.warning("Error during UR5 disconnect: %s", e)
        finally:
            self._connected = False

    def _ensure_connected(self) -> None:
        """Verify that we have an active connection."""
        if not self._connected:
            raise RuntimeError(
                "UR5 is not connected. Call connect() first."
            )

    # -------------------------------------------------------------------------
    # Motion Commands — Blocking
    # -------------------------------------------------------------------------

    def move_to(self, pose: Pose, velocity: Optional[float] = None,
                acceleration: Optional[float] = None) -> None:
        """
        Move the TCP to the target pose using a linear (moveL) motion.
        Blocks until the motion completes.
        
        This is the primary motion command for data collection — moveL produces
        straight-line Cartesian paths which are easier for a VLA to predict.
        
        Args:
            pose:         Target Cartesian pose.
            velocity:     Override default velocity (m/s).
            acceleration: Override default acceleration (m/s²).
        """
        self._ensure_connected()
        vel = velocity or self.velocity
        acc = acceleration or self.acceleration
        target = pose.to_list()

        if self.dry_run:
            logger.info(
                "[DRY RUN] moveL to [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] "
                "v=%.2f a=%.2f",
                *target, vel, acc
            )
            # Simulate movement time based on distance
            if self._current_pose is not None:
                dist = np.linalg.norm(pose.position() - self._current_pose.position())
                sim_time = max(0.1, dist / vel)
                time.sleep(min(sim_time, 0.5))  # Cap simulation delay
            self._current_pose = pose
            self._update_simulated_joints(pose)
            return

        # Execute real moveL command
        logger.info("moveL to %s v=%.2f a=%.2f", target, vel, acc)
        self._rtde_control.moveL(target, vel, acc, False)

        # Wait until the move completes
        while not self._rtde_control.isSteady():
            time.sleep(0.01)

        # Update internal pose from actual robot state
        self._current_pose = self.get_tcp_pose()

    def move_to_home(self, home_pose: Pose) -> None:
        """Move the robot to the home/rest pose."""
        logger.info("Moving to home pose ...")
        self.move_to(home_pose)

    # -------------------------------------------------------------------------
    # Motion Commands — Non-Blocking (Async)
    # -------------------------------------------------------------------------

    def move_to_async(self, pose: Pose, velocity: Optional[float] = None,
                      acceleration: Optional[float] = None) -> None:
        """
        Start a non-blocking moveL motion to the target pose.
        
        Returns immediately after issuing the motion command. Use `is_moving()`
        to check whether the motion is still in progress. This enables the
        ExperimentRunner to record observations continuously during motion.
        
        Args:
            pose:         Target Cartesian pose.
            velocity:     Override default velocity (m/s).
            acceleration: Override default acceleration (m/s²).
        """
        self._ensure_connected()
        vel = velocity or self.velocity
        acc = acceleration or self.acceleration
        target = pose.to_list()

        if self.dry_run:
            logger.info(
                "[DRY RUN] moveL (async) to [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] "
                "v=%.2f a=%.2f",
                *target, vel, acc
            )
            # Set up interpolation state for dry-run simulation
            start = self._current_pose or Pose(0, 0, 0, 0, 0, 0)
            dist = np.linalg.norm(pose.position() - start.position())
            duration = max(0.3, dist / vel)  # Minimum 0.3s for meaningful interpolation

            self._async_start_pose = start
            self._async_target_pose = pose
            self._async_start_time = time.time()
            self._async_duration = duration
            self._async_moving = True
            return

        # Execute real moveL command (non-blocking: asynchronous=True)
        logger.info("moveL (async) to %s v=%.2f a=%.2f", target, vel, acc)
        self._rtde_control.moveL(target, vel, acc, True)
        self._async_target_pose = pose
        self._async_moving = True

    def is_moving(self) -> bool:
        """
        Check whether an async moveL motion is still in progress.
        
        In dry-run mode, this checks elapsed time against the simulated
        duration. When the move completes, the current pose is updated.
        
        Returns:
            True if the robot is still moving, False if the motion is complete.
        """
        if not self._async_moving:
            return False

        if self.dry_run:
            elapsed = time.time() - self._async_start_time
            if elapsed >= self._async_duration:
                # Motion complete — snap to target
                self._current_pose = self._async_target_pose
                self._update_simulated_joints(self._async_target_pose)
                self._async_moving = False
                return False
            # Still moving — update interpolated pose
            t = elapsed / self._async_duration
            self._current_pose = Pose.lerp(
                self._async_start_pose, self._async_target_pose, t
            )
            self._update_simulated_joints(self._current_pose)
            return True

        # Real robot: check if still in motion
        if self._rtde_control.isSteady():
            self._current_pose = self.get_tcp_pose()
            self._async_moving = False
            return False
        return True

    # -------------------------------------------------------------------------
    # Gripper Commands
    # -------------------------------------------------------------------------

    def open_gripper(self, wait_time: float = 0.5) -> None:
        """
        Open the servo gripper by sending the open angle to the Arduino.
        
        Args:
            wait_time: Seconds to wait for the servo to finish moving.
        """
        self._ensure_connected()

        if self._gripper_state == self.GRIPPER_OPEN:
            logger.debug("Gripper is already OPEN. Ignoring command.")
            return

        if self.dry_run:
            logger.info("[DRY RUN] Opening gripper.")
            time.sleep(min(wait_time, 0.2))
            self._gripper_state = self.GRIPPER_OPEN
            return
        else:
            logger.debug(
                "Sending OPEN command (%d°) to Arduino", self.OPEN_ANGLE
            )
            self._send_servo_angle(self.OPEN_ANGLE)
        time.sleep(wait_time)
        self._gripper_state = self.GRIPPER_OPEN

    def close_gripper(self, wait_time: float = 0.5) -> None:
        """
        Close the servo gripper by sending the close angle to the Arduino.
        
        Args:
            wait_time: Seconds to wait for the servo to finish moving.
        """
        self._ensure_connected()

        if self._gripper_state == self.GRIPPER_CLOSED:
            logger.debug("Gripper is already CLOSED. Ignoring command.")
            return

        if self.dry_run:
            logger.info("[DRY RUN] Closing gripper.")
            time.sleep(min(wait_time, 0.2))
            self._gripper_state = self.GRIPPER_CLOSED
            return
        else:
            logger.debug(
                "Sending CLOSE command (%d°) to Arduino", self.CLOSED_ANGLE
            )
            self._send_servo_angle(self.CLOSED_ANGLE)
        time.sleep(wait_time)
        self._gripper_state = self.GRIPPER_CLOSED

    def _send_servo_angle(self, angle: int) -> None:
        """Send a servo angle command to the Arduino over serial.
        
        The Arduino expects an integer angle followed by a newline character.
        
        Args:
            angle: Target servo angle in degrees.
            
        Raises:
            RuntimeError: If the serial write fails (e.g. Arduino disconnected).
        """
        if self._arduino is None or not self._arduino.is_open:
            raise RuntimeError(
                "Arduino serial port is not open. Cannot send gripper command."
            )

        try:
            command = f"{angle}\n".encode()
            self._arduino.write(command)
            self._arduino.flush()
            logger.debug("Sent servo command: %d°", angle)
        except Exception as e:
            raise RuntimeError(
                f"Failed to write to Arduino serial port: {e}"
            ) from e

    # -------------------------------------------------------------------------
    # State Queries
    # -------------------------------------------------------------------------

    def get_tcp_pose(self) -> Pose:
        """
        Read the current TCP (Tool Center Point) pose from the robot.
        
        Returns:
            Current Cartesian pose of the end-effector.
        """
        self._ensure_connected()

        if self.dry_run:
            if self._current_pose is None:
                # Return a default pose if we haven't moved yet
                return Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return self._current_pose

        tcp = self._rtde_receive.getActualTCPPose()
        return Pose.from_list(tcp)

    def get_joint_positions(self) -> List[float]:
        """
        Read the current joint positions (6 joints) from the robot.
        
        Returns:
            List of 6 joint angles in radians [j1, j2, j3, j4, j5, j6].
        """
        self._ensure_connected()

        if self.dry_run:
            return self._simulated_joints.copy()

        return list(self._rtde_receive.getActualQ())

    def get_gripper_state(self) -> float:
        """
        Get the current gripper state normalized to [0, 1].
        
        Returns:
            0.0 = fully closed, 1.0 = fully open.
        """
        return self._gripper_state

    def get_full_state(self) -> dict:
        """
        Get the complete end-effector and joint state as a flat dictionary.
        
        This is the primary state-query method used by the DataCollector at
        every timestep. The keys map directly to CSV column names and
        subsequently to RLDS observation fields.
        
        Returns:
            Dictionary with ee position, orientation, gripper state,
            and joint positions (joint_1 through joint_6).
        """
        pose = self.get_tcp_pose()
        joints = self.get_joint_positions()
        return {
            "ee_pos_x": pose.x,
            "ee_pos_y": pose.y,
            "ee_pos_z": pose.z,
            "ee_rot_rx": pose.rx,
            "ee_rot_ry": pose.ry,
            "ee_rot_rz": pose.rz,
            "gripper_state": self.get_gripper_state(),
            "joint_1": joints[0],
            "joint_2": joints[1],
            "joint_3": joints[2],
            "joint_4": joints[3],
            "joint_5": joints[4],
            "joint_6": joints[5],
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _update_simulated_joints(self, pose: Pose) -> None:
        """
        Update simulated joint values based on TCP pose.
        
        Uses a simple deterministic mapping from Cartesian pose to approximate
        joint values. These are NOT kinematically accurate — they serve only
        to produce realistic-looking joint columns in dry-run data.
        """
        p = pose.to_array()
        # Create plausible joint values via a simple linear transform
        self._simulated_joints = [
            float(np.arctan2(p[1], p[0])),         # j1: base rotation
            float(-1.57 + p[2] * 2.0),              # j2: shoulder
            float(1.57 - p[2] * 1.5),               # j3: elbow
            float(p[3] * 0.5 - 1.57),               # j4: wrist 1
            float(p[4] * 0.5 - 1.57),               # j5: wrist 2
            float(p[5] * 0.3),                       # j6: wrist 3
        ]

    @property
    def is_connected(self) -> bool:
        """Check if the robot connection is active."""
        return self._connected
