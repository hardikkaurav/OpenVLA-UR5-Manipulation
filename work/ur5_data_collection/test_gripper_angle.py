#!/usr/bin/env python3

from src.ur5_controller import UR5Controller

robot = UR5Controller(
    robot_ip="169.254.76.5",
    dry_run=False,
)

print("Connecting...")
robot.connect()

try:
    input("Move the robot to the desired position, then press ENTER...")

    pose = robot.get_tcp_pose()

    print("\nCurrent TCP Pose")
    print("----------------------------------------")
    print(f"x  = {pose.x:.6f}")
    print(f"y  = {pose.y:.6f}")
    print(f"z  = {pose.z:.6f}")
    print(f"rx = {pose.rx:.6f}")
    print(f"ry = {pose.ry:.6f}")
    print(f"rz = {pose.rz:.6f}")

    print("\nCopy into poses.json:")
    print(
        f"[{pose.x:.6f}, {pose.y:.6f}, {pose.z:.6f}, "
        f"{pose.rx:.6f}, {pose.ry:.6f}, {pose.rz:.6f}]"
    )

finally:
    robot.disconnect()
    print("\nDisconnected.")