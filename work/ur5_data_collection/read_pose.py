#!/usr/bin/env python3

"""
Test one complete pick-and-place cycle.

Sequence:

Home
↓
Pickup Approach
↓
Pickup Grasp
↓
Close Gripper
↓
Pickup Approach
↓
Place 1 Approach
↓
Place 1
↓
Open Gripper
↓
Place 1 Approach
↓
Home
"""

import json
import time

from src.ur5_controller import UR5Controller, Pose

# --------------------------------------------------------
# Load configuration
# --------------------------------------------------------

with open("config/poses.json", "r") as f:
    config = json.load(f)

home_pose = Pose.from_list(config["home_pose"])
grasp_pose = Pose.from_list(config["pick_pose"]["grasp"])

# Use ONLY Place Pose 1
place_pose = Pose.from_list(config["place_poses"][0])

offset = config["approach_height_offset"]

pickup_approach = grasp_pose.with_z(grasp_pose.z + offset)
place_approach = place_pose.with_z(place_pose.z + offset)

move = config["move_parameters"]

# --------------------------------------------------------
# Connect Robot
# --------------------------------------------------------

robot = UR5Controller(
    robot_ip="169.254.76.5",
    velocity=move["velocity"],
    acceleration=move["acceleration"],
    blend_radius=move["blend_radius"],
    dry_run=False,
)

print("Connecting...")
robot.connect()

try:

    input("\nPlace the object at the pickup position.\nPress ENTER to start...")

    # ----------------------------------------------------

    print("\n1. Home")
    robot.move_to(home_pose)
    time.sleep(1)

    # ----------------------------------------------------

    print("\n2. Pickup Approach")
    robot.move_to(pickup_approach)
    time.sleep(1)

    # ----------------------------------------------------

    print("\n3. Pickup Grasp")
    robot.move_to(grasp_pose)
    time.sleep(1)

    # ----------------------------------------------------

    print("\n4. Close Gripper")
    robot.close_gripper()
    time.sleep(1)

    # ----------------------------------------------------

    print("\n5. Pickup Approach")
    robot.move_to(pickup_approach)
    time.sleep(1)

    # ----------------------------------------------------

    print("\n6. Place Approach")
    robot.move_to(place_approach)
    time.sleep(1)

    # ----------------------------------------------------

    print("\n7. Place Pose")
    robot.move_to(place_pose)
    time.sleep(1)

    # ----------------------------------------------------

    print("\n8. Open Gripper")
    robot.open_gripper()
    time.sleep(1)

    # ----------------------------------------------------

    print("\n9. Place Approach")
    robot.move_to(place_approach)
    time.sleep(1)

    # ----------------------------------------------------

    print("\n10. Home")
    robot.move_to(home_pose)
    time.sleep(1)

    print("\n===================================")
    print("TEST COMPLETED SUCCESSFULLY")
    print("===================================")

finally:

    robot.disconnect()
    print("Robot disconnected.")