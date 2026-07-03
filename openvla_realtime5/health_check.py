import subprocess
import torch
import sys

print("=" * 60)
print("OPENVLA SYSTEM HEALTH CHECK")
print("=" * 60)

# --------------------------------------------------
# GPU
# --------------------------------------------------
print("\n[1] GPU CHECK")

try:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode == 0:
        print("✓ NVIDIA GPU detected")
        print(result.stdout.split("\n")[8])
    else:
        print("✗ nvidia-smi failed")

except Exception as e:
    print("✗ GPU check failed:", e)

print("\nPyTorch CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# RealSense
# --------------------------------------------------
print("\n[2] REALSENSE CHECK")

try:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("✗ No RealSense camera found")
    else:
        for dev in devices:
            print("✓ Camera:", dev.get_info(rs.camera_info.name))
            print("  Serial:", dev.get_info(rs.camera_info.serial_number))

except Exception as e:
    print("✗ RealSense error:", e)

# --------------------------------------------------
# URBasic
# --------------------------------------------------
print("\n[3] URBASIC CHECK")

try:
    import URBasic

    print("✓ URBasic imported")
    print("  Location:", URBasic.__file__)

except Exception as e:
    print("✗ URBasic import failed:", e)

# --------------------------------------------------
# OpenVLA libraries
# --------------------------------------------------
print("\n[4] OPENVLA DEPENDENCIES")

packages = [
    "transformers",
    "accelerate",
    "PIL",
    "numpy",
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except Exception as e:
        print(f"✗ {pkg}: {e}")

# --------------------------------------------------
# Ping UR5
# --------------------------------------------------
print("\n[5] UR5 NETWORK CHECK")

ROBOT_IP = "169.254.76.5"

try:
    result = subprocess.run(
        ["ping", "-c", "2", ROBOT_IP],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode == 0:
        print("✓ Robot reachable")
    else:
        print("✗ Robot not reachable")

except Exception as e:
    print("✗ Ping failed:", e)

print("\nDone.")
