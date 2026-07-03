from src.ur5_controller import UR5Controller

robot = UR5Controller(robot_ip="169.254.76.5")
robot.connect()

try:
    while True:
        angle = int(input("Servo angle: "))
        robot._send_servo_angle(angle)

finally:
    robot.disconnect()