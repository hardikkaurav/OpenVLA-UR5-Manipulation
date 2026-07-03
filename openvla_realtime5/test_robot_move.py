import URBasic
import time

ROBOT_IP = "169.254.76.5"

robotModel = URBasic.robotModel.RobotModel()

robot = URBasic.urScriptExt.UrScriptExt(
    host=ROBOT_IP,
    robotModel=robotModel
)

robot.reset_error()

time.sleep(2)

pose = robotModel.ActualTCPPose()

print("Current pose:")
print(pose)

target = pose.copy()

target[0] += 0.05   # 5 cm

print("Target pose:")
print(target)

robot.movel(target, a=0.01, v=0.01)

print("Move command sent")

time.sleep(5)

robot.close()
