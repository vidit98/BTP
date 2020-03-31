#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Header

def talker():
    pub = rospy.Publisher('/panda_arm_controller/command', JointTrajectory, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    msg = JointTrajectory()
    point = JointTrajectoryPoint()
    while not rospy.is_shutdown():
        msg.joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7"]
        #msg.header.stamp = rospy.get_rostime()
        point.positions =[0,0,0,0,0,0,0]# [0.01,0.001,0.1,0.1,0.1,0.1,0.1]
        point.velocities = [0,0,0,0,0,0,0]
        #point.effort =[0,0,0,-1,2,5,0]
        point.time_from_start.secs = 1
        msg.points= [point]
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass