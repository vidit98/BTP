#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

def callback(data):
	pose_goal = geometry_msgs.msg.Pose()
	pose_goal.orientation.w = 1.0
	pose_goal.position.x = 0.4
	pose_goal.position.y = 0.1
	pose_goal.position.z = 0.4
	
	group.set_pose_target(pose_goal)
	
	plan = group.go(wait=True)
	# Calling `stop()` ensures that there is no residual movement
	group.stop()
	group.clear_pose_targets()

	group.execute(plan, wait=True)

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial',
                anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_name = "panda_arm"
group = moveit_commander.MoveGroupCommander(group_name)

rospy.Subscriber("PCD_planner", String, callback)
rospy.spin()