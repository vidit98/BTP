import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import message_filters
import os
import numpy as np

bridge = CvBridge()




# def get_data(clr):
# 		color_raw = bridge.imgmsg_to_cv2(clr, desired_encoding="passthrough")
# 		color_raw = np.array(color_raw).astype("uint8")
		
# def get_d(d):
# 		depth_raw = bridge.imgmsg_to_cv2(d, desired_encoding="passthrough")
# 		depth_raw = np.array(depth_raw).astype("uint16")

val = input()
i =0
while(val != 'e'):
	rospy.init_node('listener')
	clr = rospy.wait_for_message("/r200/camera/color/image_raw", Image, timeout=None)
	color_raw = bridge.imgmsg_to_cv2(clr, desired_encoding="passthrough")
	color_raw = np.array(color_raw).astype("uint8")
	cv2.imwrite("./data/rgb"+str(i)+ ".png" ,color_raw,[cv2.IMWRITE_PNG_COMPRESSION, 0])

	d = rospy.wait_for_message("/r200/camera/depth/image_raw", Image, timeout=None)
	depth_raw = bridge.imgmsg_to_cv2(d, desired_encoding="passthrough")
	depth_raw = np.array(depth_raw).astype("uint16")
	#cv2.imwrite("./data/depth"+str(i)+".bmp" ,depth_raw)
	cv2.imwrite("./data/depth"+str(i)+".png", depth_raw, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	#cv2.imshow("asdad", depth_raw)
	#cv2.waitKey(0)
	i+=1
	val = input()

#rospy.Subscriber("/r200/camera/depth/image_raw", Image , surface.get_d)
#rospy.Subscriber("/r200/camera/color/image_raw", Image, surface.get_data)
#rospy.spin()