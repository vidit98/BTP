import rospy
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
import numpy as np
import os
import cv2
from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array

bridge = CvBridge()

probe_pose = PointStamped()
probe_pose.header.frame_id = "/panda_hand"
probe_pose.point.x = 0.5
probe_pose.point.y = 0
probe_pose.point.z = 0

probe_pose_d = PointStamped()
probe_pose_d.header.frame_id = "/panda_hand"
probe_pose_d.point.x = 0
probe_pose_d.point.y = 0.5
probe_pose_d.point.z = 0.5

hand_pose = PointStamped()
hand_pose.header.frame_id = "/panda_hand"
hand_pose.point.x = 0
hand_pose.point.y = 0
hand_pose.point.z = 0

vec_1 = []
vec_2 = []
topic = "/gazebo/link_states"

temp = []
for i in range(128):
    file = "xy_{}.png".format(i)
    im = cv2.imread(os.path.join("XY", file), 0)
    temp.append(im)

vol = np.stack(temp[:], axis=0)
print(vol.shape)
	
rospy.init_node('Ultrasound_Volume', anonymous=True)
listener = tf.TransformListener()
img_pub = rospy.Publisher("panda/ultrasound_image", Image, queue_size=1)
rate = rospy.Rate(10)

while not rospy.is_shutdown():
	try:
		p_probe = listener.transformPoint("/human", probe_pose)
		p_probe_d = listener.transformPoint("/human", probe_pose_d)
		p_hand = listener.transformPoint("/human", hand_pose)
	except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
		continue
	vec_1 = [p_probe.point.x - p_hand.point.x, p_probe.point.y - p_hand.point.y, p_probe.point.z - p_hand.point.z]
	vec_2 = [p_probe_d.point.x - p_hand.point.x, p_probe_d.point.y - p_hand.point.y, p_probe_d.point.z - p_hand.point.z]

	normal = np.cross(vec_1, vec_2)
	a,b,c = normal
	d = np.dot(normal, np.array((p_probe.point.x, p_probe.point.y, p_probe.point.z)))
	x = np.linspace(0,127,128)
	y = np.linspace(0,127,256)
	X,Y = np.meshgrid(x,y)
	X = X.flatten()
	Y = Y.flatten()
	Z = (d - a * X - b * Y) *1.0/ c
	points = np.column_stack((X,Y,Z))
	Z = Z.reshape(256,128)	
	temp = points[points[:,2] >= 0]
	
	temp = temp[temp[:,2] <= 127]
	
	x = linspace(0,127,128)
	y = linspace(0,127,128)
	z = linspace(0,127,128)

	fn = RegularGridInterpolator((x,y,z), vol, method = "nearest")
	
	outputpoints = fn(temp)
	k=0
	for i in range(256):
		for j in range(128):
			if Z[i][j] >= 0 and Z[i][j] <=127:
				Z[i][j] = outputpoints[k]
				k+=1
			else:
				Z[i][j] = 0
	Z = np.around(Z)
	Z = Z.astype(np.uint8)
	slc = cv2.flip(Z, 0)
	img_pub.publish(bridge.cv2_to_imgmsg(slc, encoding="mono8"))
	rate.sleep()
