import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys
import rospy
#sys.path.append('/usr/local/lib/python3.4/site-packages')


from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
import std_msgs
from cv_bridge import CvBridge, CvBridgeError

import cv2
import message_filters

class SurfaceWrapper(object):

	def __init__(self):
		print("Initalized")
		self.Yl = 21
		self.Cbl = 77
		self.Crl = 83

		self.Yh = 88
		self.Cbh = 150
		self.Crh = 146

		self.clr = None
		self.depth = None
		self.cnt = 0
		self.bridge = CvBridge()

		self.h = std_msgs.msg.Header()
		self.h.stamp = rospy.Time.now()
		self.h.frame_id = "map"
		self.pub = rospy.Publisher('pcl', PointCloud2, queue_size=10)
		self.ptcld = None

		self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
          		PointField('y', 4, PointField.FLOAT32, 1),
          		PointField('z', 8, PointField.FLOAT32, 1)]


	def segment(self, img):

	    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

	    frame_threshold = cv2.inRange(img, (self.Yl, self.Cbl, self.Crl), (self.Yh, self.Cbh, self.Crh))
	    # cv2.imshow("s", frame_threshold)
	    # cv2.waitKey(0)
	    image, contours, _ = cv2.findContours(frame_threshold,  cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	    l = []
	    for c in contours:
	        
	        m = np.mean(c, axis=0)[0]
	        if m[0] > 250 and m[0] < 290:
	            if m[1] > 260 and m[1] < 350:
	                l.append(c)
	    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
	    ignore_mask_color = 256*256 -1
	    print(l[0].shape)
	    cv2.fillPoly(mask, l, ignore_mask_color)
	   
	    return mask

	def to_pcl(self):
		


		color_raw = self.bridge.imgmsg_to_cv2(self.clr, desired_encoding="passthrough")
		depth_raw = self.bridge.imgmsg_to_cv2(self.depth, desired_encoding="passthrough")
		# print("INITIAL" ,np.max(np.array(color_raw)), np.min(np.array(depth_raw)))
		color_raw = np.array(color_raw).astype("uint8")
		depth_raw = np.array(depth_raw).astype("uint16")
		mask = self.segment(color_raw)
		
		depth_raw = cv2.bitwise_and(depth_raw, mask)

		color_raw = o3d.geometry.Image(color_raw)
		depth_raw = o3d.geometry.Image(depth_raw)
		
		rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0)
	

		intrinsic = o3d.camera.PinholeCameraIntrinsic()
		intrinsic.set_intrinsics(640, 480, 554.3826904296859, 554.3826904296875, 320, 240)
		cloud = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image,intrinsic)
		print(cloud.points)
		#print(np.min(np.array(cloud.points)[:,0]))
		
		self.ptcld = point_cloud2.create_cloud(self.h, self.fields, cloud.points)
		

		self.pub.publish(self.ptcld)

	# def get_data(self, clr, dpth):
	# 	self.clr = msg
	# 	self.depth = dpth
	# 	self.to_pcl()

	def get_data(self, clr):
		self.clr = clr
		self.to_pcl()	
	def get_d(self, d):
		self.depth = d


	
def callback(img, dpth):
	print("aevev")


rospy.init_node('listener')

surface = SurfaceWrapper()

# image_sub = message_filters.Subscriber("/r200/camera/color/image_raw", Image)
# depth_sub = message_filters.Subscriber("/r200/camera/depth/image_raw", Image)

# ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
# ts.registerCallback(callback)

rospy.Subscriber("/r200/camera/depth/image_raw", Image , surface.get_d)
rospy.Subscriber("/r200/camera/color/image_raw", Image, surface.get_data)

rospy.spin()





