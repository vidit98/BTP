import numpy as np
import open3d as o3d
import tensorflow as tf
from os import listdir


def pc_normalize(pc):
	centroid = np.mean(pc, axis=0)
	pc = pc - centroid
	m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
	pc = pc / m
	return pc

def get_data(path):

	files = [f for f in listdir(path)]
	data = []
	for f in files:
		pcd = o3d.io.read_point_cloud(path + f)
		pts = pc_normalize(np.array(pcd.points, dtype=np.float32))
		if pts.shape[0] > 999:
			
			np.random.shuffle(pts)
			pts = pts[0:1000,:]
			data.append(pts)

	data = np.array(data)

	return data



# pc_data = get_data("/home/vidit/implementation/BTP/data/probe/pcl/")
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (pc_data)).shuffle(10000).batch(32)

# for images in train_ds:
# 	print(images.shape)	


