import open3d as o3d
from os import walk
from os import listdir
import numpy as np

import h5py
print(o3d.__version__)

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(640, 480, 554.3826904296859, 554.3826904296875, 320, 240)

mypath = "./probe/d"
desti = "./probe/pcl/"
files = [f for f in listdir(desti)]

f = h5py.File("/home/vidit/implementation/BTP/src/pointnet/data/modelnet40_ply_hdf5_2048/ply_data_train4.h5")

data = f['data'][:]
label = f['label'][:]

i = 0
mi = 1000000
ma = 0

# for cl in list(label):
# 	if cl == 2:
		
# 		pts = data[i, 0:1024, :]
# 		pcd1 = o3d.geometry.PointCloud()
# 		pcd1.points = o3d.utility.Vector3dVector(pts)

# 		o3d.visualization.draw_geometries([pcd1])

# 	i+=1
s = ""
for filenames in files:
	# print(filenames)
	# depth = o3d.io.read_image(mypath + filenames)
	pcd = o3d.io.read_point_cloud(desti + filenames)
	pcd.paint_uniform_color([0, 0.706, 1])
	pts = np.array(pcd.points)
	# if pts.shape[0] > 999:
	# 	print(pts.shape[0])
	# 	ma+=1
	# 	np.random.shuffle(pts)
	# 	pts = pts[0:1000,:]
	# 	pcd1 = o3d.geometry.PointCloud()
	# 	pcd1.points = o3d.utility.Vector3dVector(pts)
	# 	print(np.array(pcd1.points).shape)
	# 	# pcl = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic,  depth_scale=1000.0, depth_trunc=1000.0, stride=1)
	# 	o3d.visualization.draw_geometries([pcd, pcd1])
	# o3d.io.write_point_cloud(desti + "ptcld" + str(i) + ".pcd", pcl)
	# i += 1
	if np.array(pcd.points).shape[0] > ma:
		ma = np.array(pcd.points).shape[0]
		s = filenames
	if np.array(pcd.points).shape[0] < mi:
		mi = np.array(pcd.points).shape[0]

print(mi, ma,s)