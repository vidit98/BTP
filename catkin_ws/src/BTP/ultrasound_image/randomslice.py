import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


# to reaf the files from XY dir or XY_Cart dir.
temp = []
for i in range(128):
    file = "xy_{}.png".format(i)
    im = cv2.imread(os.path.join("XY", file), 0)
    temp.append(im)

vol = np.stack(temp[:], axis=0)
print(vol.shape)


# define the place which cuts the volume
from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array

idy = 0
idz = 0

#for a plane which is 45 degress, and cuts the volume from the middle 
inc = 128.0/(128**2 + 128**2)**(1.0/2)

points = []
while(idy<=127 and idz<=127):
	for i in range(128):
		points.append([idz, idy, i])

	idy +=inc
	idz +=inc
name = "downmin_cart.png"
print(len(points))


x = linspace(0,127,128)
y = linspace(0,127,128)
z = linspace(0,127,128)
#print(x)
fn = RegularGridInterpolator((x,y,z), vol, method = "nearest")

outputpoints = fn(points)
slc = outputpoints.reshape(-1, 128)
slc = cv2.flip(slc, 0)
cv2.imwrite(name, slc)
plt.imshow(slc, cmap='gray')
plt.show()



# some other 45 degree cuts.
"""
idy = 64
idz = 0

#for 45 degress down
inc = 64/(64**2 + 64**2)**(1/2)

points = []
while(idy<=127 and idz<=63):
	for i in range(128):
		points.append([idz, idy, i])

	idy +=inc
	idz +=inc
name = "45down.png"
//////////////////////////

idy = 64
idz = 0

#for 45 degress up
inc = 64/(64**2 + 64**2)**(1/2)

points = []
while(idy>=0 and idz<=63):
	for i in range(128):
		points.append([idz, idy, i])

	idy -=inc
	idz +=inc
name = "45up.png"
/////////////////////////
idy = 64
idz = 127

#for 45 degress rev up 
inc = 64/(64**2 + 64**2)**(1/2)

points = []
while(idy>=0 and idz>=63):
	for i in range(128):
		points.append([idz, idy, i])

	idy -=inc
	idz -=inc
name = "45revup.png"
///////////////////////////
idy = 64
idz = 127

#for 45 degress rev down 
inc = 64/(64**2 + 64**2)**(1/2)

points = []
while(idy<=127 and idz>=63):
	for i in range(128):
		points.append([idz, idy, i])

	idy +=inc
	idz -=inc
name = "45revdown.png"
"""

