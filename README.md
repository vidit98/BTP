# BTP
Autonomous Ultrasound using 7 DOF manipulator and RGBD camera

* install ros kinetic in local env 
* create venv with python3 
```git clone https://github.com/ros-perception/vision_opencv.git
```
* copy cv_bridge from vision_opencv to src
* run ```catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so```

* make a symlink- cd /venv/lib/python3.5/site-packages the run command below

 ln -s /usr/local/lib/python3.5/dist-packages/cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
 
 
