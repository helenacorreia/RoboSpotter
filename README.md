# RoboSpotter
Install:

$ git clone https://www.github.com/ildoonet/tf-pose-estimation

$ cd tf-pose-estimation

$ pip3 install -r requirements.txt



## PAF Process

Download and Extract SWIG (swigwin-3.0.12)

Add Swig to Path

Reboot

$ cd tf_pose/pafprocess

$ swig -python -c++ pafprocess.i

$ python3 setup.py build_ext --inplace


## Copy coco API

$ cd PythonAPI

$ python setup.py build_ext --inplace
