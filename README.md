
# RoboSpotter

A violence detector in videos

https://user-images.githubusercontent.com/60934172/118996253-aa0e9680-b97f-11eb-9bfc-393eb08bca07.mp4

Helena Correia

José Henrique Brito

## Dataset:

https://ipcapt-my.sharepoint.com/:f:/g/personal/jbrito_ipca_pt/EnwrKzWGR4ZPuR3RgH9uWQ0Bbqv15lpKx3apaFLXu98Ghg?e=6bJr6P

Install:

$ pip install -r requirements.txt

## TF Pose
$ git clone https://www.github.com/ildoonet/tf-pose-estimation

$ cd tf-pose-estimation


## PAF Process

Download and Extract SWIG (swigwin-3.0.12) https://sourceforge.net/projects/swig/files/swigwin/swigwin-3.0.12/swigwin-3.0.12.zip/download

Add Swig to Path

Reboot

$ cd tf_pose/pafprocess

$ swig -python -c++ pafprocess.i

$ python setup.py build_ext --inplace


## Copy coco API

$ cd cocoapi/PythonAPI

$ python setup.py build_ext --inplace
