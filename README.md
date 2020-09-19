# DLMR

A muti resolution image registration method using convolutional neural network features written in Python2, Tensorflow API r1.5.0. 

# Requirements

* numpy
* scipy
* opencv-python
* matplotlib
* tensorflow (with or without gpu)
* lap

To install all the requirements run
```
pip install -r requirements.txt
```
Prior to doing so, in some Linux distributions, you may need to install software packages such as the following:

* pip
* python2 development package
* python-setup

and you may need to do "pip install wheel".

Pretrained VGG16 parameters file `vgg16partial.npy` is available at `https://drive.google.com/file/d/1o1xjU9F58x83iR91LoFjLOlBdLN3bPnm/view?usp=sharing`.
Please download and put it under the `src/` directory.

# Usage
See src/execute.py for ANHIR submission code. See src/demo.py for individual pair registration. 

# Project Code

The following sections of code were developed/modified for DLMR:
-utils/tps-warp
-Image slicer/slice
-registration.py localised matching and background filter sections
-main.py
-execute.py
-demo.py
-blank.py

All other functions/scripts originate from https://github.com/yzhq97/cnn-registration
(https://ieeexplore.ieee.org/document/8404075/).
