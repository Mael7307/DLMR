import Registration
from utils.utils import *
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import image_slicer
import numpy
import pandas as pd
from main import main
from pre_rotation import pre_rotation
from pre_rotation import mse
from time import process_time
import os
import csv
import math

tf.compat.v1.disable_eager_execution()

# designate image path here
# Target
IX = cv2.imread('/Users/admin/Desktop/imag reg CNN multi/src/img/HE.jpg')
# Source
IY = cv2.imread('/Users/admin/Desktop/imag reg CNN multi/src/img/S2.jpg')
Y = IY
# import source landmarks
ylm = pd.read_csv('/Users/admin/Desktop/imag reg CNN multi/src/img/S2.csv')

# pre rotation step
IY, ylm = pre_rotation(IX, IY, ylm)

#  1 is base method , 2 is DLMR, 3 is 16 x resolution method
iterations = 1

# registration
registered, nlm = main(IX, IY, ylm, iterations)

# Plot the source image, warped source and target image respectively
plt.subplot(133)
plt.imshow(cv2.cvtColor(Y, cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.imshow(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))
plt.subplot(131)
plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))
plt.show()





