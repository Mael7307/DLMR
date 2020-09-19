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

# import spreadsheet names
data = pd.read_csv('/Users/admin/Desktop/dataset_medium.csv', header=None)
data = np.array(data)

# iterate through every point in the spreadsheet
for i in range(0, data.shape[0]):

    # record execution time
    t1_start = process_time()

    # designate image path here
    IX = cv2.imread(os.path.join('/Users/admin/Desktop/Data', data[i, 3]))
    IY = cv2.imread(os.path.join('/Users/admin/Desktop/Data', data[i, 5]))

    # import source landmarks
    ylm = pd.read_csv(os.path.join('/Users/admin/Desktop/Data', data[i, 4]))

    # pre rotation step
    IY, ylm = pre_rotation(IX, IY, ylm)

    #  1 is base method , 2 is DLMR, 3 is 16 x resolution method
    iterations = 2

    # registration
    registered, nlm = main(IX, IY, ylm, iterations)


    t1_stop = process_time()
    print('time elapsed : ', t1_stop - t1_start)

    # output results in submission format
    index = np.array([range(nlm.shape[0])]).T
    for j in range(nlm.shape[0]):
        results.append([index[j], nlm[j][0], nlm[j][1]])

    # save warped source landmarks in csv
    numpy.savetxt(os.path.join("/Users/admin/Desktop/Warped target landmarks", str(i) + '.csv'), results, delimiter=",")

    # update registration results csv
    data[i, 9] = os.path.join("Warped target landmarks", str(i) + '.csv')
    data[i, 10] = str(t1_stop - t1_start)
    numpy.savetxt('/Users/admin/Desktop/Evaluation Data.csv', data, fmt='%s', delimiter=",")












