import cv2
import numpy as np
from PIL import Image


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def pre_rotation(IX, IY, ylm):

    X = cv2.cvtColor(IX, cv2.COLOR_BGR2GRAY)
    Y = cv2.cvtColor(IY, cv2.COLOR_BGR2GRAY)

    xdim = (1000,1000)
    X = cv2.resize(X, xdim, interpolation=cv2.INTER_AREA)
    Y = cv2.resize(Y, xdim, interpolation=cv2.INTER_AREA)
    Y = Image.fromarray(Y)
    IY = Image.fromarray(IY)

    ylm = np.array(ylm)
    ylm = ylm[:, 1:]

    err = []

    # calculate the mse for every iteration of rotation
    for i in range(120):
        temp = Y.rotate(3 * i)
        temp = np.array(temp)
        err.append(mse(temp, X))

    # select best angle and rotate landmarks accordingly
    angle = err.index(min(err))*3
    IY = IY.rotate(angle, fillcolor=(255,255,255))
    IY = np.array(IY)
    ylm = ylm - [IY.shape[1]/2, IY.shape[0]/2]
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

    for i in range(ylm.shape[0]):
        [q, t] = ylm[i, :]
        q = q * c - t * s
        t = q * s + t * c
        ylm[i, :] = [q, t]

    ylm = ylm + [IY.shape[1] / 2, IY.shape[0] / 2]

    return IY, ylm

