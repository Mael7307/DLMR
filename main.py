import Registration
from utils.utils import *
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import image_slicer
import numpy
import pandas as pd
import math
from blank import blank


def main(IX, IY, ylm, iterations):

    # resize images

    scale_percent = 10  # percent of original size might have to make an if to reduce every img to same size
    width = int(IX.shape[1] * scale_percent / 100)
    height = int(IX.shape[0] * scale_percent / 100)
    xdim = (width, height)
    width = int(IY.shape[1] * scale_percent / 100)
    height = int(IY.shape[0] * scale_percent / 100)
    ydim = (width, height)
    IX = cv2.resize(IX, xdim, interpolation=cv2.INTER_AREA)
    IY = cv2.resize(IY, ydim, interpolation=cv2.INTER_AREA)

    # import and rescale landmarks

    ylm = ylm * scale_percent / 100
    ylm = np.round(ylm)

    # initialize

    reg = Registration.CNN()

    if iterations == 1:

        # Base Method

        X, Y, Z = reg.register(IX, IY)

        # generate registered image using TPS

        registered, nlm = tps_warp(Y, Z, IY, IX.shape, ylm)

    elif iterations == 2:

        # DLMR

        X, Y, Z = reg.register(IX, IY)

        # generate registered image using TPS

        registered0, nlm = tps_warp(Y, Z, IY, IX.shape, ylm)

        # slice image into 4
        SLIX = Image.fromarray(IX)
        SRES = Image.fromarray(registered0)

        [i, j, k, l] = image_slicer.slice(SRES, 4, save=False)
        [m, n, o, q] = image_slicer.slice(SLIX, 4, save=False)

        i = numpy.array(i.image)
        j = numpy.array(j.image)
        k = numpy.array(k.image)
        l = numpy.array(l.image)
        m = numpy.array(m.image)
        n = numpy.array(n.image)
        o = numpy.array(o.image)
        q = numpy.array(q.image)

        xlength = registered0.shape[1] / 2
        ylength = registered0.shape[0] / 2

        # register each image slice

        Y = []
        Z = []

        if blank(i):
            Xi, Yi, Zi = reg.register(m, i)
            Y.append(Yi)
            Z.append(Zi)

        if blank(j):
            Xj, Yj, Zj = reg.register(n, j)
            Yj = Yj + [0, xlength]
            Zj = Zj + [0, xlength]
            Y.append(Yj)
            Z.append(Zj)

        if blank(k):
            Xk, Yk, Zk = reg.register(o, k)
            Yk = Yk + [ylength, 0]
            Zk = Zk + [ylength, 0]
            Y.append(Yk)
            Z.append(Zk)

        if blank(l):
            Xl, Yl, Zl = reg.register(q, l)
            Yl = Yl + [ylength, xlength]
            Zl = Zl + [ylength, xlength]
            Y.append(Yl)
            Z.append(Zl)

        Yf = numpy.concatenate(Y, axis=0)

        Zf = numpy.concatenate(Z, axis=0)

        # interpolate final image

        registered, nlm = tps_warp(Yf, Zf, registered0, IX.shape, nlm)

    elif iterations == 3:

        # 16 x resolution

        X, Y, Z = reg.register(IX, IY)

        # generate registered image using TPS

        registered0, temp1 = tps_warp(Y, Z, IY, IX.shape, ylm)

        # slice image into 4

        SLIX = Image.fromarray(IX)
        SRES = Image.fromarray(registered0)

        [i, j, k, l] = image_slicer.slice(SRES, 4, save=False)
        [m, n, o, q] = image_slicer.slice(SLIX, 4, save=False)

        i = numpy.array(i.image)
        j = numpy.array(j.image)
        k = numpy.array(k.image)
        l = numpy.array(l.image)
        m = numpy.array(m.image)
        n = numpy.array(n.image)
        o = numpy.array(o.image)
        q = numpy.array(q.image)

        # register each image slice

        Xi, Yi, Zi = reg.register(m, i)

        Xj, Yj, Zj = reg.register(n, j)

        Xk, Yk, Zk = reg.register(o, k)

        Xl, Yl, Zl = reg.register(q, l)

        # combine Z of each slice into one

        xlength = registered0.shape[1] / 2
        ylength = registered0.shape[0] / 2

        Yj = Yj + [0, xlength]
        Yk = Yk + [ylength, 0]
        Yl = Yl + [ylength, xlength]
        Zj = Zj + [0, xlength]
        Zk = Zk + [ylength, 0]
        Zl = Zl + [ylength, xlength]

        Yf = numpy.concatenate((Yi, Yj, Yk, Yl), axis=0)

        Zf = numpy.concatenate((Zi, Zj, Zk, Zl), axis=0)

        # interpolate intermediate image

        registered1, nlm = tps_warp(Yf, Zf, registered0, IX.shape, temp1)

        SLIX = Image.fromarray(IX)
        SRES = Image.fromarray(registered1)

        [xa, xb, xc, xd, xe, xf, xg, xh, xi, xj, xk, xl, xm, xn, xo, xp] = image_slicer.slice(SLIX, 16, save=False)
        [ya, yb, yc, yd, ye, yf, yg, yh, yi, yj, yk, yl, ym, yn, yo, yp] = image_slicer.slice(SRES, 16, save=False)

        xa, xb, xc, xd, xe, xf, xg, xh, xi, xj, xk, xl, xm, xn, xo, xp = numpy.array(xa.image), numpy.array(xb.image), \
            numpy.array(xc.image), numpy.array(xd.image), numpy.array(xe.image), numpy.array(xf.image),\
            numpy.array(xg.image),numpy.array(xh.image), numpy.array(xi.image), numpy.array(xj.image), \
            numpy.array(xk.image), numpy.array(xl.image), numpy.array(xm.image), numpy.array(xn.image), \
            numpy.array(xo.image), numpy.array(xp.image)

        ya, yb, yc, yd, ye, yf, yg, yh, yi, yj, yk, yl, ym, yn, yo, yp = numpy.array(ya.image), numpy.array(yb.image), \
            numpy.array(yc.image), numpy.array(yd.image), numpy.array(ye.image), numpy.array(yf.image),\
            numpy.array(yg.image),numpy.array(yh.image), numpy.array(yi.image), numpy.array(yj.image), \
            numpy.array(yk.image), numpy.array(yl.image), numpy.array(ym.image), numpy.array(yn.image), \
            numpy.array(yo.image), numpy.array(yp.image)

        Xa, Ya, Za = reg.register(xa, ya)
        Xb, Yb, Zb = reg.register(xb, yb)
        Xc, Yc, Zc = reg.register(xc, yc)
        Xd, Yd, Zd = reg.register(xd, yd)
        Xe, Ye, Ze = reg.register(xe, ye)
        Xf, Yf, Zf = reg.register(xf, yf)
        Xg, Yg, Zg = reg.register(xg, yg)
        Xh, Yh, Zh = reg.register(xh, yh)
        Xi, Yi, Zi = reg.register(xi, yi)
        Xj, Yj, Zj = reg.register(xj, yj)
        Xk, Yk, Zk = reg.register(xk, yk)
        Xl, Yl, Zl = reg.register(xl, yl)
        Xm, Ym, Zm = reg.register(xm, ym)
        Xn, Yn, Zn = reg.register(xn, yn)
        Xo, Yo, Zo = reg.register(xo, yo)
        Xp, Yp, Zp = reg.register(xp, yp)

        xlength = IX.shape[1] / 4
        ylength = IX.shape[0] / 4

        Yb, Zb = Yb + [0, xlength] , Zb + [0, xlength]
        Yc, Zc = Yc + [0, 2*xlength] , Zc + [0, 2*xlength]
        Yd, Zd = Yd + [0, 3*xlength] , Zd + [0, 3*xlength]
        Ye, Ze = Ye + [ylength, 0] , Ze + [ylength, 0]
        Yf, Zf = Yf + [ylength, xlength] , Zf + [ylength, xlength]
        Yg, Zg = Yg + [ylength, 2*xlength] , Zg + [ylength, 2*xlength]
        Yh, Zh = Yh + [ylength, 3*xlength] , Zh + [ylength, 3*xlength]
        Yi, Zi = Yi + [2*ylength, 0] , Zi + [2*ylength, 0]
        Yj, Zj = Yj + [2*ylength, xlength] , Zj + [2*ylength, xlength]
        Yk, Zk = Yk + [2*ylength, 2*xlength] , Zk + [2*ylength, 2*xlength]
        Yl, Zl = Yl + [2*ylength, 3*xlength] , Zl + [2*ylength, 3*xlength]
        Ym, Zm = Ym + [3*ylength, 0] , Zm + [3*ylength, 0]
        Yn, Zn = Yn + [3*ylength, xlength] , Zn + [3*ylength, xlength]
        Yo, Zo = Yo + [3*ylength, 2*xlength] , Zo + [3*ylength, 2*xlength]
        Yp, Zp = Yp + [3*ylength, 3*xlength] , Zp + [3*ylength, 3*xlength]

        Yr = numpy.concatenate((Ya, Yb, Yc, Yd, Ye, Yf, Yg , Yh, Yi, Yj, Yk, Yl, Ym, Yn, Yo, Yp), axis=0)

        Zr = numpy.concatenate((Za, Zb, Zc, Zd, Ze, Zf, Zg, Zh, Zi, Zj, Zk, Zl, Zm, Zn, Zo, Zp), axis=0)

        registered, nlm = tps_warp(Yr, Zr, registered1, IX.shape, nlm)


    nlm = nlm / (scale_percent / 100)


    return registered, nlm





