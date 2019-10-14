import math
import numpy as np
import scipy.signal
import utility
from filters import PREWITTX, PREWITTY, SOBELX, SOBELY


def prewitt_filter(im: np.ndarray):
    im, im2 = utility.prepare_image(im, 1, 'repeat')
    fx = scipy.signal.convolve2d(im, PREWITTX, 'valid')
    fy = scipy.signal.convolve2d(im, PREWITTY, 'valid')

    f = np.sqrt(fx**2 + fy**2)
    f = f/np.max(f) * 255
    d = np.arctan2(fy, fx)

    return fx, fy, f, d


def sobel_filter(im: np.ndarray):
    im, im2 = utility.prepare_image(im, 1, 'repeat')
    fx = scipy.signal.convolve2d(im, SOBELX, 'valid')
    fy = scipy.signal.convolve2d(im, SOBELY, 'valid')

    f = np.sqrt(fx**2 + fy**2)
    f = f/np.max(f) * 255
    d = np.arctan2(fy, fx)

    return fx, fy, f, d


def nonmax_supress(im, d):
    # convert d to proper angles
    d = d * 180.0 / np.pi
    im, i = utility.prepare_image(im, 1, 'repeat')
    dimensions = im.shape

    for j in range(1, dimensions[0] - 2):
        for k in range(1, dimensions[1] - 2):
            angle = d[j, k] + 180 if d[j, k] < 0 else d[j, k]  # set negative angles to inverse for easier logic
            n1, n2 = 0, 0  # variables for numbers on each side of given point
            # use angle to find directional maxes at 0, 45, 90, 135
            # 22.5 degree increments
            if(0 <= angle < 22.5) or (157.5 <= angle <= 180):  # 0 degrees or +/- 180
                n1 = im[j, k + 1]
                n2 = im[j, k - 1]
            elif 22.5 <= angle < 67.5:  # 45 degrees or -135
                n1 = im[j + 1, k - 1]
                n2 = im[j - 1, k + 1]
            elif 67.5 <= angle < 112.5:  # 90 degrees or -90
                n1 = im[j + 1, k]
                n2 = im[j - 1, k]
            elif 112.5 <= angle < 157.5:  # 135 degrees or -45
                n1 = im[j - 1, k - 1]
                n2 = im[j + 1, k + 1]
            if (im[j, k] < n1) or (im[j, k] < n2):
                i[j, k] = 0
            else:
                i[j, k] = im[j, k]

    return i


