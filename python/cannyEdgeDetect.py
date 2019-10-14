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
