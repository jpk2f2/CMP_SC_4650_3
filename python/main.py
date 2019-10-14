import cv2
import matplotlib
import utility
import cannyEdgeDetect as ced

matplotlib.use('TkAgg')

im_1016 = cv2.imread('resources/Fig1016.tif', 0)
im_1001a = cv2.imread('resources/Fig1001(a).tif', 0)
im_1001b = cv2.imread('resources/Fig1001(b).tif', 0)
im_1001c = cv2.imread('resources/Fig1001(c).tif', 0)
im_1001d = cv2.imread('resources/Fig1001(d).tif', 0)
im_1001e = cv2.imread('resources/Fig1001(e).tif', 0)
im_1001f = cv2.imread('resources/Fig1001(f).tif', 0)
im_lenna = cv2.imread('resources/Lenna.png', 0)

# code below is test setup for assignment requirements
# should be cleaned up after

# display starting image
cv2.imshow("1", im_lenna)
cv2.waitKey(0)
# apply gauss filter
im_lenna = utility.gauss_filter(im_lenna, 1)
# display gauss filtered image
cv2.imshow("2", im_lenna)
cv2.waitKey(0)

im_lenna_fx, im_lenna_fy, im_lenna_f, im_lenna_d = ced.sobel_filter(im_lenna)
# im_lenna_fx, im_lenna_fy = ced.sobel_filter(im_lenna)

# im_lenna_fx = cv2.normalize(im_lenna_fx, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
# im_lenna_fy = cv2.normalize(im_lenna_fy, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

cv2.imshow('Fx', cv2.normalize(im_lenna_fx, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
cv2.imshow('Fy', cv2.normalize(im_lenna_fy, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
cv2.imshow('F', cv2.normalize(im_lenna_f, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
cv2.imshow('D', cv2.normalize(im_lenna_d, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
cv2.waitKey(0)






