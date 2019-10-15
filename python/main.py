import cv2
import matplotlib
import utility
import cannyEdgeDetect as cED

matplotlib.use('TkAgg')

im_1016 = cv2.imread('resources/Fig1016.tif', 0)
im_1001a = cv2.imread('resources/Fig1001(a).tif', 0)
im_1001b = cv2.imread('resources/Fig1001(b).tif', 0)
im_1001c = cv2.imread('resources/Fig1001(c).tif', 0)
im_1001d = cv2.imread('resources/Fig1001(d).tif', 0)
im_1001e = cv2.imread('resources/Fig1001(e).tif', 0)
im_1001f = cv2.imread('resources/Fig1001(f).tif', 0)
im_lenna = cv2.imread('resources/Lenna.png', 0)


def action(im, sigma, t_h, t_l, name):
    im_gauss = utility.gauss_filter(im, sigma)
    im_fx, im_fy, im_f, im_d = cED.sobel_filter(im_gauss)
    im_thin = cED.nonmax_supress(im_f, im_d)
    im_hyst = cED.hysteresis(im_thin, t_h * 255, t_l * 255)

    utility.canny_display([im, im_gauss, im_fx, im_fy, im_f, im_d, im_thin,
                           im_hyst], sigma, t_h, t_l)
    utility.canny_write([im, im_gauss, im_fx, im_fy, im_f, im_d,
                         im_thin, im_hyst], sigma, t_h, t_l, name)

    return


action(im_lenna, 1, 0.10, 0.03, 'lenna')
action(im_lenna, 1, 0.15, 0.03, 'lenna')
action(im_lenna, 1, 0.20, 0.10, 'lenna')

action(im_1001a, 1, 0.10, 0.03, '1001')
action(im_1016, 1, 0.15, 0.03, '1016')
# action(im_1016, 2, 0.10, 0.03, '1016')
action(im_1016, 3, 0.10, 0.03, '1016')
# action(im_1016, 4, 0.10, 0.03, '1016')
action(im_1016, 5, 0.10, 0.03, '1016')
