import cv2
import matplotlib
import utility
import cannyEdgeDetect as cED

matplotlib.use('TkAgg')

# read in all images for use as single channel
im_1016 = cv2.imread('resources/Fig1016.tif', 0)
im_1001a = cv2.imread('resources/Fig1001(a).tif', 0)
im_1001b = cv2.imread('resources/Fig1001(b).tif', 0)
im_1001c = cv2.imread('resources/Fig1001(c).tif', 0)
im_1001d = cv2.imread('resources/Fig1001(d).tif', 0)
im_1001e = cv2.imread('resources/Fig1001(e).tif', 0)
im_1001f = cv2.imread('resources/Fig1001(f).tif', 0)
im_lenna = cv2.imread('resources/Lenna.png', 0)


# function that actually does the canny edge detection
# takes in the image to be processed, the high threshold, the low threshold, the name for saving, and a bool
# for whether or not to display the images
# retuns a a list of images taken at each step
def action(im, sigma, t_h, t_l, name, display: bool = True):
    im_gauss = utility.gauss_filter(im, sigma)  # run gauss filter on image w/ given sigma
    im_fx, im_fy, im_f, im_d = cED.sobel_filter(im_gauss)  # run sobel filter on smoothed image
    im_thin = cED.nonmax_supress(im_f, im_d)  # get thin image by supressing magnitude and direction
    im_hyst = cED.hysteresis(im_thin, t_h * 255, t_l * 255)  # run hysteresis thresholding on thin image

    # have function display images
    if display:
        # display function to display image sets
        utility.canny_display([im, im_gauss, im_fx, im_fy, im_f, im_d, im_thin,
                               im_hyst], sigma, t_h, t_l)
    # write images to file
    utility.canny_write([im, im_gauss, im_fx, im_fy, im_f, im_d,
                         im_thin, im_hyst], sigma, t_h, t_l, name)

    # return list of images
    return [im, im_gauss, im_fx, im_fy, im_f, im_d, im_thin, im_hyst]


# run all algorithms

a = action(im_lenna, 2, 0.03, 0.10, 'lenna', False)
print("set 1 finished")
b = action(im_lenna, 2, 0.03, 0.15, 'lenna', False)
print("set 2 finished")
c = action(im_lenna, 2, 0.10, 0.20, 'lenna', False)
print("set 3 finished")

d = action(im_1001a, 1, 0.03, 0.10, '1001', False)
print("set 4 finished")
e = action(im_1016, 1, 0.03, 0.10, '1016', False)
print("set 5 finished")
f = action(im_1016, 2, 0.03, 0.10, '1016', False)
print("set 6 finished")
g = action(im_1016, 3, 0.03, 0.10, '1016', False)
print("set 7 finished")
h = action(im_1016, 4, 0.03, 0.10, '1016', False)
print("set 8 finished")
i = action(im_1016, 5, 0.03, 0.010, '1016', False)
print("set 9 finished")

# display all algorithm results
utility.canny_display(a, 2, 0.03, 0.10)
utility.canny_display(b, 2, 0.03, 0.15)
utility.canny_display(c, 2, 0.10, 0.20)
utility.canny_display(d, 1, 0.03, 0.10)
utility.canny_display(e, 1, 0.03, 0.10)
utility.canny_display(f, 2, 0.03, 0.10)
utility.canny_display(g, 3, 0.03, 0.10)
utility.canny_display(h, 4, 0.03, 0.10)
utility.canny_display(i, 5, 0.03, 0.10)
